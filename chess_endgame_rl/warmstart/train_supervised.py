from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Iterable

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chess_endgame_rl.models.dueling_net import DuelingNet
from chess_endgame_rl.seed import set_seed
from chess_endgame_rl.solver.tablebase import (
    Tablebase,
    existing_tablebase_cache_path,
)
from chess_endgame_rl.warmstart.dataset import TablebaseDataset, split_dataset

ROOT = Path(__file__).resolve().parents[2]


def _batch_loss(
    q_values: torch.Tensor,
    batch: dict[str, torch.Tensor],
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "cross_entropy":
        return -(
            batch["target_distribution"] * F.log_softmax(q_values, dim=1)
        ).sum(dim=1).mean()

    mask = batch["legal_mask"].to(dtype=q_values.dtype)
    squared_error = (q_values - batch["q_targets"]).square() * mask
    return squared_error.sum() / mask.sum().clamp(min=1.0)


def _run_epoch(
    model: DuelingNet,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    optimal = 0
    keeps_win = 0
    win_to_draw = 0

    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            q_values, _, _ = model(batch["state"], batch["legal_mask"])
            loss = _batch_loss(q_values, batch, loss_type)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        chosen = q_values.argmax(dim=1)
        optimal_hits = batch["optimal_mask"].gather(1, chosen[:, None]).squeeze(1)
        chosen_target = batch["q_targets"].gather(1, chosen[:, None]).squeeze(1)
        batch_size = batch["state"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        count += batch_size
        optimal += int(optimal_hits.sum())
        keeps_win += int((chosen_target > 0).sum())
        win_to_draw += int((chosen_target == 0).sum())

    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {
        "loss": total_loss / count,
        "optimal_move_rate": optimal / count,
        "keeps_win_rate": keeps_win / count,
        "win_to_draw_rate": win_to_draw / count,
        "positions": float(count),
    }


def train(
    tablebase: Tablebase,
    output_path: Path,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    validation_fraction: float = 0.1,
    seed: int = 0,
    loss_type: str = "cross_entropy",
    device_name: str = "cpu",
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if loss_type not in {"cross_entropy", "q_regression"}:
        raise ValueError("loss_type must be 'cross_entropy' or 'q_regression'.")

    started_at = time.perf_counter()
    set_seed(seed)
    device = torch.device(device_name)
    dataset = TablebaseDataset(tablebase)
    train_set, validation_set = split_dataset(
        dataset, validation_fraction=validation_fraction, seed=seed
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = DuelingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, loss_type, optimizer)
        validation_metrics = _run_epoch(model, validation_loader, device, loss_type)
        record = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)

        improved = (
            best_metrics is None
            or validation_metrics["optimal_move_rate"] > best_metrics["optimal_move_rate"]
            or (
                validation_metrics["optimal_move_rate"] == best_metrics["optimal_move_rate"]
                and validation_metrics["loss"] < best_metrics["loss"]
            )
        )
        if improved:
            best_metrics = validation_metrics
            best_epoch = epoch
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {"in_channels": 18, "action_dim": 4096},
                    "piece_type": tablebase.piece_type,
                    "loss_type": loss_type,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_state_ids": torch.as_tensor(
                        dataset.state_ids[np.asarray(validation_set.indices)],
                        dtype=torch.int64,
                    ),
                    "validation_metrics": validation_metrics,
                },
                output_path,
            )

    result = {
        "output": str(output_path),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "config": {
            "piece_type": tablebase.piece_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "loss_type": loss_type,
            "device": str(device),
        },
        "runtime_seconds": time.perf_counter() - started_at,
        "hardware": platform.platform(),
        "torch_version": torch.__version__,
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised warm start from exact tablebase labels.")
    parser.add_argument("--kind", choices=("KQvK", "KRvK"), default="KRvK")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--loss", choices=("cross_entropy", "q_regression"), default="cross_entropy"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    kind = args.kind.upper()
    piece_type = chess.QUEEN if kind == "KQVK" else chess.ROOK
    cache_path = existing_tablebase_cache_path(piece_type)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {kind} tablebase at {cache_path}; run scripts/build_tablebase.py "
            f"--kind {kind} first."
        )
    model_name = "kqk" if piece_type == chess.QUEEN else "krk"
    output_path = args.output or ROOT / "models" / "warmstart" / f"{model_name}.pt"
    result = train(
        Tablebase.load(cache_path),
        output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        loss_type=args.loss,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
