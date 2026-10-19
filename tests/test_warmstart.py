from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from chess_endgame_rl.models.dueling_net import DuelingNet
from chess_endgame_rl.solver.tablebase import CACHE_DIR, Tablebase
from chess_endgame_rl.warmstart.dataset import TablebaseDataset, split_dataset
from chess_endgame_rl.warmstart import train_supervised
from chess_endgame_rl.warmstart.train_supervised import _run_epoch


def test_solver_dataset_has_valid_optimal_targets_and_masks():
    tablebase = Tablebase.load(CACHE_DIR / "kqk.npz")
    dataset = TablebaseDataset(tablebase)

    assert len(dataset) == int((tablebase.dtm[0::2] > 0).sum())
    sample = dataset[0]
    assert sample["state"].shape == (18, 8, 8)
    assert sample["legal_mask"].shape == (4096,)
    assert sample["optimal_mask"].shape == (4096,)
    assert sample["target_distribution"].sum().item() == 1.0
    assert torch.all(sample["optimal_mask"] <= sample["legal_mask"])
    assert torch.all(sample["target_distribution"][~sample["optimal_mask"]] == 0)
    assert torch.all(sample["q_targets"][~sample["legal_mask"]] == 0)
    assert torch.all(sample["q_targets"][sample["optimal_mask"]] > 0)


def test_position_split_is_deterministic_and_disjoint():
    dataset = TablebaseDataset(Tablebase.load(CACHE_DIR / "krk.npz"))
    train_a, validation_a = split_dataset(dataset, validation_fraction=0.1, seed=29)
    train_b, validation_b = split_dataset(dataset, validation_fraction=0.1, seed=29)
    train_ids = set(train_a.indices)
    validation_ids = set(validation_a.indices)

    assert train_ids.isdisjoint(validation_ids)
    assert len(train_a) + len(validation_a) == len(dataset)
    assert validation_a.indices == validation_b.indices
    assert train_a.indices == train_b.indices


def test_dueling_network_uses_legal_only_advantage_mean():
    model = DuelingNet(hidden_dim=32).eval()
    states = torch.zeros((2, 18, 8, 8), dtype=torch.float32)
    legal = torch.zeros((2, 4096), dtype=torch.bool)
    legal[0, [1, 17]] = True
    legal[1, [2, 3, 4]] = True

    with torch.no_grad():
        q_values, values, _ = model(states, legal)

    torch.testing.assert_close(
        (q_values[0, legal[0]].mean(), q_values[1, legal[1]].mean()),
        (values[0, 0], values[1, 0]),
    )
    assert torch.all(q_values[~legal] == -1e9)


def test_supervised_loss_labels_only_legal_moves():
    tablebase = Tablebase.load(CACHE_DIR / "kqk.npz")
    dataset = TablebaseDataset(tablebase)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = DuelingNet(hidden_dim=32).eval()
    q_values, _, _ = model(batch["state"], batch["legal_mask"])

    assert np.isfinite(q_values.detach().numpy()).all()
    assert torch.isfinite(batch["target_distribution"]).all()


def test_supervised_training_step_supports_both_losses():
    dataset = TablebaseDataset(Tablebase.load(CACHE_DIR / "krk.npz"))
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    for loss_type in ("cross_entropy", "q_regression"):
        model = DuelingNet(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = next(model.parameters()).detach().clone()
        metrics = _run_epoch(
            model,
            [batch],
            torch.device("cpu"),
            loss_type,
            optimizer,
        )
        after = next(model.parameters()).detach()

        assert np.isfinite(metrics["loss"])
        assert 0.0 <= metrics["optimal_move_rate"] <= 1.0
        assert not torch.equal(before, after)


def test_supervised_trainer_saves_best_validation_checkpoint(tmp_path, monkeypatch):
    tablebase = Tablebase.load(CACHE_DIR / "krk.npz")
    won_white = tablebase.legal[0::2] & (tablebase.dtm[0::2] > 0)
    state_ids = np.flatnonzero(won_white)[:10] * 2
    monkeypatch.setattr(
        train_supervised,
        "TablebaseDataset",
        lambda loaded: TablebaseDataset(loaded, state_ids=state_ids),
    )
    checkpoint = tmp_path / "warmstart.pt"

    result = train_supervised.train(
        tablebase,
        checkpoint,
        epochs=1,
        batch_size=4,
        seed=7,
    )

    assert checkpoint.exists()
    assert result["best_epoch"] == 1
    assert result["best_validation"]["positions"] == 1.0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert len(saved["validation_state_ids"]) == 1
