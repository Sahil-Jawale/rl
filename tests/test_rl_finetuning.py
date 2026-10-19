from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from chess_endgame_rl.agents.rl_finetuning import train_finetune
from chess_endgame_rl.models.dueling_net import DuelingNet
from chess_endgame_rl.solver.tablebase import CACHE_DIR, Tablebase


def test_finetuning_runs_truncated_transition_and_saves_best_checkpoint(tmp_path):
    tablebase = Tablebase.load(CACHE_DIR / "krk.npz")
    warmstart = tmp_path / "warmstart.pt"
    model = DuelingNet()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"in_channels": 18, "action_dim": 4096},
        },
        warmstart,
    )
    output = tmp_path / "rl.pt"

    result = train_finetune(
        tablebase,
        warmstart,
        output,
        episodes=1,
        max_plies=2,
        batch_size=1,
        eval_every_steps=1,
        eval_positions=2,
        seed=3,
    )

    assert result["environment_steps"] == 1
    assert result["config"]["eval_positions"] == 2
    assert result["config"]["early_stop_patience"] == 5
    assert output.is_file()
    metrics_path = output.with_suffix(".metrics.json")
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["best_validation"][
        "positions"
    ] == 2


def test_finetuning_requires_even_ply_horizon(tmp_path):
    tablebase = Tablebase.load(CACHE_DIR / "krk.npz")
    with pytest.raises(ValueError, match="positive"):
        train_finetune(
            tablebase,
            tmp_path / "missing.pt",
            tmp_path / "out.pt",
            max_plies=3,
        )
