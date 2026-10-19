from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_endgame_rl.eval.evaluate_all import (
    board_from_state,
    evaluate_scenario,
    evaluate_move_quality,
    play_game,
    results_markdown_table,
    run_evaluation,
    summarize_games,
    update_readme_table,
)
from scripts.regression_check import check_regression
from chess_endgame_rl.models.dueling_net import DuelingNet
from chess_endgame_rl.solver.tablebase import CACHE_DIR, Tablebase


def test_evaluation_boards_round_trip_to_tablebase_state():
    tablebase = Tablebase.load(CACHE_DIR / "krk.npz")
    state_id = int(
        next(
            idx
            for idx in range(0, tablebase.dtm.size, 2)
            if tablebase.legal[idx] and tablebase.dtm[idx] > 0
        )
    )

    board = board_from_state(tablebase, state_id)
    assert board.is_valid()
    assert board.turn == chess.WHITE
    assert tablebase.lookup(board)[0] == 1


def test_evaluation_rejects_unknown_scenario_before_loading_models():
    with pytest.raises(ValueError, match="kind must be"):
        run_evaluation(kind="invalid")


def test_game_summary_reports_separate_win_draw_loss_and_win_rate():
    summary = summarize_games(
        [
            {"outcome": "win", "plies": 9, "initial_dtm": 7},
            {"outcome": "draw", "plies": 100, "initial_dtm": 9},
            {"outcome": "loss", "plies": 5, "initial_dtm": 11},
        ]
    )

    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 1, 1)
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["mean_mate_length_to_optimal_dtm"] == pytest.approx(9 / 7)


def test_solver_move_quality_metrics_are_seeded_and_wdl_rates_are_complete():
    tablebase = Tablebase.load(CACHE_DIR / "kqk.npz")
    model = DuelingNet(hidden_dim=32).eval()
    args = {
        "model": model,
        "tablebase": tablebase,
        "sample_size": 4,
        "seed": 31,
        "device": torch.device("cpu"),
    }
    first = evaluate_move_quality(**args)
    second = evaluate_move_quality(**args, batch_size=1)

    assert first == second
    assert first["optimal_move_rate"] + first["suboptimal_move_rate"] == 1.0
    assert 0.0 <= first["win_to_draw_rate"] <= 1.0


def test_random_play_game_returns_valid_wdl_and_horizon_result():
    tablebase = Tablebase.load(CACHE_DIR / "krk.npz")
    start_id = int(
        next(
            idx
            for idx in range(0, tablebase.dtm.size, 2)
            if tablebase.legal[idx] and tablebase.dtm[idx] > 0
        )
    )

    result = play_game(
        tablebase,
        start_id,
        "random",
        model=None,
        seed=1,
        device=torch.device("cpu"),
        max_plies=2,
    )

    assert result["outcome"] in {"win", "draw", "loss"}
    assert result["plies"] <= 2
    assert result["truncated"] or result["plies"] <= 2


def test_full_evaluation_protocol_smoke_uses_shared_heldout_start(tmp_path):
    tablebase = Tablebase.load(CACHE_DIR / "kqk.npz")
    model = DuelingNet(hidden_dim=32).eval()
    heldout_id = int(
        next(
            idx
            for idx in range(0, tablebase.dtm.size, 2)
            if tablebase.legal[idx] and tablebase.dtm[idx] > 0
        )
    )

    result = evaluate_scenario(
        tablebase,
        model=model,
        games_per_defender=1,
        seed=12,
        device=torch.device("cpu"),
        quality_positions=1,
        heldout_state_ids=np.array([heldout_id]),
    )

    assert result["move_quality"]["positions"] == 1
    assert set(result["defenders"]) == {"random", "optimal"}
    for defender_metrics in result["defenders"].values():
        assert defender_metrics["model"]["games"] == 1
        assert defender_metrics["random_baseline"]["games"] == 1


def test_results_table_updates_only_marked_readme_section(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "Before\n<!-- WARMSTART_RESULTS_START -->\nold\n"
        "<!-- WARMSTART_RESULTS_END -->\nAfter\n",
        encoding="utf-8",
    )
    results = {
        "scenarios": {
            "krk": {
                "move_quality": {"optimal_move_rate": 0.7},
                "defenders": {
                    "random": {
                        "model": {
                            "wins": 2, "draws": 1, "losses": 0,
                            "win_rate": 2 / 3, "draw_rate": 1 / 3,
                        },
                        "random_baseline": {
                            "wins": 0, "draws": 3, "losses": 0,
                            "win_rate": 0.0, "draw_rate": 1.0,
                        },
                    }
                },
            }
        }
    }

    table = results_markdown_table(results)
    update_readme_table(readme, table)
    content = readme.read_text(encoding="utf-8")

    assert "| Warm start | krk | random | 2 | 1 | 0 | 0.667 | 0.700 | 0.333 |" in content
    assert content.startswith("Before\n")
    assert content.endswith("After\n")


def test_regression_gate_requires_identical_protocol_and_checks_thresholds():
    baseline = {
        "scenario": "krk",
        "seed": 3,
        "games_per_defender": 150,
        "quality_positions": 10_000,
        "win_rate_vs_optimal": 0.4,
        "optimal_move_rate": 0.8,
    }
    current = {**baseline, "win_rate_vs_optimal": 0.3}

    assert len(check_regression(current, baseline)) == 1
    with pytest.raises(ValueError, match="protocol mismatch"):
        check_regression({**current, "seed": 9}, baseline)
