from __future__ import annotations

import chess
import torch
from fastapi.testclient import TestClient

from chess_endgame_rl.models.dueling_net import DuelingNet
from chess_endgame_rl.ui import server
from chess_endgame_rl.ui.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_root_disables_html_caching_and_uses_versioned_ui_assets():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "/static/app.js?v=20261020-draw-reasons" in response.text
    assert "Dueling network warm-start" in response.text
    assert "Non-terminal step" in response.text
    assert "Agent Outcomes vs Optimal Defender" in response.text
    assert 'id="chart-agent-outcomes"' in response.text
    assert "rook captures, repetitions" in response.text
    assert "Zero losses are omitted" in response.text
    assert "Win / D / L" not in response.text
    assert "Q-Learning" not in response.text
    assert "PPO" not in response.text
    assert "Reward-trained baseline" in response.text
    assert "DQN learns from reward feedback alone" in response.text
    assert "compact 193-value input" in response.text
    assert "36 compact actions" in response.text
    assert "copying the tablebase" in response.text


def test_scenarios_endpoint():
    response = client.get('/api/scenarios')
    assert response.status_code == 200
    data = response.json()
    assert 'scenarios' in data
    assert 'krk' in data['scenarios']


def test_results_endpoint_returns_json_list():
    response = client.get('/api/results')
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_agent_catalog_marks_only_existing_checkpoints_available(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODELS_DIR", tmp_path)
    (tmp_path / "dqn.npz").write_bytes(b"legacy")
    response = client.get("/api/agents")

    assert response.status_code == 200
    data = response.json()
    agents = {agent["name"]: agent for agent in data["agents"]}
    assert agents["DQN"]["trained"] is True
    assert agents["Dueling network (tablebase warm-start)"]["trained"] is False
    assert data["default_agent"] == "DQN"

    warmstart = tmp_path / "warmstart"
    warmstart.mkdir()
    (warmstart / "krk.pt").write_bytes(b"checkpoint")
    assert client.get("/api/agents").json()["default_agent"] == (
        "Dueling network (tablebase warm-start)"
    )

    (tmp_path / "rl").mkdir()
    (tmp_path / "rl" / "krk.pt").write_bytes(b"fine-tuned checkpoint")
    assert client.get("/api/agents").json()["default_agent"] == (
        "Dueling network (tablebase warm-start)"
    )


def test_dueling_ui_inference_returns_a_legal_move_with_q_value(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(server, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(server, "_loaded_agents", {})
    monkeypatch.setattr(server, "_loaded_agent_mtimes", {})
    checkpoint_path = tmp_path / "models" / "warmstart" / "krk.pt"
    checkpoint_path.parent.mkdir(parents=True)
    model = DuelingNet().eval()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"in_channels": 18, "action_dim": 4096},
        },
        checkpoint_path,
    )
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")

    loaded_model = server._get_agent("Dueling network (tablebase warm-start)")
    assert isinstance(loaded_model, DuelingNet)
    move, analysis = server._dueling_move(loaded_model, board)

    assert move in board.legal_moves
    assert isinstance(analysis["q_value"], float)


def test_comparison_endpoint_reports_solver_model_rows():
    response = client.get("/api/comparison")

    assert response.status_code == 200
    comparison = response.json()
    models = comparison["solver_models"]
    assert [model["name"] for model in models] == [
        "DQN",
        "Dueling network (tablebase warm-start)",
        "Dueling Double DQN (fine-tuned)",
    ]
    assert all("checkpoint_available" in model for model in models)
    assert all("vs_random" in model["evaluation"] for model in models)
    dqn, warmstart, fine_tuned = models
    assert dqn["evaluation"]["vs_optimal"]["games"] == 150
    assert dqn["evaluation"]["vs_random"]["outcomes"] == {
        "wins": 0,
        "losses": 0,
        "rook_captures": 73,
        "repetitions": 42,
        "horizon_draws": 35,
        "other_draws": 0,
    }
    assert dqn["evaluation"]["vs_optimal"]["outcomes"]["rook_captures"] == 129
    assert dqn["evaluation"]["vs_optimal"]["outcomes"]["repetitions"] == 21
    assert warmstart["evaluation"]["vs_optimal"]["games"] == 150
    assert fine_tuned["evaluation"]["vs_optimal"]["games"] == 150
    assert warmstart["evaluation"]["move_quality"]["positions"] == 10000
    assert warmstart["evaluation"]["move_quality"]["optimal_move_rate"] == 0.9406
    assert warmstart["evaluation"]["vs_optimal"]["wins"] == 124
    assert warmstart["evaluation"]["vs_optimal"]["outcomes"]["rook_captures"] == 19
    assert fine_tuned["evaluation"]["move_quality"]["optimal_move_rate"] == 0.9145
    assert fine_tuned["evaluation"]["vs_optimal"]["wins"] == 84
    assert set(comparison) == {"solver_models"}


def test_comparison_uses_checkpoint_validation_metrics_without_sidecar(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path / "results")
    checkpoint_path = (
        tmp_path / "models" / "warmstart" / "krk.pt"
    )
    checkpoint_path.parent.mkdir(parents=True)
    validation_metrics = {
        "optimal_move_rate": 0.9,
        "keeps_win_rate": 0.98,
        "win_to_draw_rate": 0.02,
    }
    torch.save({"validation_metrics": validation_metrics}, checkpoint_path)

    response = client.get("/api/comparison")

    assert response.status_code == 200
    models = response.json()["solver_models"]
    assert models[1]["validation"] == validation_metrics
