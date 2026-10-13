from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chess_endgame_rl.env import EndgameEnv
from chess_endgame_rl.env.scenarios import SCENARIOS
from chess_endgame_rl.eval.suite import tablebase_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "chess_endgame_rl" / "ui" / "static"
RESULTS_PATH = PROJECT_ROOT / "results" / "benchmark_summary.json"

app = FastAPI(title="Chess Endgame RL")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/scenarios")
def scenarios():
    return {"scenarios": list(SCENARIOS.keys())}


@app.get("/agents")
def agents():
    return {
        "agents": [
            "Tabular Q-Learning",
            "DQN",
            "PPO",
            "Random",
            "Greedy",
            "Tablebase-optimal",
        ]
    }


@app.get("/benchmark")
def benchmark():
    return {
        "status": "ready",
        "tablebase": tablebase_status(),
        "scenarios": {
            key: {
                "move_cap": value.get("move_cap", 0),
                "action_count": value.get("action_count", 0),
                "white_pieces": value.get("white_pieces", []),
                "black_pieces": value.get("black_pieces", []),
            }
            for key, value in SCENARIOS.items()
        },
    }


@app.get("/api/scenarios")
def api_scenarios():
    return scenarios()


@app.get("/api/agents")
def api_agents():
    return agents()


@app.get("/api/results")
def api_results():
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return []


@app.get("/api/status")
def api_status():
    return {
        "tablebase": tablebase_status(),
        "benchmark_ready": RESULTS_PATH.exists(),
    }


@app.get("/api/fen")
def api_fen(scenario: str = "krk"):
    env = EndgameEnv(scenario_id=scenario, defender="random", seed=7)
    return {"scenario": scenario, "fen": env.render()}


@app.get("/api/compare")
def api_compare(scenario: str = "krk"):
    seeds = [7, 11, 13]
    labels = ["Q-Learning", "DQN", "PPO"]
    return {
        "scenario": scenario,
        "tablebase": tablebase_status(),
        "positions": [
            {
                "label": label,
                "fen": EndgameEnv(scenario_id=scenario, defender="random", seed=seed).render(),
            }
            for label, seed in zip(labels, seeds)
        ],
    }
