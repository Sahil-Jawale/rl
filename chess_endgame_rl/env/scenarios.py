from __future__ import annotations

from pathlib import Path

import chess
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenarios.yaml"

_PIECE_MAP = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
    "P": chess.PAWN,
}


def _load_scenarios() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("scenarios", {})


SCENARIOS = _load_scenarios()


def normalize_scenario_id(scenario_id: str | None) -> str:
    return (scenario_id or "").lower().strip()


def list_scenarios() -> list[str]:
    return list(SCENARIOS)


def scenario_piece_types(scenario_id: str | None, color: chess.Color) -> list[int]:
    key = normalize_scenario_id(scenario_id)
    scenario = SCENARIOS.get(key, {})
    label_key = "white_pieces" if color == chess.WHITE else "black_pieces"
    pieces = scenario.get(label_key, [])
    if not pieces:
        return []
    return [_PIECE_MAP[str(piece).upper()] for piece in pieces]


def get_scenario(scenario_id: str) -> dict:
    scenario_key = normalize_scenario_id(scenario_id)
    if scenario_key not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario '{scenario_id}'. Valid scenarios: {valid}")
    return SCENARIOS[scenario_key]
