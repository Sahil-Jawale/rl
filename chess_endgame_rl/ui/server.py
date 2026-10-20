"""FastAPI server for the Chess Endgame RL interactive UI.

Provides endpoints for:
1. Interactive play: User plays Black King, RL agent plays White K+R
2. Training results: Fetch metrics and training curves for comparison
3. Game state management: New game, make moves, agent responses
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_endgame_rl.agents import DeepQNetwork, PPOAgent, TabularQAgent
from chess_endgame_rl.env.encoding import (
    board_to_observation,
    legal_action_mask,
    action_to_move,
    state_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "chess_endgame_rl" / "ui" / "static"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

app = FastAPI(title="Chess Endgame RL — Learning to Checkmate")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ======================================================================
# In-memory game state for active sessions
# ======================================================================
games: dict[str, dict] = {}

# Loaded RL agents (lazy-loaded)
_loaded_agents: dict[str, object] = {}
_loaded_agent_mtimes: dict[str, int] = {}
_loaded_krk_tablebase = None


def _get_agent(agent_name: str):
    """Load a trained RL agent from disk (cached)."""
    model_paths = {
        "Dueling network (tablebase warm-start)": MODELS_DIR / "warmstart" / "krk.pt",
        "Dueling Double DQN (fine-tuned)": MODELS_DIR / "rl" / "krk.pt",
    }
    if agent_name in model_paths:
        path = model_paths[agent_name]
        if not path.is_file():
            return None
        modified_ns = path.stat().st_mtime_ns
        if (
            agent_name in _loaded_agents
            and _loaded_agent_mtimes.get(agent_name) == modified_ns
        ):
            return _loaded_agents[agent_name]

        import torch

        from chess_endgame_rl.models.dueling_net import DuelingNet

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model_config = checkpoint.get("model_config", {})
        agent = DuelingNet(
            in_channels=int(model_config.get("in_channels", 18)),
            action_dim=int(model_config.get("action_dim", 4096)),
            hidden_dim=int(model_config.get("hidden_dim", 256)),
        )
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError(f"Checkpoint {path} has no model_state_dict.")
        agent.load_state_dict(state_dict)
        agent.eval()
        _loaded_agents[agent_name] = agent
        _loaded_agent_mtimes[agent_name] = modified_ns
        return agent

    if agent_name in _loaded_agents:
        return _loaded_agents[agent_name]

    if agent_name == "Q-Learning":
        path = MODELS_DIR / "q_learning.json"
        if path.is_file():
            agent = TabularQAgent.load(path)
            _loaded_agents[agent_name] = agent
            return agent
    elif agent_name == "PPO":
        path = MODELS_DIR / "ppo.npz"
        if path.is_file():
            agent = PPOAgent.load(path)
            _loaded_agents[agent_name] = agent
            return agent
    elif agent_name in ("DQN", "DQN (legacy)"):
        path = MODELS_DIR / "dqn.npz"
        if path.is_file():
            agent = DeepQNetwork.load(path)
            _loaded_agents[agent_name] = agent
            return agent
    return None


def _agent_catalog() -> list[dict[str, str | bool]]:
    """Return the intentionally small, player-facing agent list."""
    definitions = [
        ("DQN", "Legacy DQN baseline", MODELS_DIR / "dqn.npz", "dqn"),
        (
            "Dueling network (tablebase warm-start)",
            "Tablebase-supervised Dueling network",
            MODELS_DIR / "warmstart" / "krk.pt",
            "warmstart",
        ),
        (
            "Dueling Double DQN (fine-tuned)",
            "Warm-start checkpoint fine-tuned with Double DQN",
            MODELS_DIR / "rl" / "krk.pt",
            "rl",
        ),
    ]
    agents = []
    for name, description, path, kind in definitions:
        agents.append(
            {
                "name": name,
                "description": description,
                "trained": path.is_file(),
                "model_path": str(path),
                "kind": kind,
            }
        )
    return agents


def _dueling_move(agent, board: chess.Board) -> tuple[chess.Move, dict]:
    """Select a legal move and report its Q-value and available solver context."""
    import torch

    from chess_endgame_rl.env.encoding import encode_board, index_to_move, legal_mask

    mask = legal_mask(board)
    state = torch.as_tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)
    legal = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.inference_mode():
        q_values, _, _ = agent(state, legal)
    action = int(q_values.argmax(dim=1).item())
    move = index_to_move(action, board)
    if move not in board.legal_moves:
        raise RuntimeError(f"Dueling Double DQN selected illegal action {action}.")
    analysis: dict[str, object] = {"q_value": float(q_values[0, action].item())}

    tablebase_path = PROJECT_ROOT / "data" / "tablebases" / "krk.npz"
    if tablebase_path.is_file():
        tablebase = _krk_tablebase()
        wdl, dtm = tablebase.lookup(board)
        if wdl == 1:
            analysis["solver_dtm"] = dtm
            analysis["optimal_move"] = move in tablebase.optimal_moves(board)
    return move, analysis


def _krk_tablebase():
    global _loaded_krk_tablebase
    from chess_endgame_rl.solver.tablebase import Tablebase

    if _loaded_krk_tablebase is None:
        _loaded_krk_tablebase = Tablebase.load(
            PROJECT_ROOT / "data" / "tablebases" / "krk.npz"
        )
    return _loaded_krk_tablebase


def _evaluate_krk(board: chess.Board, move_count: int = 0) -> dict:
    """Calculate KRK position evaluation and win-probability for UI eval bar."""
    if board.is_checkmate():
        if board.turn == chess.BLACK:
            return {"score": 100.0, "display": "#", "winning": "white", "eval_percent": 100}
        else:
            return {"score": -100.0, "display": "-#", "winning": "black", "eval_percent": 0}

    if board.is_stalemate() or board.is_insufficient_material() or move_count >= 100:
        return {"score": 0.0, "display": "0.0", "winning": "draw", "eval_percent": 50}

    wk_sq = board.king(chess.WHITE)
    bk_sq = board.king(chess.BLACK)
    rooks = list(board.pieces(chess.ROOK, chess.WHITE))

    if not rooks or wk_sq is None or bk_sq is None:
        return {"score": 0.0, "display": "0.0", "winning": "draw", "eval_percent": 50}

    # Black king distance from center (higher = closer to edge = closer to checkmate)
    bk_file = chess.square_file(bk_sq)
    bk_rank = chess.square_rank(bk_sq)
    center_dist = max(abs(bk_file - 3.5), abs(bk_rank - 3.5))

    # Distance between kings
    king_dist = chess.square_distance(wk_sq, bk_sq)

    # Base KRK eval
    score = 5.0 + (center_dist * 1.2) + max(0.0, (7.0 - king_dist) * 0.4)
    if board.is_check():
        score += 1.0

    percent = min(98, max(52, int(50 + score * 4.2)))

    return {
        "score": round(score, 1),
        "display": f"+{score:.1f}",
        "winning": "white",
        "eval_percent": percent,
    }


def _board_to_dict(board: chess.Board, move_count: int = 0) -> dict:
    """Convert board state to JSON-serializable dict."""
    # Get legal moves for the current side
    legal_moves_list = []
    for move in board.legal_moves:
        legal_moves_list.append({
            "uci": move.uci(),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
        })

    # Check game status
    status = "playing"
    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    elif board.is_insufficient_material():
        status = "insufficient_material"
    elif move_count >= 100:  # 50 full moves = 100 half moves
        status = "draw_50_moves"

    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal_moves": legal_moves_list,
        "move_count": move_count,
        "full_moves": move_count // 2,
        "is_check": board.is_check(),
        "status": status,
        "evaluation": _evaluate_krk(board, move_count),
    }


def _generate_krk_position(seed: int | None = None) -> chess.Board:
    """Generate a random legal KRK position."""
    rng = random.Random(seed)
    for _ in range(50_000):
        board = chess.Board(None)
        board.turn = chess.WHITE
        board.clear_board()

        squares = list(range(64))
        rng.shuffle(squares)

        # Place White King, White Rook, Black King
        wk_sq = squares.pop()
        wr_sq = squares.pop()
        bk_sq = squares.pop()

        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wr_sq, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        if not board.is_valid():
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue
        if len(list(board.legal_moves)) == 0:
            continue

        # Kings should be reasonably separated
        king_dist = chess.square_distance(wk_sq, bk_sq)
        if king_dist <= 2:
            continue

        # Rook should not be immediately capturable
        if board.is_attacked_by(chess.BLACK, wr_sq):
            continue

        return board

    raise ValueError("Could not generate a legal KRK position")


# ======================================================================
# API Routes
# ======================================================================

@app.get("/")
def root():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/scenarios")
def list_scenarios_endpoint():
    return {"scenarios": ["krk", "knk"]}


@app.get("/api/agents")
def list_agents():
    """List the DQN agents offered by the play UI and their checkpoint status."""
    agents = _agent_catalog()
    preferred = "Dueling network (tablebase warm-start)"
    secondary = "Dueling Double DQN (fine-tuned)"
    ready_names = [str(agent["name"]) for agent in agents if agent["trained"]]
    default_agent = (
        preferred
        if preferred in ready_names
        else secondary
        if secondary in ready_names
        else "DQN"
        if "DQN" in ready_names
        else ready_names[0]
        if ready_names
        else None
    )
    return {"agents": agents, "default_agent": default_agent}


class NewGameRequest(BaseModel):
    agent_name: str = "DQN"
    seed: int | None = None


@app.post("/api/game/new")
def new_game(req: NewGameRequest):
    """Start a new game against a selected RL agent.
    
    The RL agent plays White (King + Rook).
    The user plays Black (lone King).
    White moves first.
    """
    agent = _get_agent(req.agent_name)

    seed = req.seed if req.seed is not None else random.randint(0, 999999)
    board = _generate_krk_position(seed)
    game_id = f"game_{seed}_{random.randint(0, 9999)}"

    # Agent makes the first move (White)
    move_count = 0
    move_history = []
    agent_move_uci, agent_analysis = _agent_make_move(
        agent, req.agent_name, board, move_count
    )
    if agent_move_uci:
        agent_move = chess.Move.from_uci(agent_move_uci)
        agent_san = board.san(agent_move)
        board.push(agent_move)
        move_count += 1
        move_history.append({
            "side": "white",
            "uci": agent_move_uci,
            "san": agent_san,
            "from": chess.square_name(agent_move.from_square),
            "to": chess.square_name(agent_move.to_square),
            "move_number": 1,
        })

    games[game_id] = {
        "board": board,
        "agent_name": req.agent_name,
        "move_count": move_count,
        "move_history": move_history,
        "agent_analysis": agent_analysis,
    }

    state = _board_to_dict(board, move_count)
    state["game_id"] = game_id
    state["agent_name"] = req.agent_name
    state["agent_move"] = agent_move_uci
    state["agent_analysis"] = agent_analysis
    state["move_history"] = games[game_id]["move_history"]
    return state


class MoveRequest(BaseModel):
    game_id: str
    move_uci: str


@app.post("/api/game/move")
def make_move(req: MoveRequest):
    """User makes a move (Black King), then RL agent responds (White)."""
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[req.game_id]
    board = game["board"]
    agent_name = game["agent_name"]

    # Validate it's Black's turn
    if board.turn != chess.BLACK:
        raise HTTPException(status_code=400, detail="It's not your turn (Black)")

    # Validate and apply user's move
    try:
        move = chess.Move.from_uci(req.move_uci)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid move format: {req.move_uci}")

    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail=f"Illegal move: {req.move_uci}")

    user_san = board.san(move)
    board.push(move)
    game["move_count"] += 1
    game["move_history"].append({
        "side": "black",
        "uci": req.move_uci,
        "san": user_san,
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "move_number": (game["move_count"] + 1) // 2,
    })

    # Check if game is over after user's move
    status = _check_game_status(board, game["move_count"])
    if status != "playing":
        state = _board_to_dict(board, game["move_count"])
        state["game_id"] = req.game_id
        state["agent_name"] = agent_name
        state["agent_move"] = None
        state["user_move"] = req.move_uci
        state["status"] = status
        state["move_history"] = game["move_history"]
        state["agent_analysis"] = game.get("agent_analysis", {})
        return state

    # Agent responds (White move)
    agent = _get_agent(agent_name)
    agent_move_uci, agent_analysis = _agent_make_move(
        agent, agent_name, board, game["move_count"]
    )
    if agent_move_uci:
        agent_move = chess.Move.from_uci(agent_move_uci)
        agent_san = board.san(agent_move)
        board.push(agent_move)
        game["move_count"] += 1
        game["move_history"].append({
            "side": "white",
            "uci": agent_move_uci,
            "san": agent_san,
            "from": chess.square_name(agent_move.from_square),
            "to": chess.square_name(agent_move.to_square),
            "move_number": game["move_count"] // 2,
        })

    game["agent_analysis"] = agent_analysis
    state = _board_to_dict(board, game["move_count"])
    state["game_id"] = req.game_id
    state["agent_name"] = agent_name
    state["agent_move"] = agent_move_uci
    state["agent_analysis"] = agent_analysis
    state["user_move"] = req.move_uci
    state["move_history"] = game["move_history"]
    return state


class HintRequest(BaseModel):
    game_id: str


@app.post("/api/game/hint")
def get_hint(req: HintRequest):
    """Provide a defensive hint for the user (Black King)."""
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[req.game_id]
    board = game["board"]
    if board.turn != chess.BLACK:
        return {"hint_move": None, "explanation": "Wait for the RL agent to move first."}

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {"hint_move": None, "explanation": "No legal moves available."}

    best_move = legal_moves[0]
    best_score = -9999.0
    best_explanation = "Move toward the center to avoid the edge."

    for move in legal_moves:
        if board.is_capture(move):
            return {
                "hint_move": move.uci(),
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "explanation": "Capture the White Rook! This causes immediate insufficient material and secures a draw for you!"
            }

        to_sq = move.to_square
        f = chess.square_file(to_sq)
        r = chess.square_rank(to_sq)
        dist_from_rim = min(f, 7 - f, r, 7 - r)

        board.push(move)
        can_mate = any(board.is_legal(m) and (board.push(m) or True) and board.is_checkmate() and (board.pop() or True) for m in board.legal_moves)
        in_check = board.is_check()
        board.pop()

        score = dist_from_rim * 10.0
        if can_mate:
            score -= 500.0
        if in_check:
            score -= 5.0

        if score > best_score:
            best_score = score
            best_move = move
            if dist_from_rim > 0:
                best_explanation = f"Step into {chess.square_name(to_sq)} (centerward) to stay away from the board border and delay the mate net."
            else:
                best_explanation = f"Step to {chess.square_name(to_sq)} to maintain maximum mobility along the rank/file."

    return {
        "hint_move": best_move.uci(),
        "from": chess.square_name(best_move.from_square),
        "to": chess.square_name(best_move.to_square),
        "explanation": best_explanation,
    }


class ResignRequest(BaseModel):
    game_id: str


@app.post("/api/game/resign")
def resign_game(req: ResignRequest):
    """User concedes the game."""
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[req.game_id]
    board = game["board"]
    state = _board_to_dict(board, game["move_count"])
    state["status"] = "resigned"
    state["game_id"] = req.game_id
    state["agent_name"] = game["agent_name"]
    state["move_history"] = game["move_history"]
    state["agent_analysis"] = game.get("agent_analysis", {})
    return state


def _check_game_status(board: chess.Board, move_count: int) -> str:
    """Check if the game has ended."""
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if move_count >= 100:  # 50 full moves
        return "draw_50_moves"
    if board.is_insufficient_material():
        return "insufficient_material"
    return "playing"


def _agent_make_move(
    agent, agent_name: str, board: chess.Board, move_count: int = 0
) -> tuple[str | None, dict]:
    """Have the RL agent select and return a move in UCI format."""
    if board.turn != chess.WHITE:
        return None, {}

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, {}
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"{agent_name} model is unavailable. Train the model before starting a game.",
        )

    analysis: dict = {}
    if agent_name in (
        "Dueling network (tablebase warm-start)",
        "Dueling Double DQN (fine-tuned)",
    ):
        move, analysis = _dueling_move(agent, board)
        return move.uci(), analysis
    if agent_name == "Q-Learning":
        action_id = agent.select_action_greedy(
            state_index(board), legal_action_mask(board, "krk")
        )
    elif agent_name in ("DQN", "DQN (legacy)", "PPO"):
        obs = board_to_observation(board, "krk")
        move_frac = np.array([move_count / 100.0], dtype=np.float32)
        full_obs = np.concatenate([obs.astype(np.float32), move_frac])
        mask = legal_action_mask(board, "krk")
        if agent_name in ("DQN", "DQN (legacy)"):
            action_id = agent.predict_greedy(full_obs, mask)
        else:
            action_id = agent.select_action_greedy(full_obs, mask)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported agent '{agent_name}'.")

    move = action_to_move(board, action_id)
    if move is None or move not in legal_moves:
        raise RuntimeError(f"{agent_name} selected an invalid action: {action_id}")
    return move.uci(), analysis


# ======================================================================
# Results & Comparison API
# ======================================================================

@app.get("/api/results")
def get_results():
    """Get training results for all agents as a list."""
    results = []
    for algo in ["q_learning", "dqn", "ppo"]:
        path = RESULTS_DIR / f"training_{algo}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
    return results


@app.get("/api/results/{agent_name}")
def get_agent_results(agent_name: str):
    """Get training results for a specific agent."""
    # Normalize name
    name_map = {
        "q-learning": "q_learning",
        "q_learning": "q_learning",
        "dqn": "dqn",
        "ppo": "ppo",
    }
    normalized = name_map.get(agent_name.lower())
    if not normalized:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")

    path = RESULTS_DIR / f"training_{normalized}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No training results found")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/comparison")
def get_comparison():
    """Return same-protocol evaluation data for all selectable agents."""
    solver_models = []
    evaluation_path = RESULTS_DIR / "results.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.is_file()
        else {}
    )
    scenario_evaluation = evaluation.get("scenarios", {}).get("krk", {})
    full_evaluation = scenario_evaluation.get("defenders", {})
    dqn_path = MODELS_DIR / "dqn.npz"
    dqn_evaluation = scenario_evaluation.get("defenders", {})
    solver_models.append(
        {
            "name": "DQN",
            "checkpoint_available": dqn_path.is_file(),
            "training": {},
            "validation": None,
            "evaluation": {
                "move_quality": None,
                "vs_random": dqn_evaluation.get("random", {}).get("dqn_baseline"),
                "vs_optimal": dqn_evaluation.get("optimal", {}).get("dqn_baseline"),
                "protocol_version": evaluation.get("protocol_version"),
            },
        }
    )
    for name, checkpoint, run_metrics, evaluation_key in [
        (
            "Dueling network (tablebase warm-start)",
            MODELS_DIR / "warmstart" / "krk.pt",
            MODELS_DIR / "warmstart" / "krk.metrics.json",
            "model",
        ),
        (
            "Dueling Double DQN (fine-tuned)",
            MODELS_DIR / "rl" / "krk.pt",
            MODELS_DIR / "rl" / "krk.metrics.json",
            "rl_model",
        ),
    ]:
        training = (
            json.loads(run_metrics.read_text(encoding="utf-8"))
            if run_metrics.is_file()
            else {}
        )
        validation = training.get("best_validation")
        if validation is None and checkpoint.is_file():
            import torch

            checkpoint_data = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            validation = checkpoint_data.get(
                "validation_metrics", checkpoint_data.get("evaluation_metrics")
            )
        solver_models.append(
            {
                "name": name,
                "checkpoint_available": checkpoint.is_file(),
                "training": training,
                "validation": validation,
                "evaluation": {
                    "move_quality": scenario_evaluation.get(
                        "rl_move_quality" if evaluation_key == "rl_model" else "move_quality"
                    ),
                    "vs_random": full_evaluation.get("random", {}).get(evaluation_key),
                    "vs_optimal": full_evaluation.get("optimal", {}).get(evaluation_key),
                    "protocol_version": evaluation.get("protocol_version"),
                },
            }
        )
    return {"solver_models": solver_models}


@app.get("/api/summary")
def get_summary():
    """Get training summary."""
    path = RESULTS_DIR / "training_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
