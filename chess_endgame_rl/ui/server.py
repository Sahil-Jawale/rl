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
    move_to_action_id,
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


def _get_agent(agent_name: str):
    """Load a trained RL agent from disk (cached)."""
    if agent_name in _loaded_agents:
        return _loaded_agents[agent_name]

    if agent_name == "Q-Learning":
        path = MODELS_DIR / "q_learning.json"
        if path.exists():
            agent = TabularQAgent.load(path)
            _loaded_agents[agent_name] = agent
            return agent
    elif agent_name == "DQN":
        path = MODELS_DIR / "dqn.npz"
        if path.exists():
            agent = DeepQNetwork.load(path)
            _loaded_agents[agent_name] = agent
            return agent
    elif agent_name == "PPO":
        path = MODELS_DIR / "ppo.npz"
        if path.exists():
            agent = PPOAgent.load(path)
            _loaded_agents[agent_name] = agent
            return agent

    return None


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
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/scenarios")
def list_scenarios_endpoint():
    return {"scenarios": ["krk", "knk"]}


@app.get("/api/agents")
def list_agents():
    """List available RL agents and their training status."""
    agents = []
    for name, path in [
        ("Q-Learning", MODELS_DIR / "q_learning.json"),
        ("DQN", MODELS_DIR / "dqn.npz"),
        ("PPO", MODELS_DIR / "ppo.npz"),
    ]:
        agents.append({
            "name": name,
            "trained": path.exists(),
            "model_path": str(path),
        })
    return {"agents": agents}


class NewGameRequest(BaseModel):
    agent_name: str = "Q-Learning"
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
    agent_move_uci = _agent_make_move(agent, req.agent_name, board)
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
    }

    state = _board_to_dict(board, move_count)
    state["game_id"] = game_id
    state["agent_name"] = req.agent_name
    state["agent_move"] = agent_move_uci
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
        return state

    # Agent responds (White move)
    agent = _get_agent(agent_name)
    agent_move_uci = _agent_make_move(agent, agent_name, board)
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

    state = _board_to_dict(board, game["move_count"])
    state["game_id"] = req.game_id
    state["agent_name"] = agent_name
    state["agent_move"] = agent_move_uci
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


def _agent_make_move(agent, agent_name: str, board: chess.Board) -> str | None:
    """Have the RL agent select and return a move in UCI format."""
    if agent is None or board.turn != chess.WHITE:
        return None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    try:
        if agent_name == "Q-Learning":
            si = state_index(board)
            mask = legal_action_mask(board, "krk")
            action_id = agent.select_action_greedy(si, mask)
            move = action_to_move(board, action_id)
            if move and move in legal_moves:
                return move.uci()
        elif agent_name in ("DQN", "PPO"):
            obs = board_to_observation(board, "krk")
            move_frac = np.array([0.0], dtype=np.float32)  # Unknown move count
            full_obs = np.concatenate([obs.astype(np.float32), move_frac])
            mask = legal_action_mask(board, "krk")

            if agent_name == "DQN":
                action_id = agent.predict_greedy(full_obs, mask)
            else:
                action_id = agent.select_action_greedy(full_obs, mask)

            move = action_to_move(board, action_id)
            if move and move in legal_moves:
                return move.uci()
    except Exception:
        pass

    # Fallback: pick a random legal move
    return random.choice(legal_moves).uci()


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
    """Get comparative analysis data for all agents."""
    comparison = {}
    for algo, display_name in [("q_learning", "Q-Learning"), ("dqn", "DQN"), ("ppo", "PPO")]:
        path = RESULTS_DIR / f"training_{algo}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                comparison[display_name] = {
                    "metrics": data.get("metrics", {}),
                    "training_time": data.get("training_time", 0),
                    "episodes": data.get("episodes", 0),
                    "rewards_history": data.get("rewards_history", []),
                    "outcomes_history": data.get("outcomes_history", []),
                    "move_counts_history": data.get("move_counts_history", []),
                }
    return comparison


@app.get("/api/summary")
def get_summary():
    """Get training summary."""
    path = RESULTS_DIR / "training_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
