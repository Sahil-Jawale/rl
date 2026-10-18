"""Tabular Q-Learning Agent for KRK chess endgame.

Q-Learning is a model-free, off-policy RL algorithm that learns a
state-action value function Q(s, a) stored in a lookup table.

Key features:
- ε-greedy exploration with decay
- Legal action masking to ensure only valid chess moves are selected
- Compact state representation using board indexing (64³ = 262,144 states)

Hyperparameters:
- α (learning rate): Controls how fast old values are updated
- γ (discount factor): Weights future rewards vs immediate rewards
- ε (exploration rate): Probability of random action, decayed over time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TabularQAgent:
    """Q-Learning agent with tabular state-action values.
    
    Uses a Q-table of shape (state_count, action_count) to store
    estimated values for each state-action pair.
    """

    def __init__(
        self,
        state_count: int = 262144,  # 64 * 64 * 64
        action_count: int = 36,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ):
        self.state_count = state_count
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Sparse representation: only store visited states
        self._q_values: dict[int, np.ndarray] = {}

    def _get_q_row(self, state_index: int) -> np.ndarray:
        """Get Q-values for a state, initializing if not yet visited."""
        if state_index not in self._q_values:
            self._q_values[state_index] = np.zeros(self.action_count, dtype=np.float64)
        return self._q_values[state_index]

    def select_action(self, state_index: int, legal_mask: np.ndarray | None = None) -> int:
        """Select an action using ε-greedy policy with legal action masking.
        
        Args:
            state_index: Integer index of the current state.
            legal_mask: Binary array where 1 = legal action, 0 = illegal.
        
        Returns:
            The selected action ID.
        """
        if legal_mask is None:
            legal_mask = np.ones(self.action_count, dtype=np.int8)
        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            raise ValueError("No legal actions available")

        # ε-greedy: explore with probability ε
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal_actions))

        # Exploit: pick the action with highest Q-value among legal actions
        q_values = self._get_q_row(state_index)
        action_values = q_values[legal_actions]
        best_index = int(np.argmax(action_values))
        return int(legal_actions[best_index])

    def select_action_greedy(self, state_index: int, legal_mask: np.ndarray | None = None) -> int:
        """Select the greedy (best) action without exploration."""
        if legal_mask is None:
            legal_mask = np.ones(self.action_count, dtype=np.int8)
        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            raise ValueError("No legal actions available")
        q_values = self._get_q_row(state_index)
        action_values = q_values[legal_actions]
        best_index = int(np.argmax(action_values))
        return int(legal_actions[best_index])

    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_legal_mask: np.ndarray | None = None,
    ):
        """Update Q-value using the Bellman equation.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        """
        q_row = self._get_q_row(state_index)

        if done:
            target = reward
        else:
            if next_legal_mask is None:
                next_legal_mask = np.ones(self.action_count, dtype=np.int8)
            next_legal = np.flatnonzero(next_legal_mask)
            if next_legal.size == 0:
                target = reward
            else:
                next_q = self._get_q_row(next_state)
                target = reward + self.gamma * np.max(next_q[next_legal])

        q_row[action] += self.alpha * (target - q_row[action])

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path):
        """Save the Q-table to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "state_count": self.state_count,
            "action_count": self.action_count,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_values": {
                str(k): v.tolist() for k, v in self._q_values.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabularQAgent":
        """Load a Q-table from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent = cls(
            state_count=data["state_count"],
            action_count=data["action_count"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data.get("epsilon_min", 0.05),  # Use min epsilon for inference
            epsilon_min=data.get("epsilon_min", 0.05),
            epsilon_decay=data.get("epsilon_decay", 0.9995),
        )
        agent._q_values = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in data.get("q_values", {}).items()
        }
        return agent
