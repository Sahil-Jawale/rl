from __future__ import annotations

import numpy as np


class TabularQAgent:
    def __init__(self, state_count: int, action_count: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0):
        self.state_count = state_count
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_count, action_count), dtype=np.float64)

    def select_action(self, state_index: int, legal_mask: np.ndarray | None = None) -> int:
        if legal_mask is None:
            legal_mask = np.ones(self.action_count, dtype=np.int8)
        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            raise ValueError("No legal actions available for the current state")
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal_actions))
        action_values = self.q_table[state_index, legal_actions]
        best_index = int(np.argmax(action_values))
        return int(legal_actions[best_index])

    def update(self, state_index: int, action: int, reward: float, next_state: int, done: bool, legal_mask: np.ndarray | None = None):
        if legal_mask is None:
            legal_mask = np.ones(self.action_count, dtype=np.int8)
        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            return

        target = reward
        if not done:
            next_action_values = self.q_table[next_state, legal_actions]
            target += self.gamma * np.max(next_action_values)
        self.q_table[state_index, action] += self.alpha * (target - self.q_table[state_index, action])

    def decay_epsilon(self, epsilon_min: float = 0.05, decay_rate: float = 0.9995):
        self.epsilon = max(epsilon_min, self.epsilon * decay_rate)

    def train_episode(self, env, max_steps: int = 50):
        state = env.reset()
        legal_mask = env.legal_action_mask()
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.select_action(state, legal_mask)
            next_state, reward, done, info = env.step(action)
            next_mask = env.legal_action_mask()
            self.update(state, action, reward, next_state, done, legal_mask)
            total_reward += reward
            state = next_state
            legal_mask = next_mask
            if done:
                break
        return total_reward
