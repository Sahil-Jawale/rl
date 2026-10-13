from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, legal_mask):
        record = (state.copy(), int(action), float(reward), next_state.copy(), bool(done), np.asarray(legal_mask, dtype=np.int8).copy())
        if len(self.buffer) < self.capacity:
            self.buffer.append(record)
        else:
            self.buffer[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        if batch_size == 0:
            raise ValueError("Replay buffer is empty")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states = np.stack([item[0] for item in batch])
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float64)
        next_states = np.stack([item[3] for item in batch])
        dones = np.array([item[4] for item in batch], dtype=np.float64)
        legal_masks = np.stack([item[5] for item in batch])
        return states, actions, rewards, next_states, dones, legal_masks

    def __len__(self):
        return len(self.buffer)


class DeepQNetwork:
    """A compact NumPy-based DQN implementation with replay-buffer updates and a target network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, learning_rate: float = 1e-3, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995, replay_capacity: int = 4096, batch_size: int = 32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.target_update_interval = 25
        self.steps_since_target_update = 0

        rng = np.random.default_rng(0)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

    def _forward(self, state, target: bool = False):
        x = np.asarray(state, dtype=np.float64)
        if target:
            hidden = x @ self.target_w1 + self.target_b1
            q_values = hidden @ self.target_w2 + self.target_b2
        else:
            hidden = x @ self.w1 + self.b1
            q_values = hidden @ self.w2 + self.b2
        return np.maximum(hidden, 0.0), q_values

    def _forward_batch(self, states, target: bool = False):
        hidden, q_values = self._forward(states, target=target)
        return hidden, q_values

    def _mask_q_values(self, q_values, legal_mask):
        if legal_mask is None:
            return q_values
        masked = np.array(q_values, dtype=np.float64, copy=True)
        masked[legal_mask == 0] = -1e9
        return masked

    def _sync_target_network(self):
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

    def predict(self, state, legal_mask=None):
        _, q_values = self._forward(state, target=False)
        masked = self._mask_q_values(q_values, legal_mask)
        if np.random.rand() < self.epsilon:
            legal_actions = np.flatnonzero(legal_mask) if legal_mask is not None else np.arange(self.output_dim)
            if legal_actions.size == 0:
                return 0
            return int(np.random.choice(legal_actions))
        return int(np.argmax(masked))

    def update(self, state, action, reward, next_state, done, legal_mask=None):
        self.replay_buffer.push(state, action, reward, next_state, done, legal_mask)
        if len(self.replay_buffer) < max(8, self.batch_size):
            return

        states, actions, rewards, next_states, dones, legal_masks = self.replay_buffer.sample(self.batch_size)
        hidden, current_q = self._forward_batch(states, target=False)
        _, next_q = self._forward_batch(next_states, target=True)

        target_q = current_q.copy()
        for idx in range(len(states)):
            masked_next = self._mask_q_values(next_q[idx], legal_masks[idx])
            target_value = rewards[idx] + (0.0 if dones[idx] else self.gamma * np.max(masked_next))
            target_q[idx, actions[idx]] = target_value

        grad_out = (2.0 / len(states)) * (current_q - target_q)
        grad_w2 = hidden.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_hidden = grad_out @ self.w2.T
        grad_hidden[hidden <= 0.0] = 0.0
        grad_w1 = states.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)

        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_interval:
            self._sync_target_network()
            self.steps_since_target_update = 0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
