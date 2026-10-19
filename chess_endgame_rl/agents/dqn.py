"""Deep Q-Network (DQN) Agent for KRK chess endgame.

DQN extends Q-Learning by using a neural network to approximate Q(s, a),
enabling generalization across similar board states instead of memorizing
each one individually.

Key features:
- Two-layer neural network with ReLU activation (NumPy-based, no PyTorch needed)
- Experience replay buffer for stable training
- Target network with periodic sync for training stability
- Legal action masking
- ε-greedy exploration with decay

Architecture:
  Input (193) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(36, Linear)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ReplayBuffer:
    """Circular experience replay buffer for DQN training.
    
    Stores (state, action, reward, next_state, done, legal_mask) transitions
    and provides random mini-batch sampling for stable gradient updates.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: list[tuple] = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, legal_mask):
        """Add a transition to the buffer."""
        record = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
            np.asarray(legal_mask, dtype=np.int8).copy(),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(record)
        else:
            self.buffer[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample a random mini-batch from the buffer."""
        actual_size = min(batch_size, len(self.buffer))
        if actual_size == 0:
            raise ValueError("Replay buffer is empty")
        indices = np.random.choice(len(self.buffer), size=actual_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states = np.stack([item[0] for item in batch])
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch])
        dones = np.array([item[4] for item in batch], dtype=np.float32)
        legal_masks = np.stack([item[5] for item in batch])
        return states, actions, rewards, next_states, dones, legal_masks

    def __len__(self):
        return len(self.buffer)


class DeepQNetwork:
    """NumPy-based DQN with two hidden layers, target network, and replay buffer.
    
    This implementation uses pure NumPy for the neural network, making it
    dependency-free (no PyTorch/TensorFlow required). The network learns
    to approximate Q(s, a) from 193-dimensional board observations.
    
    Attributes:
        input_dim: Observation vector size (193 for KRK).
        output_dim: Number of actions (36 for KRK).
        epsilon: Current exploration rate.
    """

    def __init__(
        self,
        input_dim: int = 193,
        output_dim: int = 36,
        hidden1: int = 256,
        hidden2: int = 128,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_double_dqn: bool = True,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.train_steps = 0

        # Initialize weights with He initialization
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden1)).astype(np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2)).astype(np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.w3 = rng.normal(0, np.sqrt(2.0 / hidden2), (hidden2, output_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

        # Target network (copy of main network)
        self._sync_target()

    def _forward(self, x: np.ndarray, use_target: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network.
        
        Returns hidden activations and Q-values.
        """
        x = np.asarray(x, dtype=np.float32)
        if use_target:
            h1 = np.maximum(0, x @ self.tw1 + self.tb1)  # ReLU
            h2 = np.maximum(0, h1 @ self.tw2 + self.tb2)  # ReLU
            q = h2 @ self.tw3 + self.tb3
        else:
            h1 = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
            h2 = np.maximum(0, h1 @ self.w2 + self.b2)  # ReLU
            q = h2 @ self.w3 + self.b3
        return h1, h2, q

    def _sync_target(self):
        """Copy main network weights to target network."""
        self.tw1 = self.w1.copy()
        self.tb1 = self.b1.copy()
        self.tw2 = self.w2.copy()
        self.tb2 = self.b2.copy()
        self.tw3 = self.w3.copy()
        self.tb3 = self.b3.copy()

    def predict(self, state: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        """Select an action using ε-greedy policy.
        
        Args:
            state: Observation vector.
            legal_mask: Binary mask of legal actions.
        
        Returns:
            Selected action ID.
        """
        if legal_mask is not None:
            legal_actions = np.flatnonzero(legal_mask)
        else:
            legal_actions = np.arange(self.output_dim)

        if legal_actions.size == 0:
            return 0

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal_actions))

        _, _, q_values = self._forward(state.reshape(1, -1))
        q = q_values[0].copy()
        if legal_mask is not None:
            q[legal_mask == 0] = -1e9
        return int(np.argmax(q))

    def predict_greedy(self, state: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        """Select the best action without exploration (for inference)."""
        if legal_mask is not None:
            legal_actions = np.flatnonzero(legal_mask)
        else:
            legal_actions = np.arange(self.output_dim)
        if legal_actions.size == 0:
            return 0

        _, _, q_values = self._forward(state.reshape(1, -1))
        q = q_values[0].copy()
        if legal_mask is not None:
            q[legal_mask == 0] = -1e9
        return int(np.argmax(q))

    def compute_target_q(
        self,
        current_q: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        legal_masks: np.ndarray,
    ) -> np.ndarray:
        """Compute target Q-values using Double-DQN (decoupled) or standard DQN.
        
        Double-DQN decoupled rule:
            a* = argmax_{a in LEGAL(s')} Q_online(s', a)
            target = r + gamma * (1 - done) * Q_target(s', a*)
        """
        target_q = current_q.copy()
        _, _, next_q_target = self._forward(next_states, use_target=True)

        if self.use_double_dqn:
            _, _, next_q_online = self._forward(next_states, use_target=False)
            for i in range(len(current_q)):
                if dones[i]:
                    target_q[i, actions[i]] = rewards[i]
                else:
                    masked_online = next_q_online[i].copy()
                    masked_online[legal_masks[i] == 0] = -1e9
                    best_a = int(np.argmax(masked_online))
                    target_q[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_a]
        else:
            for i in range(len(current_q)):
                masked_target = next_q_target[i].copy()
                masked_target[legal_masks[i] == 0] = -1e9
                if dones[i]:
                    target_q[i, actions[i]] = rewards[i]
                else:
                    target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(masked_target)
        return target_q

    def update(self, state, action, reward, next_state, done, legal_mask=None):
        """Store transition and train on a mini-batch from replay buffer."""
        if legal_mask is None:
            legal_mask = np.ones(self.output_dim, dtype=np.int8)
        self.replay_buffer.push(state, action, reward, next_state, done, legal_mask)

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, legal_masks = \
            self.replay_buffer.sample(self.batch_size)

        # Forward pass
        h1, h2, current_q = self._forward(states, use_target=False)

        # Compute target Q-values with Double-DQN
        target_q = self.compute_target_q(
            current_q, actions, rewards, next_states, dones, legal_masks
        )

        # Backpropagation (gradient descent)
        batch_size = len(states)
        error = (current_q - target_q) / batch_size

        # Clip gradients to prevent exploding
        error = np.clip(error, -1.0, 1.0)

        # Layer 3 gradients
        grad_w3 = h2.T @ error
        grad_b3 = error.sum(axis=0)

        # Layer 2 gradients (with ReLU derivative)
        grad_h2 = error @ self.w3.T
        grad_h2[h2 <= 0] = 0  # ReLU derivative
        grad_w2 = h1.T @ grad_h2
        grad_b2 = grad_h2.sum(axis=0)

        # Layer 1 gradients (with ReLU derivative)
        grad_h1 = grad_h2 @ self.w2.T
        grad_h1[h1 <= 0] = 0  # ReLU derivative
        grad_w1 = states.T @ grad_h1
        grad_b1 = grad_h1.sum(axis=0)

        # Gradient clipping (norm-based)
        max_norm = 1.0
        for grad in [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm

        # Update weights
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w3 -= self.learning_rate * grad_w3
        self.b3 -= self.learning_rate * grad_b3

        # Periodically sync target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self._sync_target()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path):
        """Save model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
            config=np.array([
                self.input_dim, self.output_dim, self.hidden1, self.hidden2,
                self.learning_rate, self.gamma, self.epsilon_min, self.epsilon_decay,
            ]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeepQNetwork":
        """Load model weights from disk."""
        data = np.load(path, allow_pickle=False)
        config = data["config"]
        agent = cls(
            input_dim=int(config[0]),
            output_dim=int(config[1]),
            hidden1=int(config[2]),
            hidden2=int(config[3]),
            learning_rate=float(config[4]),
            gamma=float(config[5]),
            epsilon=float(config[6]),  # Use epsilon_min for inference
            epsilon_min=float(config[6]),
            epsilon_decay=float(config[7]),
        )
        agent.w1 = data["w1"]
        agent.b1 = data["b1"]
        agent.w2 = data["w2"]
        agent.b2 = data["b2"]
        agent.w3 = data["w3"]
        agent.b3 = data["b3"]
        agent._sync_target()
        return agent
