"""Proximal Policy Optimization (PPO) Agent for KRK chess endgame.

PPO is an on-policy, actor-critic algorithm that uses a clipped surrogate
objective to prevent large policy updates, ensuring stable training.

Key features:
- Separate policy (actor) and value (critic) heads sharing a backbone
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Legal action masking via logit masking
- Mini-batch updates from collected trajectory data

Architecture:
  Input (193) → Dense(256, ReLU) → Dense(128, ReLU) → Policy head (36, Softmax)
                                                     → Value head (1, Linear)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PPOAgent:
    """NumPy-based PPO agent with actor-critic architecture.
    
    Collects trajectories during rollouts and updates policy/value networks
    using the clipped PPO objective.
    
    Attributes:
        input_dim: Observation vector size (193 for KRK).
        output_dim: Number of actions (36 for KRK).
        clip_epsilon: PPO clipping parameter.
        gamma: Discount factor for returns.
        gae_lambda: GAE parameter for advantage estimation.
    """

    def __init__(
        self,
        input_dim: int = 193,
        output_dim: int = 36,
        hidden1: int = 256,
        hidden2: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        mini_batch_size: int = 32,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        # Initialize weights with He initialization
        rng = np.random.default_rng(123)

        # Shared backbone
        self.w1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden1)).astype(np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2)).astype(np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        # Policy head
        self.w_policy = rng.normal(0, np.sqrt(2.0 / hidden2), (hidden2, output_dim)).astype(np.float32)
        self.b_policy = np.zeros(output_dim, dtype=np.float32)

        # Value head
        self.w_value = rng.normal(0, np.sqrt(2.0 / hidden2), (hidden2, 1)).astype(np.float32)
        self.b_value = np.zeros(1, dtype=np.float32)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through shared backbone + policy and value heads.
        
        Returns:
            h1: First hidden layer activations.
            h2: Second hidden layer activations.
            logits: Raw policy logits (before softmax).
            values: State value estimates.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h1 = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)  # ReLU
        logits = h2 @ self.w_policy + self.b_policy
        values = h2 @ self.w_value + self.b_value
        return h1, h2, logits, values

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _mask_logits(self, logits: np.ndarray, legal_mask: np.ndarray | None) -> np.ndarray:
        """Set illegal action logits to -inf."""
        if legal_mask is None:
            return logits
        masked = logits.copy()
        if masked.ndim == 1:
            masked[legal_mask == 0] = -1e9
        else:
            mask_2d = np.asarray(legal_mask)
            if mask_2d.ndim == 1:
                mask_2d = mask_2d.reshape(1, -1)
            masked[mask_2d == 0] = -1e9
        return masked

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray | None = None) -> tuple[int, float]:
        """Sample an action from the policy distribution.
        
        Args:
            state: Observation vector.
            legal_mask: Binary mask of legal actions.
        
        Returns:
            action: Selected action ID.
            log_prob: Log probability of the selected action.
        """
        _, _, logits, _ = self._forward(state)
        logits = logits[0]
        masked_logits = self._mask_logits(logits, legal_mask)
        probs = self._softmax(masked_logits)

        # Ensure valid probability distribution over legal actions
        if legal_mask is not None:
            legal_actions = np.flatnonzero(legal_mask)
            if legal_actions.size == 0:
                return 0, 0.0
            valid_probs = probs[legal_actions]
            valid_probs = valid_probs / (valid_probs.sum() + 1e-10)
            chosen_idx = np.random.choice(len(legal_actions), p=valid_probs)
            action = int(legal_actions[chosen_idx])
        else:
            action = int(np.random.choice(self.output_dim, p=probs))

        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob

    def select_action_greedy(self, state: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        """Select the most probable action (for inference)."""
        _, _, logits, _ = self._forward(state)
        logits = logits[0]
        masked_logits = self._mask_logits(logits, legal_mask)
        return int(np.argmax(masked_logits))

    def get_value(self, state: np.ndarray) -> float:
        """Estimate the value of a state."""
        _, _, _, values = self._forward(state)
        return float(values[0, 0])

    def update(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        dones: list[bool],
        log_probs: list[float],
        legal_masks: list[np.ndarray | None],
    ) -> dict[str, float]:
        """Update policy and value networks using PPO.
        
        Args:
            states: List of observation vectors from the trajectory.
            actions: List of actions taken.
            rewards: List of rewards received.
            dones: List of done flags.
            log_probs: List of log probabilities of actions taken.
            legal_masks: List of legal action masks.
        
        Returns:
            Dictionary with training metrics (policy_loss, value_loss, entropy).
        """
        if not states:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        n = len(states)
        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int64)
        old_log_probs = np.array(log_probs, dtype=np.float32)

        # Compute returns and advantages using GAE
        values = []
        for s in states:
            values.append(self.get_value(s))
        values = np.array(values, dtype=np.float32)

        # Bootstrap value for last state
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if dones[t] or t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - float(dones[t])) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        if n > 1:
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - advantages.mean()) / adv_std

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb_idx = indices[start:end]
                mb_states = states_arr[mb_idx]
                mb_actions = actions_arr[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_masks = [legal_masks[i] for i in mb_idx]

                policy_loss, value_loss, entropy = self._ppo_step(
                    mb_states, mb_actions, mb_old_log_probs,
                    mb_advantages, mb_returns, mb_masks,
                )
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                update_count += 1

        return {
            "policy_loss": total_policy_loss / max(update_count, 1),
            "value_loss": total_value_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
        }

    def _ppo_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        legal_masks: list,
    ) -> tuple[float, float, float]:
        """Perform one PPO gradient update on a mini-batch."""
        batch_size = len(states)
        h1, h2, logits, values = self._forward(states)

        # Mask logits for legal actions
        for i in range(batch_size):
            if legal_masks[i] is not None:
                mask = np.asarray(legal_masks[i], dtype=np.int8)
                logits[i][mask == 0] = -1e9

        probs = self._softmax(logits)

        # Current log probabilities
        new_log_probs = np.log(probs[np.arange(batch_size), actions] + 1e-10)

        # PPO clipped ratio
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = np.mean((values[:, 0] - returns) ** 2)

        # Entropy bonus (encourages exploration)
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        # --- Backpropagation ---
        # Policy gradient
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), actions] = 1.0

        # Gradient of the clipped policy loss with respect to logits.
        mask_ratio = np.where(
            ratio * advantages < clipped_ratio * advantages,
            1.0, 0.0
        ).reshape(-1, 1)
        policy_grad = (
            (probs - one_hot)
            * (advantages * ratio).reshape(-1, 1)
            * mask_ratio
            / batch_size
        )

        # Differentiate the entropy bonus through the masked policy distribution.
        entropy_per_state = -np.sum(
            probs * np.log(probs + 1e-10), axis=1, keepdims=True
        )
        entropy_grad = (
            self.entropy_coef
            * probs
            * (np.log(probs + 1e-10) + entropy_per_state)
            / batch_size
        )

        logit_grad = policy_grad + entropy_grad

        # Value gradient
        value_grad = (2.0 / batch_size) * (values[:, 0] - returns).reshape(-1, 1) * self.value_coef

        # Gradient clipping
        logit_grad = np.clip(logit_grad, -1.0, 1.0)
        value_grad = np.clip(value_grad, -1.0, 1.0)

        # Policy head gradients
        grad_w_policy = h2.T @ logit_grad
        grad_b_policy = logit_grad.sum(axis=0)

        # Value head gradients
        grad_w_value = h2.T @ value_grad
        grad_b_value = value_grad.sum(axis=0)

        # Backbone gradients (combined from both heads)
        grad_h2 = logit_grad @ self.w_policy.T + value_grad @ self.w_value.T
        grad_h2[h2 <= 0] = 0  # ReLU derivative

        grad_w2 = h1.T @ grad_h2
        grad_b2 = grad_h2.sum(axis=0)

        grad_h1 = grad_h2 @ self.w2.T
        grad_h1[h1 <= 0] = 0  # ReLU derivative

        grad_w1 = states.T @ grad_h1
        grad_b1 = grad_h1.sum(axis=0)

        # Update weights
        self.w_policy -= self.learning_rate * grad_w_policy
        self.b_policy -= self.learning_rate * grad_b_policy
        self.w_value -= self.learning_rate * grad_w_value
        self.b_value -= self.learning_rate * grad_b_value
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

        return float(policy_loss), float(value_loss), float(entropy)

    def save(self, path: str | Path):
        """Save model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w_policy=self.w_policy, b_policy=self.b_policy,
            w_value=self.w_value, b_value=self.b_value,
            config=np.array([
                self.input_dim, self.output_dim, self.hidden1, self.hidden2,
                self.learning_rate, self.gamma, self.clip_epsilon, self.gae_lambda,
            ]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
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
            clip_epsilon=float(config[6]),
            gae_lambda=float(config[7]),
        )
        agent.w1 = data["w1"]
        agent.b1 = data["b1"]
        agent.w2 = data["w2"]
        agent.b2 = data["b2"]
        agent.w_policy = data["w_policy"]
        agent.b_policy = data["b_policy"]
        agent.w_value = data["w_value"]
        agent.b_value = data["b_value"]
        return agent
