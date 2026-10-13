from __future__ import annotations

import numpy as np


class PPOAgent:
    """A lightweight, policy-gradient-inspired PPO scaffold for legal-masked action selection."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, learning_rate: float = 1e-3, gamma: float = 0.99, clip_epsilon: float = 0.2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        rng = np.random.default_rng(1)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w_policy = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
        self.b_policy = np.zeros(output_dim)
        self.w_value = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))
        self.b_value = np.zeros(1)

    def _forward(self, state):
        x = np.asarray(state, dtype=np.float64)
        hidden = x @ self.w1 + self.b1
        hidden = np.maximum(hidden, 0.0)
        logits = hidden @ self.w_policy + self.b_policy
        value = hidden @ self.w_value + self.b_value
        return hidden, logits, value

    def _mask_logits(self, logits, legal_mask):
        if legal_mask is None:
            return logits
        masked = np.array(logits, dtype=np.float64, copy=True)
        masked[legal_mask == 0] = -1e9
        return masked

    def _softmax(self, logits):
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def select_action(self, state, legal_mask=None):
        _, logits, _ = self._forward(state)
        masked_logits = self._mask_logits(logits, legal_mask)
        probs = self._softmax(masked_logits)
        actions = np.arange(self.output_dim) if legal_mask is None else np.flatnonzero(legal_mask)
        if actions.size == 0:
            return 0
        if legal_mask is not None:
            valid_probs = probs[actions]
            valid_probs /= valid_probs.sum()
            return int(np.random.choice(actions, p=valid_probs))
        return int(np.random.choice(self.output_dim, p=probs))

    def update(self, states, actions, rewards, dones, legal_masks=None):
        if not states:
            return
        returns = np.zeros(len(rewards), dtype=np.float64)
        cumulative = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            if dones[idx]:
                cumulative = 0.0
            cumulative = rewards[idx] + self.gamma * cumulative
            returns[idx] = cumulative

        legal_masks = [None] * len(states) if legal_masks is None else list(legal_masks)
        for state, action, ret, legal_mask in zip(states, actions, returns, legal_masks):
            hidden, logits, value = self._forward(state)
            if legal_mask is not None:
                legal_mask = np.asarray(legal_mask, dtype=np.int8)
                logits = self._mask_logits(logits, legal_mask)
            probs = self._softmax(logits)
            if legal_mask is not None and legal_mask[action] == 0:
                continue
            if probs[action] <= 0.0:
                continue

            advantage = ret - float(value[0])
            policy_grad = np.zeros_like(probs)
            policy_grad[action] = 1.0
            surrogate = self._softmax(logits)
            grad_logits = (surrogate - policy_grad) * advantage / max(len(states), 1)
            grad_logits = grad_logits / np.maximum(np.sum(np.abs(grad_logits)), 1e-12)
            grad_w_policy = hidden[:, None] * grad_logits[None, :]
            grad_b_policy = grad_logits
            self.w_policy -= self.learning_rate * grad_w_policy
            self.b_policy -= self.learning_rate * grad_b_policy
            self.w_value -= self.learning_rate * (2.0 * (value[0] - ret)) * hidden[:, None]
            self.b_value -= self.learning_rate * 2.0 * (value[0] - ret)

        return float(np.mean(returns))
