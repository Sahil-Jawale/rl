from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chess_endgame_rl.models.dueling_net import DuelingNet


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    legal_mask: np.ndarray
    next_state: np.ndarray
    terminated: bool
    next_legal_mask: np.ndarray


class DuelingDoubleDQN:
    """Legal-masked Double DQN with optional supervised anchor loss."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        epsilon: float = 0.05,
        replay_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_frequency: int = 1_000,
        anchor_weight: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
        hidden_dim: int = 256,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if learning_rate <= 0 or batch_size < 1 or replay_capacity < batch_size:
            raise ValueError("Learning rate and replay sizes must be positive and consistent.")
        if not 0.0 <= epsilon <= 1.0 or anchor_weight < 0 or target_update_frequency < 1:
            raise ValueError("Invalid epsilon, anchor weight, or target update frequency.")

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.anchor_weight = anchor_weight
        self.replay_capacity = replay_capacity
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.online = DuelingNet(hidden_dim=hidden_dim).to(self.device)
        self.target = DuelingNet(hidden_dim=hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[Transition] = deque(maxlen=replay_capacity)
        self.steps = 0

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError(f"Checkpoint {path} does not contain model_state_dict.")
        self.online.load_state_dict(state_dict)
        self.target.load_state_dict(state_dict)

    def save_checkpoint(self, path: str | Path, **metadata: object) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online.state_dict(),
                "model_config": {"in_channels": 18, "action_dim": 4096},
                "algorithm": "legal-masked-dueling-double-dqn",
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                **metadata,
            },
            destination,
        )

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        *,
        explore: bool = True,
    ) -> int:
        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            raise ValueError("Cannot select an action when no legal actions are available.")
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions.tolist()))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        was_training = self.online.training
        self.online.eval()
        q_values, _, _ = self.online(state_tensor, mask_tensor)
        self.online.train(was_training)
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        legal_mask: np.ndarray,
        next_state: np.ndarray,
        terminated: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        legal = np.asarray(legal_mask, dtype=np.bool_)
        next_legal = np.asarray(next_legal_mask, dtype=np.bool_)
        if legal.shape != (4096,) or next_legal.shape != (4096,):
            raise ValueError("Current and successor legal masks must have 4096 entries.")
        if not 0 <= action < 4096 or not legal[action]:
            raise ValueError("The stored action must be within the action space and legal.")
        if not terminated and not next_legal.any():
            raise ValueError("A non-terminal transition must have legal successor moves.")
        self.replay.append(
            Transition(
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                legal.copy(),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(terminated),
                next_legal.copy(),
            )
        )

    @staticmethod
    def double_dqn_targets(
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        online_next_q: torch.Tensor,
        target_next_q: torch.Tensor,
        next_legal_mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        if online_next_q.shape != target_next_q.shape or online_next_q.shape != next_legal_mask.shape:
            raise ValueError("Next-state Q values and masks must have matching shapes.")
        if torch.any(~terminated.bool() & ~next_legal_mask.any(dim=1)):
            raise ValueError("Every non-terminal successor must have at least one legal action.")

        legal_online_q = online_next_q.masked_fill(~next_legal_mask, -torch.inf)
        selected_actions = legal_online_q.argmax(dim=1)
        selected_target_q = target_next_q.gather(1, selected_actions[:, None]).squeeze(1)
        return rewards + gamma * (~terminated.bool()).to(rewards.dtype) * selected_target_q

    def train_step(
        self,
        anchor_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float] | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            [item.terminated for item in batch], dtype=torch.bool, device=self.device
        )
        next_masks = torch.as_tensor(
            np.stack([item.next_legal_mask for item in batch]), dtype=torch.bool, device=self.device
        )

        current_masks = torch.as_tensor(
            np.stack([item.legal_mask for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        current_q, _, _ = self.online(states, current_masks)
        predicted_q = current_q.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next_q, _, _ = self.online(next_states, next_masks)
            target_next_q, _, _ = self.target(next_states, next_masks)
            targets = self.double_dqn_targets(
                rewards,
                terminated,
                online_next_q,
                target_next_q,
                next_masks,
                self.gamma,
            )
        td_loss = F.smooth_l1_loss(predicted_q, targets)
        anchor_loss = torch.zeros((), dtype=torch.float32, device=self.device)

        if self.anchor_weight and anchor_batch is not None:
            anchor_batch = {key: value.to(self.device) for key, value in anchor_batch.items()}
            anchor_q, _, _ = self.online(
                anchor_batch["state"], anchor_batch["legal_mask"]
            )
            anchor_loss = -(
                anchor_batch["target_distribution"] * F.log_softmax(anchor_q, dim=1)
            ).sum(dim=1).mean()

        loss = td_loss + self.anchor_weight * anchor_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {
            "loss": float(loss.detach()),
            "td_loss": float(td_loss.detach()),
            "anchor_loss": float(anchor_loss.detach()),
        }
