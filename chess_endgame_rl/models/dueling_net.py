"""Legal-masked Dueling Double Deep Q-Network for Chess Endgames.

Architecture:
    Input: (B, 18, 8, 8) side-to-move binary feature planes
    Conv Trunk:
        Conv2d(18, 64, 3, padding=1) + BatchNorm2d(64) + ReLU
        Conv2d(64, 64, 3, padding=1) + BatchNorm2d(64) + ReLU
        Conv2d(64, 64, 3, padding=1) + BatchNorm2d(64) + ReLU
        Flatten -> 64 * 8 * 8 = 4096
        Shared FC: Linear(4096, 256) + ReLU
    Dueling Heads:
        Value Head: Linear(256, 128) -> ReLU -> Linear(128, 1)
        Advantage Head: Linear(256, 512) -> ReLU -> Linear(512, 4096)
        
    Forward with Legal-Move Mask:
        A_legal_mean = sum(A * mask, dim=-1, keepdim=True) / max(sum(mask), 1)
        Q = V + (A - A_legal_mean)
        Q[~mask] = -1e9  (safe argmax masking)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNet(nn.Module):
    """Convolutional Dueling Network with legal-only advantage baseline."""

    def __init__(
        self,
        in_channels: int = 18,
        action_dim: int = 4096,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim

        # Convolutional feature trunk
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Shared dense representation (64 * 8 * 8 = 4096)
        self.fc_shared = nn.Linear(64 * 8 * 8, hidden_dim)

        # Value stream: V(s) in R^1
        self.fc_value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream: A(s, a) in R^4096
        self.fc_advantage = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing V(s), A(s, a), and legal-masked Q(s, a).
        
        Args:
            x: Input tensor of shape (B, 18, 8, 8).
            legal_mask: Boolean mask of shape (B, 4096) where True indicates legal moves.
            
        Returns:
            (q_values, value, advantage):
                q_values: Legal-masked Q-values of shape (B, 4096).
                value: State value V(s) of shape (B, 1).
                advantage: Raw advantage values A(s, a) of shape (B, 4096).
        """
        # Feature extraction
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = h.view(h.size(0), -1)
        features = F.relu(self.fc_shared(h))

        # Stream predictions
        v = self.fc_value(features)          # (B, 1)
        a = self.fc_advantage(features)      # (B, 4096)

        if legal_mask is not None:
            mask_float = legal_mask.float()
            mask_sum = mask_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
            # Legal-only advantage mean: sum(A * mask) / sum(mask)
            a_mean_legal = (a * mask_float).sum(dim=-1, keepdim=True) / mask_sum
            q = v + (a - a_mean_legal)
            # Mask illegal moves with -1e9 for argmax safety
            q = torch.where(legal_mask, q, torch.full_like(q, -1e9))
        else:
            a_mean = a.mean(dim=-1, keepdim=True)
            q = v + (a - a_mean)

        return q, v, a
