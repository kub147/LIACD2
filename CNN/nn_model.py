"""
nn_model.py
-----------
Compact Policy-Value CNN with dropout in both heads.

- Trunk: a small Conv-BN-ReLU stack (adapt as needed)
- Policy head: logits over N*N actions
- Value head: scalar in [-1, 1]
- Dropout added to both heads to reduce overfitting
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    I added this class, which add a skip-connection, to allow the nwetwork to deepen the trunk without making the training unstable.
    Also it helps the model learn more complex spatial patterns for the game and improves the performance.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x     
        out = self.relu(out)
        return out


class PolicyValueNet(nn.Module):
    def __init__(self, board_size=15, channels=128, dropout_p=0.3):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        # ----- Trunk -----
        # Input shape: (B, 2, N, N)  -> 2 planes (current player stones, opponent stones)
        self.trunk = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

        # ----- Policy head -----
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=dropout_p),  # <--- Dropout to reduce overfitting
            nn.Linear(2 * board_size * board_size, self.action_size)
        )

        # ----- Value head -----
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * board_size * board_size, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),  # <--- Dropout to reduce overfitting
            nn.Linear(channels, 1),
            nn.Tanh(),  # map to [-1, 1]
        )

    def forward(self, x):
        """
        x: (B, 2, N, N)
        Returns:
          policy_logits: (B, N*N)
          value: (B, 1)
        """
        h = self.trunk(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value
