"""
resnet_model.py
---------------
CORRECTED:
1. start_conv changed to expect 1 input channel (to match weights).
2. Policy/Value head names changed (to match weights: policy_fc, value_fc1/2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Basic Residual Block ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# --- Policy Value ResNet Model ---
class ResNetPolicyValueNet(nn.Module):
    # NOTE: The model now requires num_channels=1 in its definition, not 2.
    def __init__(self, board_size=15, channels=64, num_res_blocks=5, dropout_p=0.3):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        # --- FIX 1: Input channel changed from 2 to 1 ---
        self.start_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=True)
        self.start_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        # --- FIX 2: Renamed policy linear layer to match weights ('policy_fc') ---
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value Head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        # --- FIX 3: Renamed value linear layers to match weights ('value_fc1', 'value_fc2') ---
        self.value_fc1 = nn.Linear(1 * board_size * board_size, channels)
        self.value_fc2 = nn.Linear(channels, 1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.num_res_blocks = num_res_blocks

    def forward(self, x):
        # Trunk
        h = F.relu(self.start_bn(self.start_conv(x)))
        h = self.res_blocks(h)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.dropout(p.flatten(start_dim=1))
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(start_dim=1)))
        v = self.dropout(v)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value