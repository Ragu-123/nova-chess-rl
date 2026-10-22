"""NovaZero Neural Network Architecture.

Features:
- Squeeze-and-Excitation Residual Trunk for fast global feature integration
- Policy Head (4096 action logits)
- Value Head (Scalar evaluation in [-1, +1])
- Auxiliary Regret Head (Predicts instantaneous action regret for RGPO loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import NUM_PLANES, ACTION_SPACE_SIZE

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for inter-channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w

class ResidualBlock(nn.Module):
    """Residual Block with Squeeze-and-Excitation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual, inplace=True)
        return out

class NovaZeroNet(nn.Module):
    """High-Throughput Policy-Value-Regret Network for Chess."""
    def __init__(self, in_channels: int = NUM_PLANES, num_filters: int = 128, num_blocks: int = 6):
        super().__init__()
        
        # Initial Convolutional Stem
        self.conv_stem = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(num_filters)
        
        # Residual Backbone
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])
        
        # 1. Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SPACE_SIZE)
        
        # 2. Value Head
        self.value_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # 3. Regret Head (Novel auxiliary head for blunder suppression)
        self.regret_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.regret_bn = nn.BatchNorm2d(16)
        self.regret_fc = nn.Linear(16 * 8 * 8, ACTION_SPACE_SIZE)

    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 19, 8, 8)
            
        Returns:
            policy_logits: (B, 4096)
            value: (B, 1) in [-1, +1]
            regret: (B, 4096) in [0, +inf)
        """
        out = F.relu(self.bn_stem(self.conv_stem(x)), inplace=True)
        for block in self.blocks:
            out = block(out)
            
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)), inplace=True)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(out)), inplace=True)
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))
        
        # Regret (ReLU ensures non-negative predicted regret)
        r = F.relu(self.regret_bn(self.regret_conv(out)), inplace=True)
        r = r.flatten(start_dim=1)
        regret = F.relu(self.regret_fc(r))
        
        return policy_logits, value, regret
