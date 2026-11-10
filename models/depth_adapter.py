"""Depth-only residual adaptation inside Samba's shared RGB/depth encoder."""

import torch
from torch import nn


class DepthAdapter(nn.Module):
    """Channel-last bottleneck with a local geometry operator and zero output."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, hidden)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = self.down(self.norm(x)).permute(0, 3, 1, 2).contiguous()
        residual = self.activation(self.spatial(residual))
        residual = self.up(residual.permute(0, 2, 3, 1).contiguous())
        return x + residual
