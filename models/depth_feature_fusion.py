"""Lightweight depth residual injection for channel-last Samba features."""

import torch
from torch import nn


class DepthResidualFusion(nn.Module):
    """Add a projected depth residual without changing the initial RGB path.

    Inputs have shape B x H x W x C. The channel scales start at zero so the
    original A model is recovered at initialization; the scales and projection
    are learned using the existing saliency objective.
    """

    def __init__(self, channels):
        super().__init__()
        self.depth_norm = nn.LayerNorm(channels)
        self.depth_projection = nn.Linear(channels, channels, bias=False)
        self.scale = nn.Parameter(torch.zeros(channels))

    def forward(self, rgb_features, depth_features):
        depth_residual = self.depth_projection(self.depth_norm(depth_features))
        scale = self.scale.tanh().view(1, 1, 1, -1)
        return rgb_features + scale * depth_residual
