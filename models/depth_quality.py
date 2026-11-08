"""Degradation-aware, selective RGB guidance for depth input to Samba.

Inspired by SigNet (CVPR 2025): a learned degradation map controls where RGB
high-frequency information may enter the depth stream. This is an adaptation
for RGB-D saliency detection, not an implementation of SigNet itself.
"""

import random

import torch
from torch import nn
from torch.nn import functional as F


class DepthQualityEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(7, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.quality_head = nn.Conv2d(24, 1, 1)
        self.residual_head = nn.Conv2d(24, 1, 1)
        self.rgb_gate_head = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.rgb_gate_head.weight)
        nn.init.constant_(self.rgb_gate_head.bias, -2.0)
        nn.init.constant_(self.quality_head.bias, 2.0)

    @staticmethod
    def _edges(x):
        dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return dx + dy

    @staticmethod
    def _blur(x, kernel_size=7):
        return F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2,
                            count_include_pad=False)

    @staticmethod
    def _synthetic_degradation(depth):
        damaged = depth.clone()
        mask = torch.zeros_like(depth)
        _, _, height, width = depth.shape
        for index in range(depth.shape[0]):
            if random.random() >= 0.7:
                continue
            patch_h = random.randint(max(4, height // 12), max(4, height // 4))
            patch_w = random.randint(max(4, width // 12), max(4, width // 4))
            top = random.randint(0, height - patch_h)
            left = random.randint(0, width - patch_w)
            region = (slice(index, index + 1), slice(None), slice(top, top + patch_h), slice(left, left + patch_w))
            mask[region] = 1.0
            operation = random.randrange(3)
            if operation == 0:
                damaged[region] = 0.0
            elif operation == 1:
                damaged[region] = (depth[region] + 0.12 * torch.randn_like(depth[region])).clamp(0, 1)
            else:
                blurred = DepthQualityEnhancer._blur(depth[index:index + 1])
                damaged[region] = blurred[:, :, top:top + patch_h, left:left + patch_w]
        return damaged, mask

    def forward(self, rgb, normalized_depth, train_aux=False):
        raw_depth = (normalized_depth[:, :1] * self.std[:, :1] + self.mean[:, :1]).clamp(0, 1)
        rgb_raw = (rgb * self.std + self.mean).clamp(0, 1)
        luminance = rgb_raw[:, :1] * 0.299 + rgb_raw[:, 1:2] * 0.587 + rgb_raw[:, 2:3] * 0.114

        if self.training and train_aux:
            depth_input, corruption_mask = self._synthetic_degradation(raw_depth)
        else:
            depth_input, corruption_mask = raw_depth, None

        depth_low = self._blur(depth_input)
        depth_high = depth_input - depth_low
        rgb_high = luminance - self._blur(luminance)
        depth_edge = self._edges(depth_input)
        rgb_edge = self._edges(luminance)
        features = self.features(torch.cat((
            depth_input, depth_low, depth_high, depth_edge,
            rgb_high, rgb_edge, (depth_edge - rgb_edge).abs(),
        ), dim=1))
        quality_logits = self.quality_head(features)
        low_quality = 1.0 - quality_logits.sigmoid()
        rgb_gate = self.rgb_gate_head(features).sigmoid()
        residual = self.residual_head(features).tanh()
        correction = 0.55 * residual + 0.35 * rgb_gate * rgb_high
        repaired = (depth_input + low_quality * correction).clamp(0, 1)
        encoded_depth = (repaired.repeat(1, 3, 1, 1) - self.mean) / self.std

        if corruption_mask is None:
            return encoded_depth, None

        clean_mask = 1.0 - corruption_mask
        quality_loss = F.binary_cross_entropy_with_logits(
            quality_logits, clean_mask, weight=1.0 + 8.0 * corruption_mask
        )
        restoration_loss = ((repaired - raw_depth).abs() * corruption_mask).sum() / corruption_mask.sum().clamp_min(1.0)
        identity_loss = ((repaired - raw_depth).abs() * clean_mask).mean()
        edge_mask = F.max_pool2d(corruption_mask, 3, stride=1, padding=1)
        edge_loss = ((self._edges(repaired) - self._edges(raw_depth)).abs() * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
        unsupported_rgb_edge = F.relu(rgb_edge - 2.0 * depth_edge.detach())
        rgb_leak_loss = (rgb_gate * unsupported_rgb_edge * clean_mask).mean()
        aux_loss = quality_loss + 2.0 * restoration_loss + 0.5 * edge_loss + 0.3 * identity_loss + 0.1 * rgb_leak_loss
        return encoded_depth, aux_loss
