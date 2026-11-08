"""Decoupled global geometry and local detail repair for Samba depth input.

Inspired by GDNet (CVPR 2025), with depth discontinuities used as a spatial
geometry prior. This lightweight RGB-D SOD adaptation is not GDNet itself.
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
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(5, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.geometry_head = nn.Conv2d(16, 1, 1)
        self.detail_head = nn.Conv2d(24, 1, 1)
        self.quality_head = nn.Conv2d(41, 1, 1)
        self.detail_gate_head = nn.Conv2d(41, 1, 1)
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)
        nn.init.constant_(self.quality_head.bias, 2.0)
        nn.init.constant_(self.detail_gate_head.bias, -1.0)

    @staticmethod
    def _edges(x):
        dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return dx + dy

    @staticmethod
    def _blur(x, kernel_size=9):
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

        # Low-resolution branch extracts coarse geometry and suppresses pixel noise.
        low_resolution = F.avg_pool2d(torch.cat((depth_input, depth_low), dim=1), 4, stride=4)
        geometry = self.geometry_encoder(low_resolution)
        geometry_full = F.interpolate(geometry, size=depth_input.shape[-2:], mode="bilinear", align_corners=False)

        # Full-resolution branch retains local depth structure with optional RGB detail.
        detail = self.detail_encoder(torch.cat((
            depth_high, depth_edge, rgb_high, rgb_edge,
            (depth_edge - rgb_edge).abs(),
        ), dim=1))
        fused = torch.cat((geometry_full, detail, depth_edge), dim=1)
        quality_logits = self.quality_head(fused)
        low_quality = 1.0 - quality_logits.sigmoid()
        detail_gate = self.detail_gate_head(fused).sigmoid()
        geometry_delta = F.interpolate(self.geometry_head(geometry).tanh(),
                                       size=depth_input.shape[-2:], mode="bilinear", align_corners=False)
        detail_delta = self.detail_head(detail).tanh()

        # Preserve measured depth discontinuities when applying coarse corrections.
        geometry_prior = torch.exp(-6.0 * depth_edge)
        correction = low_quality * (
            0.60 * geometry_prior * geometry_delta + 0.35 * detail_gate * detail_delta
        )
        repaired = (depth_input + correction).clamp(0, 1)
        encoded_depth = (repaired.repeat(1, 3, 1, 1) - self.mean) / self.std

        if corruption_mask is None:
            return encoded_depth, None

        clean_mask = 1.0 - corruption_mask
        quality_loss = F.binary_cross_entropy_with_logits(
            quality_logits, clean_mask, weight=1.0 + 8.0 * corruption_mask
        )
        restoration_loss = ((repaired - raw_depth).abs() * corruption_mask).sum() / corruption_mask.sum().clamp_min(1.0)
        identity_loss = ((repaired - raw_depth).abs() * clean_mask).mean()
        low_mask = F.max_pool2d(corruption_mask, 9, stride=1, padding=4)
        geometry_loss = ((self._blur(repaired) - self._blur(raw_depth)).abs() * low_mask).sum() / low_mask.sum().clamp_min(1.0)
        detail_mask = F.max_pool2d(corruption_mask, 3, stride=1, padding=1)
        repaired_high = repaired - self._blur(repaired)
        target_high = raw_depth - self._blur(raw_depth)
        detail_loss = ((repaired_high - target_high).abs() * detail_mask).sum() / detail_mask.sum().clamp_min(1.0)
        aux_loss = quality_loss + 2.0 * restoration_loss + 0.5 * geometry_loss + 0.5 * detail_loss + 0.2 * identity_loss
        return encoded_depth, aux_loss
