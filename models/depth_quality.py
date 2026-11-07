"""Small, task-guided depth repair module for the RGB-D Samba experiment.

The input follows Samba's existing three-channel ImageNet normalization.  The
repair operates on the original [0, 1] grayscale depth, then restores the same
three-channel normalization before the pretrained VMamba encoder.
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
            nn.Conv2d(4, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.quality_head = nn.Conv2d(24, 1, 1)
        self.residual_head = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.constant_(self.quality_head.bias, 2.0)

    @staticmethod
    def _edges(x):
        dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return dx + dy

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
                blurred = F.avg_pool2d(depth[index:index + 1], kernel_size=7, stride=1, padding=3)
                damaged[region] = blurred[:, :, top:top + patch_h, left:left + patch_w]
        return damaged, mask

    def forward(self, rgb, normalized_depth, train_aux=False):
        raw_depth = (normalized_depth[:, :1] * self.std[:, :1] + self.mean[:, :1]).clamp(0, 1)
        rgb_raw = (rgb * self.std + self.mean).clamp(0, 1)
        luminance = (rgb_raw[:, :1] * 0.299 + rgb_raw[:, 1:2] * 0.587 + rgb_raw[:, 2:3] * 0.114)

        if self.training and train_aux:
            depth_input, corruption_mask = self._synthetic_degradation(raw_depth)
        else:
            depth_input = raw_depth
            corruption_mask = None

        depth_edge = self._edges(depth_input)
        rgb_edge = self._edges(luminance)
        features = self.features(torch.cat((depth_input, depth_edge, rgb_edge, (depth_edge - rgb_edge).abs()), dim=1))
        quality_logits = self.quality_head(features)
        quality = quality_logits.sigmoid()
        residual = self.residual_head(features).tanh()
        repaired = (depth_input + (1.0 - quality) * residual).clamp(0, 1)
        encoded_depth = (repaired.repeat(1, 3, 1, 1) - self.mean) / self.std

        if corruption_mask is None:
            return encoded_depth, None

        quality_target = 1.0 - corruption_mask
        quality_weight = 1.0 + 8.0 * corruption_mask
        quality_loss = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, weight=quality_weight
        )
        restoration_loss = ((repaired - raw_depth).abs() * corruption_mask).sum() / corruption_mask.sum().clamp_min(1.0)
        identity_loss = ((repaired - raw_depth).abs() * quality_target).mean()
        aux_loss = quality_loss + 2.0 * restoration_loss + 0.1 * identity_loss
        return encoded_depth, aux_loss
