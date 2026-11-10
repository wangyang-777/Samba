"""Verify A initialization, exact initial bypass, learned adapters and inference."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'dataset'), str(ROOT / 'transform')]
from models.SalMamba_dual import Model
from rgbd_dataset import Dataset
import transform_rgbd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(1024); np.random.seed(1024); torch.manual_seed(1024)
    model = Model()
    reference = json.loads(args.reference.read_text())
    state = model.state_dict()
    assert all(hashlib.sha256(state[key].numpy().tobytes()).hexdigest() == value
               for key, value in reference.items()), 'Original A initialization changed'
    initial = {key: value.detach().clone() for key, value in model.backbone.depth_adapters.state_dict().items()}
    model.cuda().eval()
    sample = torch.randn(1, 3, 448, 448, device='cuda')
    with torch.inference_mode():
        bypass = model.backbone.vssm_r(sample)
        adapted = model.backbone.vssm_r(sample, depth_adapters=model.backbone.depth_adapters)
    for original, current in zip(bypass, adapted):
        torch.testing.assert_close(original, current, rtol=1e-6, atol=1e-6)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    changed = {}
    for index, adapter in enumerate(model.backbone.depth_adapters):
        changed[f'stage_{index + 1}'] = {
            'up_weight_mean_abs': adapter.up.weight.detach().abs().mean().item(),
            'down_weight_change': (adapter.down.weight.detach().cpu() - initial[f'{index}.down.weight']).abs().max().item(),
        }
        assert changed[f'stage_{index + 1}']['up_weight_mean_abs'] > 0
        assert changed[f'stage_{index + 1}']['down_weight_change'] > 0
    transform = transforms.Compose([transform_rgbd.FixedResize((448, 448)),
                                    transform_rgbd.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transform_rgbd.ToTensor()])
    data = Dataset(['NJU2K'], mode='test', transform=transform)[0]
    with torch.inference_mode():
        out, _ = model(data['image'][None].cuda(), data['depth'][None].cuda())
    assert out.shape == (1, 1, 448, 448) and torch.isfinite(out).all()
    report = {'a_common_initialization_tensors': len(reference), 'identity_at_initialization': True,
              'new_parameters': sum(value.numel() for value in model.backbone.depth_adapters.parameters()),
              'trained_adapters': changed, 'strict_checkpoint_reload': True, 'real_sample_inference': True}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
