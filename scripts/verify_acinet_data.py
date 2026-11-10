"""Check the formerly saturated DUT example and strict trained-model inference."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'dataset'),str(ROOT/'transform')]
from models.SalMamba_dual import Model
from rgbd_dataset import Dataset
import transform_rgbd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    preprocess=transforms.Compose([
        transform_rgbd.FixedResize((448,448)),
        transform_rgbd.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        transform_rgbd.ToTensor()])
    source=Path('rgbd_dataset/train_DUT')
    record=next(line.split() for line in (source/'train.txt').read_text().splitlines()
                if Path(line.split()[0]).stem=='0002')
    depth_path=source/record[2].lstrip('/')
    native=Image.open(depth_path)
    assert native.mode=='L' and np.asarray(native).dtype==np.uint8
    sample=preprocess({'image':Image.open(source/record[0].lstrip('/')).convert('RGB'),
                       'label':np.asarray(Image.open(source/record[1].lstrip('/')).convert('L'),dtype=np.float32)/255,
                       'depth':native.convert('RGB')})
    recovered=sample['depth'][0]*0.229+0.485
    assert torch.isfinite(recovered).all() and recovered.std().item()>.02
    model=Model().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
    dataset=Dataset(['NJU2K'],mode='test',transform=preprocess)
    test=dataset[0]
    with torch.inference_mode():
        out,saliency=model(test['image'][None].cuda(),test['depth'][None].cuda())
    assert out.shape==(1,1,448,448) and torch.isfinite(out).all() and torch.isfinite(saliency).all()
    report={'checkpoint':str(args.checkpoint.resolve()),'strict_checkpoint_reload':True,
            'real_selection_sample_inference':True,'formerly_saturated_example':str(depth_path.resolve()),
            'source_depth_range':[int(np.asarray(native).min()),int(np.asarray(native).max())],
            'recovered_depth_std':recovered.std().item(),'selection_samples':len(dataset),
            'model_parameters':sum(p.numel() for p in model.parameters())}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
