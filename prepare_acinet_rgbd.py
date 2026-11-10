"""Validate and link the exact RGB/GT/depth files configured by ACINet_best."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = {'train_DUT': ('RGB-D_train', 'train.txt', 2985),
          'DUT-RGBD': ('RGB-D_test/DUT', 'test.txt', 400),
          'NJU2K': ('RGB-D_test/NJUD', 'test.txt', 500),
          'NLPR': ('RGB-D_test/NLPR', 'test.txt', 300),
          'SIP': ('RGB-D_test/SIP', 'test.txt', 929),
          'STERE': ('RGB-D_test/STERE', 'test.txt', 1000)}


def index_files(path):
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    indexed = {p.stem: p for p in files}
    if len(indexed) != len(files):
        raise ValueError(f'Duplicate image stems: {path}')
    return indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/wmy/project/Datasets/RGBD_dataset'))
    parser.add_argument('--destination', type=Path, default=Path('../datasets_acinet_best'))
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    report = {'source': str(source), 'selection_set': 'NJU2K (500), matching prior Samba/A selection',
              'splits': {}, 'training_test_stem_overlap': {}}
    prepared = []
    train_stems = None
    for alias, (relative, manifest, expected) in SPLITS.items():
        split = source / relative
        paths = {key: index_files(split / key) for key in ('RGB', 'GT', 'depth')}
        stems = set(paths['RGB'])
        if stems != set(paths['GT']) or stems != set(paths['depth']) or len(stems) != expected:
            raise ValueError(f'Unexpected pairing/count: {alias}')
        if alias == 'train_DUT':
            train_stems = stems
        else:
            overlap = sorted(train_stems & stems)
            report['training_test_stem_overlap'][alias] = len(overlap)
            if overlap:
                raise ValueError(f'Train/test identifiers overlap in {alias}: {overlap[:10]}')
        modes = Counter()
        resolutions = Counter()
        constant_count = 0
        lines = []
        fingerprint = hashlib.sha256()
        for stem in sorted(stems):
            with Image.open(paths['RGB'][stem]) as image, Image.open(paths['GT'][stem]) as gt, Image.open(paths['depth'][stem]) as depth:
                if image.size != gt.size or image.size != depth.size:
                    raise ValueError(f'Size mismatch: {alias}/{stem}')
                if min(image.size) <= 30:
                    raise ValueError(f'Image too small for the inherited crop: {alias}/{stem}')
                modes[depth.mode] += 1
                if depth.mode not in ('L', 'RGB'):
                    raise ValueError(f'Unadapted depth bit depth: {alias}/{stem}: {depth.mode}')
                native = np.asarray(depth)
                loaded = np.asarray(depth.convert('RGB'))
                if native.dtype != np.uint8 or not np.array_equal(loaded[...,0], loaded[...,1]) or not np.array_equal(loaded[...,0], loaded[...,2]):
                    raise ValueError(f'Depth is not 8-bit grayscale: {alias}/{stem}')
                constant_count += int(loaded[...,0].min() == loaded[...,0].max())
                resolutions[f'{image.size[0]}x{image.size[1]}'] += 1
            names = [f'/{key}/{paths[key][stem].name}' for key in ('RGB', 'GT', 'depth')]
            lines.append(' '.join(names) + '\n')
            for key in ('RGB', 'GT', 'depth'):
                path = paths[key][stem]
                stat = path.stat()
                fingerprint.update(f'{path.relative_to(source)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
        if constant_count:
            raise ValueError(f'Constant depth images in {alias}: {constant_count}')
        content = ''.join(lines)
        report['splits'][alias] = {'source': str(split), 'samples': len(stems), 'depth_modes': dict(modes),
                                 'constant_depths_after_reading': constant_count, 'resolutions': dict(resolutions),
                                 'manifest_sha256': hashlib.sha256(content.encode()).hexdigest(),
                                 'path_size_mtime_fingerprint': fingerprint.hexdigest()}
        prepared.append((alias, split, manifest, content))
        print(f'{alias}: {len(stems)} triplets, modes={dict(modes)}, constant depths={constant_count}', flush=True)
    # Publish aliases only after every split passes validation.
    for alias, split, manifest, content in prepared:
        target = destination / alias
        target.mkdir(parents=True, exist_ok=True)
        for key in ('RGB', 'GT', 'depth'):
            link = target / key
            expected = (split / key).resolve()
            if link.is_symlink() and link.resolve() == expected:
                continue
            if link.exists() or link.is_symlink():
                raise FileExistsError(f'Refusing to replace an unrelated dataset link: {link}')
            link.symlink_to(expected, target_is_directory=True)
        (target / manifest).write_text(content)
    (destination / 'data_protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated ACINet_best aliases ready: {destination}', flush=True)


if __name__ == '__main__':
    main()
