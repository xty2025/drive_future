#!/usr/bin/env python3
"""Build the eight-time frozen V-JEPA2.1 supervision for original DrivoR.

index-real builds a metadata index from NAVSIM logs. encode-real and encode-cosmos
run independently per torchrun rank and write scene NPZ caches. encode-cosmos can
first extract exactly the trajectory-aligned images into scene ZIP archives.
"""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'nuplan-devkit'))
from extract_cosmos_future_frames import DEFAULT_GT, DEFAULT_ROLLOUTS, TIMES, extract_scene, scene_tokens


def index_real(args):
    import yaml
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import Scene, SceneFilter, StateSE2, convert_absolute_to_relative_se2_array
    config = yaml.load((ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml').read_text(), Loader=yaml.CSafeLoader)
    config.pop('_target_')
    config.pop('_convert_', None)
    if args.tokens_json:
        config['tokens'] = scene_tokens(args.gt_root, args.tokens_json, args.limit)
        value = json.loads(Path(args.tokens_json).read_text())
        if isinstance(value, dict) and 'train' in value:
            config['log_names'] = sorted({r['log'] for key in ('train', 'val') for r in value[key]})
    if args.limit:
        config['max_scenes'] = args.limit
    loader = SceneLoader(Path(args.navsim) / 'navsim_logs/trainval',
                         Path(args.navsim) / 'sensor_blobs/trainval', SceneFilter(**config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w') as handle:
        for token, frames in loader.scene_frames_dicts.items():
            current = config['num_history_frames'] - 1
            future = [frames[current + int(round(float(t) / 0.5))] for t in TIMES]
            origin = Scene._build_ego_status(frames[current]).ego_pose
            global_poses = np.stack([Scene._build_ego_status(f).ego_pose for f in future])
            trajectory = convert_absolute_to_relative_se2_array(StateSE2(*origin), global_poses)
            record = {'token': token, 'times': TIMES.tolist(), 'trajectory': trajectory.tolist(),
                      'images': [f['cams']['CAM_F0']['data_path'] for f in future]}
            handle.write(json.dumps(record) + '\n')
    temporary.replace(output)
    print(json.dumps({'index': str(output), 'scenes': len(loader)}), flush=True)


class FutureImages(Dataset):
    def __init__(self, records, args):
        self.records, self.args = records, args

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record, args = self.records[index], self.args
        images, assignments = [], []
        if args.mode == 'encode-real':
            from navsim.common.oss_image import load_camera_image
            token = record['token']
            trajectories = np.asarray(record['trajectory'], np.float32)[None]
            for t, path in enumerate(record['images']):
                pixels = load_camera_image(Path(args.navsim) / 'sensor_blobs/trainval', path)
                images.append(np.array(Image.fromarray(pixels).resize(tuple(args.size), Image.Resampling.BICUBIC)))
                assignments.append((0, t))
        else:
            token = record
            if args.extract_frames:
                extract_scene(token, args.rollouts, args.gt_root, args.frames_output, args.size)
            path = Path(args.frames_output) / token[:2] / (token + '.zip')
            started = time.monotonic()
            while not path.is_file() and time.monotonic() - started < args.wait_for_frames:
                time.sleep(1)
            trajectories = np.zeros((4, 8, 3), np.float32)
            with zipfile.ZipFile(path) as archive:
                info = json.loads(archive.read('metadata.json'))
                if info['size_wh'] != list(args.size) or not np.allclose(info['times'], TIMES):
                    raise ValueError(f'Wrong frame grid or resolution in {path}')
                for entry in info['slots']:
                    slot = entry['slot']
                    trajectories[slot] = np.asarray(entry['trajectory'], np.float32)
                    for t, name in enumerate(entry['images']):
                        with Image.open(BytesIO(archive.read(name))) as image:
                            images.append(np.array(image.convert('RGB')))
                        assignments.append((slot, t))
        return token, trajectories, np.stack(images), np.asarray(assignments, np.int64)


def identity(value):
    return value


def image_batches(loader, minimum_images):
    pending, count = [], 0
    for item in loader:
        pending.append(item)
        count += len(item[2])
        if count >= minimum_images:
            yield pending
            pending, count = [], 0
    if pending:
        yield pending


def save_scene_features(output, item, embedding):
    token, trajectories, pixels, assignments = item
    if len(embedding) != len(assignments) or not np.isfinite(embedding).all():
        raise ValueError(f'Invalid teacher feature batch for {token}')
    features = np.zeros((len(trajectories), 8, embedding.shape[-1]), np.float16)
    valid = np.zeros((len(trajectories), 8), bool)
    for i, (slot, t) in enumerate(assignments):
        features[slot, t] = embedding[i]
        valid[slot, t] = True
    path = Path(output) / token[:2] / (token + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, features=features, valid=valid, trajectories=trajectories,
                 times=TIMES, teacher=np.asarray('vjepa2_1_vit_large_384'))
    temporary.replace(path)


def encode(args):
    from safetensors.torch import load_file
    from navsim.agents.drivoR_action_dit.vjepa2_1_image_encoder import build_official_vjepa21
    rank, world = args.rank, args.world_size
    if args.mode == 'encode-real':
        records = []
        with open(args.index) as handle:
            for i, line in enumerate(handle):
                if i % world == rank:
                    records.append(json.loads(line))
        if args.limit:
            records = records[:args.limit]
    else:
        records = scene_tokens(args.gt_root, args.tokens_json, args.limit)[rank::world]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        records = [r for r in records if not (output / (r if isinstance(r, str) else r['token'])[:2]
                   / ((r if isinstance(r, str) else r['token']) + '.npz')).is_file()]
    dataset = FutureImages(records, args)
    kwargs = dict(batch_size=None, num_workers=args.workers, collate_fn=identity)
    if args.workers:
        kwargs.update(multiprocessing_context='spawn', prefetch_factor=2)
    loader = DataLoader(dataset, **kwargs)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    model = build_official_vjepa21('vjepa2_1_vit_large_384')
    state = load_file(args.weights, device='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False).to(device)
    # These intermediate readouts are discarded by official inference; keep
    # only the final feature output, avoiding their allocation during caching.
    model.out_layers_distillation = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    started, count, image_count, last_report = time.monotonic(), 0, 0, -1
    with torch.no_grad():
        for items in image_batches(loader, args.batch_size):
            pixels = np.concatenate([item[2] for item in items])
            embeddings = []
            for start in range(0, len(pixels), args.batch_size):
                x = torch.from_numpy(pixels[start:start + args.batch_size]).to(device).permute(0, 3, 1, 2).float() / 255.
                x = (x - mean) / std
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    embedding = model(x.unsqueeze(2)).float().mean(dim=1)
                embeddings.append(embedding.cpu())
            embedding = torch.cat(embeddings).numpy()
            offset = 0
            for item in items:
                length = len(item[2])
                save_scene_features(output, item, embedding[offset:offset + length])
                offset += length
            count += len(items)
            image_count += len(pixels)
            if count // 100 != last_report:
                last_report = count // 100
                print(json.dumps({'rank': rank, 'scenes': count, 'images': image_count,
                                  'seconds': round(time.monotonic() - started, 1)}), flush=True)
    report = {'rank': rank, 'world_size': world, 'scenes_written': count, 'images': image_count,
              'mode': args.mode, 'teacher': 'vjepa2_1_vit_large_384', 'weights': args.weights,
              'times': TIMES.tolist(), 'size_wh': args.size, 'frozen': True, 'lora': False,
              'pooling': 'final_patch_mean_fp32',
              'seconds': round(time.monotonic() - started, 1)}
    (output / f'prepare_rank_{rank:04d}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('index-real', 'encode-real', 'encode-cosmos'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--index')
    parser.add_argument('--navsim', default=str(ROOT.parent / 'data/navsim'))
    parser.add_argument('--gt-root', default=str(DEFAULT_GT))
    parser.add_argument('--rollouts', default=DEFAULT_ROLLOUTS)
    parser.add_argument('--frames-output', default=str(ROOT / 'exp/world4scorer_data/frames_cosmos'))
    parser.add_argument('--extract-frames', action='store_true')
    parser.add_argument('--wait-for-frames', type=float, default=0,
                        help='Wait up to this many seconds per scene for a concurrent frame extractor')
    parser.add_argument('--tokens-json')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--size', type=int, nargs=2, default=[384, 384])
    parser.add_argument('--weights', default=str(ROOT / 'weights/vjepa2_1/vitl/ema_encoder.safetensors'))
    parser.add_argument('--device', default='cuda:' + os.environ.get('LOCAL_RANK', '0'))
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', '0')))
    parser.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE', '1')))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.mode == 'index-real':
        index_real(args)
    else:
        if args.mode == 'encode-real' and not args.index:
            parser.error('encode-real needs --index')
        encode(args)


if __name__ == '__main__':
    main()
