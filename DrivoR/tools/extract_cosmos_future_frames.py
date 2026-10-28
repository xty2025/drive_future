#!/usr/bin/env python3
"""Extract trajectory-aligned future images from Cosmos clips (index0 is t=0).

Eight future points use zero-based video frames5,10,...,40. Each scene is stored
as one ZIP of JPEGs plus explicit slot/time metadata, avoiding millions of files.
No input files are changed and no hashes are computed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
from PIL import Image


DEFAULT_ROLLOUTS = '/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_full_allcosmos_cuda_v1_gpu64'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_GT = PROJECT / 'multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu'
TIMES = np.arange(1, 9, dtype=np.float32) * 0.5


def scene_tokens(root, token_file=None, limit=None):
    if token_file:
        value = json.loads(Path(token_file).read_text())
        if isinstance(value, dict) and 'train' in value:
            tokens = [row['token'] for key in ('train', 'val') for row in value[key]]
        else:
            tokens = value['tokens'] if isinstance(value, dict) else value
    else:
        # scandir does not materialize/stat every directory as Path.iterdir can.
        with os.scandir(root) as entries:
            tokens = [entry.name for entry in entries if len(entry.name) == 16]
    return sorted(set(tokens))[:limit]


def valid_gt(root, token):
    folder = Path(root) / token
    metadata = json.loads((folder / 'trajectory.json').read_text())
    trajectories = np.load(folder / 'gt_trajectories.npy', allow_pickle=False).astype(np.float32)
    valid = np.load(folder / 'gt_valid_mask.npy', allow_pickle=False).astype(bool)
    valid &= ~np.asarray(metadata['gt_padded'], bool)
    return trajectories, np.flatnonzero(valid), metadata


def frame_indices(pose_times, times=TIMES):
    pose_times = np.asarray(pose_times)
    indices = np.abs(pose_times[:, None] - np.asarray(times)[None]).argmin(0)
    if not np.allclose(pose_times[indices], times, atol=1e-5, rtol=0):
        raise ValueError('Requested future times are not present in pose_times_s.npy')
    return indices


def decode_frames(video, indices, size):
    width, height = size
    selection = '+'.join(f'eq(n\\,{int(index)})' for index in indices)
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(video),
               '-vf', f'select={selection},scale={width}:{height}', '-vsync', 'vfr',
               '-frames:v', str(len(indices)), '-threads', '1', '-filter_threads', '1',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f'Video decode failed: {video}: {result.stderr.decode(errors="replace")[-1000:]}')
    expected = len(indices) * height * width * 3
    if len(result.stdout) != expected:
        raise ValueError(f'Expected {len(indices)} decoded future frames in {video}')
    return np.frombuffer(result.stdout, np.uint8).reshape(len(indices), height, width, 3)


def extract_scene(token, rollouts, gt_root, output, size=(384, 384), times=TIMES):
    output = Path(output) / token[:2] / (token + '.zip')
    if output.is_file():
        with zipfile.ZipFile(output) as archive:
            metadata = json.loads(archive.read('metadata.json'))
        if metadata['size_wh'] != list(size) or not np.allclose(metadata['times'], times):
            raise ValueError(f'Existing frame cache uses another size/time grid: {output}')
        return {'token': token, 'status': 'cached', 'frames': len(metadata['slots']) * len(times)}
    trajectories, slots, gt_metadata = valid_gt(gt_root, token)
    if not len(slots):
        raise ValueError(f'No valid unpadded GTs for {token}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    manifest = {'schema': 'cosmos_trajectory_frames_v1', 'token': token,
                'times': list(map(float, times)), 'size_wh': list(size),
                'source_rollouts': str(Path(rollouts).resolve()), 'slots': []}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
        for slot in slots:
            folder = Path(rollouts) / token / f'trajectory_{slot}'
            info = json.loads((folder / 'metadata.json').read_text())
            if not info['valid'] or not info['source_valid'] or info['source_padded']:
                raise ValueError(f'Invalid video for valid GT {token}/{slot}')
            if int(info['trajectory_index']) != slot or info['trajectory_label'] != gt_metadata['gt_labels'][slot]:
                raise ValueError(f'GT/video slot mismatch for {token}/{slot}')
            indices = frame_indices(np.load(folder / 'pose_times_s.npy', allow_pickle=False), times)
            decoded = decode_frames(folder / 'future.mp4', indices, size)
            names = []
            for index, pixels in zip(indices, decoded):
                name = f'slot_{slot}/frame_{index:06d}.jpg'
                buffer = BytesIO()
                Image.fromarray(pixels).save(buffer, format='JPEG', quality=90)
                archive.writestr(name, buffer.getvalue())
                names.append(name)
            manifest['slots'].append({'slot': int(slot), 'label': info['trajectory_label'],
                                      'frame_indices': indices.tolist(), 'images': names,
                                      'trajectory': trajectories[slot].tolist()})
        archive.writestr('metadata.json', json.dumps(manifest))
    temporary.replace(output)
    return {'token': token, 'status': 'extracted', 'frames': len(slots) * len(times)}


def _extract_job(args):
    token, rollouts, gt_root, output, size = args
    try:
        return extract_scene(token, rollouts, gt_root, output, size)
    except Exception as exc:
        return {'token': token, 'status': 'error', 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', default=DEFAULT_ROLLOUTS)
    parser.add_argument('--gt-root', default=str(DEFAULT_GT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens-json')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--size', type=int, nargs=2, default=(384, 384), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    tokens = scene_tokens(args.gt_root, args.tokens_json, args.limit)[args.shard_id::args.num_shards]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {'scenes': len(tokens), 'extracted': 0, 'cached': 0, 'errors': 0, 'frames': 0}
    started = time.monotonic()
    with (output / f'extract_{args.shard_id:04d}.jsonl').open('w') as report, ProcessPoolExecutor(args.workers) as pool:
        # Keep the number of queued scenes bounded, especially on shared CPFS.
        for start in range(0, len(tokens), args.workers * 4):
            batch = tokens[start:start + args.workers * 4]
            jobs = [(token, args.rollouts, args.gt_root, args.output, args.size) for token in batch]
            for result in pool.map(_extract_job, jobs):
                report.write(json.dumps(result) + '\n')
                summary['errors' if result['status'] == 'error' else result['status']] += 1
                summary['frames'] += result.get('frames', 0)
            report.flush()
            print(json.dumps(dict(summary, seconds=round(time.monotonic() - started, 1))), flush=True)
            if summary['errors'] >= 20:
                raise RuntimeError('Repeated extraction failures; inspect the JSONL report before resuming')
    (output / f'extract_{args.shard_id:04d}.json').write_text(json.dumps(summary, indent=2) + '\n')
    return int(summary['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
