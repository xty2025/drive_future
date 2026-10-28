#!/usr/bin/env python3
"""Convert the official CLOVER candidate package to World4Scorer's four NPYs.

Uses the reference's first 64 finite candidates per valid scene, with their
precomputed labels. --preview-scenes reads a small prefix for real-data smoke;
the full conversion should omit it. No new trajectories or labels are invented.
"""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

SCORE_KEYS = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
              'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'pdm_score']


def read_scenes(path, preview=None):
    if preview is None:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        if isinstance(data, dict):
            for name in ('scenes', 'data', 'items'):
                if name in data:
                    return data[name]
            raise ValueError('CLOVER dictionary needs a scenes/data/items list')
        return data

    # The official release is a large list of scene dictionaries. A preview can
    # stop after complete scene dictionaries, without loading the entire bank.
    class Complete(Exception):
        pass

    class Preview(pickle._Unpickler):
        dispatch = pickle._Unpickler.dispatch.copy()

        def load_setitems(self):
            super().load_setitems()
            value = self.stack[-1]
            if isinstance(value, dict) and {'token', 'trajectories_relative', 'scores', 'valid'} <= value.keys():
                scenes.append(value)
                if len(scenes) >= preview:
                    raise Complete

        dispatch[pickle.SETITEMS[0]] = load_setitems

    scenes = []
    with open(path, 'rb') as handle:
        try:
            Preview(handle).load()
        except Complete:
            pass
    return scenes


def score_array(scores, count):
    if isinstance(scores, dict):
        return np.stack([np.asarray(scores[key], np.float32).reshape(count) for key in SCORE_KEYS], 1)
    return np.asarray([[row[key] for key in SCORE_KEYS] for row in scores], np.float32).reshape(count, 7)


def build(source, output, capacity=64, preview=None):
    source_scenes = read_scenes(source, preview)
    valid_scenes = [scene for scene in source_scenes if bool(np.all(scene.get('valid', True)))]
    # The official package also contains valid=True entries with no trajectories.
    # CLOVER's own loader skips these; they have no labels to train against.
    scenes = [scene for scene in valid_scenes
              if len(scene.get('trajectories_relative', [])) and len(scene.get('scores', []))]
    empty_scenes = len(valid_scenes) - len(scenes)
    if not scenes:
        raise ValueError('No valid scenes in the CLOVER package')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').unlink(missing_ok=True)
    shapes = {'tokens': ((len(scenes),), 'S64'),
              'bank_candidates': ((len(scenes), capacity, 8, 3), np.float32),
              'bank_subscores': ((len(scenes), capacity, 7), np.float32),
              'bank_mask': ((len(scenes), capacity), bool)}
    arrays = {key: np.lib.format.open_memmap(output / (key+'.npy'), mode='w+', dtype=dtype, shape=shape)
              for key, (shape, dtype) in shapes.items()}
    valid_count = unsafe_count = low_score_count = 0
    for row, scene in enumerate(scenes):
        trajectories = np.asarray(scene['trajectories_relative'], np.float32)[:, :8, :3]
        labels = score_array(scene['scores'], len(trajectories))
        keep = np.flatnonzero(np.isfinite(trajectories).all((1, 2)) & np.isfinite(labels).all(1))[:capacity]
        n = len(keep)
        arrays['tokens'][row] = str(scene['token']).encode()
        arrays['bank_candidates'][row] = 0
        arrays['bank_subscores'][row] = 0
        arrays['bank_mask'][row] = False
        arrays['bank_candidates'][row, :n] = trajectories[keep]
        arrays['bank_subscores'][row, :n] = labels[keep]
        arrays['bank_mask'][row, :n] = True
        valid_count += n
        unsafe_count += int(((labels[keep, 0] < .5) | (labels[keep, 1] < .5)).sum())
        low_score_count += int((labels[keep, -1] < .5).sum())
    for array in arrays.values():
        array.flush()
    report = {'source': str(Path(source).resolve()), 'source_repository': 'https://github.com/WilliamXuanYu/CLOVER',
              'source_file_id': '1JUKc9hQ2YH7Ck7ghjZm65OCavWVg6Po_',
              'preview': preview is not None, 'num_tokens': len(scenes), 'empty_scenes_skipped': empty_scenes,
              'bank_k': capacity,
              'bank_subscore_keys': SCORE_KEYS, 'mean_valid_per_scene': valid_count / len(scenes),
              'valid_candidates': valid_count, 'unsafe_candidates': unsafe_count,
              'candidates_pdms_below_0p5': low_score_count}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--capacity', type=int, default=64)
    parser.add_argument('--preview-scenes', type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.capacity, args.preview_scenes)), flush=True)
