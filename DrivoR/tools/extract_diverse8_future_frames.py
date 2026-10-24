#!/usr/bin/env python3
"""Warm the shared diverse8 RGB cache used by the six register experiments."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.diverse_future_frames import (
    DEFAULT_DIVERSE_GT, DEFAULT_DIVERSE_ROLLOUTS, extract_diverse_scene_frames,
)


def extract(args):
    token, gt, rollouts, output, size = args
    try:
        path = Path(output) / token[:2] / (token + '.zip')
        cached = path.is_file()
        extract_diverse_scene_frames(token, gt, rollouts, output, size)
        return dict(token=token, status='cached' if cached else 'extracted')
    except Exception as error:
        return dict(token=token, status='error', error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-root', default=DEFAULT_DIVERSE_GT)
    parser.add_argument('--rollouts', default=DEFAULT_DIVERSE_ROLLOUTS)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'exp/world4scorer_data/frames_diverse8'))
    parser.add_argument('--tokens-json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--size', nargs=2, type=int, default=(384, 384))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error('Use positive workers and 0 <= shard-id < num-shards')
    if args.tokens_json:
        source = json.loads(Path(args.tokens_json).read_text())
        if isinstance(source, dict) and 'train' in source:
            tokens = [r['token'] for key in ('train', 'val') for r in source[key]]
        else:
            tokens = source['tokens'] if isinstance(source, dict) else source
    else:
        with os.scandir(args.gt_root) as entries:
            tokens = [e.name for e in entries if len(e.name) == 16 and e.is_dir()]
    tokens = sorted(set(tokens))[:args.limit][args.shard_id::args.num_shards]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summary = dict(scenes=len(tokens), extracted=0, cached=0, error=0)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool, (root / f'extract_{args.shard_id:04d}.jsonl').open('w') as log:
        for start in range(0, len(tokens), args.workers * 4):
            jobs = [(token, args.gt_root, args.rollouts, args.output, args.size)
                    for token in tokens[start:start + args.workers * 4]]
            for result in pool.map(extract, jobs):
                summary[result['status']] += 1
                log.write(json.dumps(result) + '\n')
            log.flush()
            print(json.dumps(dict(summary, seconds=round(time.monotonic() - started, 1))), flush=True)
    (root / f'extract_{args.shard_id:04d}.json').write_text(json.dumps(summary, indent=2) + '\n')
    return int(summary['error'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
