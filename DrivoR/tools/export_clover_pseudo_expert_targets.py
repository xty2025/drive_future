#!/usr/bin/env python3
"""Materialize the final per-scene CLOVER expert set, including human fallback.

Uses the existing training selection functions and actual NAVSIM logged poses.
Only reads trajectory/log metadata; no images, GPUs or online scoring are used.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import time

import numpy as np
from pyquaternion import Quaternion
from nuplan.common.actor_state.state_representation import StateSE2

from build_world4scorer_bank import read_scenes
from navsim.agents.drivoR_pseudo_expert.selection import select_candidates, include_human_expert
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array


def load_human_log(job):
    path, rows = job
    with open(path, 'rb') as handle:
        frames = pickle.load(handle)
    index = {frame['token']: position for position, frame in enumerate(frames)}
    poses = np.asarray([
        [frame['ego2global_translation'][0], frame['ego2global_translation'][1],
         Quaternion(*frame['ego2global_rotation']).yaw_pitch_roll[0]]
        for frame in frames], dtype=np.float64)
    output = []
    for row, token in rows:
        position = index[token]
        if position + 8 >= len(frames):
            raise ValueError(f'{token} has fewer than 8 future frames in {path}')
        current = poses[position]
        future = convert_absolute_to_relative_se2_array(
            StateSE2(*current), poses[position+1:position+9])
        output.append((row, future.astype(np.float32), current))
    return output


def select_with_indices(scene, top_k, score_thr):
    trajectories = np.asarray(scene.get('trajectories_relative', []), dtype=np.float32)
    scores = np.asarray([value['pdm_score'] for value in scene.get('scores', [])])
    if not scene.get('valid', False) or not len(trajectories) or not len(scores):
        return (np.zeros((top_k,8,3),np.float32), np.zeros(top_k,bool),
                np.full(top_k,-2,np.int32), np.full(top_k,np.nan,np.float32))
    selected, valid = select_candidates(trajectories, scores, top_k, score_thr)
    eligible = np.flatnonzero(np.isfinite(trajectories).all((1,2)) & np.isfinite(scores) & (scores>=score_thr))
    ordered = eligible[np.argsort(-scores[eligible],kind='stable')]
    indices = np.full(top_k,-2,np.int32)
    selected_scores = np.full(top_k,np.nan,np.float32)
    for slot in np.flatnonzero(valid):
        # Stable score order resolves duplicate-coordinate candidates exactly
        # like select_candidates; do not accidentally attribute a lower score.
        matches = np.flatnonzero((trajectories[ordered]==selected[slot]).all((1,2)))
        original_index = ordered[matches[0]]
        indices[slot] = original_index
        selected_scores[slot] = scores[original_index]
    return selected, valid, indices, selected_scores


def export(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    print(f'Loading candidates: {args.source}',flush=True)
    source_scenes = read_scenes(args.source,args.preview_scenes)
    scenes = [scene for scene in source_scenes if scene.get('split')=='navtrain']
    if not scenes:
        raise ValueError('Source contains no navtrain scenes')
    tokens = [str(scene['token']) for scene in scenes]
    if len(set(tokens))!=len(tokens):
        raise ValueError('Source has duplicate scene tokens')
    count = len(scenes)
    top_k = args.top_k
    specs = {
        'tokens': ((count,), 'S64'),
        'log_names': ((count,), 'S128'),
        'pseudo_experts': ((count,top_k,8,3), np.float32),
        'pseudo_expert_mask': ((count,top_k), bool),
        'pseudo_expert_is_human': ((count,top_k), bool),
        'pseudo_expert_scores': ((count,top_k), np.float32),
        'source_candidate_indices': ((count,top_k), np.int32),
        'scored_expert_count': ((count,), np.int16),
        'pseudo_expert_fallback': ((count,), bool),
        'human_trajectories': ((count,8,3), np.float32),
        'ego_current_global': ((count,3), np.float64),
    }
    arrays = {name:np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
              for name,(shape,dtype) in specs.items()}
    arrays['tokens'][:] = np.asarray(tokens,dtype='S64')
    groups = defaultdict(list)
    for row,scene in enumerate(scenes):
        # Official source stores .../<log_name>/unknown/<token>/metric_cache.pkl.
        log_name = Path(scene['cache_path']).parents[2].name
        arrays['log_names'][row] = log_name.encode()
        groups[args.log_root/(log_name+'.pkl')].append((row,tokens[row]))
    print(f'navtrain scenes={count}; reading human GT from {len(groups)} local logs',flush=True)
    jobs = list(groups.items())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        completed = 0
        for log_number,records in enumerate(pool.map(load_human_log,jobs),1):
            for row,human,current in records:
                arrays['human_trajectories'][row] = human
                arrays['ego_current_global'][row] = current
            completed += len(records)
            if log_number%100==0 or log_number==len(jobs):
                print(f'Human GT: logs={log_number}/{len(jobs)}, scenes={completed}/{count}',flush=True)

    reference = None
    if args.reference_pack:
        reference = {
            'index':{token.decode():row for row,token in enumerate(np.load(args.reference_pack/'tokens.npy'))},
            'trajectories':np.load(args.reference_pack/'pseudo_experts.npy',mmap_mode='r'),
            'mask':np.load(args.reference_pack/'pseudo_expert_mask.npy',mmap_mode='r'),
        }
    scored_histogram = Counter()
    final_histogram = Counter()
    human_added = fallback = verified = source_candidates = 0
    for row,scene in enumerate(scenes):
        experts,mask,indices,scores = select_with_indices(scene,top_k,args.score_thr)
        source_candidates += len(scene.get('trajectories_relative',[]))
        scored_count = int(mask.sum())
        scored_histogram[scored_count] += 1
        arrays['scored_expert_count'][row] = scored_count
        arrays['pseudo_expert_fallback'][row] = scored_count==0
        if reference and tokens[row] in reference['index']:
            old_row = reference['index'][tokens[row]]
            np.testing.assert_array_equal(experts,reference['trajectories'][old_row])
            np.testing.assert_array_equal(mask,reference['mask'][old_row])
            verified += 1
        experts,final_mask = include_human_expert(experts,mask,arrays['human_trajectories'][row])
        is_human = final_mask & ~mask
        indices[is_human] = -1
        arrays['pseudo_experts'][row] = experts
        arrays['pseudo_expert_mask'][row] = final_mask
        arrays['pseudo_expert_is_human'][row] = is_human
        arrays['pseudo_expert_scores'][row] = scores
        arrays['source_candidate_indices'][row] = indices
        human_added += int(is_human.sum())
        fallback += int(scored_count==0)
        final_histogram[int(final_mask.sum())] += 1
        if (row+1)%10000==0 or row+1==count:
            print(f'Fixed expert sets: {row+1}/{count}',flush=True)
    for value in arrays.values():
        value.flush()
    valid = arrays['pseudo_expert_mask']
    if not valid.any(-1).all() or not np.isfinite(arrays['pseudo_experts'][valid]).all():
        raise ValueError('Final expert set contains an empty scene or a non-finite trajectory')
    for row in np.flatnonzero(arrays['pseudo_expert_is_human'].any(-1)):
        np.testing.assert_array_equal(arrays['pseudo_experts'][row,arrays['pseudo_expert_is_human'][row]][0],
                                      arrays['human_trajectories'][row])
    report = dict(
        status='complete', source=str(args.source.resolve()), split='navtrain',
        log_root=str(args.log_root.resolve()), selection='score_filter_then_greedy_xy_fps',
        top_k=top_k, score_thr=args.score_thr, fps_seed='highest_score_stable_order',
        distance='mean(abs(xy_a-xy_b)) over 8 time points and 2 coordinates',
        diversity_min_distance=0.05, human_coverage_distance=0.5, human_inclusion='offline_finalized',
        human_rule='append only if uncovered and capacity remains; human-only when no scored candidate',
        coordinate_frame='current_ego_rear_axle_relative', axes=['x_m','y_m','heading_rad'],
        target_times=(np.arange(1,9)*.5).tolist(), trajectory_normalized=False,
        num_tokens=count, source_candidates=source_candidates, human_added_scenes=human_added,
        human_only_fallback_scenes=fallback, source_total_scenes=len(source_scenes),
        scored_expert_count_histogram={str(k):scored_histogram[k] for k in range(top_k+1)},
        final_expert_count_histogram={str(k):final_histogram[k] for k in range(top_k+1)},
        total_valid_trajectories=int(valid.sum()),
        reference_scored_cache=str(args.reference_pack.resolve()) if reference else None,
        reference_identical_scenes=verified, preview=args.preview_scenes is not None,
        elapsed_seconds=round(time.monotonic()-started,2),
        files={name+'.npy':dict(shape=list(shape),dtype=str(np.dtype(dtype))) for name,(shape,dtype) in specs.items()},
    )
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'README.md').write_text(f'''# 固定 pseudo-expert 轨迹集合

按当前训练规则选取，已包含必要的 human GT 补入；每个场景最终 1～{top_k} 条。
先筛选 PDMS ≥ {args.score_thr}，以最高分候选为起点进行最远点采样。
轨迹之间使用 8 点平均 XY 绝对差；最大剩余距离 <0.05m 时停止。
FPS 后若不足 {top_k} 条且没有距离 human GT <0.5m 的候选，则补入 human GT。
没有合格候选的场景只使用 human GT；满 {top_k} 条时不替换候选。

轨迹为当前帧后轴坐标系的 `[x米,y米,heading弧度]`，未归一化，
8 个点对应未来 `0.5,1.0,...,4.0` 秒，不包含当前点。不插值或生成新轨迹。
`ego_current_global.npy` 保存坐标变换基准，`human_trajectories.npy` 来自实际 NAVSIM 日志，
与训练时 `Scene.get_future_trajectory(8)` 的坐标转换相同。

所有 NPY 按同一行顺序对应 `tokens.npy`，可用 mmap 读取：

```python
from pathlib import Path
import numpy as np

root = Path({str(output)!r})
tokens = np.load(root / "tokens.npy")
index = {{token.decode(): i for i, token in enumerate(tokens)}}
trajectories = np.load(root / "pseudo_experts.npy", mmap_mode="r")
valid = np.load(root / "pseudo_expert_mask.npy", mmap_mode="r")
i = index["357a415dcb8b55a6"]
scene_trajectories = trajectories[i][valid[i]]  # [K,8,3]
```

- `pseudo_expert_is_human.npy`：本次按规则补入的 human 标记。
- `pseudo_expert_scores.npy`：原 PKL 中候选的 PDMS；补入 human 与 padding 为 NaN，没有伪造评分。
- `source_candidate_indices.npy`：该场景原 PKL 的候选下标；human=-1，padding=-2。
- `scored_expert_count.npy`：补入 human 前的高分 FPS 候选数量。
- `pseudo_expert_fallback.npy`：是否没有高分 FPS 候选、仅回退 human。
- `log_names.npy`：每场景对应的实际 NAVSIM 日志名。
- `manifest.json`：筛选规则、形状、数量与核对统计。

该目录是独立导出结果；现有训练配置仍指向原缓存。
''')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--log-root',type=Path,required=True)
    parser.add_argument('--reference-pack',type=Path)
    parser.add_argument('--top-k',type=int,default=8)
    parser.add_argument('--score-thr',type=float,default=.8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--preview-scenes',type=int)
    export(parser.parse_args())
