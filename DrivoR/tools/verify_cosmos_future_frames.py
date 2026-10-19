#!/usr/bin/env python3
"""Audit all frame archives and compare sampled images/poses with their sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from extract_cosmos_future_frames import DEFAULT_GT, DEFAULT_ROLLOUTS, TIMES, scene_tokens, decode_frames


def inspect_archive(item):
    token, root = item
    try:
        with zipfile.ZipFile(Path(root)/token[:2]/(token+'.zip')) as archive:
            meta=json.loads(archive.read('metadata.json'))
            assert meta['token']==token and meta['size_wh']==[384,384]
            assert np.allclose(meta['times'],TIMES,atol=1e-6,rtol=0)
            slots=meta['slots'];seen=set();names=set(archive.namelist())
            assert 1<=len(slots)<=4
            for entry in slots:
                slot=entry['slot']
                assert slot in range(4) and slot not in seen
                seen.add(slot)
                assert entry['frame_indices']==[5,10,15,20,25,30,35,40]
                expected=[f'slot_{slot}/frame_{i:06d}.jpg' for i in entry['frame_indices']]
                assert entry['images']==expected and set(expected)<=names
                assert all(archive.getinfo(name).file_size>0 for name in expected)
                assert np.asarray(entry['trajectory']).shape==(8,3)
        return {'token':token,'gt_count':len(slots),'frames':len(slots)*8}
    except Exception as exc:
        return {'token':token,'error':repr(exc)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',required=True)
    parser.add_argument('--gt-root',default=str(DEFAULT_GT))
    parser.add_argument('--rollouts',default=DEFAULT_ROLLOUTS)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sample-scenes',type=int,default=32)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    tokens=scene_tokens(args.gt_root)
    errors=[];gt_count=0;frame_count=0
    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(0,len(tokens),512):
            for result in pool.map(inspect_archive,[(t,args.frames) for t in tokens[start:start+512]]):
                if 'error' in result:errors.append(result)
                else:gt_count+=result['gt_count'];frame_count+=result['frames']
            if start%8192==0:print(json.dumps({'checked':min(start+512,len(tokens)),'errors':len(errors)}),flush=True)
    report={'scenes':len(tokens),'valid_gt_trajectories':gt_count,'frames':frame_count,
            'frame_indices':[5,10,15,20,25,30,35,40],'metadata_errors':errors,
            'sampled_scenes':0,'sampled_images':0,'max_pose_error':0.,'max_jpeg_mean_absolute_error':0.}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if not errors:
        sample=np.random.default_rng(17).choice(tokens,min(args.sample_scenes,len(tokens)),replace=False)
        for token in sample:
            gt_meta=json.loads((Path(args.gt_root)/token/'trajectory.json').read_text())
            gt=np.load(Path(args.gt_root)/token/'gt_trajectories.npy',allow_pickle=False)
            mask=np.load(Path(args.gt_root)/token/'gt_valid_mask.npy',allow_pickle=False).astype(bool)
            mask &= ~np.asarray(gt_meta['gt_padded'],bool)
            with zipfile.ZipFile(Path(args.frames)/token[:2]/(token+'.zip')) as archive:
                meta=json.loads(archive.read('metadata.json'))
                assert [e['slot'] for e in meta['slots']]==np.flatnonzero(mask).tolist()
                for entry in meta['slots']:
                    slot=entry['slot'];folder=Path(args.rollouts)/token/f'trajectory_{slot}'
                    pose=np.load(folder/'ego_pose.npy',allow_pickle=False)[entry['frame_indices']]
                    error=float(np.abs(pose-gt[slot]).max())
                    report['max_pose_error']=max(report['max_pose_error'],error)
                    np.testing.assert_allclose(pose,gt[slot],atol=1e-3,rtol=1e-5)
                    # Archive/model coordinates deliberately use float32; compare
                    # with the same representation rather than a float64 source.
                    np.testing.assert_array_equal(entry['trajectory'],gt[slot].astype(np.float32))
                    decoded=decode_frames(folder/'future.mp4',[5,40],(384,384))
                    for j,name in enumerate((entry['images'][0],entry['images'][-1])):
                        with Image.open(BytesIO(archive.read(name))) as image:
                            assert image.size==(384,384)
                            pixels=np.array(image.convert('RGB'))
                        mae=float(np.abs(pixels.astype(np.float32)-decoded[j]).mean())
                        report['max_jpeg_mean_absolute_error']=max(report['max_jpeg_mean_absolute_error'],mae)
                        assert mae<12, (token,slot,name,mae)
                        report['sampled_images']+=1
            report['sampled_scenes']+=1
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return int(bool(errors))


if __name__=='__main__':
    raise SystemExit(main())
