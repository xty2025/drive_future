#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

def gt_local(cache):
    s=TrajectorySampling(time_horizon=4, interval_length=.5)
    a=np.asarray(get_trajectory_as_array(cache.trajectory,s,cache.ego_state.time_point))
    x0,y0,h=cache.ego_state.rear_axle.x,cache.ego_state.rear_axle.y,cache.ego_state.rear_axle.heading
    d=a[1:,:2]-[x0,y0]; c,sn=np.cos(h),np.sin(h)
    return np.c_[c*d[:,0]+sn*d[:,1],-sn*d[:,0]+c*d[:,1]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--set',required=True); ap.add_argument('--metric-cache',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    def step_path(k):
        flat=a.root/f'{a.set}_step{k}.pkl'
        return flat if flat.exists() else a.root/a.set/f'step{k}.pkl'
    data={k:pickle.load(open(step_path(k),'rb')) for k in range(1,5)}; cache=MetricCacheLoader(a.metric_cache); a.output.mkdir(parents=True,exist_ok=True)
    colors=['#e41a1c','#ff7f00','#4daf4a','#377eb8']
    for tok in sorted(data[1]):
        gt=gt_local(cache.get_from_token(tok)); fig,ax=plt.subplots(figsize=(8,6)); ax.plot(gt[:,0],gt[:,1],'k--',lw=3,label='GT (8 × 0.5 s)')
        pts=[gt]
        for k,col in zip(range(1,5),colors):
            tr=np.asarray(data[k][tok]['trajectory'].poses); pts.append(tr[:,:2]); ax.plot(tr[:,0],tr[:,1],'-o',ms=2.5,lw=2,color=col,label=f'{k} step')
        z=np.concatenate(pts); pad=max(1.,.12*max(np.ptp(z[:,0]),np.ptp(z[:,1]))); ax.set_xlim(z[:,0].min()-pad,z[:,0].max()+pad); ax.set_ylim(z[:,1].min()-pad,z[:,1].max()+pad)
        label=('DrivoR refinement candidates + ChainFlow scorer' if 'drivor_proposals' in data[4][tok] else 'ChainFlow diffusion candidates + DrivoR scorer')
        ax.set_aspect('equal'); ax.grid(alpha=.25); ax.set_xlabel('longitudinal x [m]'); ax.set_ylabel('lateral y [m]'); ax.set_title(f'{a.set} | {tok}\n{label}'); ax.legend(); fig.tight_layout(); fig.savefig(a.output/f'{tok}_steps_vs_gt.png',dpi=180); plt.close(fig)
    # Diversity at step 4, all 64 candidates plus DrivoR winner and GT.
    for tok,item in data[4].items():
        cand=np.asarray(item.get('diffusion_proposals', item.get('drivor_proposals'))); gt=gt_local(cache.get_from_token(tok)); win=item.get('drivor_selected_index', item.get('chainflow_selected_index')); fig,ax=plt.subplots(figsize=(8,6))
        for tr in cand: ax.plot(tr[:,0],tr[:,1],color='#4c78a8',alpha=.16,lw=.8)
        ax.scatter(cand[:,-1,0],cand[:,-1,1],s=8,color='#4c78a8',alpha=.35,label='64 endpoints'); ax.plot(cand[win,:,0],cand[win,:,1],color='#e45756',lw=3,label=f'learned scorer winner #{win}'); ax.plot(gt[:,0],gt[:,1],'k--',lw=2.5,label='GT'); z=np.concatenate([cand[:,:,:2].reshape(-1,2),gt]); pad=max(1.,.08*max(np.ptp(z[:,0]),np.ptp(z[:,1]))); ax.set_xlim(z[:,0].min()-pad,z[:,0].max()+pad); ax.set_ylim(z[:,1].min()-pad,z[:,1].max()+pad); ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(); ax.set_title(f'{a.set} step 4 diversity | {tok}'); fig.tight_layout(); (a.output/'diversity').mkdir(exist_ok=True); fig.savefig(a.output/'diversity'/f'{tok}_64_candidates.png',dpi=180); plt.close(fig)

if __name__=='__main__': main()
