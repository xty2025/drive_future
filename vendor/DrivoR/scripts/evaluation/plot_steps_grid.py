#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    with open(path, 'rb') as f: return pickle.load(f)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pkl', nargs=4, required=True); ap.add_argument('--output', required=True); ap.add_argument('--scenes', type=int, default=16); args=ap.parse_args()
    data=[load(x) for x in args.pkl]
    common=sorted(set(data[0]) & set(data[1]) & set(data[2]) & set(data[3]))[:args.scenes]
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    colors=['#d62728','#ff7f0e','#2ca02c','#1f77b4']
    for token in common:
        fig,axs=plt.subplots(4,4,figsize=(16,16),squeeze=False,sharex=True,sharey=True)
        gt=np.asarray(data[0][token]['gt_trajectory'])
        step4_scores=np.asarray(data[3][token]['learned_scores'])
        candidate_indices=np.argsort(step4_scores)[::-1][:16]
        selected=[]
        for d in data:
            tr=np.asarray(d[token]['final_proposals'])
            selected.append(tr[candidate_indices,:,:2])
        all_xy=np.concatenate([gt[None,:,:2],*selected],axis=0).reshape(-1,2)
        x_min,y_min=all_xy.min(axis=0); x_max,y_max=all_xy.max(axis=0)
        span=max(x_max-x_min,y_max-y_min,1.0); pad=0.08*span
        for rank,(ax,candidate_idx) in enumerate(zip(axs.flat,candidate_indices),1):
            ax.plot(gt[:,0],gt[:,1],'k--',lw=2,label='GT')
            for j,d in enumerate(data):
                tr=np.asarray(d[token]['final_proposals'])
                ax.plot(tr[candidate_idx,:,0],tr[candidate_idx,:,1],color=colors[j],lw=1.8,label=f'step {j+1}')
                ax.scatter(tr[candidate_idx,-1,0],tr[candidate_idx,-1,1],color=colors[j],s=12,zorder=3)
            ax.scatter([0],[0],c='black',s=14,zorder=4)
            ax.set_xlim(x_min-pad,x_max+pad); ax.set_ylim(y_min-pad,y_max+pad)
            ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.2)
            ax.set_title(f'rank {rank} | candidate {candidate_idx} | score4={step4_scores[candidate_idx]:.3f}')
        for ax in axs[-1,:]: ax.set_xlabel('x [m]')
        for ax in axs[:,0]: ax.set_ylabel('y [m]')
        handles,labels=axs[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=5)
        fig.suptitle(f'DrivoR step comparison with GT | {token}')
        fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out/f'{token}_steps_grid.png',dpi=150); plt.close(fig)
    print(f'wrote {len(common)} scenes to {out}')

if __name__=='__main__': main()
