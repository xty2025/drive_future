from pathlib import Path
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'/mnt/workspace/codex0916/outputs/nvs_drivor_nav1_long_20261009/src')
from nvs_v6.rollout_data import load_paired_rollout_sample

ART=Path('/mnt/workspace/codex0916/outputs/nvs_drivor_nav1_long_20261009/artifacts/nav1_long_refine')
OUT=Path('/mnt/zhaozc_workspace/project/NVS/artifacts/codex0916_nav1_long_refine_visualizations'); OUT.mkdir(parents=True,exist_ok=True)
ROLL=Path('/mnt/zhaozc_workspace/project/multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2')
CACHE='/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache'
SCENES=['8f4a85d43ef351f9','b00d83c03bbf5b47']

def match(pred,gt):
 c=np.linalg.norm(pred[:,None,:,:2]-gt[None,:,:,:2],axis=-1).mean(axis=-1)
 candidate_idx, gt_idx=linear_sum_assignment(c)
 return gt_idx, candidate_idx, c
def metrics(p):
 dt=.5; xy=p[:,:2]
 vel=np.diff(xy,axis=0)/dt
 acc=np.diff(xy,n=2,axis=0)/(dt*dt)
 yaw=np.unwrap(p[:,2]); yawrate=np.diff(yaw)/dt
 return {'mean_speed_mps':float(np.linalg.norm(vel,axis=1).mean()),'max_speed_mps':float(np.linalg.norm(vel,axis=1).max()),'mean_accel_mps2':float(np.linalg.norm(acc,axis=1).mean()) if len(acc) else 0.,'max_accel_mps2':float(np.linalg.norm(acc,axis=1).max()) if len(acc) else 0.,'max_yaw_step_rad':float(np.abs(np.diff(yaw)).max()),'max_yawrate_rps':float(np.abs(yawrate).max())}

def main():
 cand=np.load(ART/'generated_trajectories.npy'); pdm=np.load(ART/'official_raw7_scores.npy')[:,:,6]; res=json.load(open(ART/'result.json')); rows=[]
 for b,scene in enumerate(SCENES):
  s=load_paired_rollout_sample(ROLL/scene,CACHE); gt=s.targets.gt_trajectory[0].numpy(); gi,ci,c=match(cand[b],gt)
  ev=next(x for x in res['evaluation'] if x['scene_id']==scene); learned=int(ev['learned_scorer_candidate']); oracle=int(ev['oracle_best_candidate']); pre=int(ev['pretrained_scorer_candidate'])
  colors=['#e41a1c','#377eb8','#4daf4a','#984ea3']
  fig,ax=plt.subplots(figsize=(10,8),dpi=180)
  for i in range(64):
   q=np.vstack([np.zeros((1,3)),cand[b,i]])
   is_sup=i in ci
   ax.plot(q[:,0],q[:,1],color=colors[int(np.where(ci==i)[0][0])] if is_sup else '#bdbdbd',alpha=.95 if is_sup else .23,lw=2.8 if is_sup else .75,zorder=5 if is_sup else 1)
  for k,(g,i) in enumerate(zip(gi,ci)):
   q=np.vstack([np.zeros((1,3)),gt[g]])
   ax.plot(q[:,0],q[:,1],color=colors[k],ls='--',lw=2.2,label=f'supervised GT{k} ↔ cand#{i} (ADE {c[i,g]:.2f}m)',zorder=6)
  for i,col,label in [(learned,'#00a651','learned'),(pre,'#ff7f00','DrivoR pretrained'),(oracle,'#000000','oracle')]:
   q=np.vstack([np.zeros((1,3)),cand[b,i]]); ax.plot(q[:,0],q[:,1],color=col,lw=3.5,label=f'{label} #{i} PDM={pdm[b,i]:.3f}',zorder=8)
  ax.scatter([0],[0],c='navy',s=35,zorder=10,label='ego origin'); ax.set_aspect('equal'); ax.grid(alpha=.2); ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)'); ax.set_title(f'{scene}: Hungarian-supervised candidates highlighted'); ax.legend(fontsize=8,loc='best'); fig.tight_layout(); fig.savefig(OUT/f'{scene}_bev_supervised_candidates.png'); plt.close(fig)
  smooth={'learned':metrics(cand[b,learned]),'oracle':metrics(cand[b,oracle]),'pretrained':metrics(cand[b,pre]),'supervised':{str(int(i)):metrics(cand[b,i]) for i in ci}}
  rows.append({'scene_id':scene,'supervised_pairs':[{'gt_index':int(g),'candidate_index':int(i),'ade_m':float(c[i,g]),'pdm':float(pdm[b,i]),'smoothness':smooth['supervised'][str(int(i))]} for g,i in zip(gi,ci)],'learned_candidate':learned,'oracle_candidate':oracle,'pretrained_candidate':pre,'smoothness':smooth})
 (OUT/'supervised_candidates_manifest.json').write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
