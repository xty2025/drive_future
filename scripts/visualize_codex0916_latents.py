from pathlib import Path
import json, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path('/mnt/workspace/codex0916/outputs/nvs_drivor_nav1_long_20261009')
ART = ROOT/'artifacts/nav1_long_refine'
SRC = ROOT/'src'
sys.path.insert(0, str(SRC))
from nvs_v6.rollout_data import load_paired_rollout_sample
from nvs_v6.model import DirectedPairDiT
from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.teacher import Ordered2x2Compressor, TimmDinoV3Teacher

ROLL = Path('/mnt/zhaozc_workspace/project/multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2')
CACHE = '/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache'
DINO = '/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors'
SCENES = ['8f4a85d43ef351f9','b00d83c03bbf5b47']
OUT = Path('/mnt/zhaozc_workspace/project/NVS/artifacts/codex0916_nav1_long_refine_visualizations')
OUT.mkdir(parents=True, exist_ok=True)

def _ordered_vectors(teacher_grid: torch.Tensor) -> torch.Tensor:
    _, _, height, width, _ = teacher_grid.shape
    vectors = []
    for chunk in range(4):
        regions = []
        for frame_index in (2 * chunk, 2 * chunk + 1):
            frame = teacher_grid[:, frame_index]
            for row in range(2):
                for col in range(2):
                    region = frame[:, row * height // 2:(row + 1) * height // 2,
                                   col * width // 2:(col + 1) * width // 2]
                    regions.append(region.mean(dim=(1, 2)))
        vectors.append(torch.cat(regions, dim=-1))
    return torch.stack(vectors, dim=1)

def cosine(a,b):
    a=a.float(); b=b.float()
    return torch.nn.functional.cosine_similarity(a,b,dim=-1)

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples=[load_paired_rollout_sample(ROLL/s,CACHE) for s in SCENES]
    # Reconstruct exactly the future target compressor used by the training run.
    teacher=TimmDinoV3Teacher(checkpoint=DINO).to(device)
    grids=[]
    with torch.no_grad():
        for sample in samples:
            per=[]
            for k in range(sample.future_rgb.shape[1]):
                per.append(teacher(sample.future_rgb[:,k].to(device)).float().cpu())
            grids.append(torch.cat(per,dim=0))
    del teacher
    comp=Ordered2x2Compressor(output_dim=256)
    comp.fit(torch.cat([_ordered_vectors(g) for g in grids]), source_hash='+'.join(SCENES), split='train')
    for sample,grid in zip(samples,grids):
        sample.targets.future_register=comp.transform(grid).unsqueeze(0)
    model=DirectedPairDiT(d_model=256,heads=8,ffn_dim=1024,depth=8,dino_checkpoint=DINO,dino_model_name='vit_large_patch16_dinov3.lvd1689m',dino_lora_rank=16,dino_lora_alpha=16.0).to(device)
    saved=torch.load(ART/'checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True); model.eval()
    modes=ImmutableModeBuffer().subset(64).to(device)
    init=StructuredSourceInitializer(jitter_std=0.0)
    result=json.load(open(ART/'result.json'))
    manifest=[]
    with torch.no_grad():
      for si,(scene,sample) in enumerate(zip(SCENES,samples)):
        sample.observation.rgb_history=sample.observation.rgb_history.to(device)
        sample.observation.history_valid=sample.observation.history_valid.to(device)
        sample.observation.ego_history=sample.observation.ego_history.to(device)
        sample.observation.route_command=sample.observation.route_command.to(device)
        sample.observation.relative_ego_poses=sample.observation.relative_ego_poses.to(device)
        modes_b=modes.expand_for_batch(1,device)
        regs=model.forward_register_only(sample.observation,modes_b)[0].cpu() # [64,4,D]
        target=sample.targets.future_register[0].cpu() # [4,4,D]
        traj=np.load(ART/'generated_trajectories.npy')[si]
        gt=sample.targets.gt_trajectory[0].numpy()
        ade=np.linalg.norm(traj[:,None,:,:2]-gt[None,:,:,:2],axis=-1).mean(axis=-1)
        ev=next(x for x in result['evaluation'] if x['scene_id']==scene)
        chosen=int(ev['learned_scorer_candidate']); oracle=int(ev['oracle_best_candidate']); pretrained=int(ev['pretrained_scorer_candidate'])
        gt_match=int(ade[chosen].argmin())
        sims=cosine(regs[chosen],target[gt_match]).numpy()
        all_sims=cosine(regs[:,None],target[None]).mean(-1).numpy()
        # BEV, with explicit current ego origin.
        fig,ax=plt.subplots(figsize=(9,8),dpi=180)
        pdm=np.load(ART/'official_raw7_scores.npy')[si,:,6]; norm=np.clip(pdm,0,1)
        for i in range(64):
            p=np.vstack([np.zeros((1,3)),traj[i]])
            ax.plot(p[:,0],p[:,1],color=plt.cm.viridis(norm[i]),alpha=.3,lw=.8)
        for j,g in enumerate(gt):
            p=np.vstack([np.zeros((1,3)),g]); ax.plot(p[:,0],p[:,1],'k--',lw=1.4,alpha=.7,label='multi-GT' if j==0 else None)
        for idx,col,label in [(chosen,'#00a651','learned'),(pretrained,'#e31a1c','DrivoR pretrained'),(oracle,'#ff9f00','oracle')]:
            p=np.vstack([np.zeros((1,3)),traj[idx]]); ax.plot(p[:,0],p[:,1],color=col,lw=3,label=f'{label} #{idx} PDM={pdm[idx]:.3f}')
        ax.scatter([0],[0],c='blue',s=30,label='ego origin',zorder=10); ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_title(f'{scene} — codex0916 nav1_long_refine'); ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)')
        fig.tight_layout(); fig.savefig(OUT/f'{scene}_bev_64_candidates.png'); plt.close(fig)
        # RGB montage + latent similarity curves.
        rgb=sample.future_rgb[0,gt_match,:,0].numpy().transpose(0,2,3,1)
        rgb=np.clip(rgb,0,1)
        fig,axs=plt.subplots(2,4,figsize=(14,6),dpi=160)
        for t,(a,im) in enumerate(zip(axs.flat,rgb)):
            a.imshow(im); a.set_title(f'future RGB t{t}'); a.axis('off')
        fig.suptitle(f'{scene} future multi-RGB (GT target #{gt_match})'); fig.tight_layout(); fig.savefig(OUT/f'{scene}_future_rgb.png'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4),dpi=180)
        ax.plot(range(4),sims,'o-',lw=2,label=f'predicted register vs GT future latent (candidate #{chosen})')
        ax.axhline(float(sims.mean()),color='#00a651',ls='--',label=f'mean={sims.mean():.4f}')
        ax.set_ylim(-1,1); ax.set_xticks(range(4),[f'chunk {i}' for i in range(4)]); ax.set_ylabel('cosine similarity'); ax.grid(alpha=.25); ax.legend(fontsize=8); ax.set_title(f'{scene} latent alignment')
        fig.tight_layout(); fig.savefig(OUT/f'{scene}_latent_similarity.png'); plt.close(fig)
        np.save(OUT/f'{scene}_predicted_register.npy',regs[chosen].numpy()); np.save(OUT/f'{scene}_future_gt_register.npy',target[gt_match].numpy())
        manifest.append({'scene_id':scene,'learned_candidate':chosen,'pretrained_candidate':pretrained,'oracle_candidate':oracle,'gt_match_for_learned':gt_match,'official_pdms':float(pdm[chosen]),'chunk_cosine':sims.tolist(),'mean_cosine':float(sims.mean()),'best_candidate_mean_cosine':float(all_sims.max()),'output_bev':str(OUT/f'{scene}_bev_64_candidates.png'),'output_future_rgb':str(OUT/f'{scene}_future_rgb.png'),'output_latent_similarity':str(OUT/f'{scene}_latent_similarity.png')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
