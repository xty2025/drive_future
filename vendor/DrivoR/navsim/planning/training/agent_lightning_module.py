from time import sleep
import os

import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = (tensor_1[:, None] == target_tensor)
    
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, for_viz = False):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None
        self.for_viz = for_viz
        self._pasp_last_prediction = None
        self._pasp_last_targets = None
        self._pasp_last_losses = None
        self._pasp_last_grad_norms = None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if logging_prefix == "train" and os.environ.get("PASP_SMOKE_DIR"):
            self._pasp_last_prediction = prediction
            self._pasp_last_targets = targets
            self._pasp_last_losses = {
                key: float(value.detach().float().cpu())
                for key, value in loss_dict.items()
                if isinstance(value, Tensor) and value.numel() == 1
            }

        if type(loss_dict) is dict:
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Emit stable rank-zero status lines for non-interactive DLC logs."""
        every = max(1, int(os.environ.get("PASP_TEXT_LOG_EVERY", "10")))
        completed_step = int(self.global_step)
        if self.global_rank != 0 or completed_step % every != 0:
            return
        metrics = self.trainer.callback_metrics
        fields = []
        for name in (
            "train/loss",
            "train/score",
            "train/best_score",
            "train/preference_loss",
            "train/unified_loss",
            "train/dpo_loss",
            "train/rank_loss",
            "train/preference_pair_rate",
        ):
            value = metrics.get(name)
            if isinstance(value, Tensor) and value.numel() == 1:
                fields.append(f"{name.removeprefix('train/')}={float(value.detach()):.6f}")
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"[pasp-train] epoch={self.current_epoch} step={completed_step} "
            + " ".join(fields)
            + f" lr={lr:.3e}",
            flush=True,
        )

    def on_after_backward(self) -> None:
        if not os.environ.get("PASP_SMOKE_DIR"):
            return
        model = self.agent._drivor_model
        groups = {
            "policy": list(model.policy_head.parameters()),
            "proposal": list(model.traj_head.parameters()),
        }
        grad_norms = {}
        for name, parameters in groups.items():
            grads = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
            if not grads:
                raise RuntimeError(f"PASP gradient check failed: no {name} gradients")
            if not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError(f"PASP gradient check failed: non-finite {name} gradients")
            norm = torch.sqrt(sum(g.detach().float().square().sum() for g in grads))
            if not torch.isfinite(norm) or float(norm) <= 0.0:
                raise RuntimeError(f"PASP gradient check failed: zero/non-finite {name} gradient norm")
            grad_norms[name] = float(norm.detach().cpu())
            self.log(f"train/{name}_grad_norm", norm, on_step=True, prog_bar=True)
        self._pasp_last_grad_norms = grad_norms

    def on_train_end(self) -> None:
        if not os.environ.get("PASP_SMOKE_DIR") or self.global_rank != 0:
            return
        pred, targets = self._pasp_last_prediction, self._pasp_last_targets
        if pred is None:
            return
        proposals = pred["proposals"][0].detach().cpu()
        gt = targets["trajectory"][0].detach().cpu()
        oracle = pred["_oracle_final_scores"][0].detach().cpu()
        learned_scores = pred["pdm_score"][0].detach().cpu()
        learned_top = int(learned_scores.argmax())
        distance = torch.linalg.norm(proposals - gt[None], dim=-1).mean(-1)
        unsafe = oracle < float(getattr(self.agent._config, "preference_unsafe_threshold", 0.2))
        rejected = int(distance.masked_fill(~unsafe, float("inf")).argmin()) if unsafe.any() else int(oracle.argmin())
        out = Path(os.environ["PASP_SMOKE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        fig, (ax, lateral_ax) = plt.subplots(1, 2, figsize=(12.0, 5.2), facecolor="white")
        for traj in proposals:
            ax.plot(traj[:, 0], traj[:, 1], color="#9ca3af", alpha=0.22, linewidth=0.8)
        ax.plot(gt[:, 0], gt[:, 1], "o-", color="#111827", label="GT", linewidth=2.6, markersize=4)
        ax.plot(proposals[learned_top, :, 0], proposals[learned_top, :, 1], "o-", color="#2563eb",
                label=f"highest scorer (slot {learned_top})", linewidth=2.2, markersize=4)
        ax.plot(proposals[rejected, :, 0], proposals[rejected, :, 1], "o--", color="#dc2626",
                label=f"negative (slot {rejected})", linewidth=2.0, markersize=4)
        ax.scatter([0.0], [0.0], marker="^", s=80, color="#059669", label="ego origin", zorder=5)
        ax.axhline(0.0, color="0.4", linewidth=0.6)
        ax.axvline(0.0, color="0.4", linewidth=0.6)
        diversity = torch.pdist(proposals[:, :, :2].reshape(proposals.shape[0], -1)).mean()
        ax.set_title("Metric geometry (equal aspect)")
        ax.set_xlabel("longitudinal x [m]")
        ax.set_ylabel("lateral y [m]")
        ax.axis("equal"); ax.grid(alpha=0.2); ax.legend(loc="best", frameon=False)

        for traj in proposals:
            lateral_ax.plot(traj[:, 0], traj[:, 1], color="#9ca3af", alpha=0.22, linewidth=0.8)
        lateral_ax.plot(gt[:, 0], gt[:, 1], "o-", color="#111827", label="GT", linewidth=2.6, markersize=4)
        lateral_ax.plot(proposals[learned_top, :, 0], proposals[learned_top, :, 1], "o-", color="#2563eb",
                        label=f"highest scorer (slot {learned_top})", linewidth=2.2, markersize=4)
        lateral_ax.plot(proposals[rejected, :, 0], proposals[rejected, :, 1], "o--", color="#dc2626",
                        label=f"negative (slot {rejected})", linewidth=2.0, markersize=4)
        lateral_ax.scatter([0.0], [0.0], marker="^", s=80, color="#059669", label="ego origin", zorder=5)
        lateral_ax.axhline(0.0, color="0.4", linewidth=0.6)
        lateral_ax.axvline(0.0, color="0.4", linewidth=0.6)
        lateral_ax.set_title("Lateral detail (expanded y-axis)")
        lateral_ax.set_xlabel("longitudinal x [m]")
        lateral_ax.set_ylabel("lateral y [m]")
        lateral_ax.grid(alpha=0.2)
        fig.suptitle(
            f"DrivoR dynamic proposals | top={learned_top}, negative={rejected}, diversity={diversity:.3f}"
        )
        fig.tight_layout()
        fig.savefig(out / "real_navsim_gt_top_negative.png", dpi=180)
        plt.close(fig)
        torch.save({"proposals": proposals, "gt": gt, "oracle_scores": oracle,
                    "learned_scores": learned_scores,
                    "highest_scorer_index": learned_top, "negative_index": rejected,
                    "proposal_diversity": diversity,
                    "global_step": int(self.global_step)}, out / "real_navsim_candidates.pt")
        summary = {
            "global_step": int(self.global_step),
            "gt_endpoint_m": gt[-1, :2].tolist(),
            "proposal_endpoint_x_range_m": [proposals[:, -1, 0].min().item(), proposals[:, -1, 0].max().item()],
            "proposal_endpoint_y_range_m": [proposals[:, -1, 1].min().item(), proposals[:, -1, 1].max().item()],
            "proposal_diversity": diversity.item(),
            "highest_scorer_index": learned_top,
            "highest_scorer_oracle_pdm": oracle[learned_top].item(),
            "oracle_pdm_mean": oracle.mean().item(),
            "oracle_pdm_max": oracle.max().item(),
            "negative_index": rejected,
            "negative_oracle_pdm": oracle[rejected].item(),
            "coordinate_frame": "ego_local_rear_axle_meters",
            "num_poses": int(proposals.shape[1]),
            "last_train_losses": self._pasp_last_losses,
            "last_gradient_norms": self._pasp_last_grad_norms,
        }
        (out / "real_navsim_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'drivor' in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            all_chosen_trajectories = predictions["trajectory"][:,None]
            all_proposed_trajectories = predictions["proposals"]
            final_score, fake_best_score, proposal_scores, l2, trajectoy_scores = self.agent.compute_score(targets, all_chosen_trajectories)
            _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(targets, all_proposed_trajectories)
            mean_score=proposal_scores.mean()

            logging_prefix="val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                best_pred_score_values = pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
                score_error = torch.abs(best_pred_score_values - proposal_scores).mean()
                self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
                best_pred_score_index = torch.argmax(pdm_score, dim=1)
                best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                best_possible_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_real_score_index]
                best_actual_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_pred_score_index]
                lost_score = torch.mean(best_possible_scores - best_actual_scores)
                self.log(f"{logging_prefix}/score_hit_rate", score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/lost_score", lost_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)
                self.log(f"{logging_prefix}/top_5_score_hit_rate", top_5_score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                # Bounded smoke-run diagnostic for the scorer-primary Top-M
                # policy fusion specified by PASP.  These are oracle-reported
                # evaluation metrics only; no oracle value enters selection.
                top_m = min(int(getattr(self.agent._config, "selection_top_m", 8)), pdm_score.shape[1])
                shortlist = torch.topk(pdm_score, k=top_m, dim=1).indices
                scorer_short = pdm_score.gather(1, shortlist)
                policy_short = predictions["policy_log_probs"].gather(1, shortlist)
                scorer_short = (scorer_short - scorer_short.mean(1, keepdim=True)) / scorer_short.std(1, keepdim=True).clamp_min(1e-6)
                policy_short = (policy_short - policy_short.mean(1, keepdim=True)) / policy_short.std(1, keepdim=True).clamp_min(1e-6)
                for weight in (0.25, 0.5, 1.0, 2.0):
                    fused_idx = shortlist.gather(1, (scorer_short + weight * policy_short).argmax(1, keepdim=True)).squeeze(1)
                    fused_score = all_proposal_scores[torch.arange(len(all_proposal_scores)), fused_idx].mean()
                    self.log(
                        f"{logging_prefix}/fusion_{str(weight).replace('.', 'p')}_score",
                        fused_score,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
                scorer_all = (pdm_score - pdm_score.mean(1, keepdim=True)) / pdm_score.std(1, keepdim=True).clamp_min(1e-6)
                policy_all = predictions["policy_log_probs"]
                policy_all = (policy_all - policy_all.mean(1, keepdim=True)) / policy_all.std(1, keepdim=True).clamp_min(1e-6)
                for weight in (0.5, 1.0, 2.0, 4.0):
                    fused_idx = (scorer_all + weight * policy_all).argmax(1)
                    fused_score = all_proposal_scores[torch.arange(len(all_proposal_scores)), fused_idx].mean()
                    self.log(
                        f"{logging_prefix}/global_fusion_{str(weight).replace('.', 'p')}_score",
                        fused_score,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
                policy_idx = predictions["policy_logits"].argmax(1)
                policy_score = all_proposal_scores[torch.arange(len(all_proposal_scores)), policy_idx].mean()
                self.log(f"{logging_prefix}/policy_score", policy_score, on_step=False, on_epoch=True, sync_dist=True)
            
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
    
    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        if len(batch) == 3:
            features, targets, tokens = batch
        elif len(batch) == 2:
            features, targets = batch
            tokens = targets.get("token")
            if tokens is None:
                raise ValueError("Prediction batch has no appended tokens and targets contain no token field")
        else:
            raise ValueError(f"Expected a 2- or 3-item prediction batch, got {len(batch)}")
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            record_candidates = os.environ.get("DRIVOR_RECORD_CANDIDATES", "0") == "1"
            record_shared_winners = os.environ.get("DRIVOR_RECORD_SHARED_WINNERS", "0") == "1"
            if record_candidates:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                learned_scores = predictions["pdm_score"]
                shared_head_proposals = predictions.get("shared_head_proposals")
                shared_head_scores = predictions.get("shared_head_scores")
            elif record_shared_winners:
                shared_head_proposals = predictions["shared_head_proposals"]
                shared_head_scores = predictions["shared_head_scores"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposal_list, 
                    'all_proposal_scores': final_scores[index],
                    'high_level_command': ego_status[index]
                }
            else:
                result[token] = {'trajectory': proposal}
                if record_candidates:
                    result[token].update({
                        'all_proposals': [x[index].detach().cpu().numpy() for x in all_proposed_trajectories],
                        'final_proposals': final_trajectories[index].detach().cpu().numpy(),
                        'learned_scores': learned_scores[index].detach().cpu().numpy(),
                        'gt_trajectory': targets['trajectory'][index].detach().cpu().numpy(),
                    })
                    if shared_head_proposals is not None:
                        result[token].update({
                            'shared_head_proposals': [x[index].detach().cpu().numpy() for x in shared_head_proposals],
                            'shared_head_scores': [x[index].detach().cpu().numpy() for x in shared_head_scores],
                        })
                elif record_shared_winners:
                    stage_scores = [x[index].detach().cpu().numpy() for x in shared_head_scores]
                    stage_proposals = [x[index].detach().cpu().numpy() for x in shared_head_proposals]
                    result[token].update({
                        'shared_head_trajectories': [
                            Trajectory(stage[int(scores.argmax())])
                            for stage, scores in zip(stage_proposals, stage_scores)
                        ],
                        'shared_head_scores': stage_scores,
                    })
        return result
