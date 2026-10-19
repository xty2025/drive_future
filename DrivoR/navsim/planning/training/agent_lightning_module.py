from time import sleep
from time import perf_counter
from collections import deque
import logging
import math

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.planning.training.rank_performance import RankPerformanceRecorder
from navsim.planning.training.random_seed import seed_model_randomness
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)


def remaining_train_batches(num_batches, max_epochs, epoch, batch_idx, max_steps, global_step, accumulation):
    """Count remaining microbatches, including partial accumulation at epoch ends."""
    accumulation = max(1, int(accumulation))
    done = int(batch_idx) + 1
    finite_epoch = math.isfinite(num_batches) and num_batches > 0
    epoch_bound = None
    if finite_epoch:
        num_batches = int(num_batches)
        current = max(0, num_batches - done)
        if max_epochs is not None and max_epochs >= 0:
            epoch_bound = current + max(0, int(max_epochs) - int(epoch) - 1) * num_batches
    if max_steps is None or max_steps < 0:
        return epoch_bound
    updates = max(0, int(max_steps) - int(global_step))
    if updates == 0:
        return 0
    if not finite_epoch:
        step_bound = max(0, updates * accumulation - done % accumulation)
    else:
        partial = done % accumulation if current else 0
        updates_this_epoch = math.ceil((current + partial) / accumulation)
        if updates <= updates_this_epoch:
            step_bound = min(current, updates * accumulation - partial)
        else:
            updates -= updates_this_epoch
            updates_per_epoch = math.ceil(num_batches / accumulation)
            full_epochs = (updates - 1) // updates_per_epoch
            last_updates = updates - full_epochs * updates_per_epoch
            step_bound = current + full_epochs * num_batches + min(num_batches, last_updates * accumulation)
    return step_bound if epoch_bound is None else min(epoch_bound, step_bound)


def _format_eta(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    hours, remainder = divmod(int(math.ceil(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_scalar(value):
    if isinstance(value, Tensor):
        value = value.detach().float().cpu().item()
    return f"{float(value):.6g}"




def _batch_size(features, targets) -> int:
    trajectory = targets.get("trajectory")
    if isinstance(trajectory, Tensor) and trajectory.ndim:
        return trajectory.shape[0]
    for value in features.values():
        if isinstance(value, Tensor) and value.ndim:
            return value.shape[0]
    raise ValueError("Cannot determine scene batch size for weighted metric logging")


def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = (tensor_1[:, None] == target_tensor)
    
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, for_viz=False, log_local_steps=True,
                 console_log_interval=25, performance_diagnostics=None, training_seed=None):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None
        self.for_viz = for_viz
        self.log_local_steps = bool(log_local_steps)
        self.console_log_interval = max(0, int(console_log_interval))
        self._console_batch_durations = deque(maxlen=100)
        self._console_previous_batch_end = None
        self._console_observed_batches = 0
        self._console_losses = {}
        self._performance_config = dict(performance_diagnostics or {})
        self._performance = None
        self.training_seed = int(training_seed) if training_seed is not None else None
        self._model_rng_seed = None

    def on_train_start(self) -> None:
        # Parameters and DDP are ready; preserve their common initialization and
        # sampler seed while giving model noise/time/dropout a rank-specific RNG.
        if self.training_seed is not None:
            self._model_rng_seed = seed_model_randomness(self.training_seed, self.global_rank)
            logger.info("Model RNG: global_rank=%s base_seed=%s process_seed=%s",
                        self.global_rank, self.training_seed, self._model_rng_seed)
        self._console_batch_durations.clear()
        self._console_previous_batch_end = None
        self._console_observed_batches = 0
        if self._performance is not None:
            self._performance.close()
            self._performance = None
        options = self._performance_config
        if options.get("enabled", False):
            from pathlib import Path
            self._performance = RankPerformanceRecorder(
                options.get("output_dir") or Path(self.trainer.default_root_dir) / "performance",
                rank=self.global_rank, world_size=self.trainer.world_size, device=self.device,
                sample_every=options.get("sample_every", 25), flush_every=options.get("flush_every", 4),
                cuda_events=options.get("cuda_events", False),
                optimizer_step_start=self.global_step,
            )

    def on_train_epoch_start(self) -> None:
        if self._performance is not None:
            self._performance.reset_wait()
        if self.global_rank == 0 and self.console_log_interval:
            self._console_previous_batch_end = perf_counter()

    def on_validation_start(self) -> None:
        if self._performance is not None:
            self._performance.reset_wait(paused=True)
        self._console_previous_batch_end = None

    def on_validation_end(self) -> None:
        # Validation/checkpoint callbacks finish before this hook. Start a new
        # interval so the next batch includes input waiting, but not validation.
        if self.global_rank == 0 and self.console_log_interval:
            self._console_previous_batch_end = perf_counter()
        if self._performance is not None:
            self._performance.reset_wait()

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self._performance is not None and self.trainer.training:
            self._performance.begin_transfer(epoch=self.current_epoch, optimizer_step=self.global_step,
                                             accumulation=self.trainer.accumulate_grad_batches)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self._performance is not None and self.trainer.training:
            self._performance.end_phase("transfer")
        return batch

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        if self._performance is not None:
            self._performance.batch_start(batch_idx)
        if self.global_rank == 0 and self.console_log_interval and self._console_previous_batch_end is None:
            self._console_previous_batch_end = perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self._log_console_progress(batch_idx)
        if self._performance is not None:
            self._performance.finish_batch(batch_idx=batch_idx, optimizer_step=self.global_step)

    def _log_console_progress(self, batch_idx: int) -> None:
        if self.global_rank != 0 or not self.console_log_interval:
            return
        now = perf_counter()
        if self._console_previous_batch_end is not None:
            duration = now - self._console_previous_batch_end
            if math.isfinite(duration) and duration > 0:
                self._console_batch_durations.append(duration)
        self._console_previous_batch_end = now
        self._console_observed_batches += 1
        trainer = self.trainer
        batches = trainer.num_training_batches
        finished = batch_idx + 1 >= batches or (trainer.max_steps >= 0 and self.global_step >= trainer.max_steps)
        if self._console_observed_batches != 1 and (batch_idx + 1) % self.console_log_interval and not finished:
            return
        remaining = remaining_train_batches(batches, trainer.max_epochs, self.current_epoch,
                                             batch_idx, trainer.max_steps, self.global_step,
                                             trainer.accumulate_grad_batches)
        mean_seconds = (sum(self._console_batch_durations) / len(self._console_batch_durations)
                        if self._console_batch_durations else None)
        eta_seconds = remaining * mean_seconds if remaining is not None and mean_seconds is not None else None
        losses = " ".join(f"{key}={_format_scalar(value)}" for key, value in self._console_losses.items())
        learning_rates = ",".join(_format_scalar(group["lr"])
                                  for optimizer in trainer.optimizers for group in optimizer.param_groups)
        logger.info("[train] rank=0 epoch=%s/%s batch=%s/%s optimizer_step=%s accum=%s "
                    "batch_s=%s ETA_train=%s remaining_train_batches=%s lr=%s %s",
                    self.current_epoch + 1, trainer.max_epochs, batch_idx + 1, batches,
                    self.global_step, trainer.accumulate_grad_batches,
                    f"{mean_seconds:.3f}" if mean_seconds is not None else "unknown",
                    _format_eta(eta_seconds), remaining if remaining is not None else "unknown",
                    learning_rates or "unknown", losses or "loss=unavailable")

    def on_train_epoch_end(self) -> None:
        # Access on EVERY rank: Lightning may perform the existing epoch metric
        # reductions here. Reading callback_metrics only on rank zero can hang.
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            keys = ("train/loss", "train/trajectory_loss", "train/future_register_loss", "train/ema_updates", "train/final_score_loss", "train/score",
                    "train/best_score", "val/score_epoch", "val/best_score", "val/mean_score",
                    "val/lost_score", "val/score_hit_rate", "val/l2", "val/future_register_loss")
            summary = " ".join(f"{key}={_format_scalar(metrics[key])}" for key in keys if key in metrics)
            logger.info("[epoch] epoch=%s scope=global %s", self.current_epoch + 1, summary)
        self._console_previous_batch_end = None
        if self._performance is not None:
            self._performance.reset_wait(paused=True)

    def on_before_backward(self, loss):
        if self._performance is not None:
            self._performance.start_phase("backward")

    def on_after_backward(self):
        if self._performance is not None:
            self._performance.end_phase("backward")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        recorder = self._performance
        if recorder is None or not recorder.active:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        def measured_closure():
            result = optimizer_closure()  # Includes forward, loss, zero_grad and backward.
            recorder.optimizer_started()
            return result

        try:
            return super().optimizer_step(epoch, batch_idx, optimizer, measured_closure)
        finally:
            recorder.end_phase("optimizer")

    def on_train_end(self):
        if self._performance is not None:
            self._performance.close()

    def teardown(self, stage):
        if self._performance is not None:
            self._performance.close()




    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        forward = getattr(self.agent, "forward_train", None) if logging_prefix == "train" else None
        forward_args = (features, targets) if forward is not None else (features,)
        forward = forward or self.agent.forward
        recorder = self._performance if logging_prefix == "train" else None
        if recorder is not None and recorder.active:
            with recorder.phase("forward"):
                prediction = forward(*forward_args)
            with recorder.phase("loss"), recorder.oracle_phases(self.agent):
                loss_dict = self.agent.compute_loss(features, targets, prediction)
        else:
            prediction = forward(*forward_args)
            loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            if logging_prefix == "train" and self.global_rank == 0 and self.console_log_interval:
                self._console_losses = {
                    key: value.detach() if isinstance(value, Tensor) else value
                    for key, value in loss_dict.items()
                    if key in ("loss", "trajectory_loss", "future_register_loss", "final_score_loss", "ema_updates")
                }
            batch_size = _batch_size(features, targets)
            for key,value in loss_dict.items():
                # Global scene-weighted means synchronize at epoch end only.
                self.log(f"{logging_prefix}/"+key, value, on_step=False, on_epoch=True,
                         prog_bar=True, sync_dist=True, batch_size=batch_size)
                if logging_prefix == "train" and self.log_local_steps and self.global_rank == 0:
                    # Step curves are explicitly rank-0-local, not global means.
                    self.log(f"train_step/{key}", value, on_step=True, on_epoch=False,
                             prog_bar=True, sync_dist=False, rank_zero_only=True, batch_size=batch_size)
            return loss_dict["loss"]
        else:
            if logging_prefix == "train" and self.global_rank == 0 and self.console_log_interval:
                self._console_losses = {"loss": loss_dict.detach() if isinstance(loss_dict, Tensor) else loss_dict}
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'drivor' in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            batch_size = _batch_size(features, targets)
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            all_chosen_trajectories = predictions["trajectory"][:,None]
            all_proposed_trajectories = predictions["proposals"]
            final_score, fake_best_score, proposal_scores, l2, trajectoy_scores = self.agent.compute_score(targets, all_chosen_trajectories)
            _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(targets, all_proposed_trajectories)
            mean_score=all_proposal_scores.mean()

            logging_prefix="val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                # pdm_score is a log utility, not a calibrated PDMS probability.
                # Compare the selected candidate's Oracle quality via lost_score.
                best_pred_score_index = torch.argmax(pdm_score, dim=1)
                best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                best_possible_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_real_score_index]
                best_actual_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_pred_score_index]
                lost_score = torch.mean(best_possible_scores - best_actual_scores)
                self.log(f"{logging_prefix}/score_hit_rate", score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
                self.log(f"{logging_prefix}/lost_score", lost_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

                top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)
                self.log(f"{logging_prefix}/top_5_score_hit_rate", top_5_score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            
            # Preserve the checkpoint monitor produced by the previous dual
            # step/epoch log without doing a collective on every validation step.
            self.log(f"{logging_prefix}/score_epoch", final_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

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
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for index, (pose, token) in enumerate(zip(poses.detach().float().cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].detach().float().cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposal_list, 
                    'all_proposal_scores': final_scores[index],
                    'high_level_command': ego_status[index]
                }
            else:
                result[token] = {'trajectory': proposal}
        return result
