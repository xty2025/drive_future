"""Plug-in agent for the original run_training.py/Lightning training interface."""

import math
import os
from pathlib import Path

import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from navsim.common.dataloader import MetricCacheLoader

from .model import DrivoRActionDiTModel


class DrivoRActionDiTAgent(DrivoRAgent):
    def __init__(self, config, lr_args, checkpoint_path="", loss=None,
                 progress_bar=True, scheduler_args=None, batch_size=1, num_gpus=1,
                 accumulate_grad_batches=1, score_worker=None):
        # Inherit DrivoR builders, sensor config, optimizer/scheduler, callbacks
        # and official online oracle, without constructing its old generator.
        AbstractAgent.__init__(self)
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path or ""
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self._configure_score_worker(score_worker)
        self.b2d = config.b2d
        self.loss = loss
        self._drivor_model = self._build_model(config)
        self._oracle_ready = False

    def _build_model(self, config):
        return DrivoRActionDiTModel(config)

    def initialize(self):
        if self._checkpoint_path:
            payload = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
            state = payload["state_dict"]
            state = {key.removeprefix("agent."): value for key, value in state.items()}
            # Only a complete checkpoint of this variant can resume it. A
            # DrivoR release scorer/trajectory checkpoint is never warm-loaded.
            self.load_state_dict(state, strict=True)

    def _initialize_oracle(self):
        if self._oracle_ready:
            return
        from navsim.agents.drivoR.score_module.compute_navsim_score import get_scores
        root = Path(os.environ["NAVSIM_EXP_ROOT"])
        metric_cache_root = Path(os.environ.get("NAVSIM_METRIC_CACHE_PATH") or root / "train_metric_cache")
        metric_cache = MetricCacheLoader(metric_cache_root)
        self.train_metric_cache_paths = metric_cache.metric_cache_paths
        self.test_metric_cache_paths = metric_cache.metric_cache_paths
        synthetic = {}
        for index in range(5):
            path = root / f"train_metric_synthetic_reaction_pdm_v1.0-{index}"
            if path.is_dir():
                synthetic.update(MetricCacheLoader(path).metric_cache_paths)
        self.train_metric_cache_paths_synthetic = synthetic
        self.test_metric_cache_paths_synthetic = synthetic
        self.get_scores = get_scores
        self._oracle_ready = True

    def compute_score(self, targets, proposals, test=True):
        self._initialize_oracle()
        return super().compute_score(targets, proposals, test=test)

    def forward_train(self, features, targets):
        # Native 5B uses ZeRO-2 with FP32 parameters/states. Keep its model
        # forward in BF16, while ordinary validation remains genuinely FP32.
        if self._config.get('action_dit', {}).get('forward_autocast_dtype') == 'bf16':
            device = next(self._drivor_model.parameters()).device
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                return self._drivor_model(features, targets=targets)
        return self._drivor_model(features, targets=targets)

    def compute_loss(self, features, targets, pred):
        # Reuse the exact X predictions that supplied the scorer's candidates.
        # No additional DiT forward or Euler sampling is needed in this phase.
        flow_loss, diagnostics = pred["_flow_loss"], pred["_flow_diagnostics"]
        loss_dict = self.loss(targets, pred, self._config, self.compute_score, flow_loss=flow_loss)
        loss_dict.update(diagnostics)
        return loss_dict

    def get_optimizers(self):
        # Preserve DrivoR's sqrt batch scaling and ramp+cosine schedule, counting
        # optimizer updates rather than microbatches when using accumulation.
        effective_batch = self.batch_size * self.num_gpus * self.accumulate_grad_batches
        lr = self._lr_args["base_lr"] * math.sqrt(effective_batch / self._lr_args["base_batch_size"])
        optimizer_type = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[self._lr_args["name"]]
        options = {}
        if self._lr_args["name"] == "AdamW":
            options["fused"] = bool(self._lr_args.get("fused", False)) and next(self._drivor_model.parameters()).is_cuda
        optimizer = optimizer_type(self._drivor_model.parameters(), lr=lr, **options)
        if self.scheduler_args is None:
            return [optimizer]
        total_steps = math.ceil(self.scheduler_args.dataset_size / effective_batch) * self.scheduler_args.num_epochs
        ramp_steps = max(1, int(total_steps * 0.1))
        ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=ramp_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - ramp_steps))
        schedule = torch.optim.lr_scheduler.SequentialLR(optimizer, [ramp, cosine], milestones=[ramp_steps])
        return [optimizer], [{"scheduler": schedule, "interval": "step"}]
