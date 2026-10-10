"""Strict DrivoR score path and online PDM oracle.

The proposal generator remains NVS/DINOv3.  Score features, decoder, six
heads, loss and selection formula come from the audited DrivoR source.  The
score path consumes current trajectories only; predicted future registers and
NVS scene tensors are deliberately ignored.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from .oracle import HEAD_INDICES, OracleLabels, load_drivor_metric_cache
from .scorer import HEAD_NAMES

OFFICIAL = Path("/mnt/zhaozc_workspace/official_DrivoR_clean")
CHECKPOINT = Path(
    "/mnt/zhaozc_workspace/project/checkpoints/DrivoR/"
    "nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth"
)
CACHE_ROOT = OFFICIAL / "data_runtime/official_metric_cache_navtrain_20261005/metric_cache"
SENSOR_ROOT = Path("/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval")
LOG_ROOT = Path("/mnt/data/navsim/trainval_navsim_logs/trainval")


class NativeTrainOracle:
    """Call DrivoR's original batch scorer once per scene for all 64 proposals."""

    def __init__(self, samples, cache_root: str | Path = CACHE_ROOT) -> None:
        from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score

        self.get_sub_score = get_sub_score
        root = Path(cache_root)
        self.paths = {
            sample.scene_id: root / sample.log_name / "unknown" / sample.scene_id / "metric_cache.pkl"
            for sample in samples
        }
        self.audit: dict[str, dict[str, object]] = {}
        required = (
            "ego_state",
            "observation",
            "centerline",
            "route_lane_ids",
            "drivable_area_map",
            "pdm_progress",
        )
        for token, path in self.paths.items():
            cache = load_drivor_metric_cache(path)
            missing = [field for field in required if not hasattr(cache, field)]
            if missing:
                raise ValueError(f"{token}: native DrivoR training cache lacks {missing}")
            self.audit[token] = {
                "path": str(path),
                "schema": type(cache).__module__ + "." + type(cache).__name__,
                "fields": sorted(vars(cache)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pdm_progress": float(cache.pdm_progress),
            }
        self.source_file = inspect.getfile(get_sub_score)
        self.source_sha256 = hashlib.sha256(Path(self.source_file).read_bytes()).hexdigest()

    @torch.no_grad()
    def evaluate(self, scene_ids: list[str], trajectories: torch.Tensor, batch_id: str = "") -> OracleLabels:
        if tuple(trajectories.shape[1:]) != (64, 8, 3) or len(scene_ids) != trajectories.shape[0]:
            raise ValueError("native DrivoR online score requires aligned [B,64,8,3] trajectories")
        poses = trajectories.detach().cpu().numpy().astype(np.float64)
        rows = [
            self.get_sub_score(str(self.paths[token]), poses[index], True)[0]
            for index, token in enumerate(scene_ids)
        ]
        raw = torch.from_numpy(np.stack(rows)).float()
        if not torch.isfinite(raw).all():
            raise RuntimeError(f"{batch_id}: non-finite official DrivoR labels")
        valid = torch.ones_like(raw, dtype=torch.bool)
        valid[..., 3] = raw[..., 3].ne(2.0)
        hashes = [[hashlib.sha256(item.tobytes()).hexdigest() for item in scene] for scene in poses]
        return OracleLabels(
            raw7=raw,
            named_six=raw[..., list(HEAD_INDICES)],
            head_valid=valid[..., list(HEAD_INDICES)],
            final_score=raw[..., 6],
            final_valid=valid[..., 6],
            scene_hashes=[str(self.audit[token]["sha256"]) for token in scene_ids],
            trajectory_hashes=hashes,
            evaluator_version="DrivoR.compute_navsim_score.get_sub_score:" + self.source_sha256,
            failure_diagnostics=[],
        )


class NativeDrivoRScorer(nn.Module):
    """Strict DrivoR DINOv2/ego features, decoder, heads, loss and utility."""

    strict_drivor_features = True

    @staticmethod
    def _load_config(official_root: Path):
        return OmegaConf.load(
            official_root / "navsim/planning/script/config/common/agent/drivoR.yaml"
        ).config

    @staticmethod
    def _compile_source_utility(drivor_model_type):
        tree = ast.parse(textwrap.dedent(inspect.getsource(drivor_model_type.forward)))
        expression = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pdm_score" for target in node.targets)
        )
        return compile(ast.Expression(expression), inspect.getfile(drivor_model_type), "eval")

    @classmethod
    def from_saved_state(
        cls,
        state: dict[str, torch.Tensor],
        scene_ids: list[str],
        device: torch.device,
        official_root: str | Path = OFFICIAL,
    ) -> "NativeDrivoRScorer":
        from navsim.agents.drivoR.drivor_model import DrivoRModel
        from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
        from navsim.agents.drivoR.score_module.scorer import Scorer
        from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer

        root = Path(official_root)
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        cfg = cls._load_config(root)
        obj.config = obj._config = cfg
        obj._source_utility = cls._compile_source_utility(DrivoRModel)
        obj.trajectory_projection = nn.Sequential(
            nn.Linear(cfg.num_poses * 3, cfg.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(cfg.tf_d_ffn, cfg.tf_d_model),
        )
        obj.scorer_attention = TransformerDecoderScorer(
            num_layers=cfg.scorer_ref_num,
            d_model=cfg.tf_d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=cfg,
        )
        obj.scorer = Scorer(cfg)
        obj.reference_projection = copy.deepcopy(obj.trajectory_projection)
        obj.reference_attention = copy.deepcopy(obj.scorer_attention)
        obj.reference_heads = copy.deepcopy(obj.scorer)
        for index in range(len(scene_ids)):
            obj.register_buffer(f"scene_{index}", torch.empty_like(state[f"scene_{index}"]))
            obj.register_buffer(f"ego_{index}", torch.empty_like(state[f"ego_{index}"]))
        obj.loss_fn = DrivoRLoss()
        obj.scene_ids = list(scene_ids)
        obj.active_scene_ids = list(scene_ids)
        obj.comparisons = []
        obj.feature_audit = {}
        obj.load_state_dict(state, strict=True)
        obj.to(device).freeze_reference()
        return obj

    def __init__(
        self,
        scene_ids: list[str],
        device: torch.device,
        log_names: list[str] | None = None,
        official_root: str | Path = OFFICIAL,
        checkpoint: str | Path = CHECKPOINT,
        sensor_root: str | Path = SENSOR_ROOT,
        log_root: str | Path = LOG_ROOT,
    ) -> None:
        super().__init__()
        from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
        from navsim.agents.drivoR.drivor_model import DrivoRModel
        from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SceneFilter, SensorConfig

        root = Path(official_root)
        cfg = self._load_config(root)
        cfg.image_backbone.model_weights = str(root / cfg.image_backbone.model_weights)
        model = DrivoRModel(cfg)
        raw = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        prefix = "agent._drivor_model."
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model.load_state_dict(state, strict=True)
        model.to(device).eval().requires_grad_(False)
        self.config = self._config = cfg
        self._source_utility = self._compile_source_utility(DrivoRModel)

        loader = SceneLoader(
            sensor_blobs_path=Path(sensor_root),
            data_path=Path(log_root),
            scene_filter=SceneFilter(
                num_history_frames=4,
                num_future_frames=10,
                frame_interval=1,
                has_route=True,
                max_scenes=None,
                log_names=log_names,
                tokens=list(scene_ids),
            ),
            sensor_config=SensorConfig(
                **{
                    key: OmegaConf.to_object(cfg[key])
                    for key in (
                        "cam_f0",
                        "cam_l0",
                        "cam_l1",
                        "cam_l2",
                        "cam_r0",
                        "cam_r1",
                        "cam_r2",
                        "cam_b0",
                        "lidar_pc",
                    )
                }
            ),
        )
        builder = DrivoRFeatureBuilder(cfg)
        self.feature_audit: dict[str, dict[str, object]] = {}
        with torch.no_grad():
            for index, token in enumerate(scene_ids):
                features = {
                    key: value.unsqueeze(0).to(device)
                    for key, value in builder.compute_features(loader.get_agent_input_from_token(token)).items()
                }
                ego = features["ego_status"].flatten(-2) if cfg.full_history_status else features["ego_status"][:, -1]
                ego_token = model.hist_encoding(ego)[:, None]
                scene = model.image_backbone(features["image"], model.scene_embeds.repeat(1, 1, 1, 1))
                self.register_buffer(f"scene_{index}", scene.detach())
                self.register_buffer(f"ego_{index}", ego_token.detach())
                self.feature_audit[token] = {
                    "scene_shape": list(scene.shape),
                    "ego_shape": list(ego_token.shape),
                    "image_shape": list(features["image"].shape),
                    "backbone": cfg.image_backbone.model_name,
                }

        self.trajectory_projection = model.pos_embed
        self.scorer_attention = model.scorer_attention
        self.scorer = model.scorer
        self.reference_projection = copy.deepcopy(self.trajectory_projection)
        self.reference_attention = copy.deepcopy(self.scorer_attention)
        self.reference_heads = copy.deepcopy(self.scorer)
        self.loss_fn = DrivoRLoss()
        self.scene_ids = list(scene_ids)
        self.active_scene_ids = list(scene_ids)
        self.comparisons: list[list[dict[str, object]]] = []
        for module in (self.trajectory_projection, self.scorer_attention, self.scorer):
            module.requires_grad_(True)
        self.freeze_reference()
        del model

    def freeze_reference(self) -> None:
        for module in (self.reference_projection, self.reference_attention, self.reference_heads):
            module.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "NativeDrivoRScorer":
        super().train(mode)
        self.freeze_reference()
        return self

    def _logits(self, trajectory: torch.Tensor, reference: bool = False) -> torch.Tensor:
        scene = torch.cat(
            [getattr(self, f"scene_{self.scene_ids.index(token)}") for token in self.active_scene_ids]
        )
        ego = torch.cat(
            [getattr(self, f"ego_{self.scene_ids.index(token)}") for token in self.active_scene_ids]
        )
        if trajectory.shape[0] != len(self.active_scene_ids):
            raise ValueError("native scorer scene identities do not align with proposals")
        modules = (
            (self.reference_projection, self.reference_attention, self.reference_heads)
            if reference
            else (self.trajectory_projection, self.scorer_attention, self.scorer)
        )
        projection, attention, heads = modules
        score_features = attention(projection(trajectory.detach().flatten(2)), scene) + ego
        logits = heads(trajectory.detach(), score_features)[0]
        return torch.stack([logits[name] for name in HEAD_NAMES], dim=-1)

    def forward(
        self,
        scene: torch.Tensor,
        trajectory: torch.Tensor,
        predicted_register: torch.Tensor | None = None,
        detach_inputs: bool = True,
    ) -> torch.Tensor:
        del scene, predicted_register, detach_inputs
        return self._logits(trajectory)

    def select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_logit = {name: logits[..., index] for index, name in enumerate(HEAD_NAMES)}
        utility = eval(self._source_utility, {"self": self, "pred_logit": pred_logit})
        return utility.argmax(dim=1), utility

    def source_loss(self, logits: torch.Tensor, labels: OracleLabels) -> torch.Tensor:
        if labels.failure_diagnostics or not labels.final_valid.all():
            raise RuntimeError("do not train DrivoR scorer on failed oracle evaluations")
        pred_logit = {name: logits[..., index] for index, name in enumerate(HEAD_NAMES)}
        # Official source maps NC/DDC classes in-place.  Float64 makes the
        # per-head casts independent tensors without editing DrivoR itself.
        return self.loss_fn.score_loss(
            pred_logit,
            None,
            None,
            None,
            labels.raw7.to(device=logits.device, dtype=torch.float64),
            None,
            None,
            None,
            None,
        )[1]

    @torch.no_grad()
    def record_comparison(self, trajectory: torch.Tensor, labels: OracleLabels) -> None:
        was_training = self.training
        self.eval()
        ours = self._logits(trajectory)
        reference = self._logits(trajectory, reference=True)
        own_indices = self.select(ours)[0].cpu()
        reference_indices = self.select(reference)[0].cpu()
        scores = labels.final_score.cpu()
        self.comparisons.append(
            [
                {
                    "scene_id": token,
                    "trained_index": int(own_indices[batch]),
                    "pretrained_index": int(reference_indices[batch]),
                    "oracle_index": int(scores[batch].argmax()),
                    "trained_pdms": float(scores[batch, own_indices[batch]]),
                    "pretrained_pdms": float(scores[batch, reference_indices[batch]]),
                    "oracle_pdms": float(scores[batch].max()),
                    "same_index": bool(own_indices[batch] == reference_indices[batch]),
                }
                for batch, token in enumerate(self.active_scene_ids)
            ]
        )
        self.train(was_training)
