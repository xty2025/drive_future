"""Human plus diverse8 future RGB, preserving original (possibly sparse) slots."""

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from navsim.common.diverse_future_frames import TIMES, FRAME_INDICES, extract_diverse_scene_frames
from navsim.agents.drivoR_action_dit.diverse_gt import DiverseTrajectoryTargetBuilder
from .targets import _human_future_targets
from .diverse_losses import future_supervision_scope


class DiverseFutureRegisterTargetBuilder(DiverseTrajectoryTargetBuilder):
    def __init__(self, diverse_gt, future):
        super().__init__(diverse_gt)
        self.supervision_scope = future_supervision_scope(future)
        # Keep A27/A28's trajectory selection in both scopes. Their D8 targets
        # come from the main root, without the separate history-3 fallback.
        self.fallback_roots = []
        if self.supervision_scope == 'human_and_diverse':
            self.frames_root = Path(future.frames_root)
            self.rollouts_root = Path(future.rollouts_root)
            self.resolved_gt_root = str(self.root.resolve())
            self.resolved_rollouts_root = str(self.rollouts_root.resolve())
        self.image_size = tuple(int(value) for value in future.image_size)
        self.sensor_blobs_path = Path(future.sensor_blobs_path)
        self.build_missing_cache = bool(future.get('build_missing_cache', True))

    def get_unique_name(self):
        w, h = self.image_size
        if self.supervision_scope == 'human_only':
            return f'drivor_diverse8_human_future_rgb_v1_k{self.top_k}_{w}x{h}'
        return f'drivor_diverse8_future_rgb_v1_k{self.top_k}_{w}x{h}'

    def compute_targets(self, scene):
        result = self.load(scene.scene_metadata.initial_token)
        result.update(_human_future_targets(scene, self.sensor_blobs_path, self.image_size))
        return result

    def load(self, token):
        result = super().load(token)
        if self.supervision_scope == 'human_only':
            return result
        w, h = self.image_size
        images = np.zeros((self.top_k, 8, 3, h, w), np.uint8)
        frame_valid = np.zeros((self.top_k, 8), bool)
        slots = np.flatnonzero(result['pseudo_expert_mask'].numpy())
        if len(slots):
            path = self.frames_root / token[:2] / (token + '.zip')
            if not path.is_file() and self.build_missing_cache:
                extract_diverse_scene_frames(token, self.root, self.rollouts_root, self.frames_root, self.image_size)
            if not path.is_file():
                raise FileNotFoundError(f'No diverse8 future frames: {path}; enable build_missing_cache or run extraction')
            source = json.loads((self.root / token / 'trajectory.json').read_text())
            with zipfile.ZipFile(path) as archive:
                metadata = json.loads(archive.read('metadata.json'))
                same_gt_root = metadata['source_gt_root'] == self.resolved_gt_root
                if not same_gt_root:
                    # A cache may have been built from hist3 before its scene
                    # or annotation files were symlinked into the main root.
                    annotation_root = (self.root / token / 'trajectory.json').resolve().parent.parent
                    same_gt_root = metadata['source_gt_root'] == str(annotation_root)
                if (metadata['schema'] != 'diverse8_future_rgb_v1' or metadata['token'] != token
                        or metadata['size_wh'] != [w, h]
                        or not np.allclose(metadata['times'], TIMES, atol=1e-6, rtol=0)
                        or not same_gt_root
                        or metadata['source_rollouts'] != self.resolved_rollouts_root):
                    raise ValueError(f'Diverse8 future cache source/geometry mismatch: {path}')
                entries = {int(entry['slot']): entry for entry in metadata['slots']}
                if len(entries) != len(metadata['slots']) or set(entries) != set(slots.tolist()):
                    raise ValueError(f'Diverse8 future slots differ from valid trajectory slots: {token}')
                for slot in slots:
                    entry = entries[int(slot)]
                    if (entry['frame_indices'] != FRAME_INDICES or len(entry['images']) != 8
                            or entry['label'] != source['gt_labels'][slot]
                            or entry['candidate_index'] != source['selected_indices'][slot]
                            or not np.allclose(entry['trajectory'], result['pseudo_experts'][slot].numpy(), atol=1e-5, rtol=0)):
                        raise ValueError(f'Diverse8 GT/future pairing mismatch: {token}/{slot}')
                    for time, name in enumerate(entry['images']):
                        with Image.open(BytesIO(archive.read(name))) as image:
                            if image.size != (w, h):
                                raise ValueError(f'Wrong cached future frame resolution: {token}/{slot}')
                            images[slot, time] = np.asarray(image.convert('RGB')).transpose(2, 0, 1)
                        frame_valid[slot, time] = True
        result['future_register_images'] = torch.from_numpy(images)
        result['future_register_valid'] = torch.from_numpy(frame_valid)
        return result
