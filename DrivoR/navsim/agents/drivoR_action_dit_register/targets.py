"""Load trajectory-aligned Cosmos RGB frames; EMA features are computed online."""

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from navsim.agents.drivoR.multi_gt import MultiTrajectoryTargetBuilder
from navsim.common.oss_image import load_camera_image
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


def _human_future_targets(scene, sensor_blobs_path, image_size):
    token = scene.scene_metadata.initial_token
    current = scene.scene_metadata.num_history_frames-1
    images = []
    for time in range(1,9):
        camera = scene.frames[current+time].cameras.cam_f0
        pixels = camera.image
        if pixels is None:
            if camera.image_path is None:
                raise ValueError(f'Missing original human future image path for {token} at +{time*.5}s')
            pixels = load_camera_image(sensor_blobs_path,camera.image_path)
        image = Image.fromarray(pixels).resize(image_size,Image.Resampling.BICUBIC)
        images.append(np.asarray(image,dtype=np.uint8).transpose(2,0,1))
    return {'human_future_register_images':torch.from_numpy(np.stack(images)),
            'human_future_register_valid':torch.ones(8,dtype=torch.bool)}


class HumanFutureRegisterTargetBuilder(AbstractTargetBuilder):
    """Logged-human future frames only; no multi-GT files or Cosmos cache."""

    def __init__(self, future):
        self.image_size = tuple(int(value) for value in future.image_size)
        self.sensor_blobs_path = Path(future.get('sensor_blobs_path','.'))

    def get_unique_name(self):
        w,h = self.image_size
        return f'drivor_human_future_register_rgb_v1_{w}x{h}'

    def compute_targets(self, scene):
        return _human_future_targets(scene,self.sensor_blobs_path,self.image_size)


class FutureRegisterTargetBuilder(MultiTrajectoryTargetBuilder):
    def __init__(self, multi_gt, future):
        super().__init__(multi_gt)
        self.frames_root = Path(future.frames_root)
        self.image_size = tuple(int(value) for value in future.image_size)
        self.sensor_blobs_path = Path(future.get('sensor_blobs_path','.'))
        self.missing_frames = str(future.get("missing_frames", "ignore"))
        if not self.frames_root.is_dir():
            raise FileNotFoundError(f"Cosmos future-frame cache missing: {self.frames_root}")
        if self.weighting != "equal" or self.matching != "nearest":
            raise ValueError("Future-register variants use equal, independent nearest-GT supervision")

    def get_unique_name(self):
        w, h = self.image_size
        return f"drivor_future_register_rgb_v2_k{self.max_gt}_{w}x{h}"

    def compute_targets(self, scene):
        token = scene.scene_metadata.initial_token
        result = self.load(token,scene.get_future_trajectory(num_trajectory_frames=8).poses)
        result.update(_human_future_targets(scene,self.sensor_blobs_path,self.image_size))
        return result

    def load(self, token, fallback):
        result = super().load(token, fallback)
        w, h = self.image_size
        images = np.zeros((self.max_gt, 8, 3, h, w), dtype=np.uint8)
        valid = np.zeros((self.max_gt, 8), dtype=bool)
        path = self.frames_root / token[:2] / (token + '.zip')
        if not bool(result['multi_gt_fallback']) and path.is_file():
            folder = self.root / token
            info = json.loads((folder / 'trajectory.json').read_text())
            source_valid = np.load(folder / 'gt_valid_mask.npy', allow_pickle=False).astype(bool)
            slots = np.flatnonzero(source_valid & ~np.asarray(info['gt_padded'], bool))
            with zipfile.ZipFile(path) as archive:
                metadata = json.loads(archive.read('metadata.json'))
                if metadata['size_wh'] != [w, h] or not np.allclose(metadata['times'], np.arange(1,9)*.5):
                    raise ValueError(f"Future frame resolution/time mismatch: {path}")
                entries = {int(entry['slot']): entry for entry in metadata['slots']}
                if len(entries) != len(metadata['slots']) or set(entries) != set(slots.tolist()):
                    raise ValueError(f"Future-frame slots disagree with valid trajectory slots: {token}")
                for row, slot in enumerate(slots):
                    entry = entries[int(slot)]
                    if entry['frame_indices'] != list(range(5,41,5)) or len(entry['images']) != 8:
                        raise ValueError(f"Future frames must be zero-based indices 5,10,...,40: {token}/{slot}")
                    expected = result['trajectory_multi_gt'][row].numpy()
                    if not np.allclose(np.asarray(entry['trajectory'],np.float32), expected, rtol=0, atol=1e-5):
                        raise ValueError(f"GT and future-video pairing disagree: {token}/{slot}")
                    if entry['label'] != info['gt_labels'][slot]:
                        raise ValueError(f"GT and future-video labels disagree: {token}/{slot}")
                    for time, name in enumerate(entry['images']):
                        with Image.open(BytesIO(archive.read(name))) as image:
                            if image.size != (w,h):
                                raise ValueError(f"Wrong cached frame size: {token}/{slot}/{time}")
                            images[row,time] = np.asarray(image.convert('RGB')).transpose(2,0,1)
                        valid[row,time] = True
        elif self.missing_frames == 'error':
            raise FileNotFoundError(f"No paired future RGB targets for {token}: {path}")
        result['future_register_images'] = torch.from_numpy(images)
        result['future_register_valid'] = torch.from_numpy(valid)
        return result
