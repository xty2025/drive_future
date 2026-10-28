"""Trajectory-indexed diverse8 RGB caches; decode semantic frames 5,10,...,40."""

from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np
from PIL import Image


TIMES = np.arange(1, 9, dtype=np.float32) * .5
FRAME_INDICES = list(range(5, 41, 5))
DEFAULT_DIVERSE_GT = '/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64'
DEFAULT_DIVERSE_ROLLOUTS = '/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_diverse8_full_allcosmos_cuda_v1_gpu256'


def decode_future_frames(video, size):
    width, height = size
    selection = '+'.join(f'eq(n\\,{index})' for index in FRAME_INDICES)
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(video),
               '-vf', f'select={selection},scale={width}:{height}', '-vsync', 'vfr',
               '-frames:v', '8', '-threads', '1', '-filter_threads', '1',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'Cannot decode {video}: {result.stderr.decode(errors="replace")[-600:]}')
    if len(result.stdout) != 8 * height * width * 3:
        raise ValueError(f'Expected future frames 5,10,...,40 in {video}')
    return np.frombuffer(result.stdout, np.uint8).reshape(8, height, width, 3)


def extract_diverse_scene_frames(token, gt_root, rollouts_root, frames_root, size=(384, 384)):
    """Publish one complete ZIP atomically; concurrent first reads may safely race.

    Source slot/candidate identities and semantic times are checked, without
    hashing videos. RGB caches are reusable across models and EMA updates.
    """
    output = Path(frames_root) / token[:2] / (token + '.zip')
    if output.is_file():
        return output
    gt_folder, video_folder = Path(gt_root) / token, Path(rollouts_root) / token
    if not (gt_folder / 'complete.json').is_file():
        raise FileNotFoundError(f'Incomplete diverse8 trajectories: {gt_folder}')
    info = json.loads((gt_folder / 'trajectory.json').read_text())
    trajectories = np.load(gt_folder / 'gt_trajectories.npy', allow_pickle=False)
    valid = np.load(gt_folder / 'gt_valid_mask.npy', allow_pickle=False).astype(bool)
    slots = np.flatnonzero(valid & ~np.asarray(info['gt_padded'], bool))
    if len(slots):
        complete = json.loads((video_folder / 'completion.json').read_text())
        if complete.get('status') != 'COMPLETE':
            raise ValueError(f'Incomplete diverse8 videos: {video_folder}')
    if not np.allclose(info['target_times'], TIMES, atol=1e-6, rtol=0):
        raise ValueError(f'Diverse8 must use +0.5,...,+4.0 second targets: {token}')
    manifest = dict(schema='diverse8_future_rgb_v1', token=token, size_wh=list(size),
                    times=TIMES.tolist(), source_gt_root=str(Path(gt_root).resolve()),
                    source_rollouts=str(Path(rollouts_root).resolve()), slots=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=token+'.', suffix='.tmp', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for slot in slots:
                slot = int(slot)
                folder = video_folder / f'trajectory_{slot}'
                metadata = json.loads((folder / 'metadata.json').read_text())
                candidate = int(info['selected_indices'][slot])
                if (not metadata.get('valid') or not metadata.get('source_valid')
                        or metadata.get('source_padded') or metadata.get('token') != token
                        or int(metadata['trajectory_index']) != slot
                        or metadata['trajectory_label'] != info['gt_labels'][slot]
                        or int(metadata['source_selected_candidate_index']) != candidate):
                    raise ValueError(f'Diverse8 GT/video slot or candidate mismatch: {token}/{slot}')
                pose_times = np.load(folder / 'pose_times_s.npy', allow_pickle=False)
                if len(pose_times) <= 40 or not np.allclose(pose_times[FRAME_INDICES], TIMES, atol=1e-5, rtol=0):
                    raise ValueError(f'Video frame zero must be current and frame 5 must be +0.5s: {folder}')
                pixels = decode_future_frames(folder / 'future.mp4', size)
                names = []
                for index, image in zip(FRAME_INDICES, pixels):
                    name = f'slot_{slot}/frame_{index:06d}.jpg'
                    buffer = BytesIO()
                    Image.fromarray(image).save(buffer, format='JPEG', quality=90)
                    archive.writestr(name, buffer.getvalue())
                    names.append(name)
                manifest['slots'].append(dict(slot=slot, label=info['gt_labels'][slot],
                    candidate_index=candidate, trajectory=trajectories[slot].tolist(),
                    frame_indices=FRAME_INDICES, images=names))
            archive.writestr('metadata.json', json.dumps(manifest))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
