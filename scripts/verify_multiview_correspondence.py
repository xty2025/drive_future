"""Verify that /mnt/data/multi_view videos line up with their Multi-GT trajectories.

Every scene ships four ``trajectory_k`` folders.  This script checks, per scene:

1. ``scene_metadata.json`` agrees with the local NAVSIM log (token, scene_token,
   frame index, timestamp) and the directory name.
2. Each ``ego_pose.npy`` is dense [41,3] at 0.1 s in DrivoR's relative frame
   (``t0 local rear; x forward/y left/yaw CCW``) and reproduces the
   ``expected_motion`` endpoint and path length recorded in ``metadata.json``.
3. The ``trajectory_label == "human_log"`` candidate reproduces the real future
   ego poses read from the NAVSIM log at +0.5 s ... +4.0 s.  This is the check
   that ties a generated video to the scene it was conditioned on: if the video
   folder were shuffled across scenes, the human trajectory could not match.
4. ``completion.json`` reports COMPLETE with four valid videos and each
   ``future.mp4`` really decodes 41 frames.
5. ``actions_meters.npy`` -- the array that actually conditioned the Cosmos
   generation of ``future.mp4`` -- integrates back to that folder's
   ``ego_pose.npy`` to float64 machine precision.  This is the check that ties a
   video to a trajectory rather than to a scene: without it a folder could hold
   a video generated from some other candidate's actions.
6. ``ego_pose_global.npy`` is the same motion expressed in the global rear-axle
   frame (``inv(global[0]) @ global == ego_pose``) and its origin equals
   ``scene_metadata.rear_origin_global``.
7. Every sha256 recorded at generation time (input arrays, ``future.mp4``,
   ``initial_front.jpg``, camera extrinsics) still matches the bytes on disk.
8. ``initial_front.jpg`` is byte-identical to the real NAVSIM ``CAM_F0`` blob
   that the log's ``data_path`` for the t0 frame resolves to, so the video's
   first frame and the observation's last history frame are the same image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

DEFAULT_MULTIVIEW_ROOT = (
    "/mnt/data/multi_view/navtrain_full_allcosmos_cuda_v1_gpu64/"
    "navtrain_full_allcosmos_cuda_v1_gpu64"
)
DEFAULT_LOG_ROOT = "/mnt/data/navsim/trainval_navsim_logs/trainval"
DEFAULT_SENSOR_ROOT = "/mnt/data/navsim/trainval_sensor_blobs/trainval"
# The upstream generator ran off a different mount; only the tail is portable.
SOURCE_SENSOR_PREFIX = "sensor_blobs/trainval"
FUTURE_SAMPLE_INDICES = np.arange(5, 41, 5)
EXPECTED_DENSE_FRAMES = 41
MAX_GT = 4
POSE_FRAME = "t0 local rear; x forward/y left/yaw CCW; meters/radians"
GLOBAL_POSE_FRAME = "global rear axle, absolute heading"


def _relative_pose(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(reference) @ frame
    return np.asarray(
        [relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])],
        dtype=np.float64,
    )


def _se2(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = cosine, -sine, sine, cosine
    matrix[0, 3], matrix[1, 3] = pose[0], pose[1]
    return matrix


def _rotation_from_6d(columns: np.ndarray) -> np.ndarray:
    """Gram-Schmidt on the first two *columns*, the generator's convention."""

    first = columns[0:3] / np.linalg.norm(columns[0:3])
    second = columns[3:6] - np.dot(columns[3:6], first) * first
    second = second / np.linalg.norm(second)
    return np.stack([first, second, np.cross(first, second)], axis=1)


def action_chain_to_local_pose(actions_meters: np.ndarray, rear_from_camera: np.ndarray) -> np.ndarray:
    """Dead-reckon the Cosmos action condition into DrivoR's t0 local rear frame.

    ``actions_meters[k]`` is the body-frame increment from step ``k`` to ``k+1``
    expressed in the OpenCV camera frame, so the chain post-multiplies and the
    result is conjugated by the camera extrinsic to land in the rear-axle frame.
    """

    camera_from_rear = np.linalg.inv(rear_from_camera)
    current = np.eye(4)
    poses = [np.eye(4)]
    for step in range(actions_meters.shape[0]):
        increment = np.eye(4)
        increment[:3, :3] = _rotation_from_6d(actions_meters[step, 3:9])
        increment[:3, 3] = actions_meters[step, 0:3]
        current = current @ increment
        poses.append(current.copy())
    rear = rear_from_camera[None] @ np.asarray(poses) @ camera_from_rear[None]
    return np.stack(
        [rear[:, 0, 3], rear[:, 1, 3], np.arctan2(rear[:, 1, 0], rear[:, 0, 0])], axis=1
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return -1
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failures.append(f"{label}: {detail}")


def verify_scene(
    scene_dir: Path,
    log_root: Path,
    report: Report,
    check_video: bool,
    sensor_root: Path | None = None,
    check_sha256: bool = True,
) -> dict[str, object]:
    token = scene_dir.name
    metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
    report.check(metadata["token"] == token, token, f"metadata token {metadata['token']} != directory name")

    log_name = str(metadata["log_name"])
    frame_index = int(metadata["navsim_frame_index"])
    log_path = log_root / f"{log_name}.pkl"
    if not log_path.exists():
        report.failures.append(f"{token}: NAVSIM log missing: {log_path}")
        return {"token": token, "log_name": log_name, "status": "log-missing"}
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)
    if frame_index < 4 or frame_index + 8 >= len(frames):
        report.failures.append(f"{token}: frame_index {frame_index} outside log of {len(frames)} frames")
        return {"token": token, "log_name": log_name, "status": "frame-index-out-of-range"}
    frame = frames[frame_index]
    report.check(str(frame["token"]) == token, token, f"log token {frame['token']} != metadata token")
    report.check(
        str(frame["scene_token"]) == str(metadata["scene_token"]),
        token,
        f"log scene_token {frame['scene_token']} != metadata {metadata['scene_token']}",
    )
    report.check(
        int(frame["timestamp"]) == int(metadata["t0_timestamp_us"]),
        token,
        f"log timestamp {frame['timestamp']} != metadata {metadata['t0_timestamp_us']}",
    )

    completion = json.loads((scene_dir / "completion.json").read_text())
    report.check(completion["status"] == "COMPLETE", token, f"completion status {completion['status']}")
    report.check(
        completion.get("valid_videos") == MAX_GT,
        token,
        f"completion valid_videos {completion.get('valid_videos')}",
    )

    trajectory_dirs = sorted(
        (path for path in scene_dir.glob("trajectory_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    report.check(
        1 <= len(trajectory_dirs) <= MAX_GT,
        token,
        f"{len(trajectory_dirs)} trajectory directories; Multi-GT allows 1..{MAX_GT}",
    )

    reference = np.asarray(frame["ego2global"], dtype=np.float64)
    human_indices = [
        index for index in range(frame_index + 1, frame_index + 9)
    ]
    human_future = np.stack([_relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64)) for i in human_indices])

    rear_from_camera = np.load(scene_dir / "rear_from_camera_opencv.npy").astype(np.float64)
    rear_origin_global = np.asarray(metadata["rear_origin_global"], dtype=np.float64)
    if check_sha256:
        for name, recorded in metadata["scene_artifact_sha256"].items():
            digest = _sha256(scene_dir / name)
            report.check(digest == recorded, token, f"{name} sha256 {digest[:12]} != recorded {recorded[:12]}")

    labels: list[str] = []
    human_error = float("nan")
    endpoint_error = 0.0
    action_error = 0.0
    global_error = 0.0
    source_indices: list[int] = []
    slots: list[int] = []
    for trajectory_dir in trajectory_dirs:
        trajectory_metadata = json.loads((trajectory_dir / "metadata.json").read_text())
        label = str(trajectory_metadata["trajectory_label"])
        labels.append(label)
        source_indices.append(int(trajectory_metadata["source_selected_candidate_index"]))
        slots.append(int(trajectory_dir.name.rsplit("_", 1)[-1]))
        report.check(
            trajectory_metadata["token"] == token,
            token,
            f"{trajectory_dir.name} metadata token {trajectory_metadata['token']}",
        )
        report.check(
            int(trajectory_metadata["trajectory_index"]) == slots[-1],
            token,
            f"{trajectory_dir.name} trajectory_index {trajectory_metadata['trajectory_index']} != folder slot",
        )
        report.check(
            trajectory_metadata["pose_coordinate_frame"] == POSE_FRAME,
            token,
            f"{trajectory_dir.name} unexpected pose frame {trajectory_metadata['pose_coordinate_frame']!r}",
        )
        report.check(
            trajectory_metadata.get("ego_pose_global_coordinate_frame") == GLOBAL_POSE_FRAME,
            token,
            f"{trajectory_dir.name} unexpected global frame {trajectory_metadata.get('ego_pose_global_coordinate_frame')!r}",
        )
        report.check(
            trajectory_metadata.get("pose_interpolation") is False,
            token,
            f"{trajectory_dir.name} pose_interpolation {trajectory_metadata.get('pose_interpolation')!r}; NVS needs native 0.1 s poses",
        )
        report.check(
            (int(trajectory_metadata["fps"]), int(trajectory_metadata["num_frames"])) == (10, EXPECTED_DENSE_FRAMES),
            token,
            f"{trajectory_dir.name} video geometry {trajectory_metadata['fps']}fps/{trajectory_metadata['num_frames']}f",
        )

        poses = np.load(trajectory_dir / "ego_pose.npy").astype(np.float64)
        times = np.load(trajectory_dir / "pose_times_s.npy").astype(np.float64)
        report.check(poses.shape == (EXPECTED_DENSE_FRAMES, 3), token, f"{trajectory_dir.name} ego_pose shape {poses.shape}")
        report.check(
            times.shape == (EXPECTED_DENSE_FRAMES,) and np.allclose(times, np.linspace(0.0, 4.0, EXPECTED_DENSE_FRAMES)),
            token,
            f"{trajectory_dir.name} pose_times_s is not 0..4s at 0.1s",
        )
        report.check(np.allclose(poses[0], 0.0, atol=1e-9), token, f"{trajectory_dir.name} pose[0] is not the t0 origin")

        motion = trajectory_metadata["expected_motion"]
        endpoint = np.asarray(motion["endpoint_local_xy_m"], dtype=np.float64)
        endpoint_error = max(endpoint_error, float(np.abs(poses[-1, :2] - endpoint).max()))
        report.check(
            np.allclose(poses[-1, :2], endpoint, atol=1e-4),
            token,
            f"{trajectory_dir.name} endpoint {poses[-1, :2]} != expected_motion {endpoint}",
        )
        path_length = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
        report.check(
            abs(path_length - float(motion["path_length_m"])) < 0.05,
            token,
            f"{trajectory_dir.name} path length {path_length:.3f} != {motion['path_length_m']}",
        )

        # The action condition that generated this folder's video must be this
        # folder's trajectory; a mismatch means the video shows another candidate.
        actions_meters = np.load(trajectory_dir / "actions_meters.npy").astype(np.float64)
        report.check(
            actions_meters.shape == (EXPECTED_DENSE_FRAMES - 1, 9),
            token,
            f"{trajectory_dir.name} actions_meters shape {actions_meters.shape}",
        )
        reconstructed = action_chain_to_local_pose(actions_meters, rear_from_camera)
        action_error = max(action_error, float(np.abs(reconstructed - poses).max()))
        report.check(
            np.allclose(reconstructed, poses, atol=1e-6),
            token,
            f"{trajectory_dir.name} actions_meters integrates {np.abs(reconstructed - poses).max():.3e} away from ego_pose",
        )
        scale = float(trajectory_metadata["action_scale"])
        cosmos_action = np.load(trajectory_dir / "cosmos_action.npy").astype(np.float64)
        report.check(
            np.allclose(cosmos_action[:, :3], actions_meters[:, :3] * scale, atol=1e-4)
            and np.allclose(cosmos_action[:, 3:], actions_meters[:, 3:], atol=1e-6),
            token,
            f"{trajectory_dir.name} cosmos_action is not actions_meters with translation x{scale}",
        )

        # The global track is the same motion, so it must agree with the local one
        # and start exactly at the scene's recorded rear-axle origin.
        global_poses = np.load(trajectory_dir / "ego_pose_global.npy").astype(np.float64)
        report.check(
            np.allclose(global_poses[0], rear_origin_global, atol=1e-6),
            token,
            f"{trajectory_dir.name} ego_pose_global[0] {global_poses[0]} != rear_origin_global {rear_origin_global}",
        )
        origin = np.linalg.inv(_se2(global_poses[0]))
        relocal = np.stack(
            [
                (lambda matrix: np.asarray([matrix[0, 3], matrix[1, 3], np.arctan2(matrix[1, 0], matrix[0, 0])]))(origin @ _se2(entry))
                for entry in global_poses
            ]
        )
        global_error = max(global_error, float(np.abs(relocal - poses).max()))
        report.check(
            np.allclose(relocal, poses, atol=1e-6),
            token,
            f"{trajectory_dir.name} ego_pose_global relocalized {np.abs(relocal - poses).max():.3e} away from ego_pose",
        )

        if check_sha256:
            for group in ("input_artifact_sha256", "output_artifact_sha256"):
                for name, recorded in trajectory_metadata[group].items():
                    digest = _sha256(trajectory_dir / name)
                    report.check(
                        digest == recorded,
                        token,
                        f"{trajectory_dir.name}/{name} sha256 {digest[:12]} != recorded {recorded[:12]}",
                    )

        sampled = poses[FUTURE_SAMPLE_INDICES]
        if trajectory_metadata.get("is_human_log"):
            human_error = float(np.abs(sampled - human_future).max())
            report.check(
                np.allclose(sampled, human_future, atol=0.05),
                token,
                f"{trajectory_dir.name} human_log deviates {human_error:.4f} m/rad from the NAVSIM log future",
            )
        if check_video:
            count = _video_frame_count(trajectory_dir / "future.mp4")
            report.check(
                count == EXPECTED_DENSE_FRAMES,
                token,
                f"{trajectory_dir.name}/future.mp4 decoded {count} frames, expected {EXPECTED_DENSE_FRAMES}",
            )

    # The video's first frame is the observation's last history frame, so a
    # byte-identity here proves the t0 alignment end to end.
    initial_front_matches_blob: bool | None = None
    if sensor_root is not None:
        source = metadata["source_initial_front"]
        relative = Path(str(source["path"]).split(SOURCE_SENSOR_PREFIX, 1)[1])
        blob = Path(sensor_root) / relative
        if not blob.exists():
            report.failures.append(f"{token}: t0 sensor blob missing: {blob}")
        else:
            log_blob = Path(sensor_root) / Path(str(frame["cams"]["CAM_F0"]["data_path"]))
            initial_front_matches_blob = _sha256(scene_dir / "initial_front.jpg") == _sha256(blob) == source["sha256"]
            report.check(
                initial_front_matches_blob,
                token,
                "initial_front.jpg is not byte-identical to the real CAM_F0 t0 blob",
            )
            report.check(
                log_blob.exists() and log_blob.resolve() == blob.resolve(),
                token,
                f"the log's CAM_F0 data_path {log_blob} is not the recorded source blob {blob}",
            )

    degenerate = len(set(labels)) < len(labels) or len(set(source_indices)) < len(source_indices)
    return {
        "token": token,
        "log_name": log_name,
        "frame_index": frame_index,
        "status": "verified",
        "labels": labels,
        "source_indices": source_indices,
        "slots": slots,
        "degenerate_multi_gt": bool(degenerate),
        "human_log_present": bool(np.isfinite(human_error)),
        "human_log_max_abs_error": human_error,
        "endpoint_max_abs_error": endpoint_error,
        "action_chain_max_abs_error": action_error,
        "global_relocalize_max_abs_error": global_error,
        "initial_front_matches_t0_blob": initial_front_matches_blob,
        "gt_count": len(trajectory_dirs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--scenes", type=int, default=8, help="how many scenes to sample (0 = every --tokens entry)")
    parser.add_argument("--tokens", nargs="*", default=None, help="explicit scene tokens instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-video-decode", action="store_true", help="trust completion.json instead of decoding mp4")
    parser.add_argument("--skip-sha256", action="store_true", help="skip the byte-integrity digests (fastest at scale)")
    parser.add_argument("--skip-sensor-blob", action="store_true", help="skip the initial_front vs CAM_F0 t0 comparison")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    root = Path(args.multiview_root)
    log_root = Path(args.log_root)
    sensor_root = None if args.skip_sensor_blob else Path(args.sensor_root)
    if not root.is_dir():
        print(f"[FAIL] multiview root is not a directory: {root}")
        return 2
    if args.tokens:
        tokens = list(args.tokens)
    else:
        available = sorted(path.name for path in root.iterdir() if path.is_dir())
        random.seed(args.seed)
        tokens = available if args.scenes <= 0 else random.sample(available, min(args.scenes, len(available)))

    report = Report()
    summaries = []
    for token in tokens:
        summary = verify_scene(
            root / token,
            log_root,
            report,
            check_video=not args.skip_video_decode,
            sensor_root=sensor_root,
            check_sha256=not args.skip_sha256,
        )
        summaries.append(summary)
        if summary["status"] == "verified":
            human = (
                f"{summary['human_log_max_abs_error']:.4f}" if summary["human_log_present"] else "absent"
            )
            flag = " [DEGENERATE-MULTIGT]" if summary["degenerate_multi_gt"] else ""
            print(
                f"[OK]   {token} log={summary['log_name']} frame={summary['frame_index']} "
                f"gt={summary['gt_count']} human_err={human} "
                f"endpoint_err={summary['endpoint_max_abs_error']:.2e} "
                f"action_err={summary['action_chain_max_abs_error']:.2e} "
                f"global_err={summary['global_relocalize_max_abs_error']:.2e} "
                f"labels={summary['labels']}{flag}"
            )
        else:
            print(f"[SKIP] {token} status={summary['status']}")

    verified = [summary for summary in summaries if summary["status"] == "verified"]
    if verified:
        def worst(key: str) -> float:
            values = [float(summary[key]) for summary in verified if np.isfinite(summary.get(key, float("nan")))]
            return max(values) if values else float("nan")

        print(
            "\nworst over %d verified scenes: human_log %.4e  endpoint %.4e  action_chain %.4e  global_relocalize %.4e"
            % (len(verified), worst("human_log_max_abs_error"), worst("endpoint_max_abs_error"),
               worst("action_chain_max_abs_error"), worst("global_relocalize_max_abs_error"))
        )
        degenerate = sum(1 for summary in verified if summary["degenerate_multi_gt"])
        reordered = sum(1 for summary in verified if summary["slots"] != summary["source_indices"])
        print(
            "degenerate_multi_gt %d/%d   folder_slot != source_candidate_index %d/%d   "
            "initial_front==t0_blob %d/%d"
            % (
                degenerate, len(verified), reordered, len(verified),
                sum(1 for summary in verified if summary["initial_front_matches_t0_blob"]),
                sum(1 for summary in verified if summary["initial_front_matches_t0_blob"] is not None),
            )
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summaries": summaries, "failures": report.failures}, indent=2))
    print(f"\nchecks passed: {report.passed}  failures: {len(report.failures)}")
    for failure in report.failures[:40]:
        print(f"  [FAIL] {failure}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())

