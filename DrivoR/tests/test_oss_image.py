import io
import json
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from navsim.common import oss_image


@pytest.fixture(autouse=True)
def isolated_oss_environment(monkeypatch):
    for name in (
        "NAVSIM_OSS_CONFIG_PATH",
        "NAVSIM_OSS_SENSOR_BLOBS_PATH",
        "NAVSIM_OSS_IMAGE_CACHE_DIR",
        "NAVSIM_OSS_USE_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    oss_image._OSS_BUCKETS.clear()
    yield
    oss_image._OSS_BUCKETS.clear()


def image_bytes(color=(17, 31, 63)):
    stream = io.BytesIO()
    Image.new("RGB", (7, 5), color).save(stream, format="PNG")
    return stream.getvalue()


def cached_images(directory):
    return [path for path in directory.rglob("*") if path.is_file() and path.suffix != ".tmp"]


class FakeBucket:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.keys = []

    def get_object(self, key):
        self.keys.append(key)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply)


class FakeOSSError(Exception):
    def __init__(self, status):
        super().__init__("SDK detail containing a-test-secret-that-must-not-be-logged")
        self.status = status


def fake_reader(monkeypatch, replies):
    bucket = FakeBucket(replies)
    monkeypatch.setenv("NAVSIM_OSS_CONFIG_PATH", "/unused/test-config.json")
    monkeypatch.setattr(oss_image, "_oss_bucket", lambda *_: bucket)
    return bucket


def test_local_image_takes_priority_without_reading_credentials(tmp_path, monkeypatch):
    path = tmp_path / "scene" / "CAM_F0" / "frame.png"
    path.parent.mkdir(parents=True)
    Image.new("L", (7, 5), 19).save(path)
    monkeypatch.setenv("NAVSIM_OSS_CONFIG_PATH", "/missing/config.json")
    monkeypatch.setenv("NAVSIM_OSS_SENSOR_BLOBS_PATH", "oss://example-bucket/trainval")
    monkeypatch.setattr(oss_image, "_oss_bucket", lambda *_: pytest.fail("Local image used OSS"))

    pixels = oss_image.load_camera_image(tmp_path, "scene/CAM_F0/frame.png")

    assert pixels.shape == (5, 7, 3)
    assert pixels.dtype == np.uint8
    assert (pixels == 19).all()


def test_missing_local_image_maps_to_oss_and_reuses_disk_cache(tmp_path, monkeypatch):
    bucket = fake_reader(monkeypatch, [image_bytes()])
    monkeypatch.setenv("NAVSIM_OSS_SENSOR_BLOBS_PATH", "oss://example-bucket/dataset/trainval/")
    cache = tmp_path / "cache"
    monkeypatch.setenv("NAVSIM_OSS_IMAGE_CACHE_DIR", str(cache))

    first = oss_image.load_camera_image(tmp_path / "local", "scene/CAM_L0/frame.jpg")
    assert bucket.keys == ["dataset/trainval/scene/CAM_L0/frame.jpg"]
    assert cached_images(cache) == [cache / "example-bucket/dataset/trainval/scene/CAM_L0/frame.jpg"]
    assert not list(cache.rglob("*.tmp"))

    monkeypatch.delenv("NAVSIM_OSS_CONFIG_PATH")
    monkeypatch.setattr(oss_image, "_oss_bucket", lambda *_: pytest.fail("Cache hit used OSS"))
    second = oss_image.load_camera_image(tmp_path / "local", "scene/CAM_L0/frame.jpg")
    np.testing.assert_array_equal(first, second)


def test_explicit_oss_uri_is_not_joined_with_local_root(tmp_path, monkeypatch):
    bucket = fake_reader(monkeypatch, [image_bytes()])
    pixels = oss_image.load_camera_image(tmp_path, "oss://example-bucket/other/frame.jpg")
    assert bucket.keys == ["other/frame.jpg"]
    assert pixels.shape == (5, 7, 3)


def test_missing_local_image_without_fallback_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(oss_image, "_oss_bucket", lambda *_: pytest.fail("Unexpected OSS access"))
    with pytest.raises(FileNotFoundError, match="NAVSIM_OSS_SENSOR_BLOBS_PATH"):
        oss_image.load_camera_image(tmp_path, "missing.jpg")


@pytest.mark.parametrize("uri", [
    "oss://example-bucket/frame.jpg?signature=not-a-real-signature",
    "oss://example-bucket/frame.jpg#fragment",
    "oss://test-id:test-secret@example-bucket/frame.jpg",
    "oss://example-bucket",
])
def test_signed_or_malformed_uris_are_rejected_before_credentials(uri, tmp_path, monkeypatch):
    monkeypatch.setattr(oss_image, "_oss_bucket", lambda *_: pytest.fail("Invalid URI reached OSS"))
    with pytest.raises(ValueError):
        oss_image.load_camera_image(tmp_path, uri)


def test_missing_remote_object_does_not_retry_or_expose_sdk_details(tmp_path, monkeypatch):
    bucket = fake_reader(monkeypatch, [FakeOSSError(404)])
    monkeypatch.setattr(oss_image.time, "sleep", lambda *_: pytest.fail("Permanent error retried"))
    with pytest.raises(FileNotFoundError) as caught:
        oss_image.load_camera_image(tmp_path, "oss://example-bucket/missing.jpg")
    assert len(bucket.keys) == 1
    rendered = "".join(traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__))
    assert "a-test-secret-that-must-not-be-logged" not in rendered
    assert "status=404" in str(caught.value)


def test_transient_and_invalid_image_responses_retry_before_caching(tmp_path, monkeypatch):
    bucket = fake_reader(monkeypatch, [FakeOSSError(503), b"not an image", image_bytes()])
    waits = []
    monkeypatch.setattr(oss_image.time, "sleep", waits.append)
    monkeypatch.setenv("NAVSIM_OSS_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    pixels = oss_image.load_camera_image(tmp_path, "oss://example-bucket/frame.jpg")
    assert len(bucket.keys) == 3
    assert waits == pytest.approx([0.3, 0.6])
    assert pixels[0, 0].tolist() == [17, 31, 63]
    assert len(cached_images(tmp_path / "cache")) == 1


def test_exhausted_retries_report_sanitized_error(tmp_path, monkeypatch):
    bucket = fake_reader(monkeypatch, [FakeOSSError(500) for _ in range(5)])
    monkeypatch.setattr(oss_image.time, "sleep", lambda *_: None)
    with pytest.raises(OSError) as caught:
        oss_image.load_camera_image(tmp_path, "oss://example-bucket/frame.jpg")
    assert len(bucket.keys) == 5
    assert "5 attempt(s)" in str(caught.value)
    assert "a-test-secret-that-must-not-be-logged" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_corrupt_cache_is_replaced_with_a_decoded_image(tmp_path, monkeypatch):
    bucket = fake_reader(monkeypatch, [image_bytes(), image_bytes((99, 88, 77))])
    cache = tmp_path / "cache"
    monkeypatch.setenv("NAVSIM_OSS_IMAGE_CACHE_DIR", str(cache))
    uri = "oss://example-bucket/frame.jpg"
    oss_image.load_camera_image(tmp_path, uri)
    cached = cached_images(cache)[0]
    cached.write_bytes(b"broken cached image")
    pixels = oss_image.load_camera_image(tmp_path, uri)
    assert pixels[0, 0].tolist() == [99, 88, 77]
    assert len(bucket.keys) == 2
    assert not list(cache.rglob("*.tmp"))


def test_cache_is_separated_by_full_bucket_and_key(tmp_path, monkeypatch):
    fake_reader(monkeypatch, [image_bytes((1, 2, 3)), image_bytes((4, 5, 6))])
    monkeypatch.setenv("NAVSIM_OSS_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    first_uri = "oss://first-bucket/shared/frame.jpg"
    second_uri = "oss://second-bucket/shared/frame.jpg"
    first = oss_image.load_camera_image(tmp_path, first_uri)
    second = oss_image.load_camera_image(tmp_path, second_uri)
    assert len(cached_images(tmp_path / "cache")) == 2
    monkeypatch.setattr(oss_image, "_oss_bucket", lambda *_: pytest.fail("Cached URI used OSS"))
    np.testing.assert_array_equal(oss_image.load_camera_image(tmp_path, first_uri), first)
    np.testing.assert_array_equal(oss_image.load_camera_image(tmp_path, second_uri), second)


def test_cache_write_failure_does_not_leave_partial_files(tmp_path, monkeypatch):
    fake_reader(monkeypatch, [image_bytes()])
    cache = tmp_path / "cache"
    monkeypatch.setenv("NAVSIM_OSS_IMAGE_CACHE_DIR", str(cache))

    def fail_replace(*_):
        raise OSError("simulated cache filesystem failure")

    monkeypatch.setattr(oss_image.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated cache filesystem failure"):
        oss_image.load_camera_image(tmp_path, "oss://example-bucket/frame.jpg")
    assert not list(cache.rglob("*.tmp"))
    assert not cached_images(cache)


def test_bucket_clients_are_scoped_by_pid_and_proxy_mode(tmp_path, monkeypatch):
    config = tmp_path / "oss.json"
    config.write_text(json.dumps({"example-bucket": {
        "access_key_id": "unit-test-id",
        "access_key_secret": "unit-test-secret",
        "endpoint": "example.invalid",
    }}))
    created = []

    def create_bucket(auth, endpoint, name, **kwargs):
        bucket = SimpleNamespace(auth=auth, endpoint=endpoint, name=name, **kwargs)
        created.append(bucket)
        return bucket

    fake_sdk = SimpleNamespace(
        Auth=lambda key_id, key_secret: (key_id, key_secret),
        Session=lambda: SimpleNamespace(session=SimpleNamespace(trust_env=True)),
        Bucket=create_bucket,
    )
    monkeypatch.setitem(sys.modules, "oss2", fake_sdk)
    monkeypatch.setattr(oss_image.os, "getpid", lambda: 1001)
    first = oss_image._oss_bucket(str(config), "example-bucket")
    assert oss_image._oss_bucket(str(config), "example-bucket") is first
    assert first.endpoint == "https://example.invalid"
    assert first.session.session.trust_env is False
    monkeypatch.setattr(oss_image.os, "getpid", lambda: 1002)
    second = oss_image._oss_bucket(str(config), "example-bucket")
    assert second is not first
    monkeypatch.setenv("NAVSIM_OSS_USE_PROXY", "1")
    proxied = oss_image._oss_bucket(str(config), "example-bucket")
    assert proxied is not second and proxied.session.session.trust_env is True
    assert len(created) == 3


def test_cameras_use_shared_reader_only_for_selected_views(tmp_path, monkeypatch):
    from navsim.common import dataclasses

    calls = []

    def reader(root, data_path):
        calls.append((root, data_path))
        return np.zeros((5, 7, 3), dtype=np.uint8)

    monkeypatch.setattr(dataclasses, "load_camera_image", reader)
    names = ["CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"]
    records = {name: {
        "data_path": f"oss://example-bucket/{name}/frame.jpg",
        "sensor2lidar_rotation": np.eye(3),
        "sensor2lidar_translation": np.zeros(3),
        "cam_intrinsic": np.eye(3),
        "distortion": np.zeros(5),
    } for name in names}
    cameras = dataclasses.Cameras.from_camera_dict(
        tmp_path, records, ["cam_f0", "cam_l0", "cam_r0", "cam_b0"]
    )
    assert len(calls) == 4
    assert all(root == tmp_path and uri.startswith("oss://") for root, uri in calls)
    assert cameras.cam_f0.image.shape == (5, 7, 3)
    assert cameras.cam_b0.image.shape == (5, 7, 3)
    assert cameras.cam_l1.image is None
    np.testing.assert_array_equal(cameras.cam_f0.intrinsics, records["CAM_F0"]["cam_intrinsic"])
