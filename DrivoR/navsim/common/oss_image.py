"""Read NAVSIM camera images locally, with optional Alibaba Cloud OSS fallback."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image


# Like FastWAM's Waymo reader, never reuse a parent's connection after a fork.
_OSS_BUCKETS: dict[tuple[int, str, str, bool], Any] = {}
_READ_ATTEMPTS = 5
_RETRY_WAIT_SECONDS = 0.3


def _oss_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("oss://") or "?" in uri or "#" in uri:
        raise ValueError("Camera images require unsigned oss://bucket/key URIs")
    bucket, separator, key = uri[6:].partition("/")
    if not bucket or not separator or not key or bucket in {".", ".."} or "@" in bucket or ":" in bucket:
        raise ValueError("Invalid OSS camera image URI")
    return bucket, key


def _oss_bucket(config_path: str, bucket_name: str):
    try:
        import oss2
    except ImportError:
        raise RuntimeError(
            "OSS camera images require oss2; install DrivoR/requirements-oss.txt"
        ) from None

    config_path = str(Path(config_path).expanduser().resolve())
    use_proxy = os.environ.get("NAVSIM_OSS_USE_PROXY", "0").lower() in {"1", "true", "yes"}
    cache_key = (os.getpid(), config_path, bucket_name, use_proxy)
    if cache_key not in _OSS_BUCKETS:
        try:
            with open(config_path, encoding="utf-8") as stream:
                config = json.load(stream)[bucket_name]
            endpoint = config["endpoint"]
            access_key_id = config["access_key_id"]
            access_key_secret = config["access_key_secret"]
            if not all(isinstance(value, str) and value for value in (
                endpoint, access_key_id, access_key_secret
            )):
                raise ValueError("Invalid OSS configuration fields")
            if not endpoint.startswith(("http://", "https://")):
                endpoint = "https://" + endpoint
            session = oss2.Session()
            # The configured cluster OSS endpoint is reachable directly. Opt in
            # explicitly to ambient proxies instead of inheriting a SOCKS proxy.
            session.session.trust_env = use_proxy
            _OSS_BUCKETS[cache_key] = oss2.Bucket(
                oss2.Auth(access_key_id, access_key_secret),
                endpoint,
                bucket_name,
                session=session,
                connect_timeout=20,
            )
        except Exception as error:
            # SDK/config exceptions can include credentials. Report only type.
            raise RuntimeError(
                f"Cannot initialize OSS camera reader for bucket {bucket_name} "
                f"using {config_path} ({type(error).__name__})"
            ) from None
    return _OSS_BUCKETS[cache_key]


def _read_rgb(source) -> np.ndarray:
    with Image.open(source) as image:
        image.load()
        return np.array(image.convert("RGB"))


def _cache_path(uri: str) -> Path | None:
    directory = os.environ.get("NAVSIM_OSS_IMAGE_CACHE_DIR")
    if not directory:
        return None
    bucket, key = _oss_parts(uri)
    relative_key = Path(key)
    if relative_key.is_absolute() or ".." in relative_key.parts:
        raise ValueError("OSS image cache keys must stay within the bucket directory")
    return Path(directory).expanduser() / bucket / relative_key


def _write_cache(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        # Readers see a complete, decoded image even when workers race to fill it.
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_oss_image(uri: str) -> np.ndarray:
    bucket_name, key = _oss_parts(uri)
    cached = _cache_path(uri)
    if cached is not None:
        try:
            return _read_rgb(cached)
        except FileNotFoundError:
            pass
        except OSError as error:
            # PIL decode errors have no errno; genuine filesystem errors should
            # remain visible. A corrupt image cache can be downloaded again.
            if error.errno is not None:
                raise
            cached.unlink(missing_ok=True)

    config_path = os.environ.get("NAVSIM_OSS_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("Set NAVSIM_OSS_CONFIG_PATH to the OSS credential JSON file")
    bucket = _oss_bucket(config_path, bucket_name)
    for attempt in range(_READ_ATTEMPTS):
        try:
            payload = bucket.get_object(key).read()
            pixels = _read_rgb(io.BytesIO(payload))
        except Exception as error:
            status = getattr(error, "status", None)
            status = status if isinstance(status, int) else None
            permanent = status in {400, 401, 403, 404}
            if permanent or attempt + 1 == _READ_ATTEMPTS:
                exception_type = FileNotFoundError if status == 404 else OSError
                raise exception_type(
                    f"OSS camera image read failed: {uri} after {attempt + 1} attempt(s) "
                    f"({type(error).__name__}, status={status})"
                ) from None
            time.sleep(_RETRY_WAIT_SECONDS * (attempt + 1))
        else:
            if cached is not None:
                _write_cache(cached, payload)
            return pixels
    raise RuntimeError("Unreachable OSS retry state")


def load_camera_image(sensor_blobs_path: Path, data_path: str) -> np.ndarray:
    """Return RGB pixels; the OSS fallback root mirrors sensor_blobs_path.

    NAVSIM_OSS_SENSOR_BLOBS_PATH includes the split, for example
    oss://bucket/dataset/sensor_blobs/trainval. Explicit OSS data_path values
    are also accepted. Local images do not require OSS configuration or oss2.
    """
    data_path = str(data_path)
    if data_path.startswith("oss://"):
        return _read_oss_image(data_path)

    local_root = Path(sensor_blobs_path)
    local_path = local_root / data_path
    try:
        return _read_rgb(local_path)
    except FileNotFoundError:
        remote_root = os.environ.get("NAVSIM_OSS_SENSOR_BLOBS_PATH")
        if not remote_root:
            raise FileNotFoundError(
                f"Missing NAVSIM camera image: {local_path}. Configure "
                "NAVSIM_OSS_SENSOR_BLOBS_PATH and NAVSIM_OSS_CONFIG_PATH for OSS fallback."
            ) from None

    relative_path = Path(data_path)
    if relative_path.is_absolute():
        try:
            relative_path = local_path.resolve().relative_to(local_root.resolve())
        except ValueError:
            raise ValueError("An absolute camera path outside sensor_blobs_path cannot be mapped to OSS") from None
    if ".." in relative_path.parts:
        raise ValueError("Camera paths mapped to OSS cannot traverse parent directories")
    uri = remote_root.rstrip("/") + "/" + relative_path.as_posix()
    return _read_oss_image(uri)
