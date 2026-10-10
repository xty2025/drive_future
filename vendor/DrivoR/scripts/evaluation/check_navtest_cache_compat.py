"""Check whether an external navtest metric cache matches DrivoR's navtest split and MetricCache format.

Usage: python scripts/evaluation/check_navtest_cache_compat.py <metric_cache_root>
"""
import lzma
import pickle
import sys
from pathlib import Path

from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader, filter_scenes
from navsim.common.dataclasses import SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache

REPO = Path(__file__).resolve().parents[2]
NAVTEST_FILTER = REPO / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
NAVSIM_LOG_PATH = Path("/mnt/data/navsim/test_navsim_logs/test")


def main() -> None:
    cache_root = Path(sys.argv[1])
    filter_cfg = OmegaConf.to_container(OmegaConf.load(NAVTEST_FILTER))
    filter_cfg = {k: v for k, v in filter_cfg.items() if not k.startswith("_")}
    scene_filter = SceneFilter(**filter_cfg)
    scenes = filter_scenes(NAVSIM_LOG_PATH, scene_filter)
    drivor_tokens = set(scenes.keys())
    print(f"[split] drivor navtest scenes: {len(drivor_tokens)}")

    loader = MetricCacheLoader(cache_root)
    cache_tokens = set(loader.metric_cache_paths.keys())
    print(f"[cache] metric cache entries: {len(cache_tokens)}")
    print(f"[overlap] intersection: {len(drivor_tokens & cache_tokens)}")
    print(f"[missing] in split but not cached: {len(drivor_tokens - cache_tokens)}")
    print(f"[extra] cached but not in split: {len(cache_tokens - drivor_tokens)}")

    probe = loader.metric_cache_paths[next(iter(cache_tokens & drivor_tokens or cache_tokens))]
    with open(probe, "rb") as f:
        obj = pickle.loads(lzma.decompress(f.read()))
    print(f"[pickle] type: {type(obj).__module__}.{type(obj).__name__}")
    print(f"[pickle] attrs: {sorted(vars(obj).keys())}")
    expected = {f for f in MetricCache.__dataclass_fields__}
    print(f"[pickle] expected fields: {sorted(expected)}")
    print(f"[pickle] missing attrs: {sorted(expected - set(vars(obj).keys()))}")
    for attr in ("ego_state", "trajectory", "observation", "centerline", "route_lane_ids", "drivable_area_map"):
        print(f"[attr] {attr}: {type(getattr(obj, attr, None)).__name__}")


if __name__ == "__main__":
    main()
