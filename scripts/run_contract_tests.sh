#!/usr/bin/env bash
# Runs the v6 e2e contract suite.  The test module puts src/, vendor/DrivoR,
# vendor/DrivoR/nuplan-devkit and vendor/SimWAM/src on sys.path itself, so the only
# thing this wrapper owns is the interpreter: `python -m pytest`, never the venv's
# `pytest` shim, whose shebang is broken on this image.
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN="${NVS_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
exec "$PYBIN" -m pytest -q tests/contract_tests_v6_wan_e2e.py "$@"
