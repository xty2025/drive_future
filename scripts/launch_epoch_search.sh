#!/bin/bash
set -eo pipefail

source /usr/local/PPU_SDK/envsetup.sh ppu >/dev/null 2>&1
source /mnt/zhaozc_workspace/envs/driov_dpo/bin/activate
set -u

export PYTHONPATH="/mnt/zhaozc_workspace/project/NVS/src:/mnt/zhaozc_workspace/official_DrivoR_clean:/mnt/zhaozc_workspace/official_DrivoR_clean/nuplan-devkit"
export NUPLAN_MAPS_ROOT=/mnt/data/nuplan/maps
export MPLCONFIGDIR=/tmp/matplotlib-nvs-epoch-search
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

cd /mnt/zhaozc_workspace/project/NVS
exec python -u scripts/run_epoch_search.py "$@"
