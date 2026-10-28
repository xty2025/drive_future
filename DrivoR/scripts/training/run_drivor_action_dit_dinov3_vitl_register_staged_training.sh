#!/usr/bin/env bash
set -euo pipefail

# Stage1: clean GT actions + register denoising only.
# Stage2: load all Stage1/EMA weights, reset optimizer, jointly train generator/scorer.
DRIVOR_REGISTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_REGISTER_ROOT="$(cd -- "${DRIVOR_REGISTER_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_REGISTER_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-action_dit_dinov3_vitl_register_staged_seed2_run1}}"
DRIVOR_REGISTER_BASE_EXP="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_register_staged}"
DRIVOR_REGISTER_STAGE1_EXP="${DRIVOR_REGISTER_BASE_EXP}_stage1"
DRIVOR_REGISTER_STAGE2_EXP="${DRIVOR_REGISTER_BASE_EXP}_stage2"
DRIVOR_REGISTER_STAGE1_RUN="${NAVSIM_EXP_ROOT}/ke/${DRIVOR_REGISTER_STAGE1_EXP}/${DRIVOR_RUN_ID}"
DRIVOR_REGISTER_READY="${DRIVOR_REGISTER_STAGE1_RUN}/stage1_complete.json"

for DRIVOR_REGISTER_ARG in "$@"; do
    case "${DRIVOR_REGISTER_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_REGISTER_STAGE2_EXP}" \
            exec bash "${DRIVOR_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_register_stage2_training.sh" "$@" ;;
    esac
done

if [[ -z "${DRIVOR_REGISTER_INIT_CHECKPOINT:-}" ]]; then
    if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
        /usr/bin/python3 - "${DRIVOR_REGISTER_READY}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
    fi
    DRIVOR_EXPERIMENT="${DRIVOR_REGISTER_STAGE1_EXP}" \
    bash "${DRIVOR_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_register_stage1_training.sh" \
        auto_resume=false train_ckpt_path=null runtime.require_resume=false "$@"

    if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
        /usr/bin/python3 - "${DRIVOR_REGISTER_STAGE1_RUN}" "${DRIVOR_REGISTER_READY}" <<'PY'
import json
from pathlib import Path
import sys
files=list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not files:
    raise FileNotFoundError('Stage1 completed without last.ckpt in '+sys.argv[1])
path=max(files,key=lambda value:value.stat().st_mtime_ns)
ready=Path(sys.argv[2]); temporary=ready.with_suffix('.tmp')
temporary.write_text(json.dumps({'checkpoint':str(path.resolve())})+'\n')
temporary.replace(ready)
PY
    fi
    # Every DLC node waits for rank0's completed checkpoint, never a partially
    # written checkpoint discovered by another worker's directory scan.
    DRIVOR_REGISTER_INIT_CHECKPOINT="$(/usr/bin/python3 - "${DRIVOR_REGISTER_READY}" <<'PY'
import json
import os
from pathlib import Path
import sys
import time
ready=Path(sys.argv[1])
deadline=time.monotonic()+float(os.environ.get('DRIVOR_STAGE_TRANSITION_TIMEOUT','1800'))
while not ready.is_file():
    if time.monotonic()>deadline:
        raise TimeoutError('Waiting for Stage1 completion: '+str(ready))
    time.sleep(1)
print(json.loads(ready.read_text())['checkpoint'])
PY
    )"
fi
export DRIVOR_REGISTER_INIT_CHECKPOINT
DRIVOR_EXPERIMENT="${DRIVOR_REGISTER_STAGE2_EXP}" \
exec bash "${DRIVOR_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_register_stage2_training.sh" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false "$@"
