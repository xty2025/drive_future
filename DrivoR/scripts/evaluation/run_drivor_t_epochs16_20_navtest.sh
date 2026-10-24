#!/usr/bin/env bash
set -euo pipefail

# T01/T03/T05/T06/T07/T08/T09/T10, table epochs 16..20 = checkpoint epochs 15..19.
# Starts a detached local 8-GPU FP32 navtest sweep, restoring the original GPU
# occupancy processes on completion, failure, or a handled SIGINT/SIGTERM.
# bash .../run_drivor_t_epochs16_20_navtest.sh --dry-run
# bash .../run_drivor_t_epochs16_20_navtest.sh --status [operation_directory]
# bash .../run_drivor_t_epochs16_20_navtest.sh --stop operation_directory
# bash .../run_drivor_t_epochs16_20_navtest.sh --restore operation_directory
# bash .../run_drivor_t_epochs16_20_navtest.sh --resume operation_directory
DRIVOR_SWEEP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 "${DRIVOR_SWEEP_DIR}/t_epoch_navtest_sweep.py" "$@"
