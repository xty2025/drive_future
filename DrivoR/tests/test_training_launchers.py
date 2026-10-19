"""Exercise the real shell launchers with DLC's node-level environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(params=["run_drivor_training.sh", "run_drivor_action_dit_training.sh",
                        "run_drivor_action_dit_multiview_training.sh",
                        "run_drivor_multigt_training.sh", "run_drivor_action_dit_multigt_training.sh",
                        "run_drivor_action_dit_multiview_multigt_training.sh",
                        "run_drivor_multigt_hungarian_training.sh",
                        "run_drivor_action_dit_multigt_hungarian_training.sh",
                        "run_drivor_action_dit_multiview_multigt_hungarian_training.sh",
                        "run_drivor_action_dit_multigt_weighted_training.sh",
                        "run_drivor_action_dit_multiview_multigt_weighted_training.sh",
                        "run_drivor_action_dit_dinov3_vits_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_training.sh",
                        "run_drivor_action_dit_vjepa2_vitl_training.sh",
                        "run_drivor_action_dit_vjepa2_1_vitl_training.sh",
                        "run_drivor_bank_training.sh", "run_drivor_vjepa2_vitl_training.sh",
                        "run_drivor_vjepa2_1_vitl_training.sh", "run_drivor_vjepa2_world_training.sh",
                        "run_drivor_vjepa2_world_multigt_training.sh",
                        "run_drivor_pseudo_expert_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_bank_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_scorer_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_multigt_poor_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_single_gt_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_single_gt_long_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_register_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_register_scorer_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_register_scorer_single_gt_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_register_stage1_training.sh",
                        "run_drivor_action_dit_dinov3_vitl_register_stage2_training.sh"])
def launcher(request, tmp_path):
    # Record the final Python argv without starting Conda, GPUs, or rendezvous.
    conda_root = tmp_path / "conda"
    profile = conda_root / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text('conda() { export CONDA_PREFIX="$2"; }\n')
    python = conda_root / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['DRIVOR_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    record = tmp_path / "argv.json"
    environment = os.environ.copy()
    for key in ("NNODES", "NPROC_PER_NODE", "GPU_NUM", "NODE_RANK", "RANK", "WORLD_SIZE",
                "LOCAL_RANK", "GROUP_RANK", "MASTER_ADDR", "MASTER_PORT", "SLURM_NODEID",
                "SLURM_JOB_ID", "PAI_JOB_ID", "CUDA_VISIBLE_DEVICES", "DRIVOR_RUN_ID",
                "DRIVOR_EXPERIMENT"):
        environment.pop(key, None)
    environment.update(DRIVOR_CONDA_ROOT=str(conda_root),
                       DRIVOR_CONDA_ENV=str(conda_root / "env"), DRIVOR_CAPTURE=str(record))
    script = Path(__file__).resolve().parents[1] / "scripts/training" / request.param

    def run(env=None, args=()):
        result = subprocess.run(["bash", str(script), *args], env={**environment, **(env or {})},
                                capture_output=True, text=True, timeout=10, check=False)
        if result.returncode:
            return result, [], {}
        arguments = json.loads(record.read_text())
        entry = next(i for i, arg in enumerate(arguments) if arg.endswith("/run_training.py"))
        config = dict(arg.lstrip("+").split("=", 1) for arg in arguments[entry + 1:] if "=" in arg)
        return result, arguments[:entry], config

    return run


@pytest.mark.parametrize("node_rank", [0, 7])
def test_dlc_environment_launches_64_ranks_without_manual_node_arguments(launcher, node_rank):
    result, command, config = launcher(dict(WORLD_SIZE="8", RANK=str(node_rank), GPU_NUM="8",
                                            MASTER_ADDR="dlc-worker-0", MASTER_PORT="23456",
                                            PAI_JOB_ID="dlc-test-job"))
    assert result.returncode == 0, result.stderr
    assert command[:3] == ["-u", "-m", "torch.distributed.run"]
    assert "--nnodes=8" in command  # DLC WORLD_SIZE is nodes, never divide it by GPU_NUM.
    assert "--nproc_per_node=8" in command
    assert f"--node_rank={node_rank}" in command
    assert "--master_addr=dlc-worker-0" in command and "--master_port=23456" in command
    assert "--rdzv_id=dlc-test-job" in command
    assert config["experiment_uid"] == "dlc-test-job"
    assert config["trainer.params.devices"] == "8" and config["trainer.params.num_nodes"] == "8"
    assert config["agent.num_gpus"] == "64"
    assert int(config["dataloader.params.batch_size"]) * int(config["agent.num_gpus"]) * int(config["trainer.params.accumulate_grad_batches"]) == 64


def test_configuration_query_does_not_need_a_dlc_job(launcher):
    result, command, config = launcher(args=["--cfg", "job", "--package", "agent", "--resolve"])
    assert result.returncode == 0, result.stderr
    assert command == ["-u"]
    assert config["agent.num_gpus"] == "64"


def test_explicit_topology_and_hydra_overrides_take_precedence(launcher):
    result, command, config = launcher(
        dict(WORLD_SIZE="8", RANK="7", GPU_NUM="8", NNODES="1", NODE_RANK="0", NPROC_PER_NODE="2"),
        ["dataloader.params.batch_size=32", "trainer.params.accumulate_grad_batches=1"],
    )
    assert result.returncode == 0, result.stderr
    assert "--nnodes=1" in command and "--nproc_per_node=2" in command and "--node_rank=0" in command
    assert "--master_addr=127.0.0.1" in command
    assert config["trainer.params.num_nodes"] == "1" and config["trainer.params.devices"] == "2"
    assert config["agent.num_gpus"] == "2" and config["dataloader.params.batch_size"] == "32"


def test_missing_dlc_rendezvous_fails_before_starting_workers(launcher):
    result, _, _ = launcher()
    assert result.returncode != 0
    assert "MASTER_ADDR" in result.stderr and "DLC" in result.stderr
