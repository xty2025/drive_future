"""Sample rank-local host/CUDA intervals without synchronization or collectives.

Host intervals measure Python/CPU wall time, not isolated GPU kernel time. Oracle
intervals nest inside loss. Backward includes autograd, DDP work and any waits;
it is not a measurement of communication alone. CUDA events are queried later,
never synchronized, and include current-stream dependencies and idle intervals.
"""
from collections import deque
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import socket
from time import perf_counter
import uuid

import torch


LOG = logging.getLogger(__name__)
SCHEMA = "drivor_rank_perf_v1"
PHASES = ("transfer", "forward", "loss", "oracle", "oracle_cpu", "backward", "optimizer")
CUDA_PHASES = {"transfer", "forward", "loss", "backward", "optimizer", "batch"}


class RankPerformanceRecorder:
    def __init__(self, output_dir, *, rank, world_size, device, sample_every=25, flush_every=4,
                 cuda_events=False, optimizer_step_start=0, clock=perf_counter, event_factory=None):
        self.sample_every = int(sample_every)
        self.flush_every = int(flush_every)
        if self.sample_every < 1 or self.flush_every < 1:
            raise ValueError("performance sample_every and flush_every must be positive")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.clock = clock
        self.session_id = uuid.uuid4().hex
        self.cuda_enabled = bool(cuda_events) and torch.device(device).type == "cuda"
        self.event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self.current = None
        self.wait_started = None
        self.pending = deque()
        self.ready = []
        self.closed = False
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"rank_{self.rank:05d}.jsonl"
        self.stream = self.path.open("a", buffering=64 * 1024)
        environment_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "NAVSIM_ORACLE_WORKERS", "NAVSIM_CAMERA_THREADS")
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        metadata = {"type": "metadata", "schema": SCHEMA, "session_id": self.session_id,
                    "rank": self.rank, "world_size": self.world_size,
                    "optimizer_step_start": int(optimizer_step_start),
                    "distributed_attempt": {key: os.environ.get(key) for key in
                                            ("PAI_JOB_ID", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT")},
                    "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
                    "hostname": socket.gethostname(), "pid": os.getpid(), "device": str(device),
                    "cpu_affinity_count": len(affinity) if affinity is not None else None,
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "environment": {key: os.environ.get(key) for key in environment_keys},
                    "sample_every": self.sample_every, "flush_every": self.flush_every,
                    "cuda_events_requested": bool(cuda_events), "cuda_events_enabled": self.cuda_enabled,
                    "sampling": "optimizer_step_before % sample_every == 0; all microbatches in that accumulation window",
                    "semantics": {
                        "data_wait_s": "previous batch-end to before-transfer: DataLoader/host-loop gap, excluding validation/epoch pauses",
                        "transfer_s": "before/after batch-transfer hooks; host dispatch/blocking, not completed asynchronous H2D time",
                        "forward_s": "agent.forward host wall time",
                        "loss_s": "agent.compute_loss host wall time, including Oracle",
                        "oracle_s": "compute_score host wall time, including tensor transfers/synchronization; subset of loss_s",
                        "oracle_cpu_s": "worker_map host wall time including CPU Oracle scheduling; subset of oracle_s and loss_s",
                        "backward_s": "backward hooks: autograd + DDP work/waits; not communication alone",
                        "optimizer_s": "optimizer_step after its training closure: includes precision/hooks/clipping; null on accumulation-only batches",
                        "batch_s": "data_wait plus before-transfer to batch-end; host wall time, no CUDA synchronization",
                        "cuda_ms": "current-stream event intervals including dependencies/idle; not isolated kernel or NCCL time",
                    }}
        self.stream.write(json.dumps(metadata, allow_nan=False) + "\n")
        self.stream.flush()

    @property
    def active(self):
        return self.current is not None

    def reset_wait(self, *, paused=False):
        self.wait_started = None if paused else self.clock()

    def _event(self):
        event = self.event_factory()
        event.record()
        return event

    def begin_transfer(self, *, epoch, optimizer_step, accumulation):
        if int(optimizer_step) % self.sample_every:
            self.current = None
            return
        now = self.clock()
        wait = max(0.0, now - self.wait_started) if self.wait_started is not None else None
        self.current = {
            "row": {"type": "batch", "schema": SCHEMA, "session_id": self.session_id,
                    "rank": self.rank, "world_size": self.world_size, "epoch": int(epoch), "batch_idx": None,
                    "optimizer_step_before": int(optimizer_step), "optimizer_step_after": None,
                    "accumulation": int(accumulation), "optimizer_step_called": False,
                    "data_wait_s": wait, **{f"{phase}_s": None for phase in PHASES},
                    "phase_calls": {phase: 0 for phase in PHASES}},
            "batch_started": self.wait_started if self.wait_started is not None else now,
            "starts": {}, "events": {}, "open_events": {},
        }
        if self.cuda_enabled:
            self.current["open_events"]["batch"] = self._event()
        self.start_phase("transfer", now=now)

    def batch_start(self, batch_idx):
        if self.active:
            self.current["row"]["batch_idx"] = int(batch_idx)

    def start_phase(self, phase, *, now=None):
        if not self.active:
            return
        self.current["starts"][phase] = self.clock() if now is None else now
        if self.cuda_enabled and phase in CUDA_PHASES:
            self.current["open_events"][phase] = self._event()

    def end_phase(self, phase):
        if not self.active or phase not in self.current["starts"]:
            return
        duration = max(0.0, self.clock() - self.current["starts"].pop(phase))
        row = self.current["row"]
        row[f"{phase}_s"] = (row[f"{phase}_s"] or 0.0) + duration
        row["phase_calls"][phase] += 1
        if self.cuda_enabled and phase in self.current["open_events"]:
            start = self.current["open_events"].pop(phase)
            self.current["events"].setdefault(phase, []).append((start, self._event()))

    @contextmanager
    def phase(self, name):
        self.start_phase(name)
        try:
            yield
        finally:
            self.end_phase(name)

    @contextmanager
    def oracle_phases(self, agent):
        """Temporary instance wrappers are removed even when scoring raises."""
        restored = []
        try:
            for attribute, phase in (("compute_score", "oracle"), ("worker_map", "oracle_cpu")):
                original = getattr(agent, attribute, None)
                if not callable(original):
                    continue
                existed = attribute in agent.__dict__
                prior_value = agent.__dict__.get(attribute)

                def timed(*args, _original=original, _phase=phase, **kwargs):
                    with self.phase(_phase):
                        return _original(*args, **kwargs)

                setattr(agent, attribute, timed)
                restored.append((attribute, existed, prior_value))
            yield
        finally:
            for attribute, existed, prior_value in reversed(restored):
                if existed:
                    setattr(agent, attribute, prior_value)
                else:
                    delattr(agent, attribute)

    def optimizer_started(self):
        if self.active:
            self.current["row"]["optimizer_step_called"] = True
            self.start_phase("optimizer")

    def finish_batch(self, *, batch_idx, optimizer_step):
        now = self.clock()
        if self.active:
            item = self.current
            row = item["row"]
            row.update(batch_idx=int(batch_idx), optimizer_step_after=int(optimizer_step),
                       batch_s=max(0.0, now - item["batch_started"]))
            if self.cuda_enabled:
                item["events"]["batch"] = [(item["open_events"].pop("batch"), self._event())]
            self.pending.append((row, item["events"]))
            self.current = None
        self._drain()
        # Exclude this recorder's own JSON/flush work from the next host gap.
        self.wait_started = self.clock()

    def _drain(self, *, closing=False):
        remaining = deque()
        while self.pending:
            row, events = self.pending.popleft()
            available = not self.cuda_enabled or all(end.query() for pairs in events.values() for _, end in pairs)
            if not available and not closing:
                remaining.append((row, events))
                continue
            row["cuda_status"] = "disabled" if not self.cuda_enabled else "ready" if available else "not_ready_at_close"
            row["cuda_ms"] = ({name: sum(start.elapsed_time(end) for start, end in pairs)
                               for name, pairs in events.items()} if self.cuda_enabled and available else {})
            self.ready.append(row)
        self.pending = remaining
        if len(self.ready) >= self.flush_every or closing:
            self._flush()

    def _flush(self):
        if not self.ready:
            return
        rows, self.ready = self.ready, []
        self.stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        self.stream.flush()
        if self.rank == 0:
            means = {}
            for key in ("data_wait_s", "transfer_s", "forward_s", "loss_s", "oracle_cpu_s", "backward_s", "optimizer_s", "batch_s"):
                values = [row[key] for row in rows if row.get(key) is not None and math.isfinite(row[key])]
                if values:
                    means[key] = sum(values) / len(values)
            LOG.info("[perf] rank=0 scope=local sampled_batches=%s optimizer_step=%s %s",
                     len(rows), rows[-1]["optimizer_step_after"],
                     " ".join(f"{key}={value:.4f}" for key, value in means.items()))

    def close(self):
        if not self.closed:
            self._drain(closing=True)
            self.stream.close()
            self.closed = True
