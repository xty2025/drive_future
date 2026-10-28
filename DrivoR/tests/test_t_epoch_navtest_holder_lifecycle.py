"""CPU subprocess checks for the signal inheritance bug in local eval cleanup."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluation/t_epoch_navtest_sweep.py'
spec = importlib.util.spec_from_file_location('epoch_sweep_lifecycle', PATH)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('CPU subprocess did not become ready')


def test_stop_escalates_only_for_captured_ignored_term_process(tmp_path, monkeypatch):
    ready = tmp_path / 'ready'
    source = ('import signal,time; from pathlib import Path; '
              'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
              f'Path({str(ready)!r}).touch(); time.sleep(60)')
    child = subprocess.Popen([sys.executable, '-c', source])
    monkeypatch.setattr(sweep, 'gpu_apps', lambda: [])
    try:
        wait_for(ready.exists)
        result = sweep.stop_holders({'holders_before': [sweep.process(child.pid)]},
                                    grace_seconds=.05, release_seconds=.5)
        assert result['forced_pids'] == [child.pid]
        assert child.wait(timeout=2) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_changed_process_identity_is_never_signalled(monkeypatch):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    monkeypatch.setattr(sweep, 'gpu_apps', lambda: [])
    try:
        row = sweep.process(child.pid)
        row['start_ticks'] += 1
        with pytest.raises(RuntimeError, match='identity changed'):
            sweep.stop_holders({'holders_before': [row]})
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_restored_child_resets_parent_ignored_signals(tmp_path, monkeypatch):
    holder = tmp_path / 'cpu_holder.py'
    ready = tmp_path / 'child.ready'
    holder.write_text('import time\nfrom pathlib import Path\n'
                      f'Path({str(ready)!r}).touch()\ntime.sleep(60)\n')
    monkeypatch.setattr(sweep, 'HOLDER', str(holder))
    monkeypatch.setattr(sweep, 'gpu_apps', lambda: [{'gpu':0, 'host_pid':0, 'memory_mib':'0'}]
                        if ready.exists() and sweep.holders() else [])
    original = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    child_pid = None
    try:
        result = sweep.restore_holders({'directory':str(tmp_path), 'holders_before':[
            {'gpu':0, 'argv':[sys.executable, str(holder), '--gpu_id', '0', '--ratio', '0.4'],
             'gpu_environment':{}, 'stdout':str(tmp_path/'holder.log'), 'cwd':str(tmp_path), 'nice':0}]})
        child_pid = result['processes'][0]['pid']
        assert result['verified']
        status = Path(f'/proc/{child_pid}/status').read_text()
        ignored = int(next(line.split()[1] for line in status.splitlines() if line.startswith('SigIgn:')), 16)
        assert not ignored & (1 << (signal.SIGTERM - 1))
        assert not ignored & (1 << (signal.SIGINT - 1))
        os.kill(child_pid, signal.SIGTERM)
        wait_for(lambda: not sweep.process(child_pid) or sweep.process(child_pid)['state'] == 'Z')
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)
        if child_pid and sweep.process(child_pid) and sweep.process(child_pid)['state'] != 'Z':
            os.kill(child_pid, signal.SIGKILL)
