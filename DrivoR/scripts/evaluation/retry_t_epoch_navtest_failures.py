"""Retry failed entries once after the active sweep has restored its GPU holders."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    directory = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location('epoch_retry',Path(__file__).with_name('t_epoch_navtest_sweep.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (directory/'.auto_retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record_path = directory/'auto_retry.json'
        if record_path.exists() and json.loads(record_path.read_text()).get('attempted'):
            return
        record = {'pid':os.getpid(),'phase':'waiting_for_current_sweep','attempted':False,
                  'updated_at':module.now(),'max_additional_attempts':1}
        module.write_json(record_path,record)
        while True:
            state = json.loads((directory/'state.json').read_text())
            if not module.same_process(state['process_identity']):
                break
            time.sleep(10)
        plan = json.loads((directory/'plan.json').read_text())
        failed = [task['id'] for task in plan['tasks'] if task['status']=='failed']
        if state['phase']!='complete_with_failures' or not failed:
            record.update(phase='skipped',reason=state['phase'],updated_at=module.now())
            module.write_json(record_path,record)
            return
        record.update(phase='launching_retry',attempted=True,failed_tasks=failed,updated_at=module.now())
        module.write_json(record_path,record)
        result = subprocess.run(['bash',str(Path(__file__).with_name('run_drivor_t_epochs16_20_navtest.sh')),
                                 '--resume',str(directory)],capture_output=True,text=True,timeout=45)
        record.update(phase='retry_launched' if result.returncode==0 else 'retry_launch_failed',
                      returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,updated_at=module.now())
        module.write_json(record_path,record)


if __name__=='__main__':
    main()
