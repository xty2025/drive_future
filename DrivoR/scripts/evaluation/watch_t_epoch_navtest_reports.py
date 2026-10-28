"""Keep epoch-25 references in reports while a legacy in-memory runner finishes."""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    spec = importlib.util.spec_from_file_location('epoch_sweep_report',Path(__file__).with_name('t_epoch_navtest_sweep.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (directory/'.epoch25_reporter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stamp = None
        refreshed = 0
        while True:
            plan = json.loads((directory/'plan.json').read_text())
            state = json.loads((directory/'state.json').read_text())
            latest_stamp = ((directory/'plan.json').stat().st_mtime_ns,
                            (directory/'epoch25_reference.json').stat().st_mtime_ns)
            markdown = (directory/'RESULTS.md').read_text() if (directory/'RESULTS.md').exists() else ''
            table = (directory/'results.csv').read_text() if (directory/'results.csv').exists() else ''
            alive = module.same_process(state.get('process_identity',{})) if state.get('process_identity') else False
            if stamp != latest_stamp or '第25轮（已有）' not in markdown or table.count(',25,24,reference,') != 8 or not alive:
                module.reports(plan)
                stamp = latest_stamp
                refreshed += 1
                module.write_json(directory/'epoch25_reporter.json',{
                    'pid':os.getpid(),'updated_at':module.now(),'phase':'watching' if alive and not args.once else 'finished',
                    'refreshes':refreshed,'epoch25_references':8,'evaluation_runner_untouched':True})
                print(json.dumps({'updated':module.now(),'refreshes':refreshed,'evaluation_runner_alive':alive}),flush=True)
            if args.once or not alive:
                return
            time.sleep(2)


if __name__=='__main__':
    main()
