"""Local, resumable T-series checkpoint evaluation with GPU-holder restoration."""

import argparse
import csv
from datetime import datetime
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
SUMMARY = PROJECT / 'artifacts/t_epoch_val_summary_20261023/summary.json'
LOCK = Path('/tmp/drive_future_t_epoch_navtest.lock')
LATEST = Path('/tmp/drive_future_t_epoch_navtest.latest')
PYTHON = '/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python'
HOLDER = '/root/occupy_gpu.py'
GPU_ENV = ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','NVIDIA_VISIBLE_DEVICES',
           'LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTORCH_CUDA_ALLOC_CONF',
           'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
IDS = ('T01','T03','T05','T06','T07','T08','T09','T10')
AGENTS = {
    'T01':'drivoR_query_register_dinov3_vitl_single_gt',
    'T03':'drivoR_query_register_dinov3_vitl_scorer_single_gt',
    'T05':'drivoR_query_dinov3_vitl_single_gt_long',
    'T06':'drivoR_query_dinov3_vitl_single_gt_long_register',
    'T07':'drivoR_query_dinov3_vitl_human_diverse8_l1',
    'T08':'drivoR_query_dinov3_vitl_human_diverse8_l1_register',
    'T09':'drivoR_query_dinov3_vitl_human_diverse8_l1_register_scorer',
    'T10':'drivoR_query_dinov3_vitl_single_gt',
}


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def process(pid):
    base = Path('/proc') / str(pid)
    try:
        args = base.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        stat = base.joinpath('stat').read_text().rsplit(') ',1)[1].split()
        return {'pid':int(pid),'argv':args,'start_ticks':int(stat[19]),'state':stat[0]}
    except (OSError, ValueError):
        return None


def holders():
    result = []
    for base in Path('/proc').iterdir():
        if not base.name.isdecimal():
            continue
        row = process(base.name)
        if not row or len(row['argv']) < 2 or row['argv'][1] != HOLDER:
            continue
        args = row['argv']
        if '--gpu_id' not in args or '--ratio' not in args:
            continue
        row['gpu'] = int(args[args.index('--gpu_id')+1])
        row['cwd'] = os.readlink(base/'cwd')
        row['nice'] = os.getpriority(os.PRIO_PROCESS, row['pid'])
        env = dict(x.split('=',1) for x in (base/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
        row['gpu_environment'] = {key:env[key] for key in GPU_ENV if key in env}
        row['stdout'] = os.readlink(base/'fd/1')
        result.append(row)
    return sorted(result,key=lambda r:r['gpu'])


def same_process(row):
    live = process(row['pid'])
    return bool(live and live['state'] != 'Z' and live['start_ticks'] == row['start_ticks']
                and live['argv'] == row['argv'])


def gpu_apps():
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping = {uuid.strip():int(index) for index,uuid in csv.reader(devices.splitlines())}
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'],text=True)
    return [{'host_pid':int(pid),'gpu':mapping[uuid.strip()],'memory_mib':memory.strip()}
            for pid,uuid,memory in csv.reader(text.splitlines()) if pid.strip()]


def acquire_lock():
    fd = os.open(LOCK,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError('An epoch sweep or restoration is already active; use --status.')
    return fd


def prepare():
    summary = json.loads(SUMMARY.read_text())
    assert tuple(summary['scope']) == IDS
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = PROJECT / 'artifacts/t_navtest_epochs16_20' / stamp
    directory.mkdir(parents=True,exist_ok=False)
    current = holders()
    apps = gpu_apps()
    # NVML PIDs are host PIDs; /proc PIDs are container PIDs. Compare GPU slots,
    # not PID numbers, and refuse to stop unrelated workloads.
    for gpu in range(8):
        if sum(a['gpu']==gpu for a in apps) != sum(h['gpu']==gpu for h in current):
            raise RuntimeError(f'GPU {gpu} has a workload not identified as {HOLDER}')
    tasks = []
    for rid in IDS:
        source = summary['records'][rid]
        checkpoints = Path(source['navtest_checkpoint']).parent
        parent = Path(source['navtest_csv']).parents[1]
        for display_epoch in range(16,21):
            epoch = display_epoch-1
            step = int(source['raw_rows'][epoch]['global_step'])
            checkpoint = checkpoints / f'epoch={epoch}-step={step}.ckpt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            run_id = f'epochs16_20_local_{stamp}_e{display_epoch:02d}_ckpt{epoch:02d}'
            output = parent / run_id
            command = ['bash',str(ROOT/'scripts/evaluation/run_drivor_pdm_score_evaluation.sh'),str(checkpoint),
                f'agent={AGENTS[rid]}','agent.config.multi_gt.enabled=false',
                'agent.config.long_trajectory_additional_poses=0',
                'agent.config.future_register.init_checkpoint=null',
                'trainer.params.precision=32-true','dataloader.params.batch_size=1']
            tasks.append({'id':f'{rid}_e{display_epoch:02d}','experiment':rid,
                'display_epoch':display_epoch,'checkpoint_epoch':epoch,'global_step':step,
                'checkpoint':str(checkpoint),'agent':AGENTS[rid],'eval_experiment':parent.name,
                'run_id':run_id,'output':str(output),'log':str(output/'launch.log'),
                'command':command,'status':'pending'})
    required = [Path(PYTHON),Path(HOLDER),PROJECT/'data/navsim/navsim_logs/test',
                PROJECT/'data/navsim/sensor_blobs/test',PROJECT/'data/metric_cache_navtest/metadata',
                ROOT/'weights/dinov3/vitl16/model.safetensors']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    plan = {'created_at':now(),'directory':str(directory),'scope':list(IDS),
        'display_epochs':[16,17,18,19,20],'checkpoint_epochs':[15,16,17,18,19],
        'precision':'32-true','gpus':list(range(8)),'score_workers':32,
        'holders_before':current,'gpu_apps_before':apps,'tasks':tasks}
    write_json(directory/'plan.json',plan)
    write_json(directory/'state.json',{'phase':'prepared','updated_at':now(),'completed':0,'failed':0,'total':40})
    return directory,plan


def stop_holders(plan, *, grace_seconds=10, release_seconds=20):
    for row in plan['holders_before']:
        if not same_process(row):
            raise RuntimeError(f'Holder identity changed before stopping PID {row["pid"]}')
    for row in plan['holders_before']:
        os.kill(row['pid'],signal.SIGTERM)
    forced = []
    deadline = time.monotonic()+grace_seconds
    while time.monotonic()<deadline:
        if not any(same_process(h) for h in plan['holders_before']) and not gpu_apps():
            return {'forced_pids':forced}
        time.sleep(min(1, grace_seconds))
    # Old restorations inherited SIG_IGN from the worker's cleanup section.
    # Escalate only for the same recorded holder (PID, start time and argv),
    # never for a host PID reported by NVML or an unrelated GPU process.
    for row in plan['holders_before']:
        if same_process(row):
            try:
                os.kill(row['pid'],signal.SIGKILL)
                forced.append(row['pid'])
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+release_seconds
    while True:
        if not any(same_process(h) for h in plan['holders_before']) and not gpu_apps():
            return {'forced_pids':forced}
        if time.monotonic() >= deadline:
            break
        time.sleep(min(1, release_seconds))
    raise RuntimeError('GPU holders did not release their resources after TERM/KILL cleanup')


def restore_holders(plan):
    directory = Path(plan['directory'])
    existing = {row['gpu']:row for row in holders()}
    needed = [row for row in plan['holders_before'] if row['gpu'] not in existing]
    deadline = time.monotonic()+45
    while any(app['gpu'] in {r['gpu'] for r in needed} for app in gpu_apps()):
        if time.monotonic() >= deadline:
            raise RuntimeError('Restoration deferred: a GPU still has an active evaluation or another workload')
        time.sleep(1)
    restored = []
    for row in plan['holders_before']:
        if row['gpu'] in existing:
            restored.append({'gpu':row['gpu'],'pid':existing[row['gpu']]['pid'],'already_running':True})
            continue
        env = os.environ.copy()
        for key in GPU_ENV:
            env.pop(key,None)
        env.update(row['gpu_environment'])
        env['PYTHONUNBUFFERED']='1'
        log = Path(row['stdout'])
        if not str(log).startswith('/') or str(log).startswith(('/dev/','/proc/')):
            log = directory/f'holder_gpu{row["gpu"]}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('ab') as output:
            # The worker ignores INT/TERM during cleanup; reset them in the
            # child before exec so a later resume can stop these holders.
            child = subprocess.Popen(['/usr/bin/env','--default-signal=INT,TERM',
                '/usr/bin/nice','-n',str(row['nice']),*row['argv']],
                cwd=row['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True)
        restored.append({'gpu':row['gpu'],'pid':child.pid,'log':str(log),'already_running':False})
    write_json(directory/'holders_restored.json',{'started_at':now(),'processes':restored})
    deadline = time.monotonic()+180
    while time.monotonic()<deadline:
        live = {h['gpu'] for h in holders()}
        occupied = {a['gpu'] for a in gpu_apps()}
        expected = {r['gpu'] for r in plan['holders_before']}
        if expected <= live and expected <= occupied:
            result = {'verified_at':now(),'processes':restored,'gpu_apps':gpu_apps(),'verified':True}
            write_json(directory/'holders_restored.json',result)
            return result
        time.sleep(2)
    raise RuntimeError('Holder processes were launched, but GPU occupancy was not verified within 180 seconds')


def eval_environment(task, plan):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DRIVOR_','TORCHELASTIC_'))}
    for key in ('LOCAL_RANK','LOCAL_WORLD_SIZE','RANK','WORLD_SIZE','NODE_RANK','GROUP_RANK','ROLE_RANK','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    env.update(DRIVOR_PYTHON=PYTHON,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',NPROC_PER_NODE='8',
        DRIVOR_EVAL_WORKER='single_machine_thread_pool',DRIVOR_SCORE_WORKERS=str(plan['score_workers']),
        DRIVOR_CPU_THREADS='1',DRIVOR_EXPERIMENT=task['eval_experiment'],DRIVOR_RUN_ID=task['run_id'],
        DRIVOR_EVAL_PRECISION='32-true',NAVSIM_EXP_ROOT=str(ROOT/'exp'),
        OPENSCENE_DATA_ROOT=str(PROJECT/'data/navsim'),
        NAVSIM_TEST_METRIC_CACHE_PATH=str(PROJECT/'data/metric_cache_navtest'),
        DRIVOR_DINOV3_VITL_WEIGHTS=str(ROOT/'weights/dinov3/vitl16/model.safetensors'),
        SUBSCORE_PATH=str(Path(task['output'])/'predictions'),PYTHONDONTWRITEBYTECODE='1')
    return env


def verify_result(task):
    paths = sorted(Path(task['output']).glob('*.csv'),key=lambda p:p.stat().st_mtime_ns)
    if not paths:
        raise RuntimeError('Evaluation returned without a result CSV')
    with paths[-1].open() as handle:
        rows = list(csv.DictReader(handle))
    scenes = [r for r in rows if r.get('token')!='average']
    averages = [r for r in rows if r.get('token')=='average']
    if len(scenes)!=12146 or len({r['token'] for r in scenes})!=12146 or len(averages)!=1:
        raise RuntimeError('Evaluation does not contain exactly 12,146 unique navtest scenes and one average')
    if not all(r.get('valid','').lower()=='true' for r in scenes):
        raise RuntimeError('Evaluation contains failed scenarios')
    score = float(averages[0]['score'])
    if not math.isfinite(score):
        raise RuntimeError('Non-finite navtest PDMS')
    return {'csv':str(paths[-1]),'score':score,'pdms':score*100,'valid_scenes':len(scenes),'average':averages[0]}


def reports(plan):
    directory = Path(plan['directory'])
    baseline_path = directory/'epoch25_reference.json'
    baseline = json.loads(baseline_path.read_text())['experiments'] if baseline_path.exists() else {}
    handle = io.StringIO(newline='')
    writer = csv.writer(handle)
    writer.writerow(['experiment','display_epoch','checkpoint_epoch','status','navtest_pdms','result_csv','checkpoint'])
    for t in plan['tasks']:
        writer.writerow([t['experiment'],t['display_epoch'],t['checkpoint_epoch'],t['status'],
            t.get('result',{}).get('pdms',''),t.get('result',{}).get('csv',''),t['checkpoint']])
    for rid in IDS:
        if rid in baseline:
            r = baseline[rid]
            writer.writerow([rid,25,24,'reference',r['pdms'],r['csv'],r['checkpoint']])
    lines = ['# T 系列第 16～20 轮与第 25 轮：FP32 navtest', '',
        '轮次沿用逐轮验证表的 1-based 编号，对应 checkpoint 文件 `epoch=15～19`。每组使用 8 张 GPU；结果必须包含 12,146 个有效场景。', '',
        '第 25 轮（checkpoint `epoch=24`）引用各实验已有的 FP32 navtest 结果，仅作对照，不新增评测、不计入本轮 40 个任务。CSV 中这些行标记为 `reference`。', '',
        '| 实验 | 第16轮 | 第17轮 | 第18轮 | 第19轮 | 第20轮 | 第25轮（已有） |','|---|---:|---:|---:|---:|---:|---:|']
    for rid in IDS:
        cells = []
        for t in [t for t in plan['tasks'] if t['experiment']==rid]:
            if t['status']=='complete':
                cells.append(f'[{t["result"]["pdms"]:.4f}]({t["result"]["csv"]})')
            else:
                cells.append({'pending':'待测','running':'运行中','failed':'失败'}.get(t['status'],t['status']))
        cells.append(f'[{baseline[rid]["pdms"]:.4f}]({baseline[rid]["csv"]})' if rid in baseline else '—')
        lines.append('| '+rid+' | '+' | '.join(cells)+' |')
    lines += ['', '[任务与路径](plan.json) · [进度](state.json) · [CSV 汇总](results.csv) · [第25轮来源](epoch25_reference.json)', '']
    for name,content in [('results.csv',handle.getvalue()),('RESULTS.md','\n'.join(lines))]:
        with tempfile.NamedTemporaryFile(mode='w',dir=directory,prefix=name+'.',suffix='.tmp',delete=False) as temporary:
            temporary.write(content)
            temp_path=Path(temporary.name)
        temp_path.replace(directory/name)


class Interrupted(Exception):
    pass


def worker(directory, lock_fd):
    if lock_fd is None:
        lock_fd = acquire_lock()
    plan = json.loads((directory/'plan.json').read_text())
    state = {'phase':'starting','pid':os.getpid(),'updated_at':now(),
             'completed':sum(t['status']=='complete' for t in plan['tasks']),
             'failed':0,'total':len(plan['tasks'])}
    state['process_identity'] = process(os.getpid())
    child = None
    def interrupted(signum, frame):
        raise Interrupted(f'Received signal {signum}')
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,interrupted)
    restore_needed = False
    outcome = 'failed'
    try:
        write_json(directory/'state.json',state)
        restore_needed = True
        stopped = stop_holders(plan)
        write_json(directory/'gpus_released.json',{'at':now(),'gpu_apps':gpu_apps(),**stopped})
        for index,task in enumerate(plan['tasks']):
            if task['status']=='complete':
                continue
            output = Path(task['output']);output.mkdir(parents=True,exist_ok=True)
            task.update(status='running',started_at=now())
            state.update(phase='evaluating',task=task['id'],task_index=index+1,output=task['output'],log=task['log'],updated_at=now())
            write_json(directory/'state.json',state);write_json(directory/'plan.json',plan);reports(plan)
            print(json.dumps({'starting':task['id'],'checkpoint':task['checkpoint'],'log':task['log']}),flush=True)
            with Path(task['log']).open('ab') as log:
                child = subprocess.Popen(task['command'],cwd=PROJECT,env=eval_environment(task,plan),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=child.pid
                write_json(directory/'state.json',state)
                while True:
                    try:
                        code=child.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        state['updated_at']=now();write_json(directory/'state.json',state)
            child = None
            task.update(returncode=code,finished_at=now())
            try:
                if code:
                    raise RuntimeError(f'Evaluation process returned {code}; see {task["log"]}')
                task['result']=verify_result(task)
                task.pop('error',None)
                task['status']='complete';state['completed']+=1
            except Exception as error:
                task.update(status='failed',error=str(error));state['failed']+=1
            write_json(directory/'plan.json',plan);reports(plan)
            print(json.dumps({'finished':task['id'],'status':task['status'],'pdms':task.get('result',{}).get('pdms')}),flush=True)
        outcome = 'complete' if not state['failed'] else 'complete_with_failures'
    except Interrupted as error:
        outcome='interrupted';state['error']=str(error)
    except BaseException as error:
        state['error']=repr(error)
        print(state['error'],flush=True)
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(phase='restoring_holders',outcome=outcome,updated_at=now())
        write_json(directory/'state.json',state)
        try:
            if restore_needed:
                state['restoration']=restore_holders(plan)
            state['phase']=outcome
        except BaseException as error:
            state.update(phase='restore_failed',restoration_error=repr(error))
        state['finished_at']=now();state['updated_at']=now()
        write_json(directory/'state.json',state);reports(plan)
        os.close(lock_fd)
        print(json.dumps(state,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--status',nargs='?',const='latest')
    parser.add_argument('--stop',type=Path)
    parser.add_argument('--restore',type=Path)
    parser.add_argument('--resume',type=Path,help='Resume unfinished entries in a previous sweep')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        directory=Path(LATEST.read_text().strip()) if args.status=='latest' else Path(args.status)
        print((directory/'state.json').read_text());return
    if args.stop:
        state=json.loads((args.stop/'state.json').read_text())
        if not same_process(state['process_identity']):
            raise RuntimeError('Recorded worker is no longer the same running process')
        os.kill(state['pid'],signal.SIGTERM)
        print('Stop requested; the worker will stop its evaluation and restore holders.');return
    if args.restore:
        fd=acquire_lock()
        try:print(json.dumps(restore_holders(json.loads((args.restore/'plan.json').read_text())),ensure_ascii=False))
        finally:os.close(fd)
        return
    if args.worker:
        worker(args.worker,args.lock_fd);return
    fd=acquire_lock()
    try:
        if args.resume:
            directory=args.resume.resolve()
            plan=json.loads((directory/'plan.json').read_text())
            current=holders();apps=gpu_apps()
            for gpu in range(8):
                if sum(a['gpu']==gpu for a in apps)!=sum(h['gpu']==gpu for h in current):
                    raise RuntimeError(f'GPU {gpu} has an unrecognized workload')
            for task in plan['tasks']:
                if not Path(task['checkpoint']).is_file():
                    raise FileNotFoundError(task['checkpoint'])
                if task['status']!='complete':task['status']='pending'
            plan.update(holders_before=current,gpu_apps_before=apps,resumed_at=now())
            write_json(directory/'plan.json',plan)
        else:
            directory,plan=prepare()
        reports(plan)
        if args.dry_run:
            print(json.dumps({'dry_run':True,'directory':str(directory),'tasks':len(plan['tasks']),
                'occupancy_gpus':[h['gpu'] for h in plan['holders_before']],'gpu_processes_stopped':False},ensure_ascii=False));return
        with (directory/'runner.log').open('ab') as log:
            process_handle=subprocess.Popen(['/usr/bin/python3',str(Path(__file__).resolve()),'--worker',str(directory),'--lock-fd',str(fd)],
                cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(fd,))
        LATEST.write_text(str(directory)+'\n')
        print(json.dumps({'started':True,'pid':process_handle.pid,'directory':str(directory),
            'runner_log':str(directory/'runner.log'),'tasks':40,
            'display_epochs':plan['display_epochs'],'checkpoint_epochs':plan['checkpoint_epochs'],
            'restore_occupancy_gpus':[h['gpu'] for h in plan['holders_before']]},ensure_ascii=False))
    finally:
        os.close(fd)


if __name__=='__main__':
    main()
