"""Serial Kinova pipeline; GPU work obeys the recorded exclusive/shared policy."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import shutil
import uuid

from resource_guard import inspect_resources, load_policy, runtime_resource_snapshot, IdlePagecacheReclaimer
from phase_memory import MemorySampler, mark_phase

ROOT = Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', project_path('kinova', '')))
PYTHON = project_path('deploy', '.venv/bin/python')
SCRIPTS = ROOT/'scripts'
BASE = project_path('base_params', '')
SOURCE_ROOT = Path(project_path('archive', 'run_20261102_215627_resume_commit32'))


def atomic(path, data):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n'); temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def manifest():
    paths = [*sorted(SCRIPTS.glob('*.py')), ROOT/'env.sh', ROOT/'configs/experiment.json', ROOT/'configs/validation_plan.json',
             ROOT/'configs/resource_policy.json']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def rebuild_report():
    builder = SCRIPTS/'build_report.py'
    if builder.is_file():
        result = subprocess.run([PYTHON, str(builder)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        if result.returncode:
            print('Report builder failed:', result.stderr[-1500:], flush=True)


def resource_deferral_gate(returncode, log_path):
    """Accept only the explicit pre-JAX CLI resource rejection contract."""
    if returncode != 75:
        return None
    prefix = 'KINOVA_RESOURCE_DEFERRED '
    lines = [line[len(prefix):] for line in Path(log_path).read_text().splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        return None
    try:
        gate = json.loads(lines[0])
    except ValueError:
        return None
    return gate if isinstance(gate, dict) and gate.get('eligible') is False else None


def validate_same_graph_e0(root=None):
    """Require a newly evaluated zero-update LoRA baseline before training."""
    root = Path(root or ROOT)
    summary = json.loads((root/'evaluations/E0_wrist_frozen/summary.json').read_text())
    plan = json.loads((root/'configs/validation_plan.json').read_text())
    baseline = summary.get('config_metadata', {})
    hashes = baseline.get('parameter_hashes', {})
    base_hashes, adapter_hashes = hashes.get('base', {}), hashes.get('adapter', {})
    checks = {
        'same_graph_baseline_kind': baseline.get('baseline_kind') == 'E0_same_graph_zero_update',
        'same_graph_architecture': baseline.get('graph_kind') == 'zero_lora_same_as_training',
        'seed42': baseline.get('seed') == 42,
        'zero_update_model': baseline.get('optimizer_step') == 0 and baseline.get('optimizer_updates_executed') == 0,
        'all_lora_b_zero': baseline.get('all_lora_b_zero') is True,
        'parameter_hashes_recorded': baseline.get('parameter_hashes_recorded') is True,
        'base51_adapter20_hashes': len(base_hashes) == 51 and len(adapter_hashes) == 20 and not set(base_hashes)&set(adapter_hashes),
        'all71_parameter_leaf_counts': baseline.get('parameter_leaf_counts') == {'base': 51, 'adapter': 20, 'all': 71},
        'native_initialized_adapters': baseline.get('missing_adapters_filled_by_model_load') is False,
        'original_base': baseline.get('base_params') == BASE and summary.get('checkpoint') == BASE,
        'evaluation_complete': summary.get('status') == 'complete' and summary.get('evaluation_valid') is True,
        'all60_blocks': summary.get('expected_block_seed_count') == summary.get('records_completed') == 60,
        'all20_validation_episodes': summary.get('episode_count') == 20,
        'zero_errors_or_missing': summary.get('errors') == summary.get('missing_records') == 0,
        'zero_nonfinite_outputs': summary.get('nonfinite_output_count') == summary.get('flow_matching_nonfinite_term_count') == 0,
        'no_test_split': summary.get('test_split_used') is False,
        'validation_loader': 'build_dataset(validation)' in summary.get('config_metadata', {}).get('input_pipeline', ''),
        'step0_E0_label': summary.get('optimizer_step') == 0 and summary.get('label') == 'E0_wrist_frozen',
        'fixed_plan': summary.get('plan_sha256') == plan['plan_sha256'],
        'train_only_normalization': summary.get('norm_stats_sha256') == plan['normalization']['sha256'],
        'actual_validation_inputs_recorded': bool(summary.get('validation_input_and_noise_sha256')),
    }
    report = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'baseline_kind': baseline.get('baseline_kind'),
              'summary': str(root/'evaluations/E0_wrist_frozen/summary.json'),
              'scope': 'Explicitly reused deterministic same-graph step0 validation; lineage/source hashes must pass CPU preflight. Legacy no-LoRA E0 is not a substitute'}
    atomic(root/'reports/same_graph_E0_validation.json', report)
    if report['status'] != 'passed':
        raise RuntimeError('Same-graph E0 evidence failed: '+', '.join(name for name, passed in checks.items() if not passed))
    return report


def archive_resource_deferral(log_path):
    """Preserve the rejected prelaunch log, freeing the normal attempt name."""
    source = Path(log_path)
    target = source.with_name(f'{source.stem}.resource_deferred_{uuid.uuid4().hex}.log')
    # Exclusive destination protects all prior evidence.  Remove the source
    # only after the archive has been completely written.
    with target.open('xb') as output, source.open('rb') as original:
        shutil.copyfileobj(original, output)
    source.unlink()
    return target


def run(args, monitor):
    cpu = json.loads((ROOT/'reports/cpu_preflight.json').read_text())
    if cpu['status'] != 'passed':
        raise RuntimeError('CPU preflight has not passed')
    recorded = json.loads((ROOT/'manifests/workflow_code.json').read_text())
    if recorded != manifest():
        raise RuntimeError('Prepared code/config manifest changed; rerun preparation and review before starting GPU work')
    status_path = ROOT/'workflow.json'
    old = json.loads(status_path.read_text()) if status_path.exists() else {}
    if old.get('status') == 'complete':
        print('Workflow already complete; nothing started'); return
    state = {'status': 'waiting_resources', 'started_at': now(), 'updated_at': now(), 'pid': os.getpid(),
             'current_job': None, 'child_pid': None, 'completed_jobs': old.get('completed_jobs', []),
             'smoke_optimizer_steps': 32, 'pilot_optimizer_steps': 500, 'robot_commands_sent': 0,
             'smoke_step_source': 'Previously committed source checkpoint32; not updates executed by this recovery run',
             'smoke_updates_executed_this_run': 0, 'smoke_recovery_status': 'verified_in_source_reused',
             'pilot_resume_checkpoint_step': 500, 'pilot_updates_executed_this_run': 0,
             'test_model_evaluation': False, 'resource_deferrals': old.get('resource_deferrals', []),
             'idle_reclaims': old.get('idle_reclaims', []),
             'retry_policy': 'Only explicit pre-JAX resource rejections requeue; no model/training retries or config changes; any GPU-stage error stops this branch'}
    atomic(status_path, state)
    active = None

    def update(**values):
        state.update(values); state['updated_at'] = now(); atomic(status_path, state)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Kinova workflow received signal {signum}; only its own child is stopped')

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    def wait_idle(*, queued_since=None, reclaimer=None):
        begin = time.monotonic() if queued_since is None else queued_since
        stable = 0
        policy = load_policy()
        required = max(2, int(policy.get('resource_stable_checks', 2)))
        eligible_poll = float(policy.get('resource_eligible_poll_seconds', 10))
        blocked_poll = float(policy.get('resource_blocked_poll_seconds', 30))
        if min(eligible_poll, blocked_poll) <= 0:
            raise ValueError('Resource polling intervals must be positive')
        while True:
            monitor.assert_healthy()
            gate = inspect_resources()
            update(status='waiting_resources', gpu_gate=gate, child_pid=None, current_job=None)
            if not gate['eligible'] and reclaimer is not None:
                outcome = reclaimer.maybe_reclaim(monitor, gate=gate, own_child=active)
                if outcome['attempted']:
                    state['idle_reclaims'].append(outcome)
                    update(last_idle_reclaim=outcome)
            rebuild_report()
            if gate['eligible']:
                stable += 1
                if stable >= required:
                    return
            else:
                stable = 0
            if not args.wait:
                raise RuntimeError('GPU gate blocked; pass --wait to keep this stdlib-only process waiting without GPU/model imports')
            if time.monotonic()-begin > args.max_wait_hours*3600:
                raise TimeoutError('Resource wait expired; no competing GPU job was stopped')
            time.sleep(eligible_poll if gate['eligible'] else blocked_poll)

    def execute(name, command, *, timeout=86400, runtime_job=None):
        nonlocal active
        if any(row['name'] == name for row in state['completed_jobs']):
            return
        queued_since = time.monotonic()
        reclaimer = IdlePagecacheReclaimer(name+'_'+uuid.uuid4().hex)
        while True:
            if time.monotonic()-queued_since > args.max_wait_hours*3600:
                raise TimeoutError('Resource wait expired across prelaunch deferrals; no GPU-stage error was retried')
            wait_idle(queued_since=queued_since, reclaimer=reclaimer)
            if recorded != manifest():
                raise RuntimeError('Queued code/config changed before child launch; no GPU work started')
            monitor.assert_healthy()
            path = ROOT/'logs'/f'{name}.log'
            if path.exists():
                raise FileExistsError(f'Prior attempt exists, preserving it: {path}; inspect and explicitly resume/revise this branch')
            started = time.monotonic()
            update(status='running', current_job=runtime_job or name, current_segment=name if runtime_job else None,
                   job_started_at=now(), log=str(path))
            rebuild_report()
            with path.open('x') as stream:
                active = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                phase_sampler = MemorySampler(ROOT, active.pid, label=name)
                try:
                    try:
                        phase_sampler.start()
                    except Exception as error:
                        # Observability must not change resource eligibility or
                        # turn a fast pre-JAX deferral into a model failure.
                        mark_phase(ROOT, 'sampler.start_error', pid=active.pid, job=name,
                                   error=f'{type(error).__name__}: {error}')
                    update(child_pid=active.pid)
                    while active.poll() is None:
                        resources = runtime_resource_snapshot(monitor)
                        update(resource_monitor=resources)
                        if not resources['eligible']:
                            update(resource_abort={'job': name, **resources,
                                                   'action': 'stop own GPU child only; no existing jobs are touched'})
                            raise RuntimeError('ResourceHeadroomAbort: reserved GPU/host/Windows memory headroom exhausted')
                        if time.monotonic()-started > timeout:
                            raise TimeoutError(f'{name} exceeded {timeout}s')
                        for stage in ['smoke', 'pilot']:
                            progress = ROOT/'reports'/f'{stage}_training_status.json'
                            if progress.exists():
                                value = json.loads(progress.read_text())
                                state[stage+'_optimizer_steps'] = value.get('optimizer_step', state[stage+'_optimizer_steps'])
                        state['pilot_updates_executed_this_run'] = max(0, state['pilot_optimizer_steps']-500)
                        update()
                        time.sleep(2)
                    code = active.returncode
                    active = None
                finally:
                    phase_sampler.close()
            deferred_gate = resource_deferral_gate(code, path)
            if deferred_gate is not None:
                archive = archive_resource_deferral(path)
                state['resource_deferrals'].append({'job': name, 'at': now(), 'exit_code': code,
                    'marker': 'KINOVA_RESOURCE_DEFERRED', 'gate': deferred_gate, 'log': str(archive),
                    'scope': 'Child rejected before JAX initialization; no model/training retry'})
                update(status='waiting_resources', current_job=None, child_pid=None, log=str(archive))
                rebuild_report()
                continue
            if code != 0:
                raise RuntimeError(f'{name} failed with code {code}; retained log: {path}')
            state['completed_jobs'].append({'name': name, 'finished_at': now(), 'seconds': time.monotonic()-started,
                                            'log': str(path), 'exit_code': 0})
            update(current_job=None, current_segment=None, child_pid=None)
            rebuild_report()
            return

    plan = str(ROOT/'configs/validation_plan.json')
    try:
        # CPU preparation verifies every reused E0 file against its explicit
        # lineage. It is source evidence, not a new execution in this run.
        validate_same_graph_e0()
        if not any(job.get('name') == 'E0_wrist_frozen'
                   and (job.get('reused_from_run') == str(SOURCE_ROOT) or job.get('reused_from_root') == str(SOURCE_ROOT))
                   for job in state['completed_jobs']):
            raise RuntimeError('Verified source E0 must be explicitly registered as reused, not silently skipped')
        smoke = json.loads((ROOT/'reports/smoke_training_evidence.json').read_text())
        if smoke.get('optimizer_step') != 32 or not smoke.get('frozen_parameters_unchanged') or not smoke.get('changed_lora_tensors'):
            raise RuntimeError('Recovered committed32 checkpoint failed source update/freeze checks')
        for filename in ['full_state_restore.json', 'checkpoint_prediction_restore.json', 'base_zero_lora_equivalence.json']:
            if json.loads((ROOT/'reports'/filename).read_text()).get('status') != 'passed':
                raise RuntimeError(f'Required smoke evidence failed: {filename}')
        complete = json.loads((ROOT/'reports/smoke_complete.json').read_text())
        smoke = json.loads((ROOT/'reports/smoke_training_evidence.json').read_text())
        if (complete.get('optimizer_step') != 32 or complete.get('recovery_updates') != 0
                or smoke.get('status') != 'complete'
                or smoke.get('independent_full_restore_verified') is not True
                or smoke.get('independent_prediction_restore_verified') is not True):
            raise RuntimeError('Independent checkpoint32 recovery did not produce a valid completion marker')
        update(smoke_recovery_status='verified_in_source_reused', smoke_updates_executed_this_run=0)
        for stop_at in (1000, 1500, 2000):
            execute(f'pilot_segment_{stop_at}', [PYTHON, str(SCRIPTS/'train_kinova.py'), '--mode', 'pilot',
                    '--resume', '--stop-at', str(stop_at)], runtime_job='E1_wrist_pilot_train2000')
            segment = json.loads((ROOT/'reports/pilot_segments'/f'step_{stop_at}_complete.json').read_text())
            checkpoint = ROOT/'checkpoints/E1_wrist_pilot/seed42'/str(stop_at)
            if (segment.get('status') != 'segment_complete'
                    or segment.get('initial_optimizer_step') != stop_at-500
                    or segment.get('optimizer_step') != stop_at
                    or segment.get('optimizer_steps_executed') != 500
                    or segment.get('total_target_steps') != 2000
                    or Path(segment.get('checkpoint', '')) != checkpoint
                    or not (checkpoint/'_CHECKPOINT_METADATA').is_file()
                    or not (checkpoint/'params/_METADATA').is_file()):
                raise RuntimeError(f'Pilot segment{stop_at} failed committed checkpoint/step/budget validation')
            update(pilot_optimizer_steps=stop_at, pilot_updates_executed_this_run=stop_at-500)
        pilot = json.loads((ROOT/'reports/pilot_training_evidence.json').read_text())
        if pilot.get('optimizer_step') != 2000 or pilot.get('base_params') != BASE:
            raise RuntimeError('Pilot did not complete exactly 2000 updates from the original base')
        # Evaluate all predeclared checkpoints only after training; validation
        # never appears in gradient updates. No test model evaluation is run.
        summaries = [str(ROOT/'evaluations/E0_wrist_frozen/summary.json')]
        for step in [500, 1000, 1500, 2000]:
            checkpoint = ROOT/'checkpoints/E1_wrist_pilot/seed42'/str(step)
            if not (checkpoint/'params/_METADATA').exists():
                raise RuntimeError(f'Required checkpoint is missing: {checkpoint}')
            output = ROOT/'evaluations/E1_wrist_pilot'/f'step_{step}'
            execute(f'E1_validation_{step}', [PYTHON, str(SCRIPTS/'evaluate_offline.py'), 'run', '--plan', plan,
                    '--label', 'E1_wrist_pilot', '--optimizer-step', str(step), '--checkpoint', str(checkpoint),
                    '--output-dir', str(output), '--resource-approved'])
            summaries.append(str(output/'summary.json'))
        # Selection is pure CPU and does not need to hold a GPU reservation.
        subprocess.run([PYTHON, str(SCRIPTS/'evaluate_offline.py'), 'select', '--plan', plan,
                        '--summaries', *summaries, '--output', str(ROOT/'configs/checkpoint_selection.json')], check=True, cwd=ROOT)
        selection = json.loads((ROOT/'configs/checkpoint_selection.json').read_text())
        if selection.get('status') != 'selected' or selection.get('all_declared_steps_evaluated') is not True:
            raise RuntimeError('Checkpoint selection did not validate all four declared checkpoints')
        update(status='complete', current_job=None, child_pid=None, smoke_optimizer_steps=32,
               pilot_optimizer_steps=2000, pilot_updates_executed_this_run=1500, finished_at=now())
        rebuild_report()
    except BaseException as error:
        if active is not None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
                active.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL); active.wait(timeout=15)
        update(status='failure', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), child_pid=None,
               finished_at=now())
        rebuild_report()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Remain queued until the recorded GPU/host/Windows resource policy permits launch')
    parser.add_argument('--max-wait-hours', type=float, default=72)
    parser.add_argument('--freeze-manifest', action='store_true', help='CPU-only: record reviewed code/config hashes; start nothing')
    args = parser.parse_args()
    if args.freeze_manifest:
        path = ROOT/'manifests/workflow_code.json'
        if path.exists():
            raise FileExistsError('Manifest already frozen; do not silently change a queued protocol')
        atomic(path, manifest()); print(path); return
    with (ROOT/'.workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another Kinova workflow owns this experiment lock') from error
        from windows_memory import WindowsMemoryMonitor
        monitor = WindowsMemoryMonitor(ROOT)
        monitor.start()
        try:
            run(args, monitor)
        finally:
            monitor.close()


if __name__ == '__main__':
    main()
