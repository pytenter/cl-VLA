"""CPU mock test of the native reduced-info metrics mirror; no model steps."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import argparse
import json
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
from scripts import train_cl_lora as runtime
from plain_replay_training import install_training_fixes, uninstall_training_fixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original_init, original_step, real_log = runtime.init_train_state, runtime.train_step_cl_lora, runtime.wandb.log
    real_wandb_init = runtime.init_wandb
    with tempfile.TemporaryDirectory(prefix='plain_replay_metrics_cpu_') as directory:
        path = Path(directory)/'metrics.jsonl'
        with mock.patch.object(runtime.wandb, 'log'), mock.patch.object(runtime, 'init_wandb') as native_init:
            delegate = mock.Mock(return_value='native-forwarded')
            native_init.side_effect = lambda *args, **kwargs: setattr(runtime.wandb, 'log', delegate)
            try:
                install_training_fixes(metrics_path=path)
                runtime.init_wandb('mock-config', resuming=False, enabled=False)
                native_init.assert_called_once_with('mock-config', resuming=False, enabled=False)
                data = {'loss_task': np.asarray(1., np.float32), 'loss_rehearsal': np.asarray(2., np.float32),
                        'loss_total': np.asarray(2., np.float32), 'grad_norm': np.asarray(.25, np.float32),
                        'lambda_kd': np.asarray(0., np.float32), 'loss_kd': np.asarray(0., np.float32)}
                assert runtime.wandb.log(data, step=4) == 'native-forwarded'
                delegate.assert_called_once_with(data, step=4)
                assert delegate.call_args.args[0] is data
                recorded = json.loads(path.read_text().splitlines()[0])
                assert recorded['step'] == 4 and recorded['optimizer_step'] == 5 and recorded['finite'] is True
                for key, value in data.items():
                    assert recorded[key] == value.item()
                broken = dict(data, grad_norm=np.asarray(np.nan, np.float32))
                try:
                    runtime.wandb.log(broken, step=5)
                except FloatingPointError:
                    pass
                else:
                    raise AssertionError('Non-finite native metrics must be retained and rejected')
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                assert rows[1]['finite'] is False and rows[1]['grad_norm'] == 'nan'
                assert delegate.call_count == 1
            finally:
                uninstall_training_fixes()
            assert runtime.wandb.log is delegate
            assert runtime.init_wandb is native_init
    assert runtime.init_train_state is original_init and runtime.train_step_cl_lora is original_step
    assert runtime.wandb.log is real_log
    assert runtime.init_wandb is real_wandb_init
    result = {'status': 'passed', 'scope': 'CPU mocks of native logging only; no model/optimizer/GPU',
              'checks': {'native_values_and_step_recorded': True, 'original_log_forwarded_unchanged': True,
                         'nonfinite_metrics_recorded_without_invalid_JSON_and_rejected': True,
                         'sink_reattaches_after_native_wandb_init_rebinds_log': True,
                         'uninstall_restores_original_init_step_and_logger': True}, 'pi05_optimizer_steps': 0}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
