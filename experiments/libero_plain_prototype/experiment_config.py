"""Ordinary LoRA plus the frozen original prototype memories, in a new run."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import dataclasses
import os
from pathlib import Path
import re

import baseline_experiment_config as baseline
from openpi.training import config as cfg


ROOT = Path(os.environ.get('PI05_LIBERO_ROOT', project_path('libero_plain_prototype', '')))
OLD_ROOT = Path(project_path('libero_fourway', ''))
BASE_PARAMS = Path(project_path('base_params', ''))
GROUPS = ('plain_prototype',)
TASKS = {
    'A': ('libero_spatial', 6, 'pick up the black bowl next to the cookie box and place it on the plate'),
    'B': ('libero_spatial', 0, 'pick up the black bowl between the plate and the ramekin and place it on the plate'),
    'C': ('libero_object', 8, 'pick up the chocolate pudding and place it in the basket'),
    'D': ('libero_goal', 7, 'turn on the stove'),
}
FIXED_REPLAY_SAMPLE_COUNTS = {'A': 219, 'B': 225, 'C': 261}


def latest_params(group, stage):
    if group != 'plain_prototype' or stage not in TASKS:
        raise ValueError(f'Unknown new group/stage: {group}/{stage}')
    folder = ROOT/'checkpoints'/f'pi05_libero_{group}_{stage}'/'seed42'
    steps = sorted(int(path.name) for path in folder.iterdir()
                   if path.is_dir() and path.name.isdigit() and (path/'params/_METADATA').is_file()) if folder.exists() else []
    if not steps:
        raise FileNotFoundError(f'No completed NEW-group checkpoint at {folder}')
    return folder/str(steps[-1])/'params'


def make_config(group='plain_prototype', stage='A', steps=None, *, weights=None, resume=False,
                batch_size=2, smoke=False):
    if group != 'plain_prototype' or stage not in TASKS:
        raise ValueError(f'Unknown new group/stage: {group}/{stage}')
    index = 'ABCD'.index(stage)
    if weights is None:
        weights = BASE_PARAMS if index == 0 else latest_params(group, 'ABCD'[index-1])
    # Explicit weights avoids baseline's old-group previous-stage lookup.
    # Its ordinary model, initialization, freeze boundary and LR are unchanged.
    original = baseline.make_config('plain', stage, steps=steps, weights=weights, resume=resume,
                                    batch_size=batch_size, smoke=smoke)
    repo = f'libero_fourway/task_{stage}'
    data = dataclasses.replace(original.data,
        assets=cfg.AssetsConfig(assets_dir=str(OLD_ROOT/'assets'), asset_id=repo))
    buffers = tuple(str(OLD_ROOT/'buffers/prototype'/f'task_{old}') for old in 'ABCD'[:index])
    rehearsal = dataclasses.replace(original.rehearsal, enabled=index > 0, source='offline_buffer',
                                    buffer_dirs=buffers, loss_weight=.5, replay_every_n_steps=4,
                                    sample_strategy='round_robin')
    result = dataclasses.replace(original,
        name=f'pi05_libero_plain_prototype_{stage}'+('_smoke' if smoke else ''),
        data=data, rehearsal=rehearsal,
        log_interval=1 if smoke else original.log_interval,
        assets_base_dir=str(ROOT/'config_assets'),
        checkpoint_base_dir=str(ROOT/('smoke_checkpoints' if smoke else 'checkpoints')),
        overwrite=False)
    model = result.model
    if (model.paligemma_variant, model.action_expert_variant) != ('gemma_2b_lora', 'gemma_300m_lora'):
        raise ValueError('Expected the original ordinary-LoRA variants')
    if any(getattr(model, field) for field in ('cl_pali_lora_bank', 'cl_action_lora_bank',
            'cl_pali_block_weight_bank', 'cl_action_block_weight_bank', 'cl_freeze_lora_a',
            'cl_freeze_shared_lora', 'cl_freeze_pali_lora_b', 'cl_freeze_action_lora_b')):
        raise ValueError('CL bank/gate/freezing options are forbidden in ordinary LoRA')
    if result.lambda_kd != 0 or result.ema_decay is not None:
        raise ValueError('The new baseline requires KD=0 and EMA disabled')
    return result


def config_from_name(name):
    match = re.fullmatch(r'pi05_libero_plain_prototype_([ABCD])', name)
    if not match:
        raise ValueError(name)
    return make_config(stage=match.group(1))
