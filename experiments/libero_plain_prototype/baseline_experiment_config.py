"""Explicit four-way π0.5 LIBERO migration protocol; no hidden training side effects."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import dataclasses,os,re
from pathlib import Path
import numpy as np
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as cfg,optimizer,weight_loaders
from openpi.models import model as model_lib
from libero_integration import LiberoDataFactory
ROOT=Path(os.environ.get('PI05_LIBERO_ROOT',project_path('libero_fourway', '')))
TASKS={
 'A':('libero_spatial',6,'pick up the black bowl next to the cookie box and place it on the plate'),
 'B':('libero_spatial',0,'pick up the black bowl between the plate and the ramekin and place it on the plate'),
 'C':('libero_object',8,'pick up the chocolate pudding and place it in the basket'),
 'D':('libero_goal',7,'turn on the stove')}
GROUPS=('plain','noreplay','prototype','uniform')

@dataclasses.dataclass(frozen=True)
class CompleteWeightLoader:
    params_path:str
    def load(self,params):
        loaded=model_lib.restore_params(self.params_path,restore_type=np.ndarray)
        return weight_loaders._merge_params(loaded,params,missing_regex=r'.*(?:lora|cl_block_weights).*')

def latest_params(group,stage):
 p=ROOT/'checkpoints'/f'pi05_libero_{group}_{stage}'/'seed42'
 steps=sorted(int(x.name) for x in p.iterdir() if x.is_dir() and x.name.isdigit() and (x/'params/_METADATA').is_file()) if p.exists() else []
 if not steps:raise FileNotFoundError(f'No completed checkpoint at {p}')
 return p/str(steps[-1])/'params'

def make_config(group,stage,steps=None,*,weights=None,resume=False,batch_size=2,smoke=False):
 assert group in GROUPS and stage in TASKS
 bank='ABCD'.index(stage);is_cl=group!='plain';repo=f'libero_fourway/task_{stage}'
 model=Pi0Config(pi05=True,dtype='bfloat16',action_horizon=10,max_token_len=200,discrete_state_input=True,
   paligemma_variant='gemma_2b_cl_lora' if is_cl else 'gemma_2b_lora',
   action_expert_variant='gemma_300m_cl_lora' if is_cl else 'gemma_300m_lora',
   cl_task_id=bank,cl_num_tasks=4,cl_pali_lora_bank=is_cl,cl_action_lora_bank=is_cl,
   cl_pali_block_weight_bank=is_cl,cl_action_block_weight_bank=is_cl,cl_freeze_lora_a=is_cl,
   cl_freeze_shared_lora=False)
 data=LiberoDataFactory(repo_id=repo,target_task_name=TASKS[stage][2],
   assets=cfg.AssetsConfig(assets_dir=str(ROOT/'assets'),asset_id=repo),extra_delta_transform=False)
 if weights is None:
  weights=Path(project_path('base_params', '')) if bank==0 else latest_params(group,'ABCD'[bank-1])
 buffers=tuple(str(ROOT/'buffers'/group/f'task_{s}') for s in 'ABCD'[:bank])
 rehearsal=cfg.RehearsalConfig(enabled=group in ['prototype','uniform'] and bank>0,source='offline_buffer',buffer_dirs=buffers,
   loss_weight=.5,replay_every_n_steps=4,sample_strategy='round_robin')
 return cfg.TrainConfig(name=f'pi05_libero_{group}_{stage}'+('_smoke' if smoke else ''),exp_name='seed42',
   model=model,data=data,weight_loader=CompleteWeightLoader(str(weights)),
   freeze_filter=model.get_balanced_lora_freeze_filter(allow_block_weights=is_cl),
   lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=500,peak_lr=5e-5,decay_steps=10000,decay_lr=5e-6),
   optimizer=optimizer.AdamW(clip_gradient_norm=1.0,weight_decay=0.0),ema_decay=None,lambda_kd=0.0,
   rehearsal=rehearsal,seed=42,batch_size=batch_size,num_workers=0,num_train_steps=steps or 10000,
   log_interval=10 if smoke else 100,save_interval=1 if smoke else 500,keep_period=None,
   assets_base_dir=str(ROOT/'config_assets'),checkpoint_base_dir=str(ROOT/'smoke_checkpoints' if smoke else ROOT/'checkpoints'),
   overwrite=False,resume=resume,wandb_enabled=False,fsdp_devices=1)

def config_from_name(name):
 match=re.fullmatch(r'pi05_libero_(plain|noreplay|prototype|uniform)_([ABCD])',name)
 if not match:raise ValueError(name)
 return make_config(*match.groups())
