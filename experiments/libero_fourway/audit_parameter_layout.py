"""CPU abstract-shape audit only: never materialize the complete model parameters."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
import json,math,time
from pathlib import Path
import jax
from flax import nnx
from experiment_config import make_config,ROOT,GROUPS
from stable_training import bank_axis,mask_other_task_banks

started=time.monotonic()
assert all(device.platform=='cpu' for device in jax.devices())
base=project_path('base_params', '')
result={'scope':'CPU jax.eval_shape only; full 3B arrays not materialized; no full optimizer update',
        'devices':[str(x) for x in jax.devices()],'layouts':{},'cases':{}}
for name,group in [('plain','plain'),('cl','noreplay')]:
    config=make_config(group,'A',weights=base)
    state=jax.eval_shape(lambda key:nnx.state(config.model.create(key)),jax.random.key(0))
    trainable=state.filter(config.trainable_filter)
    def numel(v):return math.prod(v.value.shape)
    total=sum(numel(v) for v in state.flat_state().values())
    trainable_count=sum(numel(v) for v in trainable.flat_state().values())
    bank_rows=[];frozen_bank_count=0
    for path,v in state.flat_state().items():
        axis=bank_axis(path,tuple(v.value.shape),4)
        if axis is not None:
            bank_rows.append({'path':'/'.join(map(str,path)),'shape':list(v.value.shape),'task_axis':axis,
                              'task_capacity':v.value.shape[axis],'total_parameters':numel(v)})
    for path,v in trainable.flat_state().items():
        if bank_axis(path,tuple(v.value.shape),4) is not None:
            frozen_bank_count+=numel(v)*3//4
    if name=='cl':
        # Exercise the same mask on the REAL abstract parameter pytree, not
        # only on the small diagnostic's representative paths.
        masked=jax.eval_shape(lambda tree:mask_other_task_banks(tree,task_id=1,num_tasks=4),trainable)
        assert jax.tree.structure(masked)==jax.tree.structure(trainable)
    result['layouts'][name]={'total_parameters':total,'trainable_filter_parameter_count':trainable_count,
      'trainable_tensor_count':len(trainable.flat_state()),'frozen_filter_parameter_count':total-trainable_count,
      'old_and_future_bank_parameters_masked':frozen_bank_count,
      'optimizer_allowed_parameter_count_upper_bound':trainable_count-frozen_bank_count,
      'runtime_stop_gradient_lora_a':config.model.cl_freeze_lora_a,
      'count_caveat':'trainable filter includes all allocated banks and runtime-stop-gradient A; upper bound ignores unused scanned specific/shared rows',
      'bank_tensors':bank_rows,'real_abstract_mask_trace_passed':name=='cl'}
    print(name,'abstract_total',total,'trainable_filter',trainable_count,'banks',len(bank_rows),flush=True)
for group in GROUPS:
 for stage in 'ABCD':
    c=make_config(group,stage,weights=base)
    result['cases'][f'{group}_{stage}']={'layout':'plain' if group=='plain' else 'cl','task_id':c.model.cl_task_id,
      'action_horizon':c.model.action_horizon,'action_dim':c.model.action_dim,'max_token_len':c.model.max_token_len,
      'discrete_state_input':c.model.discrete_state_input,'optimizer_weight_decay':c.optimizer.weight_decay,
      'same_balanced_freeze':True,'state':'configuration created with explicit base override for audit; not stage initialization'}
result['seconds']=time.monotonic()-started
path=ROOT/'reports/parameter_layout.json';path.write_text(json.dumps(result,indent=2));print(path,flush=True)
