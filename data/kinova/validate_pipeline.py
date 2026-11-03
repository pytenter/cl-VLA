"""Verify the deployed data/model path without constructing an optimizer or updating weights."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import os
from pathlib import Path
RUN=Path(os.environ['KINOVA_CANDIDATE_ROOT'])
os.environ['HF_LEROBOT_HOME']=str(RUN/'lerobot')
os.environ['HF_DATASETS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
import copy,hashlib,json,time
import numpy as np
import pyarrow.parquet as pq
from openpi.training import data_loader
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models import model as model_lib
from openpi.shared import normalize,nnx_utils
import kinova_adapter as adapter

model_config=Pi0Config(pi05=True,dtype='bfloat16')
mapping=json.loads((RUN/'reports/trajectory_mapping.json').read_text())
report={'parameter_updates':0,'optimizer_created':False,'datasets':{},'issues':[],'pixel_checks':0}
for split in ['train','validation','test']:
 c=adapter.data_config(split);ds=data_loader.create_torch_dataset(c,50,model_config)
 transformed=data_loader.transform_dataset(ds,c)
 rows=[m for m in mapping if m['split']==split];offset=0;boundaries=[]
 for m in rows:
  eid=m['original_episode_id'];a=np.load(RUN/'staging'/split/eid/'numeric.npz');n=m['steps']
  tab=pq.read_table(m['parquet_path']);assert tab.num_rows==n
  assert np.array_equal(np.stack(tab['observation.state'].to_pylist()),a['state'])
  assert np.array_equal(np.stack(tab['action'].to_pylist()),a['action'])
  assert np.array_equal(np.stack(tab['source_time_ns'].to_pylist()),a['source_time_ns'])
  for name in ['primary','wrist']:
   images=tab['observation.images.'+name].to_pylist()
   for i,img in enumerate(images):
    assert img['bytes'].startswith(b'\x89PNG\r\n\x1a\n')
    assert hashlib.sha256(img['bytes']).digest()==hashlib.sha256((RUN/'staging'/split/eid/name/f'{i:06d}.png').read_bytes()).digest()
    report['pixel_checks']+=1
  for i in [0,n//2,n-1]:
   item=ds[offset+i];expected=a['action'][np.minimum(np.arange(i,i+50),n-1)]
   assert np.array_equal(np.asarray(item['action']),expected)
   pad=np.asarray(item['action_is_pad']);assert np.array_equal(pad,np.arange(i,i+50)>=n)
   out=transformed[offset+i]
   assert out['state'].shape==(32,) and out['actions'].shape==(50,32)
   assert np.all(out['actions'][:,7:]==0) and np.all(out['state'][28:]==0)
   assert out['image_mask']=={'base_0_rgb':True,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False}
  boundaries.append({'episode_id':eid,'length':n,'last_step_pad_count':49,'padding':'repeat last observed action; no cross-episode lookup'})
  offset+=n
 report['datasets'][split]={'episodes':len(rows),'steps':len(ds),'tested_boundary_samples':len(rows)*3,'boundaries':boundaries}
print('ALL_SPLITS_READ_AND_BOUNDARIES_VERIFIED',flush=True)
stats=normalize.load(RUN/'assets');train=np.load(RUN/'assets/train_vectors.npz')
norm=transforms.Normalize(stats,use_quantiles=True);unnorm=transforms.Unnormalize(stats,use_quantiles=True)
errors={}
for key in ['state','actions']:
 x=train[key];y=norm({key:x.copy()})[key]
 back=transforms.Unnormalize({key:stats[key]},use_quantiles=True)({key:y.copy()})[key]
 errors[key]={'max_roundtrip_abs_error':float(np.max(np.abs(back-x))),'normalized_min':y.min(0).tolist(),'normalized_max':y.max(0).tolist(),
 'outside_minus1_plus1_fraction':((y<-1)|(y>1)).mean(0).tolist(),'normalizer_clips':False}
report['normalization']=errors
tok=PaligemmaTokenizer(200);lengths=[];truncated=[];prompt=None
for m in mapping:
 p=RUN/'staging'/m['split']/m['original_episode_id'];meta=json.loads((p/'metadata.json').read_text());prompt=meta['language_instruction']
 a=np.load(p/'numeric.npz');s=norm({'state':a['state']})['state']
 for i,row in enumerate(s):
  bins=np.digitize(row,bins=np.linspace(-1,1,257)[:-1])-1
  full=f'Task: {prompt.strip().replace("_"," ")}, State: {" ".join(map(str,bins))};\nAction: '
  ids=tok._tokenizer.encode(full,add_bos=True);lengths.append(len(ids))
  if len(ids)>200:truncated.append([m['original_episode_id'],i,len(ids)])
report['language']={'instruction':prompt,'samples':len(lengths),'min_tokens':min(lengths),'max_tokens':max(lengths),'limit':200,'truncated_count':len(truncated),'truncated_examples':truncated[:20]}
ds=data_loader.create_torch_dataset(adapter.data_config('train'),50,model_config);raw=ds[60]
baseline=transforms.compose(adapter.input_transforms())(copy.deepcopy(raw))
changes=[]
for i in range(28):
 v=copy.deepcopy(raw);s=np.asarray(v['observation.state']).copy()
 lo=stats['state'].q01[i];hi=stats['state'].q99[i]
 s[i]=lo if abs(s[i]-lo)>abs(s[i]-hi) else hi
 if hi-lo<1e-6:s[i]=s[i]+1
 v['observation.state']=s
 altered=transforms.compose(adapter.input_transforms())(v)
 changed=np.flatnonzero(baseline['tokenized_prompt']!=altered['tokenized_prompt'])
 changes.append({'state_dimension':i,'changed_token_positions':changed.tolist(),'changed':bool(len(changed))})
assert all(v['changed'] for v in changes)
wrist=transforms.compose(adapter.input_transforms(view_mode='wrist_only'))(copy.deepcopy(raw))
assert wrist['image_mask']=={'base_0_rgb':False,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False}
report['state_control']={'same_images_and_language':True,'deterministic_preprocessing':True,'all_28_dimensions_change_tokens':True,'dimensions':changes,'tokenize_before_padding':True,'discrete_state_count':28,'continuous_state_shape':[32]}
report['input_shapes']={'dual':{k:list(v.shape) for k,v in baseline['image'].items()},'state':list(baseline['state'].shape),'actions':list(baseline['actions'].shape),'tokens':list(baseline['tokenized_prompt'].shape),'dual_masks':{k:bool(v) for k,v in baseline['image_mask'].items()},'wrist_only_masks':{k:bool(v) for k,v in wrist['image_mask'].items()}}
report['loss_scope']={'source':'openpi/models/pi0.py compute_loss averages squared error across all 32 dimensions; training script reduces all horizon steps',
 'action_dimensions':32,'observed_dimensions':7,'padded_dimensions':25,'padded_dimensions_masked_in_loss':False,
 'tail_padding_mask_created_by_lerobot':True,'tail_padding_mask_retained_by_repack':False,'training_objective_modified':False}
(RUN/'reports/pipeline_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('TOKEN_AND_NORMALIZATION_VERIFIED',report['language'],flush=True)

# One actual loaded sample through the deployed frozen base model; no train/loss/gradient calls.
import jax,jax.numpy as jnp
params=model_lib.restore_params(project_path('base_params', ''),dtype=jnp.bfloat16)
model=model_config.load(params);del params
infer=nnx_utils.module_jit(model.sample_actions,static_argnames=('num_steps',))
results={}
for name,item in [('dual_provisional',baseline),('wrist_only',wrist)]:
 item={k:v for k,v in item.items() if k!='actions'}
 batched=jax.tree.map(lambda x:jnp.asarray(x)[None,...],item);obs=model_lib.Observation.from_dict(batched)
 started=time.monotonic();output=np.asarray(jax.block_until_ready(infer(jax.random.key(42),obs,num_steps=10)))
 assert output.shape==(1,50,32) and np.isfinite(output).all()
 recovered=transforms.compose(adapter.output_transforms())({'actions':output[0],'state':np.asarray(batched['state'])[0]})['actions']
 assert recovered.shape==(50,7) and np.isfinite(recovered).all()
 np.savez(RUN/'reports'/f'inference_{name}.npz',model_actions_normalized=output,restored_actions_7d=recovered)
 results[name]={'status':'passed','seconds':time.monotonic()-started,'model_output_shape':list(output.shape),'recovered_shape':list(recovered.shape),'all_finite':True,'seed':42,'denoise_steps':10,'source_episode':mapping[0]['original_episode_id'],'source_step':60}
report['inference']={'device':str(jax.devices()[0]),'model':'official deployed pi05_base; not fine-tuned','cases':results,'parameter_updates':0,'robot_commands_sent':0,'scope':'input/output execution only; no claim of synchronization, task learning, or robot execution readiness'}
(RUN/'reports/pipeline_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report['inference'],ensure_ascii=False),flush=True)
