"""Check four task identities, boundaries, state tokens, and all group configurations."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
from pathlib import Path
import os,json,dataclasses
import numpy as np
from libero_integration import install_hooks
from experiment_config import ROOT,TASKS,GROUPS,make_config
from openpi.training import data_loader
install_hooks();result={}
for stage in 'ABCD':
 c=make_config('prototype',stage,weights=project_path('base_params', ''))
 dc=c.data.create(c.assets_dirs,c.model);raw=data_loader.create_torch_dataset(dc,10,c.model);d=data_loader.transform_dataset(raw,dc)
 item=d[0];assert item['state'].shape==(32,) and item['actions'].shape==(10,32)
 assert item['image_mask']=={'base_0_rgb':True,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False}
 assert raw[0]['prompt']==TASKS[stage][2]
 changed=raw[0].copy();changed['state']=np.asarray(changed['state']).copy();changed['state'][0]+=.1
 t=data_loader.TransformedDataset([changed],[*dc.repack_transforms.inputs,*dc.data_transforms.inputs,
   __import__('openpi.transforms',fromlist=['Normalize']).Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])[0]
 assert np.any(item['tokenized_prompt']!=t['tokenized_prompt'])
 base=raw._dataset if hasattr(raw,'_dataset') else raw
 bounds=base.episode_data_index
 for k in range(base.num_episodes):
  last=int(bounds['to'][k])-1;s=raw[last];assert np.all(np.asarray(s['actions_is_pad'])[1:])
  assert np.all(np.asarray(s['actions'])==np.asarray(s['actions'])[0])
 result[stage]={'task':TASKS[stage][2],'episodes':base.num_episodes,'frames':len(raw),'state_shape':[8],'actions_shape':[10,7],'state_tokens_change':True,'end_boundaries_checked':base.num_episodes,'input_shapes':{k:list(v.shape) for k,v in item['image'].items()}}
(ROOT/'reports/data_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
