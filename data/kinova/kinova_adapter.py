"""Candidate-only Kinova mapping. No robot interfaces, optimizer, or absolute-to-delta transform."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import dataclasses
import os
from pathlib import Path
import numpy as np
from openpi import transforms
from openpi.training import config as config_lib
from openpi.models.pi0_config import Pi0Config
from openpi.shared import normalize

RUN=Path(os.environ.get('KINOVA_CANDIDATE_ROOT',project_path('candidate', '')))

@dataclasses.dataclass(frozen=True)
class KinovaInputs:
    view_mode: str = 'dual_provisional'
    def __call__(self,data):
        def image(x):
            x=np.asarray(x)
            if x.shape[0]==3:x=np.moveaxis(x,0,-1)
            if np.issubdtype(x.dtype,np.floating):x=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
            assert x.ndim==3 and x.shape[-1]==3 and x.dtype==np.uint8
            return x
        base=image(data['primary']);wrist=image(data['wrist'])
        active=self.view_mode=='dual_provisional'
        assert self.view_mode in ['dual_provisional','wrist_only']
        result={'state':np.asarray(data['state'],np.float32),
          'image':{'base_0_rgb':base if active else np.zeros_like(base),'left_wrist_0_rgb':wrist,'right_wrist_0_rgb':np.zeros_like(base)},
          'image_mask':{'base_0_rgb':np.bool_(active),'left_wrist_0_rgb':np.True_,'right_wrist_0_rgb':np.False_},
          'prompt':data['prompt']}
        assert result['state'].shape==(28,)
        if 'actions' in data:result['actions']=np.asarray(data['actions'],np.float32)
        return result

@dataclasses.dataclass(frozen=True)
class KinovaOutputs:
    def __call__(self,data):return {'actions':np.asarray(data['actions'])[...,:7]}

def data_config(split='train',view_mode='dual_provisional'):
    assert split in ['train','validation','test']
    model=Pi0Config(pi05=True,dtype='bfloat16')
    return config_lib.DataConfig(repo_id=f'kinova_candidate/{split}',asset_id='kinova_gen3_candidate_v1',
        norm_stats=normalize.load(RUN/'assets'),use_quantile_norm=True,
        repack_transforms=transforms.Group(inputs=(transforms.RepackTransform({
          'primary':'observation.images.primary','wrist':'observation.images.wrist',
          'state':'observation.state','actions':'action','prompt':'prompt'}),)),
        data_transforms=transforms.Group(inputs=(KinovaInputs(view_mode),),outputs=(KinovaOutputs(),)),
        model_transforms=config_lib.ModelTransformFactory()(model),
        action_sequence_keys=('action',),prompt_from_task=True)

def input_transforms(split='train',view_mode='dual_provisional'):
    c=data_config(split,view_mode)
    return (*c.repack_transforms.inputs,*c.data_transforms.inputs,
      transforms.Normalize(c.norm_stats,use_quantiles=c.use_quantile_norm),*c.model_transforms.inputs)

def output_transforms():
    return (transforms.Unnormalize(normalize.load(RUN/'assets'),use_quantiles=True),KinovaOutputs())
