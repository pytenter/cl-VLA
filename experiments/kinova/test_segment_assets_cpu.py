"""Production release branch and native asset callback, without a model."""
import ast,gc,hashlib,json,os,tempfile,weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from openpi.training import checkpoints
from openpi.shared import normalize
import checkpoint_io
root=Path(os.environ['KINOVA_EXPERIMENT_ROOT']);source=Path(__file__).with_name('train_kinova.py')
tree=ast.parse(source.read_text());branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Delete) and any(isinstance(y,ast.Name) and y.id=='iterator' for y in x.targets) for x in n.body))
code=compile(ast.Module(body=[branch],type_ignores=[]),str(source),'exec')
class Owner:pass
class Loader:
 def __init__(self,config):self.config=config;self.dataset=Owner()
 def data_config(self):return self.config
class Iterator:
 def __init__(self,loader):self.loader=loader
config=SimpleNamespace(norm_stats={'actions':normalize.NormStats(mean=np.zeros(7),std=np.ones(7),q01=-np.ones(7),q99=np.ones(7))},asset_id='kinova')
loader=Loader(config);loader_ref=weakref.ref(loader);data_ref=weakref.ref(loader.dataset)
iterator=Iterator(loader);iterator_ref=weakref.ref(iterator)
ns={'args':SimpleNamespace(mode='pilot'),'stop_at':1000,'state':SimpleNamespace(step=1000),'loader':loader,'iterator':iterator,'micros':[Owner()],'batch':Owner(),'evidence':{},'ROOT':root,'mark_phase':lambda *a,**kw:None}
del loader,iterator
with patch.object(checkpoint_io,'_release',side_effect=gc.collect):exec(code,ns)
checks={'old_loader_collected':loader_ref() is None,'dataset_collected':data_ref() is None,'iterator_collected':iterator_ref() is None,'input_names_removed':all(k not in ns for k in ['iterator','micros','batch']),'asset_config_identity_preserved':ns['loader'].data_config() is config,'evidence_recorded':ns['evidence'].get('checkpoint_inputs_released_before_save') is True}
class Capture:
 def __init__(self,path):self.path=path
 def save(self,step,items):items['assets'](self.path)
with tempfile.TemporaryDirectory() as d:
 d=Path(d);a=d/'original';b=d/'proxy';a.mkdir();b.mkdir()
 with patch.object(checkpoints,'_split_params',return_value=({},{})):
  checkpoints.save_state(Capture(a),ns['state'],Loader(config),1000)
  checkpoints.save_state(Capture(b),ns['state'],ns['loader'],1000)
 files_a={str(p.relative_to(a)):p.read_bytes() for p in a.rglob('*') if p.is_file()};files_b={str(p.relative_to(b)):p.read_bytes() for p in b.rglob('*') if p.is_file()}
 checks['actual_native_normalization_assets_byte_identical']=bool(files_a) and files_a==files_b
result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'trainer_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'GPU_used':False,'scope':'Actual production release AST plus native normalization asset callback; model-state split/manager mocked, no model arrays or global cache actions'}
(root/'reports/segment_assets_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if result['status']!='passed':raise SystemExit(1)
