"""CPU-only actual routing block with all model processes replaced by fixtures."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import ast, hashlib, json, tempfile, traceback, os
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', Path(__file__).parent))
source=Path(__file__).with_name('run_workflow.py')
run=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
sequence=next(n for n in reversed(run.body) if isinstance(n,ast.Try))
compiled=compile(ast.Module(body=[sequence],type_ignores=[]),str(source),'exec')
BASE=project_path('base_params', '')
SRC=Path(project_path('archive', 'run_20261102_215627_resume_commit32'))
results=[]
for case in ['success','wrong_start','wrong_budget','wrong_update_count','uncommitted','smoke_not_verified','wrong_E0_source']:
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory); (root/'reports').mkdir();(root/'configs').mkdir(); jobs=[]
  def put(path,value):
   p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
  put('reports/smoke_training_evidence.json',{'optimizer_step':32,'frozen_parameters_unchanged':True,'changed_lora_tensors':['B'],'status':'complete','independent_full_restore_verified':case!='smoke_not_verified','independent_prediction_restore_verified':True})
  put('reports/smoke_complete.json',{'optimizer_step':32,'recovery_updates':0})
  for name in ['full_state_restore.json','checkpoint_prediction_restore.json','base_zero_lora_equivalence.json']:put('reports/'+name,{'status':'passed'})
  state={'completed_jobs':[{'name':'E0_wrist_frozen','reused_from_run':str(SRC) if case!='wrong_E0_source' else '/wrong'}]}
  def execute(name,command,**kw):
   jobs.append(name)
   if name.startswith('pilot_segment_'):
    assert kw['runtime_job']=='E1_wrist_pilot_train2000'
    assert command[-3:] == ['--resume','--stop-at',name.rsplit('_',1)[1]]
    stop=int(command[-1]); cp=root/'checkpoints/E1_wrist_pilot/seed42'/str(stop)
    put(str(cp.relative_to(root)/'params/_METADATA'),{})
    if case!='uncommitted' or stop!=1000:put(str(cp.relative_to(root)/'_CHECKPOINT_METADATA'),{})
    put(f'reports/pilot_segments/step_{stop}_complete.json',{'status':'segment_complete','initial_optimizer_step':0 if case=='wrong_start' else stop-500,'optimizer_step':stop,'optimizer_steps_executed':499 if case=='wrong_update_count' else 500,'total_target_steps':1000 if case=='wrong_budget' else 2000,'checkpoint':str(cp)})
    if stop==2000:
     put('reports/pilot_training_evidence.json',{'optimizer_step':2000,'base_params':BASE})
     put('checkpoints/E1_wrist_pilot/seed42/500/params/_METADATA',{})
   else:assert name.startswith('E1_validation_')
  def select(*a,**kw):put('configs/checkpoint_selection.json',{'status':'selected','all_declared_steps_evaluated':True})
  ns={'ROOT':root,'SOURCE_ROOT':SRC,'BASE':BASE,'SCRIPTS':root/'scripts','PYTHON':'python','plan':str(root/'configs/validation_plan.json'),'state':state,'active':None,'execute':execute,'validate_same_graph_e0':lambda:None,'update':lambda **kw:state.update(kw),'json':json,'Path':Path,'subprocess':SimpleNamespace(run=select),'traceback':traceback,'rebuild_report':lambda:None,'now':lambda:'CPU-fixture'}
  error=None
  try:exec(compiled,ns)
  except RuntimeError as exc:error=str(exc)
  expected=['pilot_segment_1000','pilot_segment_1500','pilot_segment_2000','E1_validation_500','E1_validation_1000','E1_validation_1500','E1_validation_2000']
  passed=(error is None and jobs==expected and state.get('status')=='complete') if case=='success' else (error is not None and 'E1_validation_500' not in jobs and 'pilot_segment_1500' not in jobs)
  results.append({'case':case,'passed':passed,'jobs':jobs,'error':error})
result={'status':'passed' if all(x['passed'] for x in results) else 'failed','scope':'Actual production routing block, model/child/resource execution stubbed; no GPU/model/checkpoint500 arrays. Prefix/copy validated separately.','cases':results,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'no_E0_or_smoke_GPU_jobs':True,'optimizer_updates_executed':0}
p=ROOT/'reports/segmented_workflow_cpu.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(result['status'],len(results));
if result['status']!='passed':raise SystemExit(1)
