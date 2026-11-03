"""Reuse existing maintenance fixtures; never sync/drop the real kernel cache."""
import argparse, hashlib, json
from pathlib import Path
import memory_maintenance as m
from test_memory_maintenance_cpu import fixture

def configure(f):
 f['policy'].update(resume_training_clean_cache_reclaim=True,native_cache_reclaim_min_windows_commit_free_gib=12,native_cache_reclaim_wait_seconds=60)
 f['workflow']['current_job']='E1_wrist_pilot_train2000';f['save_policy']();f['save_workflow']()

def refused(f):
 try:m.reclaim_before_resumed_training(f['root'])
 except m.CacheMaintenanceError:return True
 return False

def run(output):
 checks={}
 with fixture() as f:
  configure(f);f['policy']['user_authorized']=False;f['save_policy']()
  checks['unauthorized_no_sync_drop']=refused(f) and not f['sync'].called and not f['drop_call'].called
 with fixture() as f:
  configure(f);f['workflow']['child_pid']+=1;f['save_workflow']()
  checks['wrong_owner_no_drop']=refused(f) and not f['drop_call'].called
 with fixture() as f:
  configure(f);f['state']['pre_drop_only']=True
  checks['fresh_but_predrop_sample_waits_and_aborts']=refused(f) and f['clock'].value==160 and f['drop_call'].call_count==1
 with fixture() as f:
  configure(f);f['policy']['native_cache_reclaim_min_windows_commit_free_gib']=8;f['save_policy']();f['state']['headroom_gib']=11.9
  checks['absolute12_floor_not_lowered']=refused(f) and f['clock'].value==160
 with fixture() as f:
  configure(f);f['state']['headroom_gib']=12.5
  result=m.reclaim_before_resumed_training(f['root'])
  checks['postdrop12_5_success_distinct_precompile_report_once']=(result['status']=='ready_after_reclaim' and result['required_windows_commit_free_bytes']==12*m.GIB and result['phase_scope']=='resumed_training.precompile_cache' and 'GPU checkpoint state restored' in result['caller_contract'] and (f['root']/'reports/resumed_training_cache_reclaim.jsonl').is_file() and not (f['root']/'reports/device_transfer_reclaim.jsonl').exists() and f['drop'].read_text()=='1\n' and refused(f) and f['drop_call'].call_count==1)
 report={'status':'passed' if all(checks.values()) else 'failed','schema':'kinova-resumed-training-cache-cpu-v1','checks':checks,'check_count':len(checks),'source_module_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'actual_sync_executed':False,'actual_global_drop_executed':False,'model_loaded':False,'scope':'Existing temporary proc/Windows fixtures, fake clock and mocked sync; drop only temporary ordinary file; no GPU or actual cache effectiveness test'}
 p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],len(checks))
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
