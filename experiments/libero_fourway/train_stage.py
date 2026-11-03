"""Run one explicitly configured π0.5 LIBERO stage, with isolated checkpoints."""
import argparse,dataclasses,json,logging,os,time,sys
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=['plain','noreplay','prototype','uniform'],required=True);p.add_argument('--stage',choices=list('ABCD'),required=True);p.add_argument('--steps',type=int);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--weights');p.add_argument('--replay-buffer',action='append');a=p.parse_args()
 logging.basicConfig(level=logging.INFO)
 from libero_integration import install_hooks
 from experiment_config import make_config,ROOT
 install_hooks();cfg=make_config(a.group,a.stage,steps=a.steps,batch_size=a.batch_size,resume=a.resume,smoke=a.smoke,weights=a.weights)
 if a.replay_buffer:cfg=dataclasses.replace(cfg,rehearsal=dataclasses.replace(cfg.rehearsal,buffer_dirs=tuple(a.replay_buffer)))
 (ROOT/'reports/configurations').mkdir(exist_ok=True)
 (ROOT/'reports/configurations'/f'{cfg.name}.json').write_text(json.dumps(dataclasses.asdict(cfg),default=str,indent=2))
 started=time.monotonic()
 if a.group=='plain':
  from scripts.train import main as train
 else:
  from stable_training import install_training_fixes
  install_training_fixes()
  from scripts.train_cl_lora import main as train
 train(cfg)
 (ROOT/'reports'/f'{cfg.name}_complete.json').write_text(json.dumps({'group':a.group,'stage':a.stage,'steps':cfg.num_train_steps,'batch_size':cfg.batch_size,'seconds':time.monotonic()-started,'checkpoint_dir':str(cfg.checkpoint_dir),'smoke':a.smoke},indent=2))
if __name__=='__main__':main()
