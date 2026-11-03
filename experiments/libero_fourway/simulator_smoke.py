"""Reset one LIBERO simulation, run no-ops, and verify offscreen dual RGB observations."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--gl',choices=['egl','osmesa','glx'],default='egl')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--task-label',choices=['A','B','C','D'],default='A')
    args=parser.parse_args()
    root=Path(args.root).resolve();dest=root/'reports/simulator'/f'smoke_{args.gl}_{args.task_label}'
    dest.mkdir(parents=True,exist_ok=True)
    os.environ['MUJOCO_GL']=args.gl
    if args.gl in ['egl','osmesa']:os.environ['PYOPENGL_PLATFORM']=args.gl
    if args.gl=='osmesa':os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
    os.environ['LIBERO_CONFIG_PATH']=str(root/'cache/libero_config')
    started=time.monotonic();env=None
    result={'status':'running','backend':args.gl,'seed':args.seed,'task_label':args.task_label,'parameter_updates':0,'model_loaded':False,'real_robot_commands':0}
    try:
        import numpy as np
        import yaml
        from PIL import Image
        specification=importlib.util.find_spec('libero')
        package=Path(next(iter(specification.submodule_search_locations)))/'libero'
        assets=root/'simulator_assets'
        target=package/'assets'
        if not target.exists():target.symlink_to(assets,target_is_directory=True)
        if target.resolve()!=assets.resolve():raise RuntimeError(f'Unexpected simulator assets target {target}')
        directory=root/'cache/libero_config';directory.mkdir(parents=True,exist_ok=True)
        paths={'benchmark_root':str(package),'bddl_files':str(package/'bddl_files'),'init_states':str(package/'init_files'),'datasets':str(root/'datasets/libero'),'assets':str(assets)}
        (directory/'config.yaml').write_text(yaml.safe_dump(paths))
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.transform_utils import quat2axisangle
        tasks={'A':('libero_spatial',6,'pick up the black bowl next to the cookie box and place it on the plate'),
               'B':('libero_spatial',0,'pick up the black bowl between the plate and the ramekin and place it on the plate'),
               'C':('libero_object',8,'pick up the chocolate pudding and place it in the basket'),
               'D':('libero_goal',7,'turn on the stove')}
        suite_name,task_index,language=tasks[args.task_label]
        suite=benchmark.get_benchmark_dict()[suite_name](task_order_index=0);task=suite.get_task(task_index)
        assert task.language==language,(task.language,language)
        initial=suite.get_task_init_states(task_index)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(args.seed);env.reset();obs=env.set_init_state(initial[0])
        for _ in range(10):obs,reward,done,info=env.step([0.0]*6+[-1.0])
        images={}
        sys.path.insert(0,project_path('deploy', 'source/pi05/packages/openpi-client/src'))
        from openpi_client import image_tools
        for name,key in [('agentview','agentview_image'),('wrist','robot0_eye_in_hand_image')]:
            image=np.asarray(obs[key]);assert image.shape==(256,256,3) and image.dtype==np.uint8
            assert np.std(image)>1.0 and np.count_nonzero(image)>1000,(name,'blank image')
            Image.fromarray(image).save(dest/f'{name}_raw.png')
            rotated=np.ascontiguousarray(image[::-1,::-1])
            policy_image=image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated,224,224))
            Image.fromarray(policy_image).save(dest/f'{name}_policy.png')
            images[name]={'shape_hwc':list(image.shape),'dtype':str(image.dtype),'min':int(image.min()),'max':int(image.max()),'std':float(image.std()),'policy_image_shape':list(policy_image.shape),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
            images[name+'_array']=policy_image
        state=np.concatenate([obs['robot0_eef_pos'],quat2axisangle(np.array(obs['robot0_eef_quat'],copy=True)),obs['robot0_gripper_qpos']]).astype(np.float32)
        assert state.shape==(8,) and np.isfinite(state).all()
        np.savez_compressed(dest/'sample_input.npz',image=images.pop('agentview_array'),wrist_image=images.pop('wrist_array'),state=state,prompt=np.asarray(language))
        result.update(status='passed',suite=suite_name,task_index=task_index,language=language,initial_states=len(initial),initial_state_index=0,
            initial_state_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(initial[0])).tobytes()).hexdigest(),no_op_steps=10,
            state=state.tolist(),images=images,observation_keys=sorted(obs),action_dim=int(env.env.action_dim),
            versions={p:importlib.metadata.version(p) for p in ['hf-libero','robosuite','mujoco','numba','numpy','torch','PyOpenGL']},
            config=paths,simulator_task_map_sha256=hashlib.sha256((package/'benchmark/libero_suite_task_map.py').read_bytes()).hexdigest())
    except BaseException:
        result.update(status='failed',error=traceback.format_exc())
    finally:
        if env is not None:
            try:env.close()
            except BaseException:result['cleanup_error']=traceback.format_exc()
        result['seconds']=time.monotonic()-started
        (dest/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
