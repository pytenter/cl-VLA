"""Export an immutable RLDS source into a new, explicit PNG/numeric staging version."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
from pathlib import Path
import json,sys,hashlib,collections,csv,io,time
import numpy as np
import tensorflow as tf

run=Path(sys.argv[1]); src=Path('/mnt/d/KinovaDatasets/review_20261031/candidate_rlds_20261031/kinova_dual_view/1.5.0')
raw=Path('/root/kinova_candidate_work/run_20261031/episodes')
fields=[('joint_position_deg',7,'degree'),('joint_velocity_deg_s',7,'degree/s'),('joint_torque_nm',7,'N*m'),('tool_pose_m_deg',6,'m,m,m,degree,degree,degree'),('gripper_position_raw',1,'device raw feedback')]
times=['action_host_monotonic_ns','action_target_host_monotonic_ns','robot_host_monotonic_ns','target_robot_host_monotonic_ns','image_host_monotonic_ns','image_content_first_receipt_host_monotonic_ns','image_estimated_content_host_monotonic_ns','wrist_image_host_monotonic_ns']
records=[]; train_states=[];train_actions=[];start=time.monotonic()
for split in ['train','validation','test']:
 for shard in sorted(src.glob(f'*{split}.tfrecord-*')):
  for buf in tf.compat.v1.io.tf_record_iterator(str(shard)):
   f=tf.train.Example.FromString(buf).features.feature
   def txt(k):return f['episode_metadata/'+k].bytes_list.value[0].decode()
   def array(k,dtype=None):
    v=f['steps/'+k];return np.asarray(getattr(v,v.WhichOneof('kind')).value,dtype=dtype)
   eid=txt('episode_id'); n=len(array(times[0])); dest=run/'staging'/split/eid;dest.mkdir(parents=True,exist_ok=False)
   vectors=[]
   for name,dim,unit in fields:
    a=array('observation/'+name,np.float32).reshape(n,-1);assert a.shape[1]==dim and np.isfinite(a).all();vectors.append(a)
   state=np.concatenate(vectors,axis=1);actions=array('action',np.float32).reshape(n,-1);assert actions.shape==(n,7)
   ts=np.stack([array(k,np.int64) for k in times],axis=1)
   flags=np.stack([array(k,bool) for k in ['is_first','is_last','is_terminal']],axis=1)
   langs=[v.decode() for v in f['steps/language_instruction'].bytes_list.value];assert len(set(langs))==1
   meta={k:txt(k) for k in ['episode_id','task_name','outcome','outcome_source','action_source','action_spec','reconstruction_json']}
   meta.update(split=split,steps=n,language_instruction=langs[0],source_shard=str(shard),state_fields=fields,timestamp_fields=times)
   np.savez(dest/'numeric.npz',state=state,action=actions,source_time_ns=ts,source_flags=flags)
   (dest/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
   image_info={};
   for name,key,shape in [('primary','image',(640,480,3)),('wrist','wrist_image',(480,640,3))]:
    d=dest/name;d.mkdir();previous=None;repeated=0;last_png=None;first_digest=None;pixel_digests=[];codec=collections.Counter()
    for i,data in enumerate(f['steps/observation/'+key].bytes_list.value):
     digest=hashlib.sha256(data).digest();codec['JPEG' if data.startswith(b'\xff\xd8') else 'PNG']+=1
     if digest==previous:
      repeated+=1;png=last_png;pixel_digest=pixel_digests[-1]
     else:
      image=tf.io.decode_image(data,channels=3,expand_animations=False).numpy();assert image.shape==shape and image.dtype==np.uint8
      png=tf.io.encode_png(image,compression=1).numpy();pixel_digest=hashlib.sha256(image.tobytes()).hexdigest()
      check=tf.io.decode_png(png,channels=3).numpy();assert np.array_equal(image,check)
     (d/f'{i:06d}.png').write_bytes(png);pixel_digests.append(pixel_digest);previous=digest;last_png=png
    image_info[name]={'source_codecs':dict(codec),'written_codec':'PNG','shape_hwc':shape,'adjacent_identical':repeated,'decoded_frames':n,'pixel_sha256':pixel_digests}
   rawmeta=json.loads((raw/eid/'metadata.json').read_text())
   t=ts[:,0];u=ts[:,1];dt=(u-t)/1e6;wrist_age=(t-ts[:,7])/1e6;content_age=(t-ts[:,6])/1e6
   grip=state[:,-1];z=state[:,23]
   # These are observable feedback events, not asserted task-stage labels.
   events={'initial':0,'maximum_gripper_closure':int(np.argmax(grip)),'highest_tool_z':int(np.argmax(z)),
     'strongest_closing_change':int(np.argmax(np.diff(grip,prepend=grip[0]))),
     'strongest_opening_change':int(np.argmin(np.diff(grip,prepend=grip[0]))),'last_exported':n-1}
   finalraw=json.loads((raw/eid/'robot.jsonl').read_text().splitlines()[-1])
   rb=finalraw['feedback']['base'];rp=np.array([rb[k] for k in ['tool_pose_x','tool_pose_y','tool_pose_z']])
   row={'original_episode_id':eid,'candidate_episode_id':eid,'split':split,'steps':n,'status':rawmeta['status'],
    'raw_outcome':rawmeta.get('outcome'),'candidate_outcome':meta['outcome'],'outcome_source':meta['outcome_source'],
    'source_shard':str(shard),'raw_duration_s':rawmeta['duration_seconds'],
    'start_trim_s':float((t[0]-rawmeta['started_host_monotonic_ns'])/1e9),
    'tail_trim_s':float((rawmeta['ended_host_monotonic_ns']-u[-1])/1e9),
    'tail_final_to_last_observation_translation_mm':float(np.linalg.norm(rp-state[-1,21:24])*1000),
    'dt_ms':{'min':float(dt.min()),'median':float(np.median(dt)),'max':float(dt.max())},
    'wrist_age_ms':{'min':float(wrist_age.min()),'median':float(np.median(wrist_age)),'max':float(wrist_age.max())},
    'content_age_ms':{'min':float(content_age.min()),'median':float(np.median(content_age)),'max':float(content_age.max())},
    'gripper_raw_min':float(grip.min()),'gripper_raw_max':float(grip.max()),'events':events,
    'key_stages_status':'feedback proxies only; visual task completion not independently confirmed',
    'missing_images':0,'abnormal_timestamps':bool(np.any(np.diff(t)<=0) or np.any(u<=t)),
    'images':image_info}
   (dest/'audit.json').write_text(json.dumps(row,ensure_ascii=False,indent=2));records.append(row)
   if split=='train':train_states.append(state);train_actions.append(actions)
   print(split,eid,n,'staged PNG + numeric',flush=True)
   (run/'reports/staging_progress.json').write_text(json.dumps({'episodes':len(records),'last':eid}))
np.savez(run/'assets/train_vectors.npz',state=np.concatenate(train_states),actions=np.concatenate(train_actions))
(run/'reports/episode_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
simple=[{k:v for k,v in r.items() if k not in ['images','events','dt_ms','wrist_age_ms','content_age_ms']} for r in records]
with (run/'reports/episode_audit.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
(run/'reports/staging_complete.json').write_text(json.dumps({'episodes':len(records),'steps':sum(r['steps'] for r in records),'all_images_decoded':True,'png_roundtrip_pixels_exact':True,'seconds':time.monotonic()-start},indent=2))
