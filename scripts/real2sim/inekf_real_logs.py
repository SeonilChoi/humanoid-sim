"""Offline contact-aided InEKF (olaf/inekf.py) on the real KB logs (after fk_real_logs.py), for a noise-parameter grid:

    python scripts/real2sim/inekf_real_logs.py <dir with *_fk.npz> <accel_noise list> <contact_noise list> <fk_noise list> [v]
    e.g. ... . 0.5,2.0 0.02,0.05 0.002,0.004

Prints per setting: mean NIS of the FK innovations, walking forward velocity estimate vs command, vertical velocity
jitter, tilt gap to the MTi-3 attitude, estimated biases (v = per log). Result 11-17 (7 logs 11-13..11-16):
accel 2.0 / contact 0.05 / fk 0.004 -> tilt gap 1.86 deg, vz std 0.064 m/s, |b_a| 0.042, |b_g| 0.022, vx 0.010 vs 0.007.
NIS stays < 1 for every setting (the slip / impact model is conservative); the filter is the same in sim and on the robot.
"""
import sys, glob, numpy as np, torch, itertools
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../source/humanoid_sim/humanoid_sim/tasks/direct/olaf'))
from inekf import InEKF
from scipy.spatial.transform import Rotation as Rot
files=sorted(glob.glob(sys.argv[1]+'/*_fk.npz'))
def run(D, an, cn, fn, gn=0.02):
    t,s,g,a,att=D['t'],torch.tensor(D['s']),torch.tensor(D['gyro']),torch.tensor(D['acc']),D['att']
    e=InEKF(1,'cpu',gyro_noise=gn,accel_noise=an,gyro_bias_noise=.002,accel_bias_noise=.02,contact_noise=cn,fk_noise=fn,
            init_std={"rot":.03,"vel":.05,"pos":.01,"gyro_bias":.03,"accel_bias":.2})
    R0=Rot.from_quat(att[0,[1,2,3,0]]).as_matrix()
    e.reset(torch.ones(1,dtype=torch.bool),torch.tensor(R0)[None],s[0:1])
    V=[];N=[];TL=[];C=[]
    for k in range(1,len(t)):
        dt=float(np.clip(t[k]-t[k-1],1e-3,0.05))
        e.propagate(g[k:k+1],a[k:k+1],dt)
        rel=(e.R[0]@s[k].T).T; c=(rel[:,2]-rel[:,2].min())<0.012
        e.correct(s[k:k+1],c[None]); N.append(e.nis[0].numpy()); C.append(c.numpy())
        V.append((e.R[0].T@e.v[0]).numpy())
        Ra=Rot.from_quat(att[k,[1,2,3,0]]).as_matrix(); TL.append(np.degrees(np.arccos(np.clip(Ra[2]@e.R[0,2].numpy(),-1,1))))
    V=np.array(V);N=np.array(N)
    cmd=D['cmd'][1:]; walk=np.abs(D['qd'][1:]).mean(1)>0.3
    return dict(nis=np.nanmean(N), vx_err=np.mean(V[walk,0])-np.mean(cmd[walk,0]) if walk.any() else np.nan,
                vx=np.mean(V[walk,0]) if walk.any() else np.nan, cmdx=np.mean(cmd[walk,0]) if walk.any() else np.nan,
                vz_std=V[:,2].std(), tilt=np.mean(TL), bg=e.bg[0].numpy(), ba=e.ba[0].numpy(), vfinal=np.linalg.norm(V[-60:].mean(0)))
Ds=[dict(np.load(f)) for f in files]
for an,cn,fn in itertools.product(*[list(map(float,x.split(','))) for x in sys.argv[2:5]]):
    rs=[run(D,an,cn,fn) for D in Ds]
    print(f"accel {an} contact {cn} fk {fn} | NIS {np.mean([r['nis'] for r in rs]):.2f} | walk vx est {np.nanmean([r['vx'] for r in rs]):.3f} vs cmd {np.nanmean([r['cmdx'] for r in rs]):.3f} | vz std {np.mean([r['vz_std'] for r in rs]):.3f} | tilt vs MTi {np.mean([r['tilt'] for r in rs]):.2f} deg | |ba| {np.mean([np.linalg.norm(r['ba']) for r in rs]):.3f} |bg| {np.mean([np.linalg.norm(r['bg']) for r in rs]):.4f}", flush=True)
    if len(sys.argv)>5:
        for f,r in zip(files,rs): print("   ",f.split('/')[-1][12:27], {k:(np.round(v,3) if not isinstance(v,float) else round(v,3)) for k,v in r.items()})
