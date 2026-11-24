"""Leg kinematics of the real ARTI-H1 KB logs for the offline InEKF check (Isaac Lab venv, pinocchio):

    python scripts/real2sim/fk_real_logs.py <logs>/arti_h1_vel_run_2026111[36]_*.csv

Writes <log>_fk.npz next to the working directory: soles relative to the IMU (pelvis axes; sole centre offset and IMU
mount as robot_olaf.yaml), raw gyro / accelerometer, attitude, command, joint velocities. The logs come from the
humanoid_controller git history (data/real_log_index.txt).
"""
import os, sys, glob, csv, numpy as np, pinocchio as pin
class DF:
    def __init__(s,f):
        r=list(csv.reader(open(f))); s.cols=r[0]; s.a=np.array([[float(x) if x not in ('','nan') else np.nan for x in row] for row in r[1:]])
        s.columns=s.cols
    def __len__(s): return len(s.a)
    def __getitem__(s,k):
        if isinstance(k,list): return V(s.a[:,[s.cols.index(c) for c in k]])
        return V(s.a[:,s.cols.index(k)])
    def __getattr__(s,k): return s[k]
class V:
    def __init__(s,a): s.values=a

from humanoid_sim.utils.motion_generator import robot_folder
urdf=os.path.join(robot_folder("arti_h1"), "arti_h1.urdf")
model=pin.buildModelFromUrdf(urdf); data=model.createData()
fl,fr=model.getFrameId("left_foot_link"),model.getFrameId("right_foot_link")
off=np.array([-0.026,0,-0.051]); imu=np.array([0.04688,0.0,0.34667])
names=[model.names[i] for i in range(1,model.njoints)]
print("root joint:", model.joints[1].shortname(), "nq", model.nq, names[:3])
for f in sys.argv[1:]:
    d=DF(f); q=np.zeros((len(d),model.nq))
    for i in range(1,model.njoints):
        q[:,model.joints[i].idx_q]=d[f"pos_{model.names[i]}"].values
    S=np.zeros((len(d),2,3))
    for k in range(len(d)):
        pin.framesForwardKinematics(model,data,q[k])
        for j,fid in enumerate((fl,fr)):
            M=data.oMf[fid]; S[k,j]=M.translation+M.rotation@off-imu
    out=f.replace('.csv','_fk.npz').split('/')[-1]
    np.savez(out,t=d.t.values,s=S,gyro=d[['imu_gx','imu_gy','imu_gz']].values,acc=d[['imu_ax','imu_ay','imu_az']].values,
             att=d[['att_qw','att_qx','att_qy','att_qz']].values,cmd=d[[c for c in d.columns if c.startswith('obs_')]].values[:,-3:],
             qd=d[[c for c in d.columns if c.startswith('vel_')]].values)
    print(out,len(d), "mean sole z", S[:,:,2].mean(0).round(3))
