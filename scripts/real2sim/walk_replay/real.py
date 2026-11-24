"""Real Olaf controller logs (humanoid_controller arti_h1_olaf_run_*.csv) -> walk segments for metrics.py."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from metrics import *
def load_real(f, seg_index=None):
    d = pd.read_csv(f)
    w = d.olaf_walk.values; ch = np.r_[0, np.flatnonzero(np.diff(w)) + 1, len(d)]
    segs = [(a, b) for a, b in zip(ch[:-1], ch[1:]) if w[a] == 1]
    out = []
    for a, b in segs:
        x = d.iloc[a:b]
        qw, qx, qy, qz = [x[c].values for c in ("imu_qw", "imu_qx", "imu_qy", "imu_qz")]
        R = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))); P = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1)))
        fall = np.flatnonzero((np.abs(R) > 20) | (np.abs(P) > 20)); n = fall[0] - 25 if len(fall) else len(x)  # cut 0.5 s before a fall
        x = x.iloc[:n]; R = R[:n]; P = P[:n]
        names = [f"{s}_{j}" for j in JT for s in ("left", "right")]
        D = dict(t=x.t.values, g=x[["g0", "g1", "g2"]].values, gyro=x[["imu_gx", "imu_gy", "imu_gz"]].values, rp=np.stack([R, P], 1),
                 qdes={k: x[f"q_des_{k}"].values for k in names}, q={k: x[f"pos_{k}"].values for k in names},
                 tau={k: x[f"tau_{k}"].values for k in names}, act={k: x[f"action_{k}"].values for k in names},
                 contact=x[["contact_l", "contact_r"]].values, fell=bool(len(fall)), g_full=d.iloc[a:b][["g0", "g1", "g2"]].values)
        out.append(D)
    return out
