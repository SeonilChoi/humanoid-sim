"""In-place yaw rate of a walk_replay raw result (res_<tag>_raw.npz), 4 s windows: true yaw, estimator yaw, b_g,z.
    python scripts/eval/replay_yaw.py scripts/real2sim/walk_replay/res_<tag>_raw.npz [t0_s]
t0_s = log time of the first replay frame (default 34.7, log 20261123_132540)."""
import sys

import numpy as np

d = np.degrees
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 34.7
z = np.load(sys.argv[1])
Y, al = np.unwrap(z["s0_yaw"], axis=0), z["s0_alive"]
E = np.unwrap(z["s0_eyaw"], axis=0) if "s0_eyaw" in z.files else None
for a in range(0, Y.shape[0] - 200, 200):
    b = a + 200
    ok = al[b]
    r = d(Y[b, ok] - Y[a, ok]) / 4.0
    s = f"t {a * 0.02 + t0:6.1f}  true {r.mean():+5.1f} (p10 {np.percentile(r, 10):+5.1f} p90 {np.percentile(r, 90):+5.1f}) deg/s"
    if E is not None:
        s += f"  est {d(E[b, ok] - E[a, ok]).mean() / 4:+5.1f}  b_g,z {d(z['s0_bgz'][b, ok]).mean():+4.1f}"
    print(s, f" alive {ok.sum()}")
