"""MTi-3 accelerometer statistics of the real ARTI-H1 logs (60 Hz controller rows), for the sim accelerometer model:

    python scripts/real2sim/accel_real_logs.py <logs dir>

still (observe, policy off): std, high-pass std, |a|; KB walking (11-13..11-16): high-pass std on walking windows
(mean |joint velocity| > 0.3 rad/s over 1 s), p99 of | |a| - g |, high-pass std when still under the policy.
High-pass = residual from a 5-sample centred moving average (~12 Hz at 60 Hz).
Result 11-17: still sensor 0.005-0.02 m/s^2, still under a policy 0.02-0.09, walking (1.0, 1.45, 4.0) m/s^2, p99 15 m/s^2.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd


def hf(a, k=5):
    ker = np.ones(k) / k
    sm = np.stack([np.convolve(a[:, i], ker, "same") for i in range(3)], 1)
    return (a - sm)[k:-k]


root = sys.argv[1]
print("== still (observe, policy off)")
for f in ["arti_h1_run_20261028_151912.csv", "arti_h1_run_20261029_132143.csv", "arti_h1_run_20261029_133723.csv"]:
    d = pd.read_csv(os.path.join(root, f))
    a = d[["imu_ax", "imu_ay", "imu_az"]].values
    w = 1200
    s0 = min(range(0, len(a) - w, 60), key=lambda s: a[s:s + w].std(0).sum())
    seg = a[s0:s0 + w]
    print(f, "std", seg.std(0).round(4), "hf std", hf(seg).std(0).round(4), "|a| %.4f" % np.linalg.norm(seg, axis=1).mean())
print("== KB walking / standing under the policy")
for f in sorted(glob.glob(os.path.join(root, "arti_h1_vel_run_2026111[36]_*.csv"))):
    d = pd.read_csv(f)
    a = d[["imu_ax", "imu_ay", "imu_az"]].values
    qd = np.abs(d[[c for c in d.columns if c.startswith("vel_")]].values).mean(1)
    mv = pd.Series(qd).rolling(60, center=True).mean().values
    walk, still = mv > 0.3, mv < 0.05
    r = hf(a)
    print(os.path.basename(f), "walk frac %.2f" % walk.mean(), "hf std walk", r[walk[5:-5]].std(0).round(3),
          "p99 |a|-g %.2f" % np.percentile(np.abs(np.linalg.norm(a, axis=1) - 9.8), 99),
          "still hf", r[still[5:-5]].std(0).round(3) if still.sum() > 200 else None)
