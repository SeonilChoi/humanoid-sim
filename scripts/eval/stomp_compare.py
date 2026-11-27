"""Touchdown impact (stomp) + torso shake, real robot vs sim replay, per foot (2026-11-27).
  python scripts/eval/stomp_compare.py --real data/real_logs/arti_h1_olaf_run_2026112*.csv --sim res_<tag>_raw.npz [...]
The sim file comes from walk_replay.py with RAW=1 (records acc / imp / fvz / qd since 11-27); its segments are the walk
segments of the replayed logs, in order. Both sides use the same 50 Hz definitions:
  shake: gyro rms (roll / pitch / yaw axis, body frame) over the walk segment after the first 0.5 s, before a fall
  touchdown: rising edge of the contact flag after >= 8 steps (0.16 s) of swing
    acc  = max |IMU specific force| over the touchdown step and the next 2, minus g (m/s^2)
    knee = max |knee qd| over the 2 steps before touchdown and the touchdown step (rad/s)
    stop = knee speed lost in the touchdown: max over steps -2..0 minus min over +1..+2 of the knee velocity (sign of the
           approach), rad/s. Leg-local, so it does not depend on which foot the pelvis IMU peak belongs to: on the 11-26
           logs the pelvis acc peak puts the right foot harder, the knee stop (and the operator's eye) the left
    (sim only) vz = foot sink speed just before touchdown (m/s), dv = impact dv rms of the touchdown step (cm/s)
Real contact is the controller estimator's flag, sim contact is true sole contact; 50 Hz under-samples impact peaks."""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--real", nargs="*", default=[]); ap.add_argument("--sim", nargs="*", default=[])
args = ap.parse_args()
SKIP, SW = 25, 8


def touchdowns(c, ok):
    td = np.flatnonzero(np.diff(c.astype(int)) == 1) + 1
    return np.array([i for i in td if SW <= i < len(c) - 3 and ok[i] and ok[i - SW] and c[i - SW:i].sum() <= 1], int)


def kstop(qd, i):
    v = qd[i - 2:i + 3]; v = v * np.sign(v[:3].mean() or 1.0)
    return v[:3].max() - v[3:].min()


def summarize(name, gyro, feet):
    r = np.sqrt(np.mean(np.square(gyro), 0)) if len(gyro) else np.full(3, np.nan)
    s = f"{name:28s} gyro rms r/p/y {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} |"
    for side in ("L", "R"):
        f = feet[side]; n = len(f["acc"])
        s += f" {side} n{n:4d} acc {np.median(f['acc']) if n else np.nan:5.2f} knee {np.median(f['knee']) if n else np.nan:4.2f} stop {np.median(f['stop']) if n else np.nan:4.2f}"
        if "vz" in f: s += f" vz {np.median(f['vz']) if n else np.nan:4.2f} dv {np.median(f['dv']) if n else np.nan:5.1f}"
        s += " |"
    a = [np.median(feet[k]["acc"]) if len(feet[k]["acc"]) else np.nan for k in "LR"]
    k = [np.median(feet[x]["stop"]) if len(feet[x]["stop"]) else np.nan for x in "LR"]
    print(s + f" L/R acc {a[0] / a[1]:.2f} stop {k[0] / k[1]:.2f}")


def real(f):
    d = pd.read_csv(f); w = d.olaf_walk.values
    ch = np.r_[0, np.flatnonzero(np.diff(w)) + 1, len(d)]
    gy, feet = [], {k: {"acc": [], "knee": [], "stop": []} for k in "LR"}
    g0 = np.median(np.sqrt(d.imu_ax**2 + d.imu_ay**2 + d.imu_az**2)[w == 0]) if (w == 0).any() else 9.81
    for a, b in zip(ch[:-1], ch[1:]):
        if w[a] != 1: continue
        x = d.iloc[a:b]
        qw, qx, qy, qz = [x[c].values for c in ("imu_qw", "imu_qx", "imu_qy", "imu_qz")]
        R = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))); P = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1)))
        fall = np.flatnonzero((np.abs(R) > 20) | (np.abs(P) > 20)); n = fall[0] - 25 if len(fall) else len(x)
        x = x.iloc[:n]
        if n <= SKIP + 25: continue
        ok = np.arange(n) >= SKIP
        gy.append(x[["imu_gx", "imu_gy", "imu_gz"]].values[ok])
        acc = np.sqrt(x.imu_ax**2 + x.imu_ay**2 + x.imu_az**2).values
        for k, s, S in (("L", "l", "left"), ("R", "r", "right")):
            kq = x[f"vel_{S}_knee"].values
            for i in touchdowns(x[f"contact_{s}"].values, ok):
                feet[k]["acc"].append(acc[i:i + 3].max() - g0); feet[k]["knee"].append(np.abs(kq[i - 2:i + 1]).max()); feet[k]["stop"].append(kstop(kq, i))
    summarize("real " + f.split("_run_")[-1][:-4], np.concatenate(gy) if gy else [], feet)


def sim(f):
    z = np.load(f); segs = sorted({int(k.split("_")[0][1:]) for k in z.files})
    by_log = {}
    for i in segs:
        if f"s{i}_acc" not in z.files: raise SystemExit(f"{f}: no acc record (re-run walk_replay.py with RAW=1, 11-27 or later)")
        by_log.setdefault(str(z[f"s{i}_log"]), []).append(i)
    names = list(z[f"s{segs[0]}_names"]); kn = [names.index("left_knee"), names.index("right_knee")]
    tot_g, tot_f = [], {k: {"acc": [], "knee": [], "stop": [], "vz": [], "dv": []} for k in "LR"}
    for log, ii in by_log.items():
        gy, feet = [], {k: {"acc": [], "knee": [], "stop": [], "vz": [], "dv": []} for k in "LR"}
        for i in ii:
            alive, gyro, c = z[f"s{i}_alive"], z[f"s{i}_gyro"], z[f"s{i}_contact"] > 0.5
            acc = np.linalg.norm(z[f"s{i}_acc"], axis=-1); imp, fvz, qd = z[f"s{i}_imp"], z[f"s{i}_fvz"], z[f"s{i}_qd"]
            T, N = alive.shape
            for n in range(N):
                ok = alive[:, n] & (np.arange(T) >= SKIP)
                if not alive[-1, n]: ok &= np.arange(T) < np.argmin(alive[:, n]) - 25  # cut 0.5 s before a fall, as real
                gy.append(gyro[ok, n])
                for j, k in enumerate("LR"):
                    for t in touchdowns(c[:, n, j], ok):
                        feet[k]["acc"].append(acc[t:t + 3, n].max() - 9.81); feet[k]["knee"].append(np.abs(qd[t - 2:t + 1, n, kn[j]]).max()); feet[k]["stop"].append(kstop(qd[:, n, kn[j]], t))
                        feet[k]["vz"].append(-fvz[t - 2:t, n, j].min()); feet[k]["dv"].append(np.sqrt(imp[t, n, j]))
        summarize(f"sim {f.split('res_')[-1][:-8]} {log}", np.concatenate(gy), feet)
        tot_g.append(np.concatenate(gy)); [tot_f[k][m].extend(feet[k][m]) for k in "LR" for m in feet[k]]
    summarize(f"sim {f.split('res_')[-1][:-8]} ALL", np.concatenate(tot_g), tot_f)


for f in args.real: real(f)
for f in args.sim: sim(f)
