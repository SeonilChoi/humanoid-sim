"""Per-stride fall precursors, same definitions for the real logs and sim replays (walk_replay.py with RAW=1).
  python scripts/real2sim/walk_replay/fall_analysis.py <res_tag_raw.npz ...>   (real logs are read from the npz 'log' keys)
Per stride (one walk period, 0.8 s, cut at phase 0): torso pitch / roll range, |gz| peak, hip-yaw action range, stance foot
yaw slip (estimated as torso yaw - stance hip yaw over the single-support window: identical estimator real and sim; the sim
true foot yaw is printed alongside to validate it). Distributions over the strides that end > 2 s before a fall
('steady') and over the last 3 strides before each fall ('pre-fall')."""
import sys, numpy as np, pandas as pd
CTRL = "data/real_logs/arti_h1_olaf_run_20261121_{}.csv"
FEAT = ["pitch_rng", "roll_rng", "gz_pk", "gx_pk", "hy_act_rng", "slip_est", "slip_true", "knee_act_sat"]


def strides(ph, yaw, P, R, gz, gx, hyL, hyR, ahyL, ahyR, aknL, aknR, fyL=None, fyR=None, cut=None):
    """ph phase (T,), angles rad except P/R deg. Returns (start index, feature dict) per full stride before cut."""
    T = len(ph) if cut is None else cut
    st = np.flatnonzero(np.diff(ph[:T]) < -0.5) + 1  # phase wrap
    out = []
    for a, b in zip(st[:-1], st[1:]):
        f = {"pitch_rng": np.ptp(P[a:b]), "roll_rng": np.ptp(R[a:b]), "gz_pk": np.abs(gz[a:b]).max(), "gx_pk": np.abs(gx[a:b]).max(),
             "hy_act_rng": max(np.ptp(ahyL[a:b]), np.ptp(ahyR[a:b])),
             "knee_act_sat": float(np.mean((np.abs(aknL[a:b]) > 0.95) | (np.abs(aknR[a:b]) > 0.95)))}
        sl, slt = [], []
        for lo, hi, hy, fy in ((0.08, 0.36, hyL, fyL), (0.58, 0.86, hyR, fyR)):
            m = np.flatnonzero((ph[a:b] >= lo) & (ph[a:b] < hi)) + a
            if len(m) < 6: continue
            e = np.unwrap(yaw[m]) - hy[m]
            sl.append(abs(e[-1] - e[0]))
            if fy is not None: slt.append(abs(np.unwrap(fy[m])[-1] - np.unwrap(fy[m])[0]))
        f["slip_est"] = np.degrees(max(sl)) if sl else np.nan
        f["slip_true"] = np.degrees(max(slt)) if slt else np.nan
        out.append((a, f))
    return out


def real_series(log):
    d = pd.read_csv(CTRL.format(log)); d = d[d.olaf_walk == 1].reset_index(drop=True)
    w = np.r_[0, np.flatnonzero(np.diff(d.t.values) > 0.1) + 1, len(d)]
    res = []
    for a, b in zip(w[:-1], w[1:]):
        x = d.iloc[a:b]
        qw, qx, qy, qz = [x[c].values for c in ("imu_qw", "imu_qx", "imu_qy", "imu_qz")]
        R = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))); P = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        fall = np.flatnonzero((np.abs(R) > 20) | (np.abs(P) > 20))
        res.append(dict(ph=x.phase.values, yaw=yaw, P=P, R=R, gz=x.imu_gz.values, gx=x.imu_gx.values,
                        hyL=x.pos_left_hip_yaw.values, hyR=x.pos_right_hip_yaw.values, ahyL=x.action_left_hip_yaw.values,
                        ahyR=x.action_right_hip_yaw.values, aknL=x.action_left_knee.values, aknR=x.action_right_knee.values,
                        g=x[["g0", "g1", "g2"]].values, fall=int(fall[0]) if len(fall) else None))
    return res


def sim_series(z, i):
    k = lambda n: z[f"s{i}_{n}"]
    names = list(k("names")); pn = list(k("pol_names")); alive = k("alive"); T, N = alive.shape
    out = []
    for n in range(N):
        fall = int(np.argmin(alive[:, n])) if not alive[-1, n] else None
        q = k("q")[:, n]; a = k("act")[:, n]
        out.append(dict(ph=k("phase")[:, n], yaw=k("yaw")[:, n], P=k("rp")[:, n, 1], R=k("rp")[:, n, 0], gz=k("gyro")[:, n, 2], gx=k("gyro")[:, n, 0],
                        hyL=q[:, names.index("left_hip_yaw")], hyR=q[:, names.index("right_hip_yaw")],
                        ahyL=a[:, pn.index("left_hip_yaw")], ahyR=a[:, pn.index("right_hip_yaw")],
                        aknL=a[:, pn.index("left_knee")], aknR=a[:, pn.index("right_knee")],
                        fyL=k("fyaw")[:, n, 0], fyR=k("fyaw")[:, n, 1], g=k("g"), fall=fall))
    return out


def collect(series):
    steady, pre = [], []
    for s in series:
        cut = s["fall"]
        st = strides(s["ph"], s["yaw"], s["P"], s["R"], s["gz"], s["gx"], s["hyL"], s["hyR"], s["ahyL"], s["ahyR"], s["aknL"], s["aknR"],
                     s.get("fyL"), s.get("fyR"), cut)
        turn = np.abs(s["g"][:, 2]) > 0.15
        for j, (a, f) in enumerate(st):
            if cut is not None and a > cut - 100: pre.append(f) if j >= len(st) - 3 else None
            elif not turn[min(a, len(turn) - 1)]: steady.append(f)
    return steady, pre


def table(groups):
    print(f"{'':16s}" + "".join(f"{g:>26s}" for g in groups))
    for ft in FEAT:
        row = f"{ft:16s}"
        for g, fs in groups.items():
            v = np.array([f[ft] for f in fs if not np.isnan(f[ft])])
            row += f"{'-':>26s}" if len(v) == 0 else f"  {np.percentile(v, 50):6.2f} {np.percentile(v, 90):6.2f} {np.percentile(v, 99):6.2f} n{len(v):<4d}"
        print(row)
    print("(p50 p90 p99, n strides)")


if __name__ == "__main__":
    groups = {}
    logs = ["095329", "095641", "100031"]
    rs = [s for lg in logs for s in real_series(lg)]
    groups["real steady"], groups["real pre-fall"] = collect(rs)
    for p in sys.argv[1:]:
        z = np.load(p, allow_pickle=True); n_seg = len({k.split("_")[0] for k in z.files})
        ss = [s for i in range(n_seg) for s in sim_series(z, i)]
        tag = p.split("res_")[-1].replace("_raw.npz", "")
        groups[f"{tag} steady"], groups[f"{tag} pre-fall"] = collect(ss)
        print(tag, "sim falls", sum(s["fall"] is not None for s in ss), "/", len(ss))
    table(groups)
