"""Series-compliance signature (2026-11-22), same statistic for the real logs and sim replays (walk_replay.py RAW=1):
in single support, the torso pitch / roll change over 4 control steps vs the rigid leg kinematics of the encoder angles
(weights W fitted on a RIGID sim replay), then the compliance c (rad/Nm) that explains the residual with the joint torques
(link = encoder - c tau). Real 11-21: corr 0.28 / -0.35 -> 0.89 / 0.91 with c hip_roll 0.021, ankle_pitch 0.016.
  python scripts/real2sim/walk_replay/compliance_fit.py <rigid_raw.npz> [other_raw.npz ...]"""
import sys, numpy as np, pandas as pd
CTRL = "data/real_logs/arti_h1_olaf_run_20261121_{}.csv"
CH = {"pitch": (("hip_pitch", "knee", "ankle_pitch"), 1), "roll": (("hip_roll", "ankle_roll"), 0)}
SW = {"left": (0.10, 0.34), "right": (0.60, 0.84)}


def windows(ph, a, b):
    m = (ph >= a) & (ph < b); idx = np.flatnonzero(m); br = np.flatnonzero(np.diff(idx) > 1)
    return [s for s in np.split(idx, br + 1) if len(s) >= 8]


def sim_sets(p):
    z = np.load(p, allow_pickle=True); names = list(z["s0_names"]); out = []
    for i in range(len({k.split("_")[0] for k in z.files})):
        q, rp, ph, al = z[f"s{i}_q"], np.radians(z[f"s{i}_rp"]), z[f"s{i}_phase"], z[f"s{i}_alive"]
        tau = z[f"s{i}_taul"] if f"s{i}_taul" in z.files else np.zeros_like(q)
        for n in range(q.shape[1]):
            ok = al[:, n] & (np.abs(rp[:, n]).max(-1) < np.radians(15))
            out.append(dict(q=q[:, n], tau=tau[:, n], ang=rp[:, n], ph=ph[:, n], ok=ok))
    return names, out


def real_sets(names):
    out = []
    for f in ("095329", "095641", "100031"):
        d = pd.read_csv(CTRL.format(f)); d = d[d.olaf_walk == 1]
        qw, qx, qy, qz = [d[c].values for c in ("imu_qw", "imu_qx", "imu_qy", "imu_qz")]
        R = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)); P = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        out.append(dict(q=d[[f"pos_{n}" for n in names]].values, tau=d[[f"tau_{n}" for n in names]].values, ang=np.stack([R, P], -1),
                        ph=d.phase.values, ok=np.maximum(np.abs(R), np.abs(P)) < np.radians(15)))
    return out


def samples(sets, names, ch, ax, side):
    a, b = SW[side]; jj = [names.index(f"{side}_{j}") for j in ch]; X, Y, Tq = [], [], []
    for s in sets:
        for w in windows(s["ph"], a, b):
            if not s["ok"][w].all(): continue
            for k in range(0, len(w) - 4, 4):
                i0, i1 = w[k], w[k + 4]
                X.append(s["q"][i1, jj] - s["q"][i0, jj]); Y.append(s["ang"][i1, ax] - s["ang"][i0, ax]); Tq.append(s["tau"][i1, jj] - s["tau"][i0, jj])
    return np.array(X), np.array(Y), np.array(Tq)


if __name__ == "__main__":
    names, rigid = sim_sets(sys.argv[1])
    W = {(c, sd): np.linalg.lstsq(*samples(rigid, names, CH[c][0], CH[c][1], sd)[:2], rcond=None)[0] for c in CH for sd in SW}
    data = [("real", real_sets(names))] + [(p.split("res_")[-1].replace("_raw.npz", ""), sim_sets(p)[1]) for p in sys.argv[1:]]
    for c, (ch, ax) in CH.items():
        print(f"== {c} chain {ch}")
        for nm, sets in data:
            A, y, base = [], [], []
            for sd in SW:
                X, Y, Tq = samples(sets, names, ch, ax, sd)
                y.append(Y - X @ W[(c, sd)]); base.append(Y); A.append(-W[(c, sd)] * Tq)
            A, y, base = np.concatenate(A), np.concatenate(y), np.concatenate(base)
            cc, *_ = np.linalg.lstsq(A, y, rcond=None); r = y - A @ cc
            print(f"   {nm:14s} n {len(y):6d} | corr(torso, rigid kin) {np.corrcoef(base, base - y)[0, 1]:+.2f} -> with c {np.corrcoef(base, base - r)[0, 1]:+.2f} "
                  f"| resid {np.degrees(y.std()):.2f} -> {np.degrees(r.std()):.2f} deg | c " + " ".join(f"{j}={v:.4f}" for j, v in zip(ch, cc)))
