"""Per-segment metrics shared by real logs and sim replays. Input: dict of arrays at 50 Hz:
t, g (T,3), gyro (T,3) body, rp (T,2) deg, qdes/q/tau (T,J) dict by joint name, contact (T,2), alive mask (T,) optional."""
import numpy as np
JT = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
def segments(g, dt=0.02):
    turn = np.abs(g[:, 2]) > 0.15
    fwd = (np.abs(g[:, 0]) > 0.04) & ~turn
    still = (np.abs(g).max(1) < 0.02)
    post = np.zeros_like(turn)
    e = np.flatnonzero(turn[:-1] & ~turn[1:])
    for i in e: post[i + 1:i + 1 + int(2.5 / dt)] = True
    post &= ~turn
    return {"inplace": still & ~post, "fwd": fwd & ~post, "turn": turn, "postturn": post}
def metrics(D, mask):
    out = {}
    if mask.sum() < 25: return out
    gy = D["gyro"][mask]
    out["gyro_std_x"], out["gyro_std_y"], out["gyro_std_z"] = gy.std(0)
    out["wz_mean"] = gy[:, 2].mean()
    out["roll_std"], out["pitch_std"] = D["rp"][mask].std(0)
    out["pitch_mean"] = D["rp"][mask, 1].mean(); out["roll_mean"] = D["rp"][mask, 0].mean()
    for j in JT:
        e = np.concatenate([np.abs(D["qdes"][f"{s}_{j}"][mask] - D["q"][f"{s}_{j}"][mask]) for s in ("left", "right")])
        out[f"err_{j}"] = np.degrees(e.mean())
        tau = np.concatenate([D["tau"][f"{s}_{j}"][mask] for s in ("left", "right")])
        out[f"tau_{j}"] = np.sqrt((tau ** 2).mean())
        a = np.concatenate([D["act"][f"{s}_{j}"][mask] for s in ("left", "right")])
        out[f"sat_{j}"] = (np.abs(a) > 0.95).mean()
    from scipy.signal import welch
    g = D["gyro"].copy(); g[~mask] = 0.0  # band power over the masked samples (zeros elsewhere, rescaled)
    for ax, n in enumerate("xyz"):
        fr, Pw = welch(g[:, ax], fs=50, nperseg=128); sc = len(mask) / max(mask.sum(), 1)
        for a, b in ((0, 3), (3, 6), (6, 12), (12, 25)):
            out[f"gb_{n}_{a}_{b}"] = float(np.sqrt(Pw[(fr >= a) & (fr < b)].sum() * (fr[1] - fr[0]) * sc))
    out["contact_both"] = (D["contact"][mask].sum(1) == 2).mean()
    return out
