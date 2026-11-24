"""Phase-binned walk reference table + standing grid -> one npz the env loads (arXiv 2512.16705, Sec. V-A).

Walk: for every PlaCo clip (gen_walk_clips.py) the periodic part is binned by gait phase (bins = frames per period) and
averaged over the cycles. A clip's path frame is the centred one-period moving average of the pelvis xy and yaw (the
path that the pelvis sways around); everything is stored relative to it, so the env can rebuild the full kinematic
target x_t = f(p_PF, g, phi) from its own integrated path frame (paper eq. 1-2):
    q (12) / qd (12)               joint position / velocity, env joint order (JOINTS)
    contact (2)                    left / right reference foot contact: lowest sole corner below CONTACT_HEIGHT (the env's
                                   own test, foot_contact_height) from reference/sole_heights.py; PlaCo support phase only
                                   if a clip has no _sole.npz
    off (3)                        pelvis x, y offset in the path frame, pelvis z (world, ground = 0)
    quat (4)                       pelvis orientation relative to the path frame (x, y, z, w)
    v (3) / w (3)                  pelvis linear / angular velocity in the path frame
Key per clip: its measured path velocity (vx, vy, wz) = the walking control input g it represents.

Stand: the IK grid (gen_stand_grid.py) re-ordered to the env joint order.

    /opt/isaac/IsaacLab-main/.venv/bin/python reference/build_reference.py --walk reference/clips/walk \
        --stand reference/clips/stand_grid.npz --out source/humanoid_sim/humanoid_sim/assets/reference/olaf_kb_reference.npz
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
CONTACT_HEIGHT = 0.008  # m, ArtiH1VelocityEnvCfg.foot_contact_height
LAYOUT = {"q": (0, 12), "qd": (12, 24), "contact": (24, 26), "off": (26, 29), "quat": (29, 33), "v": (33, 36), "w": (36, 39)}


def moving_average(x, n):
    k = np.ones(n) / n
    pad = n // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], n - 1 - pad, 0)], 0)
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def rot2(yaw, v):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)


def build_clip(path):
    d = np.load(path)
    fps = int(d["fps"])
    names = [str(x) for x in d["joint_names"]]
    perm = [names.index(j) for j in JOINTS]
    q = d["q"][:, perm]
    p, quat = d["root_pos"], d["root_quat"]
    rot = R.from_quat(quat)
    yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])
    period = float(d["period"])
    n = int(round(period * fps))
    clip_path = path
    path = moving_average(np.concatenate([p[:, :2], yaw[:, None]], 1), n)
    # finite differences (central) at fps
    dt = 1.0 / fps
    qd = np.gradient(q, dt, axis=0)
    v_w = np.gradient(p, dt, axis=0)
    w_w = np.zeros_like(v_w)
    rv = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
    w_w[1:-1] = rv
    w_w[0], w_w[-1] = rv[0], rv[-1]
    psi = path[:, 2]
    off_xy = rot2(-psi, p[:, :2] - path[:, :2])
    q_rel = (R.from_euler("Z", psi[:, None]).inv() * rot).as_quat()
    v_p = np.concatenate([rot2(-psi, v_w[:, :2]), v_w[:, 2:]], 1)
    w_p = np.concatenate([rot2(-psi, w_w[:, :2]), w_w[:, 2:]], 1)
    sole_file = clip_path[:-4] + "_sole.npz"
    contact = np.load(sole_file)["sole_low"] if os.path.exists(sole_file) else None
    rows = np.concatenate([q, qd, d["contact"] if contact is None else contact, off_xy, p[:, 2:3], q_rel, v_p, w_p], 1)
    # periodic part, away from the moving-average edges
    valid = d["periodic_valid"] > 0.5
    valid[: np.argmax(valid) + n] = False
    valid[-n:] = False
    phase = d["phase"] % 1.0
    b = np.minimum((phase * n).round().astype(int) % n, n - 1)
    table = np.zeros((n, rows.shape[1]))
    for k in range(n):
        m = valid & (b == k)
        if not m.any():
            raise RuntimeError(f"{clip_path}: phase bin {k} empty")
        table[k] = rows[m].mean(0)
    if contact is None:
        table[:, 24:26] = (table[:, 24:26] > 0.5).astype(float)
    else:  # phase-binned lowest sole-corner height -> contact flag
        support = np.zeros((n, 2))
        for k in range(n):
            support[k] = d["contact"][valid & (b == k)].mean(0)
        build_clip.agree.append(((table[:, 24:26] < CONTACT_HEIGHT) == (support > 0.5)).mean())
        table[:, 24:26] = (table[:, 24:26] < CONTACT_HEIGHT).astype(float)
    qn = table[:, 29:33]
    table[:, 29:33] = qn / np.linalg.norm(qn, axis=1, keepdims=True)
    T = valid.sum() * dt
    i0, i1 = np.argmax(valid), len(valid) - 1 - np.argmax(valid[::-1])
    key_v = rot2(-psi[i0:i0 + 1], (path[i1:i1 + 1, :2] - path[i0:i0 + 1, :2]))[0] / ((i1 - i0) * dt)
    # path velocity in the (rotating) path frame: average of the per-frame path-frame velocities
    pv = rot2(-psi[valid], np.gradient(path[:, :2], dt, axis=0)[valid]).mean(0)
    key = np.array([pv[0], pv[1], np.gradient(psi, dt)[valid].mean()])
    del key_v, T
    return key, table, period, d["step"]


build_clip.agree = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walk", required=True)
    p.add_argument("--stand", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    files = sorted((f for f in glob.glob(os.path.join(a.walk, "[0-9]*.npz")) if not f.endswith("_sole.npz")),
                   key=lambda f: int(os.path.basename(f)[:-4]))
    keys, tables, periods, steps = [], [], [], []
    for f in files:
        k, t, per, st = build_clip(f)
        keys.append(k)
        tables.append(t)
        periods.append(per)
        steps.append(st)
    periods = np.array(periods)
    if np.ptp(periods) > 1e-6:
        raise RuntimeError(f"clips have different periods {np.unique(periods)}")
    keys, tables = np.array(keys), np.array(tables, dtype=np.float32)

    s = np.load(a.stand)
    names = [str(x) for x in s["joint_names"]]
    perm = [names.index(j) for j in JOINTS]
    np.savez(
        a.out,
        joints=np.array(JOINTS), layout=np.array([f"{k}:{v[0]}:{v[1]}" for k, v in LAYOUT.items()]),
        walk_keys=keys.astype(np.float32), walk_table=tables, walk_period=periods[0], walk_steps=np.array(steps),
        stand_axes_dz=s["axes_dz"], stand_axes_roll=s["axes_roll"], stand_axes_pitch=s["axes_pitch"],
        stand_axes_yaw=s["axes_yaw"], stand_q=s["q"][..., perm].astype(np.float32),
        stand_root_pos=s["root_pos"].astype(np.float32), stand_root_quat=s["root_quat"].astype(np.float32),
    )
    print(f"walk: {len(files)} clips x {tables.shape[1]} bins, period {periods[0]:.3f} s")
    print(f"  path velocity coverage vx {keys[:, 0].min():+.3f}..{keys[:, 0].max():+.3f}  vy {keys[:, 1].min():+.3f}.."
          f"{keys[:, 1].max():+.3f}  wz {keys[:, 2].min():+.3f}..{keys[:, 2].max():+.3f}")
    print(f"  max |qd| per joint {np.abs(tables[:, :, 12:24]).max((0, 1)).round(1)}")
    print(f"  pelvis z {tables[:, :, 28].min():.4f}..{tables[:, :, 28].max():.4f}, |off xy| max "
          f"{np.abs(tables[:, :, 26:28]).max():.4f}")
    if build_clip.agree:
        print(f"  contact from sole heights (< {CONTACT_HEIGHT * 1000:.0f} mm) in {len(build_clip.agree)} clips; agreement with "
              f"the PlaCo support phase {np.mean(build_clip.agree):.3f}, stance fraction {tables[:, :, 24:26].mean():.3f}")
    print(f"stand: grid {s['q'].shape[:-1]} -> {a.out}")


if __name__ == "__main__":
    main()
