"""Walk reference clips for the Olaf-style imitation policy (arXiv 2512.16705, Sec. V-A).

The paper designs stylised walk cycles with a gait generation tool ([38], heel-toe motion) and maps the control input
g = path velocity to a kinematic target f(p_PF, g, phi). We do the same with PlaCo (humanoid_sim.utils.motion_generator
MotionEngine, KB stance: walk_com_height 0.29725 -> hip 0.336 / knee -0.671 / ankle -0.336 rad, pelvis z -0.0264),
plus a heel-toe roll that the plain PlaCo walk does not have:
  * late double support: the next swing foot peels off about its toe (heel-off, toes down to -toe_off),
  * swing: the foot pitch goes -toe_off -> +heel_strike while the pivot moves toe -> heel (continuous at both ends),
  * early double support: the landed foot rolls down about its heel (heel strike -> flat).
Angles scale with the forward stride (dx / walk_max_dx_forward); backward / sideways / turning steps stay flat.

Run with the Isaac Lab venv (PlaCo installed there):
    python reference/gen_walk_clips.py --out reference/clips/walk --jobs 12
One <index>.npz per command grid point (dx, dy, dth per step) + grid.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from humanoid_sim.utils.motion_generator import gait_config_path, make_engine, robot_folder  # noqa: E402

FPS = 60
TICK = 0.001


def run_clip(job):
    idx, dx, dy, dth, gait, out_dir = job
    from scipy.spatial.transform import Rotation as R

    try:
        me = make_engine(gait)
        names = list(me.get_angles().keys())
        me.set_trajectory(dx, dy, dth)
        n_frames = int(round(gait["duration"] * FPS))
        q, root_p, root_q, contact, phase, valid, lf, rf = [], [], [], [], [], [], [], []
        next_rec = 1.0 / FPS
        with contextlib.redirect_stdout(io.StringIO()):
            while len(q) < n_frames:
                me.tick(TICK)
                if me.t + 1e-9 < next_rec:
                    continue
                next_rec += 1.0 / FPS
                T = np.asarray(me.robot.get_T_world_fbase())
                q.append(list(me.get_angles().values()))
                root_p.append(T[:3, 3].copy())
                root_q.append(R.from_matrix(T[:3, :3]).as_quat())  # x y z w
                contact.append(me.get_current_support_phase())
                phase.append(me.get_gait_phase())
                valid.append(float(me.is_periodic_phase_valid()))
                lf.append(np.asarray(me.get_T_world_left_contact())[:3, 3].copy())
                rf.append(np.asarray(me.get_T_world_right_contact())[:3, 3].copy())
        np.savez(os.path.join(out_dir, f"{idx}.npz"), joint_names=np.array(names), fps=FPS, q=np.array(q),
                 root_pos=np.array(root_p), root_quat=np.array(root_q), contact=np.array(contact, dtype=np.float32),
                 phase=np.array(phase), periodic_valid=np.array(valid), left_foot=np.array(lf), right_foot=np.array(rf),
                 step=np.array([dx, dy, dth]), period=me.period)
        return idx, None
    except Exception as e:  # noqa: BLE001  (PlaCo QP failures for infeasible steps)
        return idx, repr(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gait", default=gait_config_path("arti_h1", "gait_olaf_kb.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--only", type=int, nargs="*", default=None, help="indices to (re)generate")
    a = p.parse_args()
    gait = json.load(open(a.gait))
    gait["robot_folder"] = robot_folder(gait["robot"])
    g = gait["grid"]
    grid = lambda r: np.round(np.arange(r[0], r[1] + r[2] / 2, r[2]), 5)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    table, jobs = {}, []
    for i, (dx, dy, dth) in enumerate((x, y, t) for x in grid(g["dx"]) for y in grid(g["dy"]) for t in grid(g["dth"])):
        table[i] = [float(dx), float(dy), float(dth)]
        if a.only is None or i in a.only:
            jobs.append((i, float(dx), float(dy), float(dth), gait, a.out))
    json.dump({"gait": gait, "clips": table}, open(os.path.join(a.out, "grid.json"), "w"), indent=1)
    print(f"{len(jobs)} clips -> {a.out}", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for idx, err in ex.map(run_clip, jobs):
            if err:
                failed.append(idx)
                print(f"clip {idx} {table[idx]} FAILED: {err}", flush=True)
    print(f"done, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
