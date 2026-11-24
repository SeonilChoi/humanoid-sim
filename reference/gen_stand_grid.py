"""Standing reference for the Olaf-style standing policy (arXiv 2512.16705, eq. 2: g = (torso orientation, height)).

Olaf's standing control input is (neck target, torso orientation theta, torso height p_z); ARTI-H1 has no neck / upper
body, so g = (dz, roll, pitch, yaw) of the trunk relative to the nominal KB stance, with both feet fixed. For every grid
point PlaCo IK (feet frame tasks hard-weighted, trunk frame task) gives the joint angles; the achieved pelvis pose
(relative to the feet midpoint frame) is stored with them, so the reference stays kinematically consistent when the IK
saturates at a joint limit. The env interpolates the grid multilinearly.

    python reference/gen_stand_grid.py --out reference/clips/stand_grid.npz
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

from humanoid_sim.utils.motion_generator import gait_config_path, make_engine, robot_folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gait", default=gait_config_path("arti_h1", "gait_olaf_kb.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--dz", type=float, nargs=3, default=[-0.03, 0.015, 0.015])
    p.add_argument("--roll", type=float, nargs=3, default=[-0.10, 0.10, 0.05])
    p.add_argument("--pitch", type=float, nargs=3, default=[-0.15, 0.15, 0.075])
    p.add_argument("--yaw", type=float, nargs=3, default=[-0.30, 0.30, 0.15])
    p.add_argument("--iters", type=int, default=400)
    a = p.parse_args()
    from scipy.spatial.transform import Rotation as R

    gait = json.load(open(a.gait))
    gait["robot_folder"] = robot_folder(gait["robot"])
    gait["heel_strike_deg"] = gait["toe_off_deg"] = 0.0
    me = make_engine(gait)
    me.set_trajectory(0.0, 0.0, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(500):  # settle into the walk_com_height (KB) stance first; the URDF starts at q = 0
            me.tick(0.001)
    me._solver.enable_velocity_limits(False)
    robot = me.robot
    names = list(me.get_angles().keys())
    for side in ("left", "right"):
        me._generic_tasks[f"{side}_foot"].configure(f"{side}_foot", "hard", 1.0)
    T_trunk0 = np.asarray(robot.get_T_world_frame("trunk")).copy()
    q0 = robot.state.q.copy()
    Tl, Tr = np.asarray(me.get_T_world_left_contact()), np.asarray(me.get_T_world_right_contact())
    mid = 0.5 * (Tl[:3, 3] + Tr[:3, 3])
    mid[2] = 0.0

    axes = [np.round(np.arange(r[0], r[1] + r[2] / 2, r[2]), 5) for r in (a.dz, a.roll, a.pitch, a.yaw)]
    shape = tuple(len(x) for x in axes)
    Q = np.zeros(shape + (len(names),))
    RP = np.zeros(shape + (3,))
    RQ = np.zeros(shape + (4,))
    err = np.zeros(shape + (2,))
    for idx in np.ndindex(shape):
        dz, roll, pitch, yaw = (axes[k][i] for k, i in enumerate(idx))
        T = T_trunk0.copy()
        T[2, 3] += dz
        T[:3, :3] = R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix() @ T_trunk0[:3, :3]
        me._generic_tasks["trunk"].T_world_frame = T
        robot.state.q = q0.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(a.iters):
                robot.update_kinematics()
                me._solver.solve(True)
        robot.update_kinematics()
        Ta = np.asarray(robot.get_T_world_frame("trunk"))
        Tb = np.asarray(robot.get_T_world_fbase())
        ang = R.from_matrix(Ta[:3, :3] @ T[:3, :3].T).magnitude()
        err[idx] = (np.linalg.norm(Ta[:3, 3] - T[:3, 3]), ang)
        Q[idx] = list(me.get_angles().values())
        RP[idx] = Tb[:3, 3] - mid
        RQ[idx] = R.from_matrix(Tb[:3, :3]).as_quat()  # x y z w, feet-midpoint frame = world here
    np.savez(a.out, joint_names=np.array(names), axes_dz=axes[0], axes_roll=axes[1], axes_pitch=axes[2], axes_yaw=axes[3],
             q=Q, root_pos=RP, root_quat=RQ, ik_err=err)
    print(f"grid {shape} -> {a.out}; IK trunk error pos max {err[..., 0].max() * 1000:.1f} mm, "
          f"ang max {np.rad2deg(err[..., 1].max()):.1f} deg; >2 mm / >1 deg at {(err[..., 0] > 0.002).sum()} / "
          f"{(err[..., 1] > np.deg2rad(1)).sum()} of {np.prod(shape)} points")
    c = tuple(int(np.argmin(np.abs(x))) for x in axes)
    print("nominal q", dict(zip(names, np.round(Q[c], 4))), "root z", round(RP[c][2], 4))


if __name__ == "__main__":
    main()
