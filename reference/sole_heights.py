"""Lowest sole-corner height of every walk clip frame, for reference contact flags that use the env's own contact test.

The env counts a foot in contact when its lowest sole corner (sole centre offset +- foot_sole_half_length / width in the
foot-link frame) is below foot_contact_height (8 mm). PlaCo's support phase switches earlier (the swing foot is still
within a few mm of the ground at lift-off / touchdown), which capped the contact match at 0.90 even for exact tracking.
This writes <clip>_sole.npz (sole_low (T, 2): left, right, m) next to each clip; build_reference.py thresholds the
phase-binned heights.

    python reference/sole_heights.py reference/clips/walk
"""

import glob
import os
import sys

import numpy as np
import pinocchio as pin

from humanoid_sim.utils.motion_generator import robot_folder  # noqa: E402

URDF = os.path.join(robot_folder("arti_h1"), "arti_h1.urdf")
SOLE_OFFSET = np.array([-0.026, 0.0, -0.051])  # ARTI_H1_*_FOOT_OFFSET_FROM_FOOT_LINK
HALF_LENGTH, HALF_WIDTH = 0.057, 0.045  # foot_sole_half_length / foot_sole_half_width
CORNERS = np.array([SOLE_OFFSET + [sx * HALF_LENGTH, sy * HALF_WIDTH, 0.0] for sx in (-1, 1) for sy in (-1, 1)])


def main():
    model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    feet = [model.getFrameId("left_foot_link"), model.getFrameId("right_foot_link")]
    files = sorted(f for f in glob.glob(os.path.join(sys.argv[1], "[0-9]*.npz")) if not f.endswith("_sole.npz"))
    for f in files:
        d = np.load(f)
        names = [str(x) for x in d["joint_names"]]
        idx = [model.joints[model.getJointId(n)].idx_q for n in names]
        low = np.zeros((len(d["q"]), 2))
        for k in range(len(d["q"])):
            q = np.zeros(model.nq)
            q[:3], q[3:7] = d["root_pos"][k], d["root_quat"][k]  # free flyer: xyz, quaternion x y z w
            q[idx] = d["q"][k]
            pin.framesForwardKinematics(model, data, q)
            for j, fid in enumerate(feet):
                M = data.oMf[fid]
                low[k, j] = (M.translation[2] + (CORNERS @ M.rotation.T)[:, 2]).min()
        np.savez(f[:-4] + "_sole.npz", sole_low=low)
    print(f"{len(files)} clips: sole heights written")


if __name__ == "__main__":
    main()
