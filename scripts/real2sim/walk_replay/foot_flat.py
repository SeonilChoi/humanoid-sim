"""Kinematic consistency of the real logs (2026-11-22): put the sim robot in the logged pose (MTi-3 attitude + encoder joint
angles) at double-support instants and measure the sole tilt (roll / pitch of each foot vs the ground) and the sole height
difference. On a flat floor both soles are flat in double support, so a consistent tilt = IMU mounting offset or joint zero
error (ankle) in the real robot as the policy sees it.
  python scripts/real2sim/walk_replay/foot_flat.py --headless [--logs ...]"""
import argparse, os
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512)
ap.add_argument("--logs", nargs="+", default=[f"data/real_logs/arti_h1_olaf_run_20261121_{x}.csv"
                                              for x in ("095329", "095641", "100031")])
AppLauncher.add_app_launcher_args(ap); args, _ = ap.parse_known_args(); app = AppLauncher(args)
import gymnasium as gym, torch, numpy as np, pandas as pd
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import resolve_task_config
import humanoid_sim.tasks  # noqa

rows = []
for f in args.logs:
    d = pd.read_csv(f)
    for mode, m in (("walk_ds", (d.olaf_walk == 1) & (d.contact_l == 1) & (d.contact_r == 1) & (((d.phase - 0.47).abs() < 0.03) | ((d.phase - 0.97).abs() < 0.03))),
                    ("stand", (d.olaf_walk == 0) & (d.contact_l == 1) & (d.contact_r == 1))):
        x = d[m]
        if len(x): rows.append(x.sample(min(len(x), args.n // 6), random_state=0).assign(mode=mode))
X = pd.concat(rows).reset_index(drop=True)
cfg = resolve_task_config("HumanoidSim-Olaf-Walk-Play-v0", None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = len(X); cfg.events.push_robot = None
env = gym.make("HumanoidSim-Olaf-Walk-Play-v0", cfg=cfg); u = env.unwrapped; dev = u.device
env.reset()
names = list(u.robot.data.joint_names)
T = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev)
quat = T(X[["imu_qx", "imu_qy", "imu_qz", "imu_qw"]].values); quat = quat / quat.norm(dim=-1, keepdim=True)
q = T(X[[f"pos_{n}" for n in names]].values)
if os.environ.get("COMPL", "1") == "1":  # link = encoder - c * tau (series compliance, cfg.joint_series_compliance)
    cb = u.compl_base.clone()
    if os.environ.get("COMPL_SET"):  # "hip_roll=0.01,ankle_roll=0.02": override per joint type (rad/Nm)
        for kv in os.environ["COMPL_SET"].split(","):
            jt, v = kv.split("=")
            for i, nm in enumerate(names):
                if jt in nm: cb[0, i] = float(v)
    q = q - cb * T(X[[f"tau_{n}" for n in names]].values)
ids = torch.arange(len(X), device=dev)
pose = torch.cat((u.scene.env_origins + torch.tensor([0, 0, 1.0], device=dev), quat), -1)
u.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
u.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(len(X), 6, device=dev), env_ids=ids)
u.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
u.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
u.sim.forward(); u.scene.update(0.0)
DQ = os.environ.get("DQ")  # e.g. "left_ankle_pitch=0.05,right_ankle_roll=0.03": joint angle added (rad), for sensitivities
if DQ:
    for kv in DQ.split(","):
        nm, v = kv.split("="); q[:, names.index(nm)] += float(v)
    u.robot.write_joint_position_to_sim_index(position=q, env_ids=ids); u.sim.forward(); u.scene.update(0.0)
out = {}
for side, bi in zip(("L", "R"), u.foot_body_indices):
    Rf = matrix_from_quat(u.robot.data.body_quat_w.torch[:, bi])
    # sole normal (foot z axis) in the torso heading frame: pitch tilt > 0 = sole normal leans forward (toe down)
    n = Rf[:, :, 2]
    from humanoid_sim.tasks.direct.olaf.olaf_env import _yaw_of
    yaw = _yaw_of(quat); c, s_ = torch.cos(yaw), torch.sin(yaw)
    nx, ny = c * n[:, 0] + s_ * n[:, 1], -s_ * n[:, 0] + c * n[:, 1]
    out[f"pitch{side}"] = torch.rad2deg(torch.atan2(nx, n[:, 2])).cpu().numpy()
    out[f"roll{side}"] = torch.rad2deg(torch.atan2(-ny, n[:, 2])).cpu().numpy()
fz = (u._foot_positions()[..., 2]).cpu().numpy(); out["dz_LR_mm"] = (fz[:, 0] - fz[:, 1]) * 1000
P = np.degrees(np.arcsin(np.clip(2 * (X.imu_qw * X.imu_qy - X.imu_qz * X.imu_qx), -1, 1))).values
R = np.degrees(np.arctan2(2 * (X.imu_qw * X.imu_qx + X.imu_qy * X.imu_qz), 1 - 2 * (X.imu_qx ** 2 + X.imu_qy ** 2))).values
for mode in X["mode"].unique():
    m = (X["mode"] == mode).values
    print(f"== {mode}: n {m.sum()} | IMU pitch {P[m].mean():+.2f} roll {R[m].mean():+.2f} | ankle pitch L {np.degrees(X.pos_left_ankle_pitch[m].mean()):+.1f} R {np.degrees(X.pos_right_ankle_pitch[m].mean()):+.1f}")
    for k, v in out.items():
        print(f"   {k:9s} mean {v[m].mean():+6.2f} std {v[m].std():5.2f} p10 {np.percentile(v[m], 10):+6.2f} p90 {np.percentile(v[m], 90):+6.2f}")
print("(torso heading frame: pitch > 0 = sole normal tilted forward = toe down / heel up; roll > 0 = normal tilted to +y... sign: atan2(-ny, nz))")
env.close(); app.app.close()
