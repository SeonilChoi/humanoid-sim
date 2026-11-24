"""Sim check of the actor-side state estimate (contact-aided InEKF, olaf/inekf.py = humanoid_controller robot::OlafInEKF)
against the ground truth, closed loop.

    python scripts/check_estimator.py --task HumanoidSim-Olaf-Stand-Play-v0 [--checkpoint <run>/checkpoints/agent_N.pt]
        [--num_envs 32] [--steps 1500]

Without --checkpoint the actions hold the reference joints (stand: the IK pose of g; walk: open loop, falls).
Every 5 s of sim: pelvis-frame velocity error, tilt error (gravity direction), yaw drift, pelvis position relative to
the path frame (estimate, as the actor sees it, vs truth), contact agreement, falls.
"""

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="")
ap.add_argument("--task", default="HumanoidSim-Olaf-Stand-Play-v0")
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--ideal", action="store_true", help="noiseless, undelayed IMU (checks the filter / frames)")
ap.add_argument("--no_push", action="store_true")
ap.add_argument("--impact_scale", type=float, default=-1.0, help="fixed accelerometer touchdown-noise scale")
AppLauncher.add_app_launcher_args(ap)
a, _ = ap.parse_known_args()
app = AppLauncher(a)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import humanoid_sim.tasks  # noqa: F401, E402

cfg = resolve_task_config(a.task, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = a.num_envs
cfg.episode_length_s = 1e4
if a.ideal:
    cfg.accel_base_noise, cfg.accel_impact_noise, cfg.accel_bias_range = (0.0,) * 3, (0.0,) * 3, (0.0, 0.0)
    cfg.dr_start_accel_bias, cfg.dr_start_gyro_noise, cfg.dr_start_gyro_bias = (0.0, 0.0), 0.0, (0.0, 0.0)
    cfg.mti3_gyro_noise_std_rad_s, cfg.mti3_gyro_bias_range_rad_s, cfg.imu_gyro_extra_delay_steps = 0.0, (0.0, 0.0), 0
    cfg.mti3_gyro_bias_walk_std_rad_s, cfg.imu_latency_s = 0.0, 0.0
if a.impact_scale >= 0.0:
    cfg.accel_impact_scale_range = (a.impact_scale, a.impact_scale)
if a.no_push:
    cfg.events.push_robot = None
env = gym.make(a.task, cfg=cfg)
u = env.unwrapped
dev, n, dt = u.device, u.num_envs, u.step_dt

policy = None
if a.checkpoint:
    c = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    P, pre = c["policy"], c["observation_preprocessor"]
    idx = sorted({int(k.split(".")[1]) for k in P if k.startswith("net_container.") and k.endswith(".weight")})

    def policy(o):
        x = ((o - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8)).clamp(-5, 5)
        for j, i in enumerate(idx):
            x = x @ P[f"net_container.{i}.weight"].T + P[f"net_container.{i}.bias"]
            if j < len(idx) - 1:
                x = torch.nn.functional.elu(x)
        return x


def ref_action():
    """Actions (policy order) whose PD target is the reference joint position."""
    dq = u.ref_q - u.default_q
    scale = torch.where(dq >= 0.0, u.action_positive_scale, u.action_negative_scale)
    return u._joints_to_policy(torch.clamp(dq / scale, -1.0, 1.0))


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def rel(p, path):
    c, s = torch.cos(path[:, 2]), torch.sin(path[:, 2])
    dx, dy = p[:, 0] - path[:, 0], p[:, 1] - path[:, 1]
    return torch.stack((c * dx + s * dy, -s * dx + c * dy), -1)


obs, _ = env.reset()
stats = {k: [] for k in ("vel", "tilt", "yaw", "pos", "contact", "w_true", "vz_est")}
falls = 0
yaw0 = None
for t in range(a.steps):
    with torch.no_grad():
        act = policy(obs["policy"]) if policy else ref_action()
    obs, _, term, trunc, _ = env.step(act)
    falls += int(term.sum())
    pos, quat, lin_w, _ = u._torso()
    Rt, Re = matrix_from_quat(quat), u.est_R
    # settled after a reset and not falling (true tilt < 15 deg)
    ok = (u.episode_length_buf > 5) & (Rt[:, 2, 2] > 0.966)
    # the estimate is the pelvis frame origin velocity (link, not centre of mass)
    lin_link = u.robot.data.body_link_lin_vel_w.torch[:, u.ref_body_index]
    v_true = (Rt.transpose(1, 2) @ lin_link[..., None])[..., 0]
    v_est = (Re.transpose(1, 2) @ u.est_vel[..., None])[..., 0]
    tilt = torch.rad2deg(torch.acos((Rt[:, 2, :] * Re[:, 2, :]).sum(-1).clamp(-1, 1)))
    dyaw = wrap(torch.atan2(Re[:, 1, 0], Re[:, 0, 0]) - torch.atan2(Rt[:, 1, 0], Rt[:, 0, 0]))
    yaw0 = dyaw.clone() if yaw0 is None else torch.where(u.episode_length_buf <= 5, dyaw, yaw0)
    perr = (rel(u.est_pos, u.est_path) - rel(pos, u.path)).norm(dim=-1)
    if ok.any():
        stats["vel"].append((v_est - v_true)[ok].norm(dim=-1))
        stats["tilt"].append(tilt[ok])
        stats["yaw"].append(torch.rad2deg(wrap(dyaw - yaw0)).abs()[ok])
        stats["pos"].append(100 * perr[ok])
        stats["contact"].append((u.ekf.contact == u.feet_contact)[ok].float().mean(-1))
        stats["w_true"].append(u.robot.data.body_ang_vel_w.torch[:, u.ref_body_index][ok].norm(dim=-1))
        stats["vz_est"].append(v_est[ok, 2].abs())
    if (t + 1) % int(round(5.0 / dt)) == 0 and stats["vel"]:
        m = {k: torch.cat(v).mean().item() for k, v in stats.items()}
        mx = {k: torch.cat(v).max().item() for k, v in stats.items()}
        print(f"t {(t + 1) * dt:5.1f} s | vel err {m['vel']:.3f} m/s (max {mx['vel']:.2f}) | tilt {m['tilt']:.2f} deg "
              f"(max {mx['tilt']:.1f}) | yaw drift {m['yaw']:.2f} deg | path-rel pos {m['pos']:.2f} cm "
              f"(max {mx['pos']:.1f}) | contact agree {m['contact']:.3f} | falls {falls} | true |w| {m['w_true']:.2f} rad/s | est |vz| {m['vz_est']:.3f} m/s", flush=True)
        stats = {k: [] for k in stats}
env.close()
app.app.close()
