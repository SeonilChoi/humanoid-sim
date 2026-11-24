"""Stand Play diagnostics: falls, tilt, |w|, joint error, Olaf metrics (RESULT lines). Env: NO_PUSH, NO_TT, LPF_HZ, TRAIN_DR, NO_DR, SWITCH."""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="HumanoidSim-Olaf-Stand-Play-v0")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args()
app = AppLauncher(args)
import gymnasium as gym, torch
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import resolve_task_config
import humanoid_sim.tasks  # noqa
cfg = resolve_task_config(args.task, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = args.num_envs
import os
if os.environ.get("NO_PUSH"): cfg.events.push_robot = None
if os.environ.get("NO_TT"): cfg.torso_torque_noise_std = (0.0, 0.0, 0.0)
if os.environ.get("NO_EXTRA_DELAY"):
    cfg.actuator_delay_extra_ms = {}
if os.environ.get("IMU_START"):
    cfg.accel_impact_scale_range = cfg.dr_start_accel_impact_scale; cfg.accel_bias_range = cfg.dr_start_accel_bias
    cfg.mti3_gyro_noise_std_rad_s = cfg.dr_start_gyro_noise; cfg.mti3_gyro_bias_range_rad_s = cfg.dr_start_gyro_bias
if os.environ.get("ABL") == "gains":
    cfg.events.actuator_gains.params["stiffness_distribution_params"] = cfg.dr_start_kp
    cfg.events.actuator_gains.params["damping_distribution_params"] = cfg.dr_start_kd
if os.environ.get("ABL") == "friction_coulomb":
    cfg.actuator_coulomb_scale_range = cfg.dr_start_coulomb_scale; cfg.actuator_armature_scale_range = cfg.dr_start_armature_scale
if os.environ.get("ABL") == "contact_mass":
    cfg.events.physics_material.params["static_friction_range"] = cfg.dr_start_static_friction
    cfg.events.physics_material.params["dynamic_friction_range"] = cfg.dr_start_dynamic_friction
    cfg.events.add_pelvis_mass.params["mass_distribution_params"] = cfg.dr_start_pelvis_mass
    c = cfg.dr_start_pelvis_com; cfg.events.pelvis_com.params["com_range"] = {a: (-c, c) for a in "xyz"}
if os.environ.get("ABL") == "offset":
    cfg.joint_obs_offset_range_by_type = {k: v * cfg.dr_start_joint_offset_frac for k, v in cfg.joint_obs_offset_range_by_type.items()}
if os.environ.get("NO_DR"):
    cfg.dr_widen_steps = (10**9, 10**9 + 1); cfg.actuator_delay_extra_curriculum_steps = (10**9, 10**9 + 1)
if os.environ.get("TRAIN_DR"):
    cfg.dr_widen_steps = (24_000, 1_200_000); cfg.actuator_delay_extra_curriculum_steps = (24_000, 1_200_000); cfg.curriculum_step_offset = int(os.environ["TRAIN_DR"])
if os.environ.get("LPF_HZ"): cfg.action_lowpass_hz = float(os.environ["LPF_HZ"])
if os.environ.get("SWITCH"): cfg.reset_walk_pose_fraction = 1.0; cfg.reset_stand_pose_fraction = 1.0
env = gym.make(args.task, cfg=cfg); u = env.unwrapped; dev = u.device
c = torch.load(args.checkpoint, map_location=dev, weights_only=False)
P, pre = c["policy"], c["observation_preprocessor"]
idx = sorted({int(k.split(".")[1]) for k in P if k.startswith("net_container.") and k.endswith(".weight")})
def policy(o):
    x = ((o - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8)).clamp(-5, 5)
    for j, i in enumerate(idx):
        x = x @ P[f"net_container.{i}.weight"].T + P[f"net_container.{i}.bias"]
        if j < len(idx) - 1: x = torch.nn.functional.elu(x)
    return x
obs, _ = env.reset()
falls = resets = 0; T, W, J, BOTH = [], [], [], []; LOG = {}
for t in range(args.steps):
    with torch.no_grad(): a = policy(obs["policy"])
    obs, _, term, trunc, _ = env.step(a)
    falls += int(term.sum()); resets += int((term | trunc).sum())
    for k, v in u.extras.get("log", {}).items():
        if k.startswith("Olaf/"): LOG.setdefault(k, []).append(float(v))
    ok = u.episode_length_buf > 25
    if not ok.any(): continue
    pos, quat, lin_w, ang_w = u._torso(); R = matrix_from_quat(quat)
    T.append(torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))[ok]); W.append(ang_w[ok].norm(dim=-1))
    BOTH.append(u.feet_contact.all(-1)[ok].float())
    if hasattr(u, "ref_q"): J.append(torch.rad2deg((u.robot.data.joint_pos.torch - u.ref_q).abs().mean(-1))[ok])
c = lambda L: torch.cat(L).mean().item() if L else float("nan")
print(f"RESULT steps {args.steps} x {u.num_envs} envs | falls {falls} | resets {resets}")
print(f"RESULT tilt {c(T):.2f} deg | |w| {c(W):.3f} rad/s | both feet down {c(BOTH):.3f} | joint err {c(J):.2f} deg")
for k, v in LOG.items(): print(f"RESULT {k} {sum(v)/len(v):.3f}")
env.close(); app.app.close()
