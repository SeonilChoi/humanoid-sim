"""Walk actuator saturation: knee |a|>0.95 fraction, |da|, knee lag |q_des - q|, falls. Env as gait_check."""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="HumanoidSim-Olaf-Walk-Play-v0")
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
import numpy as np
A=[]; QD=[]; QP=[]
names=u.robot.joint_names
for t in range(args.steps):
    with torch.no_grad(): a = policy(obs["policy"])
    obs, _, term, trunc, _ = env.step(a)
    u.command_timer[:] = 1e9; u.g[:] = 0.0; u.commands[:] = 0.0
    A.append(a.clamp(-1,1).cpu().numpy()); QD.append(u.q_des.cpu().numpy().copy()); QP.append(u.robot.data.joint_pos.torch.cpu().numpy().copy())
A=np.stack(A)[25:]; QD=np.stack(QD)[25:]; QP=np.stack(QP)[25:]
print("RESULT sat frac per policy joint", (np.abs(A)>0.95).mean((0,1)).round(2))
for j,n in enumerate(names):
    if "knee" in n or "hip_pitch" in n:
        print(f"RESULT {n} q_des range {QD[:,:,j].min():.2f}..{QD[:,:,j].max():.2f} pos range {QP[:,:,j].min():.2f}..{QP[:,:,j].max():.2f} mean|qdes-pos| deg {np.degrees(np.abs(QD[:,:,j]-QP[:,:,j]).mean()):.1f}")
env.close(); app.app.close()
