"""Stand tremor check (2026-11-23): action / q_des / torso gyro spectra at zero stand command, same statistics as the
real-log check (stage-4 264k: 8.8 Hz policy-loop tremor, gyro_y power >5 Hz 0.79 vs stage-3 0.22).
  python scripts/real2sim/stand_tremor.py --checkpoint <stand.pt> --num_envs 64 --steps 1000 --headless  (env NO_PUSH, NO_TT, LPF_HZ)"""
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
if os.environ.get("NO_COMPLIANCE"): cfg.joint_series_compliance_scale_range = (0.0, 0.0)
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
if os.environ.get('EXTRA_DELAY_MS'):
    x=float(os.environ['EXTRA_DELAY_MS']); cfg.actuator_delay_ms={k:v+x for k,v in cfg.actuator_delay_ms.items()}; print('RESULT delay', cfg.actuator_delay_ms)
if os.environ.get('LPF_HZ'): cfg.action_lowpass_hz=float(os.environ['LPF_HZ'])
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
A=[]; G=[]; OB=[]; QD=[]
u.command_timer[:] = 1e9; u.g_target[:] = 0.0
for t in range(args.steps):
    with torch.no_grad(): a = policy(obs["policy"])
    obs, _, term, trunc, _ = env.step(a)
    u.command_timer[:] = 1e9; u.g_target[:] = 0.0
    A.append(a.cpu().numpy()); QD.append(u.q_des.cpu().numpy()); OB.append(obs['policy'].cpu().numpy()); G.append(u.robot.data.body_ang_vel_w.torch[:, u.ref_body_index].cpu().numpy())
A=np.stack(A)[50:]; OB=np.stack(OB)[50:]; G=np.stack(G)[50:]
da=np.abs(np.diff(A,axis=0)).mean()
f=np.fft.rfftfreq(A.shape[0],0.02); P=(np.abs(np.fft.rfft(A-A.mean(0),axis=0))**2).sum(axis=(1,2))
print("RESULT |da|/step", round(float(da),4), "per joint", np.abs(np.diff(A,axis=0)).mean(axis=(0,1)).round(3))
print("RESULT action power top freqs", f[np.argsort(P)[-5:]].round(2), "power >5Hz frac", round(float(P[f>5].sum()/P[1:].sum()),3))
print("RESULT gyro std", G.std(axis=0).mean(0).round(3))
from scipy.signal import welch
QD=np.stack(QD)[50:]
f,Pg=welch(G-G.mean(0),50,nperseg=256,axis=0); Pg=Pg.mean(1)  # (f, 3)
for k,ax in enumerate("xyz"):
    print(f"RESULT gyro_{ax} peak {f[np.argmax(Pg[1:,k])+1]:.2f} Hz | power >5Hz frac {Pg[f>5,k].sum()/Pg[1:,k].sum():.2f} | 7.5-11 Hz rms {np.sqrt(Pg[(f>7.5)&(f<11),k].sum()*(f[1]-f[0])):.3f}")
f,Pq=welch(QD-QD.mean(0),50,nperseg=256,axis=0); b=(f>7.5)&(f<11)
print("RESULT q_des 7.5-11 Hz rms deg per joint", np.degrees(np.sqrt(Pq[b].sum(0)*(f[1]-f[0])).mean(0)).round(2))
env.close(); app.app.close()
