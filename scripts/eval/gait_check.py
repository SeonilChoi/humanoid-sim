"""Walk Play gait diagnostics: contact match, swing apex, velocity tracking, tilt, |w|, falls. Env: NO_PUSH, STOCH, TRAIN_DR, NO_DR, LPF_HZ."""
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
if os.environ.get("LPF_HZ"): cfg.action_lowpass_hz = float(os.environ["LPF_HZ"])
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
obs, _ = env.reset(); LOGM = []
S = {k: [] for k in ["tilt","arate","jvel","match","drag","early","ref_st","act_st","swing_h","verr","w","vcmd","vact","ori","jerr"]}
falls = 0; resets = 0
swing_max = torch.zeros(u.num_envs, 2, device=dev); swing_heights = []
for t in range(args.steps):
    with torch.no_grad():
        a = policy(obs["policy"])
        if os.environ.get("STOCH"): a = a + torch.randn_like(a) * P["log_std_parameter"].exp()
    obs, _, term, trunc, _ = env.step(a)
    falls += int(term.sum()); resets += int((term | trunc).sum())
    LOGM.append(float(u.extras["log"]["Olaf/contact_match"])) if "Olaf/contact_match" in u.extras.get("log", {}) else None
    ok = u.episode_length_buf > 25
    h, _ = u._sole_heights()
    act = u.feet_contact; ref = u.ref_contact > 0.5
    pos, quat, lin_w, ang_w = u._torso()
    R = matrix_from_quat(quat); yaw = torch.atan2(R[:,1,0], R[:,0,0])
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    vx = cy*lin_w[:,0] + sy*lin_w[:,1]; vy = -sy*lin_w[:,0] + cy*lin_w[:,1]
    rv = u.ref_v; rvx = cy*rv[:,0]+sy*rv[:,1]; rvy = -sy*rv[:,0]+cy*rv[:,1]
    # swing height: max sole height over each airborne phase
    air = ~act
    swing_max = torch.where(air, torch.maximum(swing_max, h), swing_max)
    landed = act & (swing_max > 0)
    if (landed & ok[:, None]).any(): swing_heights.append(swing_max[landed & ok[:, None]])
    swing_max = torch.where(act, torch.zeros_like(swing_max), swing_max)
    if ok.any():
        S["match"].append((act == ref)[ok].float().mean(-1))
        S["drag"].append((act & ~ref)[ok].float().mean(-1))     # ref swing, foot still down
        S["early"].append((~act & ref)[ok].float().mean(-1))    # ref stance, foot in air
        S["ref_st"].append(ref[ok].float().mean(-1)); S["act_st"].append(act[ok].float().mean(-1))
        S["verr"].append(torch.stack((vx-rvx, vy-rvy), -1)[ok].norm(dim=-1))
        S["vcmd"].append(rvx[ok]); S["vact"].append(vx[ok])
        S["w"].append(ang_w[ok].norm(dim=-1))
        S["tilt"].append(torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))[ok])
        S["arate"].append((u.actions - u.previous_actions).abs().mean(-1)[ok])
        S["jvel"].append(u.robot.data.joint_vel.torch.abs().mean(-1)[ok])
        S["jerr"].append(torch.rad2deg((u.robot.data.joint_pos.torch - u.ref_q).abs().mean(-1))[ok])
print("ep len max", u.episode_length_buf.max().item(), "falls", falls, {k: len(v) for k, v in S.items()})
m = {k: torch.cat(v).mean().item() for k, v in S.items() if v}
sh = torch.cat(swing_heights) * 100 if swing_heights else torch.zeros(1)
print("logged metric mean", sum(LOGM)/max(len(LOGM),1), len(LOGM))
print(f"steps {args.steps} x {u.num_envs} envs | falls {falls} | resets {resets}")
print(f"contact match {m['match']:.3f} | ref-swing-but-down {m['drag']:.3f} | ref-stance-but-air {m['early']:.3f}")
print(f"stance frac ref {m['ref_st']:.3f} actual {m['act_st']:.3f}")
print(f"swing apex cm: mean {sh.mean():.2f} p10 {sh.quantile(0.1):.2f} p50 {sh.median():.2f} p90 {sh.quantile(0.9):.2f} (n {sh.numel()})")
print(f"torso tilt {m['tilt']:.2f} deg | action change/step {m['arate']:.3f} | joint |qd| {m['jvel']:.2f} rad/s")
print(f"ref vx {m['vcmd']:.3f} actual vx {m['vact']:.3f} | xy vel err vs ref {m['verr']:.3f} m/s | |w| {m['w']:.2f} rad/s | joint err {m['jerr']:.2f} deg")
# reference swing apex (from the clip contact flags + ref sole heights is not available here): print ref joint knee range
env.close(); app.app.close()
