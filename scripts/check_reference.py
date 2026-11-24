"""Reference sanity check: play the kinematic reference open loop (PD target = reference joints, no policy) and report
how well the sim follows it and how consistent the reference is with the sim robot.

    python scripts/check_reference.py --task HumanoidSim-Olaf-Walk-Play-v0 --num_envs 32 --steps 120 [--no_dr]

Per command bucket: at reset (RSI) the sole heights / torso height vs the reference, then over the open-loop rollout the
joint error, torso position / orientation error, contact match and falls. Open loop will eventually fall (no balance
feedback); the first ~1-2 s show whether the reference itself is dynamically plausible.
"""

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="HumanoidSim-Olaf-Walk-Play-v0")
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--trace", type=int, default=-1, help="print per-step errors of this env")
ap.add_argument("--no_dr", action="store_true", help="nominal robot: no pushes / delays / zero errors / yaw bias")
AppLauncher.add_app_launcher_args(ap)
a, _ = ap.parse_known_args()
app = AppLauncher(a)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import humanoid_sim.tasks  # noqa: F401, E402

cfg = resolve_task_config(a.task, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = a.num_envs
cfg.events.push_robot = None
cfg.episode_length_s = 1e4
cfg.command_resample_time_s = (1e4, 1e4)
if a.no_dr:
    cfg.joint_obs_offset_range_by_type = {}
    cfg.joint_obs_offset_range = 0.0
    cfg.yaw_torque_bias_range = (0.0, 0.0)
    cfg.actuator_delay_extra_ms = {}
    cfg.actuator_delay_jitter_ms = (0.0, 0.0)
    cfg.actuator_delay_step_jitter = 0
    cfg.target_rate_limit = 0.0
    cfg.target_lowpass_alpha = 1.0
env = gym.make(a.task, cfg=cfg)
u = env.unwrapped
dev = u.device
with torch.inference_mode():
    env.reset()
    # spread commands: walk -> a few fixed path velocities, stand -> fixed g
    n = u.num_envs
    if u.walk:
        cmds = torch.tensor([[0, 0, 0], [0.1, 0, 0], [0.15, 0, 0], [-0.1, 0, 0], [0, 0.06, 0], [0, -0.06, 0], [0, 0, 0.35], [0.08, 0, -0.3]], device=dev)
    else:
        cmds = torch.tensor([[0, 0, 0, 0], [-0.03, 0, 0, 0], [0.015, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.15, 0], [0, 0, -0.15, 0], [0, 0, 0, 0.3], [-0.02, -0.08, 0.1, -0.2]], device=dev)
    grp = torch.arange(n, device=dev) % len(cmds)
    ids = torch.arange(n, device=dev)
    if u.walk:
        idx, w = u._walk_blend(cmds[grp])
        u.g[:] = (w[..., None] * u.walk_keys[idx]).sum(1)
        u.commands[:] = u.g
    else:
        u.g_target[:] = cmds[grp]
    u._reset_idx(ids)
    u.scene.write_data_to_sim()
    u.sim.forward()
    u.scene.update(0.0)
    h, _ = u._sole_heights()
    z = u.robot.data.body_pos_w.torch[:, u.ref_body_index, 2] - u.scene.env_origins[:, 2]
    print(f"after RSI: sole lowest-corner height mean {h.mean() * 1000:.1f} mm (min {h.min() * 1000:.1f}, max {h.max() * 1000:.1f}); "
          f"pelvis z {z.mean():.4f} vs ref {(u.ref_pos[:, 2] - u.scene.env_origins[:, 2]).mean():.4f}")
    acc = {k: torch.zeros(len(cmds), device=dev) for k in ("jerr", "perr", "oerr", "cm", "cnt")}
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    for t in range(a.steps):
        dq = u.ref_q - u.default_q
        # next step's reference is one phase step ahead: use it as the PD target (open-loop playback)
        scale = torch.where(dq >= 0.0, u.action_positive_scale, u.action_negative_scale)
        act = torch.clamp(dq / scale, -1.0, 1.0)
        act_pol = act[:, u.sim_joint_ids_from_policy.argsort()] if hasattr(u, "sim_joint_ids_from_policy") else act
        _, _, term, _, _ = env.step(act_pol)
        log = u.extras["log"]
        fell |= term
        pos, quat, _, _ = u._torso()
        jerr = torch.rad2deg(torch.sqrt(torch.mean(torch.square(u.robot.data.joint_pos.torch - u.ref_q), -1)))
        perr = 100 * (pos[:, :2] - u.ref_pos[:, :2]).norm(dim=-1)
        from isaaclab.utils.math import quat_box_minus
        oerr = torch.rad2deg(quat_box_minus(quat, u.ref_quat).norm(dim=-1))
        h, _ = u._sole_heights()
        cm = ((h < u.cfg.foot_contact_height).float() == u.ref_contact).float().mean(-1)
        alive = (~fell).float()
        if a.trace >= 0 and t < 60:
            e = a.trace
            print(f'  t {t:3d} phase {u.phase[e].item() if u.walk else 0:.2f} jerr {jerr[e]:.1f} perr {perr[e]:.1f} oerr {oerr[e]:.1f} '
                  f'h {[round(x * 1000) for x in h[e].tolist()]} ref_c {u.ref_contact[e].tolist()} fell {bool(fell[e])}')
        for k, v in (("jerr", jerr), ("perr", perr), ("oerr", oerr), ("cm", cm)):
            acc[k].index_add_(0, grp, v * alive)
        acc["cnt"].index_add_(0, grp, alive)
    for i, c in enumerate(cmds.tolist()):
        m = grp == i
        cnt = acc["cnt"][i].clamp_min(1)
        print(f"g {[round(x, 3) for x in c]}: joint err {acc['jerr'][i] / cnt:.1f} deg, torso pos err {acc['perr'][i] / cnt:.1f} cm, "
              f"ori err {acc['oerr'][i] / cnt:.1f} deg, contact match {acc['cm'][i] / cnt:.2f}, fell {fell[m].float().mean():.2f} "
              f"(within {a.steps} steps)")
env.close()
app.app.close()
