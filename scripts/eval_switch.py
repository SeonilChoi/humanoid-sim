"""Walk -> stand policy switch in simulation, as the controller does it on the robot.

Two Isaac Sim processes (one env type each), run by this script:

  1. walk stage (HumanoidSim-Olaf-Walk-Play-v0): every env starts from the standing stance at walk_start_phase (the
     controller's stand -> walk switch), walks a random command for a random time, then gets the stop request: the
     command goes to zero (stick released) and the switch waits for the next touchdown in the estimator contact
     (double support starts: both soles in contact now, not both in the previous step; BD-X Sec. VIII-A), at most
     --timeout s. --switch immediate switches at the request instead. The robot state at the switch is saved.
  2. stand stage (HumanoidSim-Olaf-Stand-Play-v0): every env starts from a saved switch state (root pose / velocity,
     joints, temperatures) instead of the reference, with the controller's switch reset: path frame at the torso,
     g = 0, q_des / filter at q, action history = the pose as an action (--history pose, ArtiH1::olaf_switch_reset)
     or the walk policy's last actions (--history walk, BD-X: the history is passed through). The stand policy runs
     --stand_s s; a termination is a switch fall.

  python scripts/eval_switch.py --walk_ckpt <walk agent.pt> --stand_ckpt <stand agent.pt> [--num_envs 256]
      [--switch touchdown|immediate] [--history pose|walk] [--speed 1.0] [--push] [--out <dir>]

Switch-state bank for the stand training (env.switch_bank_file, stage 2): walk stage only, per-env command scale
uniform in [0, --speed], the states written to --bank (a bare file name goes to assets/reference):

  python scripts/eval_switch.py --walk_ckpt <walk agent.pt> --stand_ckpt - --num_envs 4096 --bank olaf_switch_bank.npz

Each checkpoint is read with <run>/params (the run's training cfg is not used: the Play task cfg is). Both stages use
the final domain randomisation of the Play tasks; the DR draws of the two stages are independent.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--walk_ckpt", required=True)
ap.add_argument("--stand_ckpt", required=True)
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--switch", choices=("touchdown", "immediate"), default="touchdown")
ap.add_argument("--history", choices=("pose", "walk"), default="pose")
ap.add_argument("--speed", type=float, default=1.0, help="scale of the random walk command range (0 = step in place)")
ap.add_argument("--walk_s", type=float, nargs=2, default=(1.5, 4.0), help="walking time before the stop request")
ap.add_argument("--timeout", type=float, default=1.0, help="s: switch anyway if no touchdown")
ap.add_argument("--stand_s", type=float, default=3.0)
ap.add_argument("--push", action="store_true", help="keep the Play tasks' random pushes (default: off)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="logs/eval_switch")
ap.add_argument("--stage", choices=("all", "walk", "stand"), default="all")
ap.add_argument("--bank", default=None, help="write the switch states as a stand-training bank (walk stage only)")
args, rest = ap.parse_known_args()

if args.stage == "all":
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, __file__] + [a for a in sys.argv[1:]]
    for stage in (("walk",) if args.bank else ("walk", "stand")):
        r = subprocess.run(base + ["--stage", stage], check=False)
        if r.returncode != 0:
            sys.exit(f"[eval_switch] {stage} stage failed ({r.returncode})")
    if not args.bank:
        print((out / "result.txt").read_text())
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import humanoid_sim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

OUT = Path(args.out)
SNAP = OUT / "switch_states.npz"


def load_policy(path, dev):
    c = torch.load(path, map_location=dev, weights_only=False)
    P, pre = c["policy"], c["observation_preprocessor"]
    idx = sorted({int(k.split(".")[1]) for k in P if k.startswith("net_container.") and k.endswith(".weight")})
    mean, std = pre["running_mean"].float(), torch.sqrt(pre["running_variance"].float()) + 1e-8

    def pi(o):
        x = ((o - mean) / std).clamp(-5, 5)
        for j, i in enumerate(idx):
            x = x @ P[f"net_container.{i}.weight"].T + P[f"net_container.{i}.bias"]
            if j < len(idx) - 1:
                x = torch.nn.functional.elu(x)
        return x

    return pi


def make_env(task, n, **over):
    cfg = resolve_task_config(task, None, overrides=["physics=isaacsim_physx"])[0]
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    if not args.push:
        cfg.events.push_robot = None
    if os.environ.get("NO_COMPLIANCE"):
        cfg.joint_series_compliance_scale_range = (0.0, 0.0)
    for k, v in over.items():
        setattr(cfg, k, v)
    env = gym.make(task, cfg=cfg)
    return env, env.unwrapped


def root_state(u):
    d = u.robot.data
    pose = d.root_pose_w.torch if hasattr(d, "root_pose_w") else torch.cat((d.root_pos_w.torch, d.root_quat_w.torch), -1)
    lin = d.root_link_lin_vel_w.torch if hasattr(d, "root_link_lin_vel_w") else d.root_lin_vel_w.torch
    return pose.clone(), torch.cat((lin, d.root_ang_vel_w.torch), -1).clone()


def est_both_down(u):
    z = u.est_feet[..., 2]
    return ((z - z.min(-1, keepdim=True).values) < u.cfg.est_contact_tolerance).all(-1)


def tilt_deg(u):
    q = u.robot.data.root_quat_w.torch  # (x, y, z, w) in Isaac Lab 3.x
    x, y = q[:, 0], q[:, 1]
    return torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


# ---------------------------------------------------------------------------------------------------------------------
def walk_stage():
    env, u = make_env("HumanoidSim-Olaf-Walk-Play-v0", args.num_envs, reset_stand_pose_fraction=1.0,
                      command_resample_time_s=(1.0e6, 1.0e6))
    dev, n, dt = u.device, u.num_envs, u.step_dt
    pi = load_policy(args.walk_ckpt, dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    g = u.walk_lo + torch.rand(n, 3, device=dev, generator=gen) * (u.walk_hi - u.walk_lo)
    scale = torch.rand(n, 1, device=dev, generator=gen) * args.speed if args.bank else args.speed
    idx, w = u._walk_blend(g * scale)
    g_walk = (w[..., None] * u.walk_keys[idx]).sum(1)
    idx, w = u._walk_blend(torch.zeros(n, 3, device=dev))
    g_zero = (w[..., None] * u.walk_keys[idx]).sum(1)
    lo, hi = args.walk_s
    t_req = ((lo + torch.rand(n, device=dev, generator=gen) * (hi - lo)) / dt).long()
    WALK, WAIT, DONE, FELL = 0, 1, 2, 3
    st = torch.full((n,), WALK, device=dev)
    wait = torch.zeros(n, device=dev, dtype=torch.long)
    timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
    snap = {k: None for k in ("pose", "vel", "q", "qd", "a", "a1", "a2", "temp", "phase", "wait_s", "tilt")}
    snap_t = {}
    obs, _ = env.reset()
    both_prev = est_both_down(u)
    max_steps = int((hi + args.timeout + 1.0) / dt) + 5
    origins = u.scene.env_origins
    for t in range(max_steps):
        u.g[:] = torch.where((st == WALK)[:, None], g_walk, g_zero)
        u.commands[:] = u.g
        with torch.no_grad():
            obs, _, term, _, _ = env.step(pi(obs["policy"]))
        fell = term & ((st == WALK) | (st == WAIT))
        st[fell] = FELL
        both = est_both_down(u)
        req = (st == WALK) & (t + 1 >= t_req)
        st[req] = WAIT
        wait[st == WAIT] += 1
        if args.switch == "immediate":
            go = st == WAIT
        else:
            touchdown = both & ~both_prev
            timeout = (st == WAIT) & (wait * dt >= args.timeout)
            go = (st == WAIT) & (touchdown | timeout)
            timed_out |= go & timeout & ~touchdown
        both_prev = both
        if go.any():
            ids = go.nonzero().flatten()
            pose, vel = root_state(u)
            pose[:, :3] -= origins
            d = u.robot.data
            rows = {"pose": pose[ids], "vel": vel[ids], "q": d.joint_pos.torch[ids], "qd": d.joint_vel.torch[ids],
                    "a": u.actions[ids], "a1": u.previous_actions[ids], "a2": u.previous_actions2[ids],
                    "temp": u.temp[ids], "phase": u.phase[ids], "wait_s": wait[ids].float() * dt,
                    "tilt": tilt_deg(u)[ids]}
            for i, e in enumerate(ids.tolist()):
                snap_t[e] = {k: v[i].cpu().numpy() for k, v in rows.items()}
            st[ids] = DONE
        if not ((st == WALK) | (st == WAIT)).any():
            break
    keys = sorted(snap_t)
    np.savez(SNAP, env=np.array(keys), timed_out=timed_out.cpu().numpy()[keys] if keys else np.zeros(0, bool),
             **{k: np.stack([snap_t[e][k] for e in keys]) if keys else np.zeros(0) for k in snap})
    stats = {"envs": n, "walk_falls": int((st == FELL).sum()), "switched": len(keys),
             "timeouts": int(timed_out.sum()), "still_walking": int(((st == WALK) | (st == WAIT)).sum())}
    if keys:
        ws = np.stack([snap_t[e]["wait_s"] for e in keys])
        stats["wait_s_mean"] = float(ws.mean())
        stats["tilt_at_switch_mean"] = float(np.mean([snap_t[e]["tilt"] for e in keys]))
    (OUT / "walk_stage.json").write_text(json.dumps(stats, indent=1))
    if args.bank and keys:
        from humanoid_sim.tasks.direct.olaf.olaf_env_cfg import REFERENCE_FILE
        bank = Path(args.bank) if os.path.dirname(args.bank) else Path(REFERENCE_FILE).parent / args.bank
        np.savez(bank, joint_names=np.array(list(u.robot.data.joint_names)), walk_ckpt=str(args.walk_ckpt),
                 **{k: np.stack([snap_t[e][k] for e in keys]) for k in ("pose", "vel", "q", "qd", "tilt", "wait_s")})
        print(f"[eval_switch] bank {bank}: {len(keys)} states")
    print("[eval_switch] walk stage", stats)
    env.close()


# ---------------------------------------------------------------------------------------------------------------------
def stand_stage():
    s = np.load(SNAP)
    m = len(s["env"])
    if m == 0:
        sys.exit("[eval_switch] no switch states")
    env, u = make_env("HumanoidSim-Olaf-Stand-Play-v0", m, reset_walk_pose_fraction=0.0)
    dev = u.device
    S = {k: torch.as_tensor(s[k], device=dev, dtype=torch.float32) for k in ("pose", "vel", "q", "qd", "a", "a1", "a2", "temp")}
    orig = u._reset_idx
    applied = {"done": False}

    def reset_from_switch(env_ids):
        orig(env_ids)
        if applied["done"] or not hasattr(u, "g"):
            return
        ids = torch.arange(m, device=dev)
        pose = S["pose"].clone()
        pose[:, :3] += u.scene.env_origins
        r = u.robot
        r.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        r.write_root_link_velocity_to_sim_index(root_velocity=S["vel"], env_ids=ids)
        r.write_joint_position_to_sim_index(position=S["q"], env_ids=ids)
        r.write_joint_velocity_to_sim_index(velocity=S["qd"], env_ids=ids)
        if u.cfg.actuator_model and hasattr(u, "actuator_buffer"):
            u.actuator_buffer[:] = S["q"].unsqueeze(0)
        u.temp[:] = S["temp"]
        # controller olaf_reset_path: path frame at the torso; STAND g = the stick target (0)
        from humanoid_sim.tasks.direct.olaf.olaf_env import _yaw_of
        u.path[:, :2] = pose[:, :2]
        u.path[:, 2] = _yaw_of(pose[:, 3:7])
        u.g[:] = 0.0
        u.g_target[:] = 0.0
        q_enc = S["q"] + u.joint_obs_offset if hasattr(u, "joint_obs_offset") else S["q"]
        for b in (u.q_target, u.q_des, u.q_des_prev, u.q_filt):
            b[:] = q_enc
        if args.history == "pose":
            dq = q_enc - u.default_q
            a = torch.clamp(dq / torch.where(dq >= 0.0, u.action_positive_scale, u.action_negative_scale), -1.0, 1.0)
            u.actions[:], u.previous_actions[:], u.previous_actions2[:] = a, a, a
        else:
            u.actions[:], u.previous_actions[:], u.previous_actions2[:] = S["a"], S["a1"], S["a2"]
        u._compute_reference(ids)
        u.ref_q_prev[:], u.ref_pos_prev[:], u.ref_quat_prev[:] = u.ref_q, u.ref_pos, u.ref_quat
        u.imu_vel_prev[:] = u._imu_point_velocity()
        applied["done"] = True

    u._reset_idx = reset_from_switch
    pi = load_policy(args.stand_ckpt, dev)
    obs, _ = env.reset()
    pose_now, _ = root_state(u)
    err_q = float((u.robot.data.joint_pos.torch - S["q"]).abs().max())
    err_p = float((pose_now[:, :3] - u.scene.env_origins - S["pose"][:, :3]).abs().max())
    print(f"[eval_switch] stand start = switch state: max |q err| {err_q:.2e} rad, max |pos err| {err_p:.2e} m")
    alive = torch.ones(m, dtype=torch.bool, device=dev)
    max_tilt = torch.zeros(m, device=dev)
    t_fall = torch.full((m,), -1.0, device=dev)
    for t in range(int(args.stand_s / u.step_dt)):
        with torch.no_grad():
            obs, _, term, _, _ = env.step(pi(obs["policy"]))
        new = alive & term
        t_fall[new] = (t + 1) * u.step_dt
        alive &= ~term
        max_tilt = torch.where(alive, torch.maximum(max_tilt, tilt_deg(u)), max_tilt)
    falls = int((~alive).sum())
    ws = json.loads((OUT / "walk_stage.json").read_text())
    to = torch.as_tensor(s["timed_out"], device=dev)
    lines = [
        f"walk ckpt  {args.walk_ckpt}",
        f"stand ckpt {args.stand_ckpt}",
        f"switch {args.switch} | history {args.history} | speed x{args.speed} | push {args.push} | envs {args.num_envs}",
        f"walk stage: falls before the switch {ws['walk_falls']} / {ws['envs']}, switched {ws['switched']}, "
        f"touchdown timeouts {ws['timeouts']}, mean wait {ws.get('wait_s_mean', 0):.2f} s, "
        f"tilt at switch {ws.get('tilt_at_switch_mean', 0):.1f} deg",
        f"stand stage ({args.stand_s:.1f} s): falls after the switch {falls} / {m} ({100.0 * falls / m:.1f} %)"
        + (f", median fall time {float(t_fall[~alive].median()):.2f} s" if falls else "")
        + f", max tilt (survivors) mean {float(max_tilt[alive].mean()) if alive.any() else float('nan'):.1f} deg",
    ]
    if args.switch == "touchdown" and bool(to.any()):
        lines.append(f"  falls among touchdown timeouts {int((~alive & to).sum())} / {int(to.sum())}")
    txt = "\n".join(lines) + "\n"
    (OUT / "result.txt").write_text(txt)
    print(txt)
    env.close()


OUT.mkdir(parents=True, exist_ok=True)
(walk_stage if args.stage == "walk" else stand_stage)()
app.close()
