"""Clearance between the legs and the Togepi exterior shell (humanoid_sim/utils/exterior.py) over a policy rollout.

    python scripts/eval/exterior_clearance.py --checkpoint <pt> [--task HumanoidSim-Olaf-Walk-Play-v0] --num_envs 64 \
        --steps 750 --headless [--out logs/eval/<MMDD>/exterior_<name>]        env: NO_PUSH=1

Final DR, random commands as in Play. The shell is not simulated (visual only in videos, no mass, no collision), so the
policy runs exactly as trained; contact is judged from the body poses, in the pelvis frame
(humanoid_sim.utils.exterior.ShellClearance: leg-link surface points vs the shell wall).
An env stops counting at its first termination (fall / timeout). Prints per link: min clearance, 1st percentile,
share of frames below 0 / 5 / 10 mm, and the worst frame (env, time, command g); saves <out>.npz.
"""
import argparse
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="HumanoidSim-Olaf-Walk-Play-v0")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=750)
ap.add_argument("--stride", type=int, default=1, help="evaluate every n-th policy step")
ap.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args()
app = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import humanoid_sim.tasks  # noqa: E402,F401
from humanoid_sim.utils.exterior import LEG_LINKS, ShellClearance  # noqa: E402

LINKS = LEG_LINKS

# ------------------------------------------------------------------ rollout
cfg = resolve_task_config(args.task, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = args.num_envs
if os.environ.get("NO_PUSH"):
    cfg.events.push_robot = None
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
dev = u.device
c = torch.load(args.checkpoint, map_location=dev, weights_only=False)
P, pre = c["policy"], c["observation_preprocessor"]
idx = sorted({int(k.split(".")[1]) for k in P if k.startswith("net_container.") and k.endswith(".weight")})


def policy(o):
    x = ((o - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8)).clamp(-5, 5)
    for j, i in enumerate(idx):
        x = x @ P[f"net_container.{i}.weight"].T + P[f"net_container.{i}.bias"]
        if j < len(idx) - 1:
            x = torch.nn.functional.elu(x)
    return x


names = list(u.robot.data.body_names)
bi_pelvis = names.index("pelvis")
bi = [names.index(n) for n in LINKS]
R_rel, t_rel, alive_log, g_log = [], [], [], []
alive = torch.ones(u.num_envs, dtype=torch.bool, device=dev)
obs, _ = env.reset()
for k in range(args.steps):
    with torch.no_grad():
        obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
    alive &= ~(term | trunc)
    if k % args.stride:
        continue
    pos = u.robot.data.body_link_pos_w.torch if hasattr(u.robot.data.body_link_pos_w, "torch") else u.robot.data.body_link_pos_w
    quat = u.robot.data.body_link_quat_w.torch if hasattr(u.robot.data.body_link_quat_w, "torch") else u.robot.data.body_link_quat_w
    Rw = matrix_from_quat(quat)  # (N, B, 3, 3)
    Rp, pp = Rw[:, bi_pelvis], pos[:, bi_pelvis]
    R_rel.append((Rp.transpose(1, 2)[:, None] @ Rw[:, bi]).cpu().numpy())
    t_rel.append((Rp.transpose(1, 2)[:, None] @ (pos[:, bi] - pp[:, None])[..., None])[..., 0].cpu().numpy())
    alive_log.append(alive.cpu().numpy())
    g_log.append(u.g.cpu().numpy().copy())
R_rel, t_rel = np.stack(R_rel), np.stack(t_rel)  # (T, N, L, 3, 3), (T, N, L, 3)
alive_log, g_log = np.stack(alive_log), np.stack(g_log)
falls = int((~alive_log[-1]).sum())
dt = u.step_dt * args.stride

# ------------------------------------------------------------------ geometry
T, N, L = R_rel.shape[:3]
clear, through = ShellClearance(LINKS)(R_rel, t_rel)  # (T, N, L)

valid = alive_log[:, :, None].repeat(L, 2)
print(f"\n{args.task}  {os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))}/{os.path.basename(args.checkpoint)}"
      f"  envs {N}  steps {args.steps} (every {args.stride})  falls {falls}  push {'off' if os.environ.get('NO_PUSH') else 'on'}")
print(f"{'link':22s} {'min mm':>8s} {'p1 mm':>7s} {'<0':>7s} {'<5mm':>7s} {'<10mm':>7s}   worst: env  t[s]  g")
for j, name in enumerate(LINKS):
    cl = clear[:, :, j][valid[:, :, j]] * 1e3
    tt, nn = np.unravel_index(np.argmin(np.where(valid[:, :, j], clear[:, :, j], np.inf)), (T, N))
    print(f"{name:22s} {cl.min():8.1f} {np.percentile(cl, 1):7.1f} {np.mean(cl < 0):7.2%} {np.mean(cl < 5):7.2%} "
          f"{np.mean(cl < 10):7.2%}   {nn:4d} {tt * dt:5.2f}  {np.round(g_log[tt, nn], 3).tolist()}")
any_c = (np.where(valid, clear, np.inf).min(-1) < 0)
print(f"RESULT frames with any leg-shell contact: {any_c[alive_log].mean():.2%}  envs with a contact: {any_c.any(0).sum()} / {N}"
      f"  (of the contact frames, link through the wall without a sampled point in it: "
      f"{(through.any(-1) & any_c)[alive_log].sum()} )")
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out + ".npz", clear=clear, through=through, alive=alive_log, g=g_log, links=np.array(LINKS),
                        dt=dt, R_rel=R_rel.astype(np.float32), t_rel=t_rel.astype(np.float32))
    print("wrote", args.out + ".npz")
env.close()
app.app.close()
