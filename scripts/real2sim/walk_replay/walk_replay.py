"""Olaf real2sim (2026-11-21): sim replay of the real walk segments (same g sequence, from the stand->walk switch), per-segment metrics vs real.
  python scripts/real2sim/walk_replay/walk_replay.py --logs <humanoid_controller arti_h1_olaf_run_*.csv> --checkpoint <walk.pt> --num_envs 64 --tag <name>   env: R2S overrides as python statements on cfg (R2S="cfg.x=..;cfg.y=..")
  DR level: start ranges (near the measured robot) unless FINAL_DR=1."""
import argparse, os, sys, json
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--tag", default="nominal"); ap.add_argument("--logs", nargs="+", required=True)
AppLauncher.add_app_launcher_args(ap); args, _ = ap.parse_known_args(); app = AppLauncher(args)
import gymnasium as gym, torch, numpy as np
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse
from isaaclab_tasks.utils import resolve_task_config
import humanoid_sim.tasks  # noqa
sys.path.insert(0, os.path.dirname(__file__)); from metrics import *; from real import load_real
cfg = resolve_task_config("HumanoidSim-Olaf-Walk-Play-v0", None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = args.num_envs; cfg.events.push_robot = None; cfg.episode_length_s = 200.0
if not os.environ.get("FINAL_DR"):
    cfg.dr_widen_steps = (10**9, 10**9 + 1); cfg.actuator_delay_extra_curriculum_steps = (10**9, 10**9 + 1)
cfg.reset_stand_pose_fraction = 1.0  # every episode starts from the stance at walk_start_phase (controller stand->walk)
if os.environ.get("R2S"): exec(os.environ["R2S"])
env = gym.make("HumanoidSim-Olaf-Walk-Play-v0", cfg=cfg); u = env.unwrapped; dev = u.device
# real controller estimator height drift (real logs 11-21: est_z -1.4..-1.5 mm/s, never reset across stand<->walk) -> obs pz
EZD, EZO = float(os.environ.get("EST_Z_DRIFT", 0.0)), float(os.environ.get("EST_Z_OFF", 0.0))
if EZD or EZO:
    _est = u._estimate
    def _estimate_drift(*a, **k):
        R = _est(*a, **k); u.est_pos[:, 2] += EZO + EZD * u.episode_length_buf.float() * u.step_dt; return R
    u._estimate = _estimate_drift
c = torch.load(args.checkpoint, map_location=dev, weights_only=False); P, pre = c["policy"], c["observation_preprocessor"]
idx = sorted({int(k.split(".")[1]) for k in P if k.startswith("net_container.") and k.endswith(".weight")})
def policy(o):
    x = ((o - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8)).clamp(-5, 5)
    for j, i in enumerate(idx):
        x = x @ P[f"net_container.{i}.weight"].T + P[f"net_container.{i}.bias"]
        if j < len(idx) - 1: x = torch.nn.functional.elu(x)
    return x
sim_names = list(u.robot.data.joint_names); pol_names = list(u.cfg.policy_joint_names)
names = [f"{s}_{j}" for j in JT for s in ("left", "right")]
res = {"real": [], "sim": [], "falls": []}; raw = []
from humanoid_sim.tasks.direct.olaf.olaf_env import _yaw_of
for f in args.logs:
    for D in load_real(f):
        G = torch.tensor(D["g_full"], dtype=torch.float32, device=dev); T = len(G)
        obs, _ = env.reset(); alive = torch.ones(u.num_envs, dtype=torch.bool, device=dev)
        rec = {k: [] for k in ("gyro", "rp", "qdes", "q", "tau", "act", "contact", "alive", "yaw", "fyaw", "phase", "obs", "taul", "eyaw", "epath", "bgz", "acc", "imp", "fvz", "qd")}
        for k in range(T):
            u.command_timer[:] = 1e9
            i_, w_ = u._walk_blend(G[k].expand(u.num_envs, 3).clone()); u.g[:] = (w_[..., None] * u.walk_keys[i_]).sum(1); u.commands[:] = u.g
            with torch.no_grad(): a = policy(obs["policy"])
            obs, _, term, trunc, _ = env.step(a)
            alive &= ~(term | trunc)
            pos, q, _, angw = u._torso(); Rm = matrix_from_quat(q)
            rec["gyro"].append(quat_apply_inverse(q, angw)); rec["alive"].append(alive.clone())
            rec["rp"].append(torch.rad2deg(torch.stack((torch.atan2(Rm[:, 2, 1], Rm[:, 2, 2]), torch.asin(-Rm[:, 2, 0].clamp(-1, 1))), -1)))
            rec["qdes"].append(u.q_des.clone())
            rec["q"].append(u.robot.data.joint_pos.torch + (u.compl * u.tau_last if getattr(u, "compl_on", False) else 0.0))  # encoder (motor side)
            rec.setdefault("taul", []).append(u.tau_last.clone())
            rec["tau"].append(torch.sqrt(u.tau_sq_acc / u.cfg.decimation)); rec["act"].append(u.actions.clone()); rec["contact"].append(u.feet_contact.float())
            rec["yaw"].append(_yaw_of(q)); rec["obs"].append(obs["policy"][:, :15].clone()); rec["phase"].append(u.phase.clone())
            rec["eyaw"].append(u.est_root_yaw.clone()); rec["epath"].append(u.est_path[:, 2].clone()); rec["bgz"].append(u.ekf.bg[:, 2].float().clone())
            # touchdown / shake diagnostics (11-27): IMU specific force (pre-delay sample), per-foot impact dv^2 sum over the
            # substeps, foot vz at the last substep (m/s), joint velocities (sim joint order)
            rec["acc"].append(u.accel_buf[:, -1].clone()); rec["imp"].append(u.impact_acc.clone())
            rec["fvz"].append(u.foot_vz_prev / u.cfg.impact_velocity_scale); rec["qd"].append(u.robot.data.joint_vel.torch.clone())
            rec["fyaw"].append(torch.stack([_yaw_of(u.robot.data.body_quat_w.torch[:, bi]) for bi in u.foot_body_indices], -1))
        R = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}  # (T, N, ...)
        Tn = len(D["t"]); sg = segments(D["g"])
        res["real"].append({s: metrics(D, m) for s, m in sg.items()})
        surv = R["alive"][-1]; falls_t = [float(np.argmin(R["alive"][:, n]) * 0.02) for n in range(u.num_envs) if not surv[n]]
        res["falls"].append({"log": f[-10:-4], "falls": int((~surv).sum()), "envs": u.num_envs, "T_s": T * 0.02, "real_fell": D["fell"], "times": sorted(falls_t)[:10]})
        per = []
        for n in range(u.num_envs):
            if not surv[n]: continue
            Dn = dict(g=D["g"], gyro=R["gyro"][:Tn, n], rp=R["rp"][:Tn, n], contact=R["contact"][:Tn, n],
                      qdes={nm: R["qdes"][:Tn, n, sim_names.index(nm)] for nm in names}, q={nm: R["q"][:Tn, n, sim_names.index(nm)] for nm in names},
                      tau={nm: R["tau"][:Tn, n, sim_names.index(nm)] for nm in names}, act={nm: R["act"][:Tn, n, pol_names.index(nm)] for nm in names})
            per.append({s: metrics(Dn, m) for s, m in sg.items()})
        res["sim"].append({s: {k: float(np.mean([p[s][k] for p in per if k in p[s]])) for k in per[0][s]} for s in sg} if per else {})
        print("FALLS", res["falls"][-1], flush=True)
        if os.environ.get("RAW"):
            raw.append({k: v for k, v in R.items() if k not in ("qdes", "tau")}); raw[-1]["names"] = np.array(sim_names); raw[-1]["pol_names"] = np.array(pol_names)
            raw[-1]["g"] = D["g_full"]; raw[-1]["log"] = f[-10:-4]
out = os.path.join(os.path.dirname(__file__), f"res_{args.tag}.json")
json.dump(res, open(out, "w"), default=float, indent=1); print("WROTE", out)
if raw:
    np.savez_compressed(out[:-5] + "_raw.npz", **{f"s{i}_{k}": v for i, r in enumerate(raw) for k, v in r.items()}); print("WROTE raw")
env.close(); app.app.close()
