"""Olaf real2sim by multiple shooting (2026-11-22): start the sim from the REAL robot state at many instants of the real walk
logs and run the same policy closed loop with the real g sequence for H seconds; compare with what the real robot did next.

  python scripts/real2sim/walk_replay/state_start.py --checkpoint <walk.pt> --tag <name> [--per_start 8] [--horizon 3.0] --headless
  env: R2S="cfg.x=..." (cfg overrides, exec'd), FINAL_DR=1 (final DR ranges; default = the start ranges near the robot),
       EST_Z_DRIFT (m/s) added to the estimator height (real controller drift).

State copied from the log row t (controller order = sim order): torso attitude (MTi-3 quaternion), torso angular velocity
(gyro) and linear velocity (estimator, obs 9:12), joint positions / velocities, temperatures, the action history
a(t-1..t-3) and the actuator chain (q_des(t-1), q_des(t-2)), gait phase, g. The torso height puts the lower sole on the
ground. The estimator restarts (as in every episode) and its path frame is then offset to the real obs (p_rel xy, yaw error,
height). The first action is the policy on the REAL obs(t).
Starts: every --every s of each walk segment + every 0.5 s in the 4 s before each real fall. Output res_ss_<tag>.json/npz."""
import argparse, os, sys, json
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--tag", default="nominal")
ap.add_argument("--per_start", type=int, default=8); ap.add_argument("--horizon", type=float, default=3.0)
ap.add_argument("--every", type=float, default=2.0); ap.add_argument("--max_envs", type=int, default=512)
ap.add_argument("--logs", nargs="+", default=[f"data/real_logs/arti_h1_olaf_run_20261121_{x}.csv"
                                              for x in ("095329", "095641", "100031")])
AppLauncher.add_app_launcher_args(ap); args, _ = ap.parse_known_args(); app = AppLauncher(args)
import gymnasium as gym, torch, numpy as np, pandas as pd, math
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul, quat_conjugate
from isaaclab_tasks.utils import resolve_task_config
import humanoid_sim.tasks  # noqa
from humanoid_sim.tasks.direct.olaf.olaf_env import _yaw_of, _yaw_quat

DT = 0.02
H = int(round(args.horizon / DT))


def rp_deg(qw, qx, qy, qz):
    R = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    P = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
    return R, P


# ---------------------------------------------------------------- real starts
segs, starts = [], []
for f in args.logs:
    d = pd.read_csv(f)
    w = d.olaf_walk.values; ch = np.r_[0, np.flatnonzero(np.diff(w)) + 1, len(d)]
    for a, b in zip(ch[:-1], ch[1:]):
        if w[a] != 1: continue
        x = d.iloc[a:b].reset_index(drop=True)
        R, P = rp_deg(x.imu_qw.values, x.imu_qx.values, x.imu_qy.values, x.imu_qz.values)
        fl = np.flatnonzero((np.abs(R) > 20) | (np.abs(P) > 20)); fall = int(fl[0]) if len(fl) else None
        si = len(segs); segs.append(dict(x=x, R=R, P=P, fall=fall, log=f[-10:-4]))
        end = (fall if fall is not None else len(x)) - 5
        for t in np.arange(int(2.0 / DT), end, int(args.every / DT)):
            starts.append((si, int(t), "steady" if fall is None or fall - t > 4.0 / DT else "late"))
        if fall is not None:
            for dtb in np.arange(4.0, 0.49, -0.5):
                t = fall - int(dtb / DT)
                if t > 5: starts.append((si, int(t), f"pre{dtb:.1f}"))
if os.environ.get("SS_ONLY"):  # "log:t[,log:t]" -> only the starts nearest to these times
    keep = []
    for item in os.environ["SS_ONLY"].split(","):
        lg, tt = item.split(":"); best = min((abs(segs[si]["x"].t.values[t] - float(tt)), (si, t, k)) for si, t, k in starts if segs[si]["log"] == lg)
        keep.append(best[1])
    starts = keep
print(f"[ss] {len(starts)} starts from {len(segs)} segments, horizon {args.horizon} s, {args.per_start} envs each", flush=True)

# ---------------------------------------------------------------- env
cfg = resolve_task_config("HumanoidSim-Olaf-Walk-Play-v0", None, overrides=["physics=isaacsim_physx"])[0]
per_batch = max(1, args.max_envs // args.per_start)
cfg.scene.num_envs = per_batch * args.per_start; cfg.events.push_robot = None; cfg.episode_length_s = 200.0
if not os.environ.get("FINAL_DR"):
    cfg.dr_widen_steps = (10**9, 10**9 + 1); cfg.actuator_delay_extra_curriculum_steps = (10**9, 10**9 + 1)
cfg.reset_stand_pose_fraction = 1.0
if os.environ.get("R2S"): exec(os.environ["R2S"])
env = gym.make("HumanoidSim-Olaf-Walk-Play-v0", cfg=cfg); u = env.unwrapped; dev = u.device; N = u.num_envs
c = torch.load(args.checkpoint, map_location=dev, weights_only=False); Pn, pre = c["policy"], c["observation_preprocessor"]
idx = sorted({int(k.split(".")[1]) for k in Pn if k.startswith("net_container.") and k.endswith(".weight")})


def policy(o):
    x = ((o - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8)).clamp(-5, 5)
    for j, i in enumerate(idx):
        x = x @ Pn[f"net_container.{i}.weight"].T + Pn[f"net_container.{i}.bias"]
        if j < len(idx) - 1: x = torch.nn.functional.elu(x)
    return x


names = list(u.robot.data.joint_names); assert names == list(u.cfg.policy_joint_names)
EZD = float(os.environ.get("EST_Z_DRIFT", 0.0))
OPENLOOP = bool(os.environ.get("SS_OPENLOOP"))  # replay the REAL actions (dynamics only) instead of the policy
ez_off = torch.zeros(N, device=dev)
_est = u._estimate


def _estimate_hook(*a, **k):
    R = _est(*a, **k)
    u.est_pos[:, 2] += ez_off + EZD * u.episode_length_buf.float() * u.step_dt
    return R


u._estimate = _estimate_hook
T = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev)
# joint zero errors, true = encoder + ZERO - c tau (foot_flat.py, standing double support): with the series compliance
# model (cfg.joint_series_compliance) / without it
ZERO = ({"left_ankle_pitch": 2.62, "right_ankle_pitch": 2.59, "left_ankle_roll": 2.36, "right_ankle_roll": 1.56} if u.compl_on else
        {"left_ankle_pitch": 3.23, "right_ankle_pitch": 1.97, "left_ankle_roll": 0.52, "right_ankle_roll": 0.79})  # deg
if os.environ.get("SS_ZERO") == "0": ZERO = {}
zero_true = T([np.radians(ZERO.get(n, 0.0)) for n in names])
foot_ids = list(u.foot_body_indices)


def sole_z():
    return u._foot_positions()[..., 2].min(-1).values


results = []
for b0 in range(0, len(starts), per_batch):
    batch = starts[b0:b0 + per_batch]
    nb = len(batch)
    rows = [segs[si]["x"].iloc[t] for si, t, _ in batch]
    prev = [[segs[si]["x"].iloc[t - k] for k in (1, 2, 3)] for si, t, _ in batch]
    rep = lambda arr: T(np.repeat(np.asarray(arr, dtype=np.float32), args.per_start, 0))  # start-major env blocks
    pad = lambda v: torch.cat((v, v[-1:].expand(N - len(v), *v.shape[1:])), 0) if len(v) < N else v
    obs, _ = env.reset()
    ids = torch.arange(N, device=dev)
    # torso attitude: MTi-3 (w, x, y, z) -> Isaac Lab (x, y, z, w)
    quat = pad(rep([[r.imu_qx, r.imu_qy, r.imu_qz, r.imu_qw] for r in rows]))
    quat = quat / quat.norm(dim=-1, keepdim=True)
    q = pad(rep([[r[f"pos_{n}"] for n in names] for r in rows])); qd = pad(rep([[r[f"vel_{n}"] for n in names] for r in rows]))
    # real joint zero errors (foot_flat.py, standing double support on the hard floor 11-21): true = encoder + ZERO
    tau_r = pad(rep([[r[f"tau_{n}"] for n in names] for r in rows]))
    q = q + zero_true[None] - (u.compl * tau_r if u.compl_on else 0.0)
    q = torch.clamp(q, u._joint_pos_lower, u._joint_pos_upper)
    v_b = pad(rep([[r.obs_9, r.obs_10, r.obs_11] for r in rows])); w_b = pad(rep([[r.imu_gx, r.imu_gy, r.imu_gz] for r in rows]))
    vel = torch.cat((quat_apply(quat, v_b), quat_apply(quat, w_b)), -1)
    pose = torch.cat((u.scene.env_origins + torch.tensor([0, 0, 1.0], device=dev), quat), -1)
    u.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
    u.robot.write_root_link_velocity_to_sim_index(root_velocity=vel * 0, env_ids=ids)
    u.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
    u.robot.write_joint_velocity_to_sim_index(velocity=qd * 0, env_ids=ids)
    u.sim.forward(); u.scene.update(0.0)
    dz = sole_z() - u.scene.env_origins[:, 2]  # lower sole height above the ground with the torso at +1 m
    pose[:, 2] -= dz - 0.001
    u.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
    u.robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
    u.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
    u.sim.forward(); u.scene.update(0.0)
    chk = (sole_z() - u.scene.env_origins[:, 2]).abs().max().item()
    if os.environ.get("SS_DEBUG") and b0 == 0:
        fp = u._foot_positions()[0] - u.scene.env_origins[0]; r0 = rows[0]
        from isaaclab.utils.math import euler_xyz_from_quat
        fq = [torch.rad2deg(torch.stack(euler_xyz_from_quat(u.robot.data.body_quat_w.torch[0:1, bi]))).flatten().tolist() for bi in foot_ids]
        print(f"DBG feet z {fp[:, 2].tolist()} contact real {r0.contact_l}{r0.contact_r} phase {r0.phase:.2f} foot rpy {fq}", flush=True)
    # actuator chain / action history (encoder frame, as the controller logs them)
    a1 = pad(rep([[p[0][f"action_{n}"] for n in names] for p in prev])); a2 = pad(rep([[p[1][f"action_{n}"] for n in names] for p in prev]))
    a3 = pad(rep([[p[2][f"action_{n}"] for n in names] for p in prev]))
    qd1 = pad(rep([[p[0][f"q_des_{n}"] for n in names] for p in prev])); qd2 = pad(rep([[p[1][f"q_des_{n}"] for n in names] for p in prev]))
    u.actions[:] = a1; u.previous_actions[:] = a2
    if hasattr(u, "previous_actions2"): u.previous_actions2[:] = a3
    # actuator-rate chain state (FOH -> slew -> low-pass -> delay line) is not logged: run the env's own chain offline over
    # the real q_des of the last WARM control steps, from the measured q at the start of that window
    WARM = 50
    hist = pad(T(np.repeat(np.stack([[[segs[si]["x"].iloc[max(t - k, 0)][f"q_des_{n}"] for n in names] for k in range(WARM + 1, 0, -1)]
                                     for si, t, _ in batch]), args.per_start, 0)))  # (N, WARM+1, J): q_des(t-WARM-1 .. t-1)
    q0 = pad(T(np.repeat(np.stack([[segs[si]["x"].iloc[max(t - WARM, 0)][f"pos_{n}"] for n in names] for si, t, _ in batch]), args.per_start, 0)))
    if hasattr(u, "joint_obs_offset"): u.joint_obs_offset[:] = -zero_true[None]  # encoder = true - ZERO (obs / targets in the encoder frame)
    qf = q0.clone()  # encoder frame (the chain runs on encoder targets)
    for kk in range(1, WARM + 1):
        for sub in range(u.cfg.decimation):
            fr = (sub + 1) / u.cfg.decimation
            uu = hist[:, kk - 1] + fr * (hist[:, kk] - hist[:, kk - 1]) if u.cfg.action_first_order_hold else hist[:, kk]
            uu = torch.clamp(uu, qf - u.slew, qf + u.slew); qf = qf + u.lp_alpha * (uu - qf)
            if hasattr(u, "actuator_buffer"):
                u.actuator_buffer_head = (u.actuator_buffer_head + 1) % u.actuator_buffer.shape[0]
                u.actuator_buffer[u.actuator_buffer_head] = qf + zero_true[None]  # the delay line holds true-frame targets
    u.q_target[:] = qf; u.q_filt[:] = qf; u.q_des[:] = qd1; u.q_des_prev[:] = qd2
    if hasattr(u, "actuator_stuck"): u.actuator_stuck[:] = False
    if u.compl_on: u.tau_last[:] = tau_r
    u.temp[:] = pad(rep([[r[f"temp_{n}"] for n in names] for r in rows])); u.temp_rate[:] = 0.0
    ph = pad(rep([[r.phase] for r in rows]))[:, 0]
    u.phase[:] = ph  # _pre_physics_step advances it to phase(t+1) with the action at t (obs(t+1) shows phase(t+1))
    u.episode_length_buf[:] = 0; u.est_init[:] = False; ez_off.zero_()
    u.path[:, :2] = pose[:, :2]; u.path[:, 2] = _yaw_of(quat)
    # real obs targets for the estimator path offsets (p_rel xy, yaw error, z) over the first steps
    G = []; OB = []; AR = []; QR = []
    for (si, t, _), r in zip(batch, rows):
        x = segs[si]["x"]; tt = np.arange(t, t + H + 1).clip(max=len(x) - 1)
        G.append(x[["g0", "g1", "g2"]].values[tt]); OB.append(x[[f"obs_{i}" for i in range(9)]].values[tt])
        AR.append(x[[f"action_{n}" for n in names]].values[tt]); QR.append(x[[f"pos_{n}" for n in names]].values[tt])
    G = pad(T(np.repeat(np.stack(G), args.per_start, 0)))
    AR = pad(T(np.repeat(np.stack(AR), args.per_start, 0))); QR = pad(T(np.repeat(np.stack(QR), args.per_start, 0)))
    OB = pad(T(np.repeat(np.stack(OB), args.per_start, 0)))
    o0 = pad(T(np.repeat(np.stack([[r[f"obs_{i}"] for i in range(80)] for r in rows]), args.per_start, 0)))
    alive = torch.ones(N, dtype=torch.bool, device=dev); rec = {k: [] for k in ("P", "R", "gz", "gx", "yaw", "pz")}
    o = o0
    for k in range(H):
        u.command_timer[:] = 1e9
        i_, w_ = u._walk_blend(G[:, k].clone()); u.g[:] = (w_[..., None] * u.walk_keys[i_]).sum(1); u.commands[:] = u.g
        with torch.no_grad(): a = AR[:, k] if OPENLOOP else policy(o)
        obs, _, term, trunc, _ = env.step(a)
        if k <= 3:  # estimator restarted on its own: re-anchor its path frame / height to the real obs of step t+k+1
            ob = OB[:, k + 1]
            e_yerr = torch.atan2(ob[:, 5], ob[:, 3])
            u.est_path[:, 2] = u.est_root_yaw - e_yerr
            c_, s_ = torch.cos(u.est_path[:, 2]), torch.sin(u.est_path[:, 2])
            u.est_path[:, 0] = u.est_pos[:, 0] - (c_ * ob[:, 0] - s_ * ob[:, 1])
            u.est_path[:, 1] = u.est_pos[:, 1] - (s_ * ob[:, 0] + c_ * ob[:, 1])
            ez_off += ob[:, 2] - u.est_pos[:, 2]; u.est_pos[:, 2] = ob[:, 2]
            obs = {"policy": u._get_observations()["policy"]} if k == 3 else obs
        o = obs["policy"]
        alive &= ~(term | trunc)
        pos, qq, _, angw = u._torso(); Rm = matrix_from_quat(qq)
        rec["R"].append(torch.rad2deg(torch.atan2(Rm[:, 2, 1], Rm[:, 2, 2]))); rec["P"].append(torch.rad2deg(torch.asin(-Rm[:, 2, 0].clamp(-1, 1))))
        wb = (Rm.transpose(1, 2) @ angw[..., None])[..., 0]; rec["gz"].append(wb[:, 2]); rec["gx"].append(wb[:, 0])
        rec["yaw"].append(_yaw_of(qq)); rec["pz"].append(o[:, 2].clone())
        rec.setdefault("q", []).append(u.robot.data.joint_pos.torch + u.joint_obs_offset + (u.compl * u.tau_last if u.compl_on else 0.0))  # encoder frame
        rec.setdefault("gy", []).append(wb[:, 1]); rec.setdefault("tau", []).append(u.tau_last.clone())
        rec.setdefault("alive", []).append(alive.clone())
        if os.environ.get("SS_DEBUG_HR") and b0 == 0 and k < 25:
            si, t, _ = batch[0]; xr = segs[si]["x"].iloc[t + k + 1]; qe = (u.robot.data.joint_pos.torch + u.joint_obs_offset)[0]; ql = u.robot.data.joint_pos.torch[0]
            print(f"HR k{k:2d} hrL enc sim {math.degrees(qe[2] + u.compl[0, 2] * u.tau_last[0, 2]):+6.1f} link {math.degrees(ql[2]):+6.1f} real enc {math.degrees(xr.pos_left_hip_roll):+6.1f} "
                  f"tau sim {u.tau_last[0, 2]:+5.2f} real {xr.tau_left_hip_roll:+5.2f} | hrR enc sim {math.degrees(qe[3] + u.compl[0, 3] * u.tau_last[0, 3]):+6.1f} real {math.degrees(xr.pos_right_hip_roll):+6.1f} "
                  f"tau sim {u.tau_last[0, 3]:+5.2f} real {xr.tau_right_hip_roll:+5.2f} | target {math.degrees(u.q_target[0, 2]):+6.1f}", flush=True)
        if os.environ.get("SS_DEBUG") and b0 == 0 and k < int(os.environ.get("SS_DEBUG")):
            si, t, _ = batch[0]; xr = segs[si]["x"].iloc[t + k + 1]; qs = u.robot.data.joint_pos.torch[0]
            print(f"DBG k{k:2d} P sim {rec['P'][-1][0]:+6.1f} real {segs[si]['P'][t + k + 1]:+6.1f} | R sim {rec['R'][-1][0]:+6.1f} real {segs[si]['R'][t + k + 1]:+6.1f} | "
                  f"gy sim {(Rm[0].T @ angw[0])[1]:+5.2f} real {xr.imu_gy:+5.2f} | knee L sim {math.degrees(qs[6]):+5.0f} real {math.degrees(xr.pos_left_knee):+5.0f} "
                  f"R sim {math.degrees(qs[7]):+5.0f} real {math.degrees(xr.pos_right_knee):+5.0f} | gz sim {(Rm[0].T @ angw[0])[2]:+5.2f} real {xr.imu_gz:+5.2f} | "
                  f"apL {math.degrees(qs[8]):+4.0f}/{math.degrees(xr.pos_left_ankle_pitch):+4.0f} apR {math.degrees(qs[9]):+4.0f}/{math.degrees(xr.pos_right_ankle_pitch):+4.0f} "
                  f"hpL {math.degrees(qs[4]):+4.0f}/{math.degrees(xr.pos_left_hip_pitch):+4.0f} hpR {math.degrees(qs[5]):+4.0f}/{math.degrees(xr.pos_right_hip_pitch):+4.0f} "
                  f"c {u.feet_contact[0].int().tolist()}/{int(xr.contact_l)}{int(xr.contact_r)} ph {u.phase[0]:.2f}/{xr.phase:.2f}", flush=True)
    Rr = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    for j, ((si, t, kind), r) in enumerate(zip(batch, rows)):
        s = segs[si]; e = slice(j * args.per_start, (j + 1) * args.per_start)
        f = s["fall"]; real_fall = f is not None and f - t <= H
        tt = np.arange(t + 1, t + H + 1).clip(max=len(s["x"]) - 1)
        al = Rr["alive"][:, e]
        sim_fall_t = [int(np.argmin(al[:, n])) for n in range(al.shape[1]) if not al[-1, n]]
        m = min(50, H)  # first second: trajectory error vs real (sim envs alive)
        res = dict(log=s["log"], t=float(s["x"].t.values[t]), kind=kind, real_fall=bool(real_fall),
                   real_fall_in=float((f - t) * DT) if f is not None else None, g2=float(r.g2), phase=float(r.phase),
                   sim_falls=len(sim_fall_t), n=args.per_start, sim_fall_s=[x * DT for x in sim_fall_t],
                   real_pitch_rng=float(np.ptp(s["P"][tt][:min(H, (f - t) if f else H)])) if (f is None or f - t > 5) else None,
                   sim_pitch_rng=float(np.median([np.ptp(Rr["P"][:, e.start + n][al[:, n]]) for n in range(al.shape[1]) if al[:, n].sum() > 5] or [np.nan])),
                   P_err_1s=float(np.sqrt(np.nanmean((Rr["P"][:m, e] - s["P"][tt][:m, None]) ** 2))),
                   gz_err_1s=float(np.sqrt(np.nanmean((Rr["gz"][:m, e] - s["x"].imu_gz.values[tt][:m, None]) ** 2))),
                   R_err_1s=float(np.sqrt(np.nanmean((Rr["R"][:m, e] - s["R"][tt][:m, None]) ** 2))), sole_chk=chk)
        res["errh"] = {}
        for hh in (5, 10, 25):
            hh = min(hh, H)
            res["errh"][hh] = {"P": float(np.sqrt(np.mean((Rr["P"][:hh, e] - s["P"][tt][:hh, None]) ** 2))),
                               "R": float(np.sqrt(np.mean((Rr["R"][:hh, e] - s["R"][tt][:hh, None]) ** 2))),
                               "gx": float(np.sqrt(np.mean((Rr["gx"][:hh, e] - s["x"].imu_gx.values[tt][:hh, None]) ** 2))),
                               "gy": float(np.sqrt(np.mean((Rr["gy"][:hh, e] - s["x"].imu_gy.values[tt][:hh, None]) ** 2)))}
        h = min(25, H); qr = s["x"][[f"pos_{n}" for n in names]].values[tt][:h]
        res["err05"] = {"P": float(np.sqrt(np.mean((Rr["P"][:h, e] - s["P"][tt][:h, None]) ** 2))),
                        "R": float(np.sqrt(np.mean((Rr["R"][:h, e] - s["R"][tt][:h, None]) ** 2))),
                        "gx": float(np.sqrt(np.mean((Rr["gx"][:h, e] - s["x"].imu_gx.values[tt][:h, None]) ** 2))),
                        "gy": float(np.sqrt(np.mean((Rr["gy"][:h, e] - s["x"].imu_gy.values[tt][:h, None]) ** 2))),
                        "gz": float(np.sqrt(np.mean((Rr["gz"][:h, e] - s["x"].imu_gz.values[tt][:h, None]) ** 2)))}
        tr = s["x"][[f"tau_{n}" for n in names]].values[tt][:h]
        res["tau_pair"] = [Rr["tau"][:h, e.start].tolist(), tr.tolist()]
        for jt in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            jj = [names.index(f"{sd}_{jt}") for sd in ("left", "right")]
            res["err05"][jt] = float(np.degrees(np.sqrt(np.mean((Rr["q"][:h, e][..., jj] - qr[:, None, jj]) ** 2))))
        results.append(res)
    print(f"[ss] batch {b0 // per_batch + 1}/{math.ceil(len(starts) / per_batch)} done, sole height check {chk * 1000:.1f} mm", flush=True)

out = os.path.join(os.path.dirname(__file__), f"res_ss_{args.tag}.json")
json.dump(results, open(out, "w"), indent=1, default=float)
by = {}
for r in results:
    k = "steady" if r["kind"] in ("steady", "late") and not r["real_fall"] else ("pre-fall" if r["real_fall"] else "other")
    by.setdefault(k, []).append(r)
for k, rs in by.items():
    print(f"SUMMARY {k:9s} starts {len(rs):3d} | real fell {sum(r['real_fall'] for r in rs):3d} | sim fall frac "
          f"{np.mean([r['sim_falls'] / r['n'] for r in rs]):.3f} | P err 1s {np.nanmean([r['P_err_1s'] for r in rs]):.2f} deg "
          f"gz err {np.nanmean([r['gz_err_1s'] for r in rs]):.2f} R err {np.nanmean([r['R_err_1s'] for r in rs]):.2f}")
ek = list(results[0]["err05"].keys())
for hh in results[0]["errh"]:
    rs = by.get("steady", [])
    print(f"ERRH steady h={int(hh) * 0.02:.2f}s " + " ".join(f"{k} {np.mean([r['errh'][hh][k] for r in rs]):.3f}" for k in ("P", "R", "gx", "gy")))
ts = np.concatenate([np.array(r["tau_pair"][0]) for r in results]); trl = np.concatenate([np.array(r["tau_pair"][1]) for r in results])
print("TAUCORR sim vs real torque per joint: " + " ".join(f"{n}={np.corrcoef(ts[:, i], trl[:, i])[0, 1]:+.2f}(sim mean {ts[:, i].mean():+.2f} real {trl[:, i].mean():+.2f})" for i, n in enumerate(names)))
for grp in ("steady", "pre-fall"):
    rs = by.get(grp, [])
    if rs: print(f"ERR05 {grp:9s} " + " ".join(f"{k} {np.mean([r['err05'][k] for r in rs]):.3f}" for k in ek))
for r in results:
    if r["real_fall"]: print(f"  PRE {r['log']} t {r['t']:6.1f} fall in {r['real_fall_in']:.2f}s g2 {r['g2']:+.2f} sim falls {r['sim_falls']}/{r['n']} at {[round(x, 2) for x in r['sim_fall_s']]}")
print("WROTE", out)
env.close(); app.app.close()
