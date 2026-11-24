"""Olaf-style imitation env for the ARTI-H1 KB robot (arXiv 2512.16705, see olaf_env_cfg.py).

Built on the copied ARTI-H1 velocity env for the robot / sim-to-real models only (scene, actuator delay / Coulomb /
armature model, MTi-3 IMU model, joint zero errors, stale bus reads, pushes, yaw bias, joint target filter, trunk
terminations). Commands, path frame, reference, observations, rewards and resets are the paper's.

Quaternions are (x, y, z, w) (Isaac Lab 3). Joint tensors are in sim order inside the env, policy I/O in
ARTI_H1_POLICY_JOINT_NAMES order (_joints_to_policy), as in the deployed ARTI-H1 controller.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_conjugate,
    quat_mul,
)

from humanoid_sim.robots.arti_h1 import JOINT_EFFORT_LIMIT

from ..base.arti_h1_velocity_env import ArtiH1VelocityEnv
from .inekf import InEKF
from .olaf_env_cfg import ArtiH1OlafEnvCfg


def _yaw_quat(psi: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(psi)
    return torch.stack((z, z, torch.sin(0.5 * psi), torch.cos(0.5 * psi)), dim=-1)


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot2(psi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(psi), torch.sin(psi)
    return torch.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]), dim=-1)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class ArtiH1OlafEnv(ArtiH1VelocityEnv):
    cfg: ArtiH1OlafEnvCfg

    REWARD_TERMS = (
        "torso_pos_xy", "torso_ori", "lin_vel_xy", "lin_vel_z", "ang_vel_xy", "ang_vel_z", "leg_joint_pos",
        "leg_joint_vel", "contact", "survival", "joint_torques", "joint_acc", "leg_action_rate", "leg_action_acc",
        "temperature", "joint_limits_lower", "joint_limits_upper", "foot_collision", "impact", "yaw_drift",
    )

    def __init__(self, cfg: ArtiH1OlafEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.walk = cfg.mode == "walk"
        names = list(self.robot.data.joint_names)
        jtype = [nm.split("_", 1)[1] for nm in names]

        # ---------------- reference (reference/build_reference.py) ----------------
        ref = np.load(cfg.reference_file)
        ref_joints = [str(x) for x in ref["joints"]]
        perm = torch.tensor([ref_joints.index(nm) for nm in names], device=dev)
        tab = torch.tensor(ref["walk_table"], device=dev, dtype=torch.float32)
        tab[..., 0:12] = tab[..., 0:12][..., perm]
        tab[..., 12:24] = tab[..., 12:24][..., perm]
        self.walk_table = tab  # (clips, bins, 39)
        self.walk_keys = torch.tensor(ref["walk_keys"], device=dev, dtype=torch.float32)
        self.walk_period = float(ref["walk_period"])
        self.walk_bins = tab.shape[1]
        rng = torch.tensor(cfg.walk_command_range, device=dev, dtype=torch.float32)
        self.walk_lo, self.walk_hi = rng[:, 0], rng[:, 1]
        self.walk_scale = 1.0 / (self.walk_hi - self.walk_lo)
        self.stand_axes = [torch.tensor(ref[f"stand_axes_{a}"], device=dev, dtype=torch.float32)
                           for a in ("dz", "roll", "pitch", "yaw")]
        self.stand_q = torch.tensor(ref["stand_q"], device=dev, dtype=torch.float32)[..., perm]
        self.stand_root_pos = torch.tensor(ref["stand_root_pos"], device=dev, dtype=torch.float32)
        self.stand_root_quat = torch.tensor(ref["stand_root_quat"], device=dev, dtype=torch.float32)
        srng = torch.tensor(cfg.stand_command_range, device=dev, dtype=torch.float32)
        self.stand_lo, self.stand_hi = srng[:, 0], srng[:, 1]
        self.stand_rate = torch.tensor(cfg.stand_command_rate, device=dev, dtype=torch.float32)
        self.switch_bank = None
        if not self.walk and cfg.switch_bank_file:
            f = cfg.switch_bank_file
            if not os.path.dirname(f):
                f = os.path.join(os.path.dirname(cfg.reference_file), f)
            b = np.load(f)
            jn = [str(x) for x in b["joint_names"]] if "joint_names" in b else names
            jp = torch.tensor([jn.index(nm) for nm in names], device=dev)
            pose = torch.tensor(b["pose"], device=dev, dtype=torch.float32)
            pose[:, :2] = 0.0  # torso over the env origin (the walk drifted it), the path frame follows it
            self.switch_bank = {
                "pose": pose, "vel": torch.tensor(b["vel"], device=dev, dtype=torch.float32),
                "q": torch.tensor(b["q"], device=dev, dtype=torch.float32)[:, jp],
                "qd": torch.tensor(b["qd"], device=dev, dtype=torch.float32)[:, jp]}
            print(f"[olaf] switch bank {f}: {len(pose)} walk -> stand states")

        # ---------------- state ----------------
        gdim = 3 if self.walk else 4
        self.g = torch.zeros(n, gdim, device=dev)  # control input the policy sees
        self.g_target = torch.zeros(n, gdim, device=dev)  # standing: g follows this with a rate limit
        self.phase = torch.zeros(n, device=dev)
        self.path = torch.zeros(n, 3, device=dev)  # x, y (world), psi
        self.ref_q = self.robot.data.default_joint_pos.torch.clone()
        self.ref_q_prev = self.ref_q.clone()
        self.ref_qd = torch.zeros(n, 12, device=dev)
        self.ref_contact = torch.ones(n, 2, device=dev)
        self.ref_pos = torch.zeros(n, 3, device=dev)
        self.ref_quat = torch.zeros(n, 4, device=dev)
        self.ref_quat[:, 3] = 1.0
        self.ref_v = torch.zeros(n, 3, device=dev)
        self.ref_w = torch.zeros(n, 3, device=dev)
        # standing: previous reference torso pose, for the reference torso velocity while g (or the path frame) moves
        self.ref_pos_prev = self.ref_pos.clone()
        self.ref_quat_prev = self.ref_quat.clone()

        # thermal model (eq. 6), per joint type
        self.th_alpha = torch.tensor([cfg.thermal_alpha[t] for t in jtype], device=dev)[None]
        self.th_beta = torch.tensor([cfg.thermal_beta[t] for t in jtype], device=dev)[None]
        lo, hi = cfg.thermal_clip
        if cfg.thermal_gamma > 0.0:
            self.th_gamma = float(cfg.thermal_gamma)
        else:
            # feasibility at the upper clip with tau = 0, slowest-cooling joint and warmest ambient (Sec. VII-A)
            self.th_gamma = float(self.th_alpha.min() * (hi - cfg.thermal_ambient_range[1]) / (hi - cfg.thermal_max))
        self.temp = torch.full((n, 12), cfg.thermal_ambient_range[0], device=dev)
        self.temp_amb = torch.full((n, 1), cfg.thermal_ambient_range[0], device=dev)
        self.temp_rate = torch.zeros(n, 12, device=dev)

        # per physics step accumulators: squared torque (thermal model, torque reward), foot vertical velocity change
        self.effort = torch.tensor([JOINT_EFFORT_LIMIT[nm] for nm in names], device=dev)[None]
        self.tau_sq_acc = torch.zeros(n, 12, device=dev)
        # band-limited torso torque noise (cfg.torso_torque_noise_*): difference of two first-order low-passes of white
        # noise, normalised to unit std at the physics rate
        self.tt_lo = torch.zeros(n, 3, device=dev)
        self.tt_hi = torch.zeros(n, 3, device=dev)
        self.tt_scale = torch.ones(n, 1, device=dev)
        pdt = cfg.sim.dt
        a_lo, a_hi, norm = [], [], []
        g = torch.Generator().manual_seed(0)
        w = torch.randn(200_000, generator=g).tolist()
        for f_lo, f_hi in cfg.torso_torque_noise_band_hz:
            al, ah = 1.0 - math.exp(-2 * math.pi * f_lo * pdt), 1.0 - math.exp(-2 * math.pi * f_hi * pdt)
            lo = hi = 0.0
            out = []
            for x in w:  # one-off filter gain on the CPU
                lo += al * (x - lo)
                hi += ah * (x - hi)
                out.append(hi - lo)
            a_lo.append(al); a_hi.append(ah); norm.append(1.0 / float(np.std(out[2000:])))
        self.tt_a_lo = torch.tensor(a_lo, device=dev)
        self.tt_a_hi = torch.tensor(a_hi, device=dev)
        self.tt_norm = torch.tensor(norm, device=dev)
        self.tt_std = torch.tensor(cfg.torso_torque_noise_std, device=dev, dtype=torch.float32)
        jt_of = lambda nm: next((t for t in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll") if t in nm), None)
        comp = getattr(cfg, "joint_series_compliance", None) or {}
        self.compl_base = torch.tensor([float(comp.get(jt_of(nm), 0.0)) for nm in names], device=dev)[None]
        self.compl = self.compl_base.expand(n, -1).clone()
        self.compl_on = bool((self.compl_base > 0).any())
        self.tau_last = torch.zeros(n, len(names), device=dev)
        self.wz_err_ema = torch.zeros(n, device=dev)  # low-passed torso yaw-rate error vs the reference (yaw_drift)
        if getattr(cfg, "est_z_offset_range", None) is not None:
            self.est_z_off = torch.zeros(n, device=dev)
            self.est_z_rate = torch.zeros(n, device=dev)
        self.substeps = 0
        self.foot_vz_prev = torch.zeros(n, 2, device=dev)
        self.impact_skip = torch.ones(n, dtype=torch.bool, device=dev)
        self.impact_acc = torch.zeros(n, 2, device=dev)
        self.applied_target = self.robot.data.default_joint_pos.torch.clone()

        # runtime chain (Sec. VII-C): policy target q_des at 50 Hz -> first-order hold between the last two targets over
        # the control period -> slew limit -> first-order low-pass at action_lowpass_hz, every physics (actuator) step
        self.q_des = self.robot.data.default_joint_pos.torch.clone()
        self.q_des_prev = self.q_des.clone()
        self.q_filt = self.q_des.clone()
        sim_dt = float(cfg.sim.dt)
        self.lp_alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.action_lowpass_hz * sim_dt) if cfg.action_lowpass_hz > 0 else 1.0
        self.slew = cfg.action_rate_limit * sim_dt if cfg.action_rate_limit > 0 else float("inf")

        # robot-side state estimate for the actor (humanoid_controller robot::OlafInEKF): contact-aided invariant EKF on
        # the MTi-3 gyro / accelerometer and the leg kinematics, in its own world frame, with its own path frame
        self.ekf = InEKF(n, dev, gyro_noise=cfg.ekf_gyro_noise, accel_noise=cfg.ekf_accel_noise,
                         gyro_bias_noise=cfg.ekf_gyro_bias_noise, accel_bias_noise=cfg.ekf_accel_bias_noise,
                         contact_noise=cfg.ekf_contact_noise, fk_noise=cfg.ekf_fk_noise, init_std=dict(cfg.ekf_init_std))
        self.imu_r = torch.tensor(cfg.imu_translation, device=dev, dtype=torch.float32)  # IMU in the pelvis frame
        # IMU point velocity accel_average_s before the observation (the Xsens acceleration output is dV / dt over its
        # output interval)
        self.accel_avg_steps = min(cfg.decimation, max(1, int(round(cfg.accel_average_s / sim_dt))))
        self.imu_vel_prev = torch.zeros(n, 3, device=dev)
        self.accel_bias = torch.zeros(n, 3, device=dev)
        self.accel_base = torch.tensor(cfg.accel_base_noise, device=dev)
        self.accel_impact = torch.tensor(cfg.accel_impact_noise, device=dev)
        self.accel_impact_scale = torch.zeros(n, 1, device=dev)
        self.accel_contact_prev = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self._dr_imu_final = (cfg.mti3_gyro_noise_std_rad_s, tuple(cfg.mti3_gyro_bias_range_rad_s),
                              tuple(cfg.accel_impact_scale_range), tuple(cfg.accel_bias_range))
        self.accel_buf = torch.zeros(n, self.mti3_imu.delay_steps + 1, 3, device=dev)
        self.accel_buf[..., 2] = 9.81
        self.est_init = torch.zeros(n, dtype=torch.bool, device=dev)
        self.est_pos = torch.zeros(n, 3, device=dev)
        self.est_vel = torch.zeros(n, 3, device=dev)
        self.est_R = torch.eye(3, device=dev).repeat(n, 1, 1)
        self.est_feet = torch.zeros(n, 2, 3, device=dev)
        self.est_feet_yaw = torch.zeros(n, 2, device=dev)
        self.est_root_yaw = torch.zeros(n, device=dev)
        self.est_path = torch.zeros(n, 3, device=dev)

        # joint limits for the CBF terms (the sim's hard limits = real motor limits, hip_roll +-15 deg)
        self.q_min = self._joint_pos_lower + cfg.joint_limit_margin
        self.q_max = self._joint_pos_upper - cfg.joint_limit_margin
        self.default_q = self.robot.data.default_joint_pos.torch
        print(f"[OLAF] mode {cfg.mode}: walk ref {tab.shape[0]} clips x {self.walk_bins} bins (period {self.walk_period:.2f} s), "
              f"stand grid {tuple(self.stand_q.shape[:-1])}; thermal gamma_T {self.th_gamma:.4f}; obs {cfg.observation_space}, "
              f"critic {cfg.state_space}", flush=True)

    # ------------------------------------------------------------------ reference
    def _walk_blend(self, g: torch.Tensor):
        d = torch.cdist(g * self.walk_scale, self.walk_keys * self.walk_scale)
        dist, idx = torch.topk(d, self.cfg.reference_knn, dim=1, largest=False)
        w = 1.0 / (dist + 1e-3)
        return idx, w / w.sum(dim=1, keepdim=True)

    def _walk_rows(self, g: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        idx, w = self._walk_blend(g)
        x = phase * self.walk_bins
        b0 = torch.floor(x).long() % self.walk_bins
        b1 = (b0 + 1) % self.walk_bins
        f = (x - torch.floor(x))[:, None, None]
        r = (1.0 - f) * self.walk_table[idx, b0[:, None].expand_as(idx)] + f * self.walk_table[idx, b1[:, None].expand_as(idx)]
        r = (w[..., None] * r).sum(1)
        q = r[:, 29:33]
        r[:, 29:33] = q / q.norm(dim=-1, keepdim=True)
        return r

    def _stand_lookup(self, g: torch.Tensor):
        """Multilinear interpolation of the IK grid at g = (dz, roll, pitch, yaw)."""
        i0s, fs = [], []
        for a, ax in enumerate(self.stand_axes):
            x = g[:, a].clamp(ax[0], ax[-1])
            i0 = torch.clamp(torch.searchsorted(ax, x.contiguous(), right=True) - 1, 0, len(ax) - 2)
            f = ((x - ax[i0]) / (ax[i0 + 1] - ax[i0])).clamp(0.0, 1.0)
            i0s.append(i0)
            fs.append(f)
        q = torch.zeros(g.shape[0], 12, device=self.device)
        p = torch.zeros(g.shape[0], 3, device=self.device)
        r = torch.zeros(g.shape[0], 4, device=self.device)
        for corner in range(16):
            bits = [(corner >> k) & 1 for k in range(4)]
            w = torch.ones(g.shape[0], device=self.device)
            ii = []
            for k in range(4):
                w = w * (fs[k] if bits[k] else 1.0 - fs[k])
                ii.append(i0s[k] + bits[k])
            q += w[:, None] * self.stand_q[ii[0], ii[1], ii[2], ii[3]]
            p += w[:, None] * self.stand_root_pos[ii[0], ii[1], ii[2], ii[3]]
            r += w[:, None] * self.stand_root_quat[ii[0], ii[1], ii[2], ii[3]]
        return q, p, r / r.norm(dim=-1, keepdim=True)

    @staticmethod
    def _resolve_walk(r: torch.Tensor, path: torch.Tensor, z0: torch.Tensor):
        """Walk reference rows (path-frame relative) -> joints, contacts, world torso pose / velocity."""
        psi = path[:, 2]
        pos = torch.cat((path[:, :2] + _rot2(psi, r[:, 26:28]), (r[:, 28] + z0)[:, None]), dim=-1)
        quat = quat_mul(_yaw_quat(psi), r[:, 29:33])
        v = torch.cat((_rot2(psi, r[:, 33:35]), r[:, 35:36]), dim=-1)
        w = torch.cat((_rot2(psi, r[:, 36:38]), r[:, 38:39]), dim=-1)
        # contact flags: the phase interpolation and the KNN blend make them fractional; the reward compares them with the
        # 0/1 sole contact by equality, so binarise (a fractional flag never matched: ~22% of the contact reward was lost)
        contact = (r[:, 24:26] > 0.5).float()
        return r[:, 0:12], r[:, 12:24], contact, pos, quat, v, w

    def _compute_reference(self, env_ids: torch.Tensor | None = None):
        """x_t = f(p_PF, g, phi) (eq. 2), resolved in the world: torso pose / velocity, joints, contacts."""
        ids = slice(None) if env_ids is None else env_ids
        path, g = self.path[ids], self.g[ids]
        psi = path[:, 2]
        z0 = self.scene.env_origins[ids, 2]
        if self.walk:
            q, qd, contact, pos, quat, v, w = self._resolve_walk(self._walk_rows(g, self.phase[ids]), path, z0)
        else:
            q, p, rq = self._stand_lookup(g)
            qd = (q - self.ref_q_prev[ids]) / self.step_dt if env_ids is None else torch.zeros_like(q)
            contact = torch.ones(q.shape[0], 2, device=self.device)
            pos = torch.cat((path[:, :2] + _rot2(psi, p[:, :2]), (p[:, 2] + z0)[:, None]), dim=-1)
            quat = quat_mul(_yaw_quat(psi), rq)
            if env_ids is None:
                # the reference moves while g follows its target (and the path frame converges): torso velocity of that
                # motion (world), finite difference over the control step like qd
                v = (pos - self.ref_pos_prev) / self.step_dt
                w = quat_box_minus(quat, self.ref_quat_prev) / self.step_dt
            else:
                v = torch.zeros_like(pos)
                w = torch.zeros_like(pos)
        self.ref_q[ids], self.ref_qd[ids], self.ref_contact[ids] = q, qd, contact
        self.ref_pos[ids], self.ref_quat[ids], self.ref_v[ids], self.ref_w[ids] = pos, quat, v, w

    # ------------------------------------------------------------------ commands (control input g)
    def _sample_commands(self, env_ids: torch.Tensor):
        k = len(env_ids)
        if k == 0 or not hasattr(self, "g"):
            return
        dev, cfg = self.device, self.cfg
        if self.walk:
            g = self.walk_lo + torch.rand(k, 3, device=dev) * (self.walk_hi - self.walk_lo)
            g[torch.rand(k, device=dev) < cfg.walk_zero_command_fraction] = 0.0
            # project onto the reference coverage: the KNN blend of the clip keys (commanded path == reference path)
            idx, w = self._walk_blend(g)
            self.g[env_ids] = (w[..., None] * self.walk_keys[idx]).sum(1)
        else:
            g = self.stand_lo + torch.rand(k, 4, device=dev) * (self.stand_hi - self.stand_lo)
            g[torch.rand(k, device=dev) < cfg.stand_nominal_fraction] = 0.0
            self.g_target[env_ids] = g
        self.commands[env_ids] = 0.0
        if self.walk:
            self.commands[env_ids] = self.g[env_ids]
        lo, hi = cfg.command_resample_time_s
        self.command_timer[env_ids] = torch.empty(k, device=dev).uniform_(lo, hi)

    # ------------------------------------------------------------------ step
    def _torso(self):
        d = self.robot.data
        b = self.ref_body_index
        # link-origin velocity (body_lin_vel_w is the centre of mass, 0.37 m above the pelvis link origin): the reference
        # torso velocity, the estimator and the RSI writes are all about the link origin
        return d.body_pos_w.torch[:, b], d.body_quat_w.torch[:, b], d.body_link_lin_vel_w.torch[:, b], d.body_ang_vel_w.torch[:, b]

    def _feet_frame(self):
        """Midpoint between the soles and their mean heading (right foot link faces -x: its yaw + pi)."""
        feet = self._foot_positions()
        d = self.robot.data
        yl = _yaw_of(d.body_quat_w.torch[:, self.foot_body_indices[0]])
        yr = _yaw_of(d.body_quat_w.torch[:, self.foot_body_indices[1]]) + math.pi
        psi = torch.atan2(torch.sin(yl) + torch.sin(yr), torch.cos(yl) + torch.cos(yr))
        return 0.5 * (feet[:, 0, :2] + feet[:, 1, :2]), psi

    def _step_path(self, path, pos_xy, yaw, mid, psi_f):
        """One control step of the path frame (in place): walking integrates g, standing converges to the feet
        midpoint / heading; then it is kept within path_max_distance / path_max_yaw of the torso (Sec. V)."""
        dt, cfg = self.step_dt, self.cfg
        if self.walk:
            psi = path[:, 2].clone()
            path[:, :2] += _rot2(psi, self.g[:, :2]) * dt
            path[:, 2] = psi + self.g[:, 2] * dt
        else:
            a = 1.0 - math.exp(-dt / cfg.stand_path_tau_s)
            path[:, :2] += a * (mid - path[:, :2])
            path[:, 2] += a * _wrap(psi_f - path[:, 2])
        e = path[:, :2] - pos_xy
        dist = e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        path[:, :2] = pos_xy + e * torch.clamp(cfg.path_max_distance / dist, max=1.0)
        path[:, 2] = yaw + torch.clamp(_wrap(path[:, 2] - yaw), -cfg.path_max_yaw, cfg.path_max_yaw)

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)  # actions, command resampling, PD target (scale, real limits)
        # the new policy target; the actuator-rate chain ramps from the previous one (_apply_action)
        self.q_des_prev.copy_(self.q_des)
        self.q_des.copy_(self.q_target)
        dt = self.step_dt
        pos, quat, _, _ = self._torso()
        if self.walk:
            self.phase = (self.phase + dt / self.walk_period) % 1.0
            mid, psi_f = None, None
        else:
            step = self.stand_rate * dt
            self.ref_q_prev.copy_(self.ref_q)
            self.ref_pos_prev.copy_(self.ref_pos)
            self.ref_quat_prev.copy_(self.ref_quat)
            self.g += torch.clamp(self.g_target - self.g, -step, step)
            mid, psi_f = self._feet_frame()
        self._step_path(self.path, pos[:, :2], _yaw_of(quat), mid, psi_f)
        if self.cfg.actor_state_estimator:
            # the robot-side path frame, integrated in the estimator frame from the last estimate (controller order)
            e_mid = self.est_feet[:, :, :2].mean(1)
            yl, yr = self.est_feet_yaw[:, 0], self.est_feet_yaw[:, 1] + math.pi
            e_psi = torch.atan2(torch.sin(yl) + torch.sin(yr), torch.cos(yl) + torch.cos(yr))
            self._step_path(self.est_path, self.est_pos[:, :2], self.est_root_yaw, e_mid, e_psi)
        self.tau_sq_acc.zero_()
        self.impact_acc.zero_()
        self.substeps = 0

    def _apply_action(self):
        # actuator-rate chain: first-order hold (linear between the last two policy targets), slew limit, low-pass
        k = (self.substeps + 1) / self.cfg.decimation
        u = self.q_des_prev + k * (self.q_des - self.q_des_prev) if self.cfg.action_first_order_hold else self.q_des
        u = torch.clamp(u, self.q_filt - self.slew, self.q_filt + self.slew)
        self.q_filt += self.lp_alpha * (u - self.q_filt)
        self.q_target.copy_(self.q_filt)
        target = self.q_target.clone()
        if hasattr(self, "joint_obs_offset"):
            target = target - self.joint_obs_offset  # encoder-frame targets (homing zero error)
        if self.cfg.actuator_model:
            target = self._apply_actuator_model(target)
        d = self.robot.data
        if self.compl_on:
            # series compliance: the motor PD acts on the encoder (motor side) q_m = q + c tau, so the link torque is
            # (kp (target - q) - kd qd) / (1 + kp c); sent as the equivalent implicit-drive target on the link
            kp, kd = d.joint_stiffness.torch.clamp_min(1e-3), d.joint_damping.torch
            q, qd = d.joint_pos.torch, d.joint_vel.torch
            tau_c = (kp * (target - q) - kd * qd) / (1.0 + kp * self.compl)
            tau_c = torch.maximum(torch.minimum(tau_c, self.effort), -self.effort)
            target = q + (tau_c + kd * qd) / kp
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_balance_support()
        self._apply_yaw_torque_bias()
        self._apply_torso_torque_noise()
        # PD torque the motor produces this physics step (implicit actuator: same law, clamped at the effort limit)
        d = self.robot.data
        tau = d.joint_stiffness.torch * (target - d.joint_pos.torch) - d.joint_damping.torch * d.joint_vel.torch
        tau = torch.maximum(torch.minimum(tau, self.effort), -self.effort)
        self.tau_last = tau
        self.tau_sq_acc += tau * tau
        # impact reduction: change of the foot vertical velocity since the previous physics step, saturated
        vz = self._foot_velocities()[..., 2] * self.cfg.impact_velocity_scale
        dv2 = torch.square(vz - self.foot_vz_prev) * (~self.impact_skip)[:, None]
        self.impact_skip.zero_()
        self.impact_acc += torch.clamp(dv2, max=self.cfg.impact_dv_max ** 2)
        self.foot_vz_prev.copy_(vz)
        if self.substeps == self.cfg.decimation - self.accel_avg_steps:
            self.imu_vel_prev.copy_(self._imu_point_velocity())
        self.substeps += 1

    def _update_dr_widen(self):
        """Base DR widening + the IMU terms (MTi-3 gyro noise / bias, accelerometer impact scale / bias), lerped from the
        dr_start_* values to the final cfg values at the same levels. The MTi-3 model and the resets read these cfg fields."""
        before = getattr(self, "_dr_level", -1)
        super()._update_dr_widen()
        level = getattr(self, "_dr_level", -1)
        if level == before or not hasattr(self, "_dr_imu_final"):
            return
        cfg = self.cfg
        fq = level / (max(2, int(cfg.dr_widen_levels)) - 1)
        L = lambda a, b: self._dr_lerp(a, b, fq)
        g_noise, g_bias, a_scale, a_bias = self._dr_imu_final
        cfg.mti3_gyro_noise_std_rad_s = L(cfg.dr_start_gyro_noise, g_noise)
        cfg.mti3_gyro_bias_range_rad_s = L(cfg.dr_start_gyro_bias, g_bias)
        cfg.accel_impact_scale_range = L(cfg.dr_start_accel_impact_scale, a_scale)
        cfg.accel_bias_range = L(cfg.dr_start_accel_bias, a_bias)
        print(f"[DR] widen level {level}: gyro noise {cfg.mti3_gyro_noise_std_rad_s:.3f} bias {cfg.mti3_gyro_bias_range_rad_s}, "
              f"accel impact scale {cfg.accel_impact_scale_range} bias {cfg.accel_bias_range}", flush=True)

    def _imu_point_velocity(self) -> torch.Tensor:
        d = self.robot.data
        q = d.body_quat_w.torch[:, self.imu_body_index]
        w = d.body_ang_vel_w.torch[:, self.imu_body_index]
        r = quat_apply(q, self.imu_r.expand(self.num_envs, 3))
        # body_lin_vel_w is the centre-of-mass velocity: the IMU point moves with the link frame origin
        return d.body_link_lin_vel_w.torch[:, self.imu_body_index] + torch.cross(w, r, dim=-1)

    def _read_accel(self) -> torch.Tensor:
        """MTi-3 accelerometer: specific force at the IMU point (IMU = pelvis axes), mean over the last accel_average_s
        (Xsens: dV / dt of the output interval), + per-episode bias + white noise, delayed like the MTi-3 orientation."""
        d = self.robot.data
        a_w = (self._imu_point_velocity() - self.imu_vel_prev) / (self.accel_avg_steps * float(self.cfg.sim.dt))
        a_w[:, 2] += 9.81
        f = quat_apply_inverse(d.body_quat_w.torch[:, self.imu_body_index], a_w)
        # touchdown on this control step (true sole contact, as the env's contact reward) -> impact noise
        td = (self.feet_contact & ~self.accel_contact_prev).any(-1, keepdim=True).float()
        self.accel_contact_prev.copy_(self.feet_contact)
        std = self.accel_base + td * self.accel_impact_scale * self.accel_impact
        f = f + self.accel_bias + std * torch.randn_like(f)
        self.accel_buf = torch.roll(self.accel_buf, -1, dims=1)
        self.accel_buf[:, -1] = f
        return self.accel_buf[:, 0]

    # ------------------------------------------------------------------ helpers
    def _foot_collision(self) -> torch.Tensor:
        """Sole rectangles (xy, oriented) overlapping within foot_collision_margin and |dz| < max_dz (separating axes)."""
        d = self.robot.data
        feet = self._foot_positions()
        hl, hw = self.cfg.foot_sole_half_length + self.cfg.foot_collision_margin, self.cfg.foot_sole_half_width + self.cfg.foot_collision_margin
        axes = []
        for bi in self.foot_body_indices:
            x = quat_apply(d.body_quat_w.torch[:, bi], self.fwd_axis_b.expand(self.num_envs, -1))[:, :2]
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            axes.append((x, torch.stack((-x[:, 1], x[:, 0]), dim=-1)))
        c = feet[:, 1, :2] - feet[:, 0, :2]
        sep = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for ax in (axes[0][0], axes[0][1], axes[1][0], axes[1][1]):
            r0 = hl * (axes[0][0] * ax).sum(-1).abs() + hw * (axes[0][1] * ax).sum(-1).abs()
            r1 = hl * (axes[1][0] * ax).sum(-1).abs() + hw * (axes[1][1] * ax).sum(-1).abs()
            sep |= (c * ax).sum(-1).abs() > r0 + r1
        close_z = (feet[:, 1, 2] - feet[:, 0, 2]).abs() < self.cfg.foot_collision_max_dz
        return (~sep & close_z).float()

    def _estimate(self, R_imu: torch.Tensor, gyro_b: torch.Tensor):
        """Twin of robot::OlafInEKF use in ArtiH1::olaf_estimate. R_imu = MTi-3 attitude (world_R_pelvis as the controller
        sees it; only used to start the filter), gyro_b = MTi-3 gyro (pelvis frame, the same one the actor sees)."""
        cfg, d = self.cfg, self.robot.data
        accel_b = self._read_accel()
        pel_p = d.body_pos_w.torch[:, self.ref_body_index]
        Rp = matrix_from_quat(d.body_quat_w.torch[:, self.ref_body_index])
        sole_b = (Rp.transpose(1, 2)[:, None] @ (self._foot_positions() - pel_p[:, None])[..., None])[..., 0]  # FK
        s = sole_b - self.imu_r  # soles relative to the IMU (the filter's body origin), pelvis axes
        # (re)start over the first control steps of an episode: the MTi-3 model only reports the reset attitude after
        # its delay line has refilled; the controller starts its filter after the IMU calibration, with the policy
        new = ~self.est_init | (self.episode_length_buf <= 2)
        ekf = self.ekf
        # one batched filter step for every env; the restarting ones are overwritten by reset() below
        ekf.propagate(gyro_b, accel_b, self.step_dt)
        rel = (ekf.R.float()[:, None] @ s[..., None])[..., 0]
        zmin = rel[..., 2].min(-1, keepdim=True).values
        contact = rel[..., 2] - zmin < cfg.est_contact_tolerance
        ekf.correct(s, contact)
        if new.any():
            ekf.reset(new, R_imu[new], s[new])
        R = ekf.R.float()
        w = gyro_b - ekf.bg.float()
        self.est_R = R
        self.est_pos = ekf.p.float() - (R @ self.imu_r[:, None])[..., 0]
        if hasattr(self, "est_z_off"):  # robot estimator height drift (cfg.est_z_offset_range / est_z_drift_range)
            self.est_pos[:, 2] += self.est_z_off + self.est_z_rate * self.episode_length_buf.float() * self.step_dt
        self.est_vel = ekf.v.float() - (R @ torch.cross(w, self.imu_r.expand_as(w), dim=-1)[..., None])[..., 0]
        self.est_feet = ekf.p.float()[:, None] + (R[:, None] @ s[..., None])[..., 0]
        R_feet = [R @ Rp.transpose(1, 2) @ matrix_from_quat(d.body_quat_w.torch[:, bi]) for bi in self.foot_body_indices]
        self.est_feet_yaw = torch.stack([torch.atan2(Rf[:, 1, 0], Rf[:, 0, 0]) for Rf in R_feet], -1)
        self.est_root_yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        if new.any():
            self.est_path[new] = torch.stack((self.est_pos[new, 0], self.est_pos[new, 1], self.est_root_yaw[new]), -1)
            self.est_init[new] = True
        return R

    def _temp_obs(self, temp: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.thermal_clip
        return (temp.clamp(lo, hi) - 0.5 * (lo + hi)) / (0.5 * (hi - lo))

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        cfg, d = self.cfg, self.robot.data
        dt = self.step_dt
        n_sub = max(self.substeps, 1)
        impact = self.impact_acc * (cfg.impact_reference_dt / float(cfg.sim.dt))  # 240 Hz-equivalent sum
        tau_sq = self.tau_sq_acc / n_sub
        # thermal model (eq. 6), explicit Euler at the control rate
        self.temp_rate = -self.th_alpha * (self.temp - self.temp_amb) + self.th_beta * tau_sq
        self.temp += dt * self.temp_rate
        self._compute_reference()

        pos, quat, lin_w, ang_w = self._torso()
        q, qd = d.joint_pos.torch, d.joint_vel.torch
        height, _ = self._sole_heights()
        contact = (height < cfg.foot_contact_height).float()
        self._feet_height_obs.copy_(height)
        self.feet_contact.copy_(contact > 0.5)

        lo, hi = cfg.thermal_clip
        tc = self.temp.clamp(lo, hi)
        h_t = -self.temp_rate + self.th_gamma * (cfg.thermal_max - tc)
        a = torch.clamp(self.actions, -1.0, 1.0)
        a1 = torch.clamp(self.previous_actions, -1.0, 1.0)
        a2 = torch.clamp(self.previous_actions2, -1.0, 1.0)
        gq = cfg.joint_limit_gamma
        self.wz_err_ema += (dt / cfg.yaw_drift_tau_s) * ((ang_w[:, 2] - self.ref_w[:, 2]) - self.wz_err_ema)
        terms = {
            "torso_pos_xy": cfg.w_torso_pos_xy * torch.exp(-cfg.k_torso_pos * torch.sum(torch.square(pos[:, :2] - self.ref_pos[:, :2]), -1)),
            "torso_ori": cfg.w_torso_ori * torch.exp(-cfg.k_torso_ori * torch.sum(torch.square(quat_box_minus(quat, self.ref_quat)), -1)),
            "lin_vel_xy": cfg.w_lin_vel_xy * torch.exp(-cfg.k_lin_vel * torch.sum(torch.square(lin_w[:, :2] - self.ref_v[:, :2]), -1)),
            "lin_vel_z": cfg.w_lin_vel_z * torch.exp(-cfg.k_lin_vel * torch.square(lin_w[:, 2] - self.ref_v[:, 2])),
            "ang_vel_xy": cfg.w_ang_vel_xy * torch.exp(-cfg.k_ang_vel * torch.sum(torch.square(ang_w[:, :2] - self.ref_w[:, :2]), -1)),
            "ang_vel_z": cfg.w_ang_vel_z * torch.exp(-cfg.k_ang_vel * torch.square(ang_w[:, 2] - self.ref_w[:, 2])),
            "leg_joint_pos": -cfg.w_leg_joint_pos * torch.sum(torch.square(q - self.ref_q), -1),
            "leg_joint_vel": -cfg.w_leg_joint_vel * torch.sum(torch.square(qd - self.ref_qd), -1),
            "contact": cfg.w_contact * torch.sum((contact == self.ref_contact).float(), -1),
            "survival": cfg.w_survival * torch.ones(self.num_envs, device=self.device),
            "joint_torques": -cfg.w_joint_torques * torch.sum(tau_sq, -1),
            "joint_acc": -cfg.w_joint_acc * torch.sum(torch.square(d.joint_acc.torch), -1),
            "leg_action_rate": -cfg.w_leg_action_rate * torch.sum(torch.square(a - a1), -1),
            "leg_action_acc": -cfg.w_leg_action_acc * torch.sum(torch.square(a - 2.0 * a1 + a2), -1),
            "temperature": -cfg.w_temperature * torch.sum(torch.clamp(h_t, max=0.0).abs(), -1),
            "joint_limits_lower": -cfg.w_joint_limits * torch.sum(torch.clamp(qd + gq * (q - self.q_min), max=0.0).abs(), -1),
            "joint_limits_upper": -cfg.w_joint_limits * torch.sum(torch.clamp(-qd + gq * (self.q_max - q), max=0.0).abs(), -1),
            "foot_collision": -cfg.w_foot_collision * self._foot_collision(),
            "impact": -cfg.w_impact * torch.sum(impact, -1),
            "yaw_drift": -cfg.w_yaw_drift * torch.square(self.wz_err_ema),
        }
        reward = sum(terms.values()) * dt
        for k, v in terms.items():
            self.episode_sums[k] += v * dt
        jerr = torch.sqrt(torch.mean(torch.square(q - self.ref_q), -1))
        self.extras["log"] = {
            **{f"Reward/{k}": (v * dt).mean() for k, v in terms.items()},
            "Olaf/joint_err_deg": torch.rad2deg(jerr).mean(),
            "Olaf/torso_pos_err_cm": 100.0 * (pos[:, :2] - self.ref_pos[:, :2]).norm(dim=-1).mean(),
            "Olaf/torso_ori_err_deg": torch.rad2deg(quat_box_minus(quat, self.ref_quat).norm(dim=-1)).mean(),
            "Olaf/contact_match": (contact == self.ref_contact).float().mean(),
            "Olaf/temp_max_C": self.temp.max(dim=-1).values.mean(),
            "Olaf/temp_over_max_frac": (self.temp > cfg.thermal_max).float().mean(),
            "Olaf/tau_rms": torch.sqrt(tau_sq.mean()),
            # rms per 240 Hz step (comparable across physics rates)
            "Olaf/impact_dv_rms_cm_s": torch.sqrt(impact.sum(-1) * cfg.impact_reference_dt / dt).mean(),
            "Olaf/foot_collision_frac": self._foot_collision().mean(),
            "Olaf/path_torso_dist_cm": 100.0 * (self.path[:, :2] - pos[:, :2]).norm(dim=-1).mean(),
        }
        return reward

    # ------------------------------------------------------------------ observations
    def _stale_joint_feedback(self):
        pos, vel = super()._stale_joint_feedback()
        if self.compl_on:
            pos = pos + self.compl * self.tau_last  # the encoder sits on the motor side of the series compliance
        return pos, vel

    def _get_observations(self) -> dict:
        cfg, d = self.cfg, self.robot.data
        n, dev = self.num_envs, self.device
        pos, quat, lin_w, ang_w = self._torso()
        psi = self.path[:, 2]
        # root pose relative to the path frame
        p_rel = torch.cat((_rot2(-psi, pos[:, :2] - self.path[:, :2]), (pos[:, 2] - self.scene.env_origins[:, 2])[:, None]), -1)
        rel_true = quat_mul(quat_conjugate(_yaw_quat(psi)), quat)
        quat_imu, ang_vel_w_imu = self.mti3_imu.read(d, self.imu_body_index)
        rel_imu = quat_mul(quat_conjugate(_yaw_quat(psi)), quat_imu)

        def rot6(qr):
            return matrix_from_quat(qr)[..., :2].reshape(n, 6)

        gyro_b = quat_apply_inverse(quat_imu, ang_vel_w_imu)
        lin_b = quat_apply_inverse(quat, lin_w)
        ang_b = quat_apply_inverse(quat, ang_w)
        if cfg.actor_state_estimator:
            R = self._estimate(matrix_from_quat(quat_imu), quat_apply_inverse(quat_imu, ang_vel_w_imu))
            e_psi = self.est_path[:, 2]
            act_p = torch.cat((_rot2(-e_psi, self.est_pos[:, :2] - self.est_path[:, :2]), self.est_pos[:, 2:3]), -1)
            c, s_ = torch.cos(e_psi), torch.sin(e_psi)
            Rz = torch.zeros(n, 3, 3, device=dev)
            Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, s_, -s_, c, 1.0
            act_rot = (Rz @ R)[..., :2].reshape(n, 6)
            act_v = (R.transpose(1, 2) @ self.est_vel[..., None])[..., 0]
        else:
            act_p, act_rot, act_v = p_rel, rot6(rel_imu), lin_b
        joint_pos, joint_vel = self._stale_joint_feedback()
        a1 = torch.clamp(self.actions, -1.0, 1.0)
        a2 = torch.clamp(self.previous_actions, -1.0, 1.0)
        temp_seen = torch.round(self.temp / cfg.thermal_obs_quantum) * cfg.thermal_obs_quantum
        P = self._joints_to_policy
        tail = [torch.stack((torch.cos(2 * math.pi * self.phase), torch.sin(2 * math.pi * self.phase)), -1), self.g] \
            if self.walk else [self.g]

        def nz(x, s):
            return x + s * torch.randn_like(x)

        policy = torch.cat((
            nz(act_p, cfg.noise_root_pos), nz(act_rot, cfg.noise_root_rot), nz(act_v, cfg.noise_lin_vel), gyro_b,
            P(nz(joint_pos - self.default_q, cfg.noise_joint_pos)), P(nz(joint_vel, cfg.noise_joint_vel)), P(a1), P(a2),
            P(self._temp_obs(temp_seen)), *tail), dim=-1)
        height = self._feet_height_obs
        rot_err = quat_box_minus(quat, self.ref_quat)
        pos_err = pos - self.ref_pos
        bias = self.yaw_torque_bias[:, None] if hasattr(self, "yaw_torque_bias") else torch.zeros(n, 1, device=dev)
        coul = self.actuator_coulomb.mean(-1, keepdim=True) if hasattr(self, "actuator_coulomb") else torch.zeros(n, 1, device=dev)
        critic = torch.cat((
            p_rel, rot6(rel_true), lin_b, ang_b, P(d.joint_pos.torch - self.default_q), P(d.joint_vel.torch), P(a1), P(a2),
            P(self._temp_obs(self.temp)), *tail,
            P((self.temp - 50.0) / 30.0), P(self.temp_rate * 10.0), height * 10.0, self.feet_contact.float(),
            P(self.ref_q - d.joint_pos.torch), self.ref_contact, pos_err, rot_err, bias, coul), dim=-1)
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------------ reset
    def _apply_torso_torque_noise(self):
        if not bool((self.tt_std > 0).any()):
            return
        w = torch.randn_like(self.tt_lo)
        self.tt_lo += self.tt_a_lo * (w - self.tt_lo)
        self.tt_hi += self.tt_a_hi * (w - self.tt_hi)
        torque = ((self.tt_hi - self.tt_lo) * (self.tt_norm * self.tt_std) * self.tt_scale)[:, None, :]
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=torch.zeros_like(torque), torques=torque, body_ids=self._support_body_ids, is_global=False)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)  # default pose, random yaw, DR draws, buffers, _sample_commands
        if not hasattr(self, "g"):
            return
        k, dev, cfg = len(env_ids), self.device, self.cfg
        d = self.robot.data
        lo, hi = cfg.torso_torque_noise_scale_range
        self.tt_scale[env_ids] = torch.empty(k, 1, device=dev).uniform_(lo, hi)
        if hasattr(self, "est_z_off"):
            lo, hi = cfg.est_z_offset_range
            self.est_z_off[env_ids] = torch.empty(k, device=dev).uniform_(lo, hi)
            lo, hi = cfg.est_z_drift_range
            self.est_z_rate[env_ids] = torch.empty(k, device=dev).uniform_(lo, hi)
        if self.compl_on:
            lo, hi = cfg.joint_series_compliance_scale_range
            self.compl[env_ids] = self.compl_base * torch.empty(k, 1, device=dev).uniform_(lo, hi)
            self.tau_last[env_ids] = 0.0
        self.wz_err_ema[env_ids] = 0.0
        self.tt_lo[env_ids] = 0.0
        self.tt_hi[env_ids] = 0.0
        # thermal state: ambient per episode, initial temperatures up to above T_max (paper: train near the limit)
        lo, hi = cfg.thermal_ambient_range
        self.temp_amb[env_ids] = torch.empty(k, 1, device=dev).uniform_(lo, hi)
        self.temp[env_ids] = self.temp_amb[env_ids] + torch.rand(k, 12, device=dev) * (cfg.thermal_init_max - self.temp_amb[env_ids])
        self.temp_rate[env_ids] = 0.0
        # path frame at the spawned torso (position, heading)
        root = d.root_pose_w.torch[env_ids] if hasattr(d, "root_pose_w") else torch.cat(
            (d.root_pos_w.torch[env_ids], d.root_quat_w.torch[env_ids]), -1)
        self.path[env_ids, :2] = self.scene.env_origins[env_ids, :2]
        self.path[env_ids, 2] = _yaw_of(root[:, 3:7])
        # Policy switches at runtime (paper Sec. VII-C, Animation Engine): part of the walking episodes start from the
        # standing stance at walk_start_phase (stand -> walk), part of the standing episodes start mid-stride from a
        # walking reference state (walk -> stand). The rest start from the own reference state (RSI).
        switch = torch.rand(k, device=dev) < (cfg.reset_stand_pose_fraction if self.walk else cfg.reset_walk_pose_fraction)
        if self.walk:
            self.phase[env_ids] = torch.rand(k, device=dev) if cfg.reset_reference_state else 0.0
            self.phase[env_ids[switch]] = cfg.walk_start_phase
        else:
            self.g[env_ids] = self.g_target[env_ids] if cfg.reset_reference_state else 0.0
            if self.switch_bank is not None:
                self.g[env_ids[switch]] = 0.0  # the controller switches with the STAND input at 0, g_target ramps in
        bank_rows = None
        if not self.walk and self.switch_bank is not None and switch.any():
            bank_rows = torch.randint(0, len(self.switch_bank["pose"]), (int(switch.sum()),), device=dev)
            # controller olaf_reset_path: path frame at the torso (origin + bank xy 0, the bank torso heading)
            self.path[env_ids[switch], 2] = _yaw_of(self.switch_bank["pose"][bank_rows, 3:7])
        self._compute_reference(env_ids)
        if cfg.reset_reference_state:
            # reference state initialisation: torso pose / velocity and joints of x_t
            pose = torch.cat((self.ref_pos[env_ids], self.ref_quat[env_ids]), -1)
            vel = torch.cat((self.ref_v[env_ids], self.ref_w[env_ids]), -1)
            q = self.ref_q[env_ids].clone()
            qd = self.ref_qd[env_ids].clone()
            if switch.any():
                s_ids = env_ids[switch]
                if self.walk:
                    # standing stance (default pose, the base reset's root pose), at rest
                    root = d.default_root_pose.torch[s_ids].clone()
                    root[:, :2] = self.scene.env_origins[s_ids, :2]
                    root[:, 2] += self.scene.env_origins[s_ids, 2]
                    root[:, 3:7] = quat_mul(_yaw_quat(self.path[s_ids, 2]), root[:, 3:7])
                    pose[switch] = root
                    vel[switch] = 0.0
                    q[switch] = self.default_q[s_ids]
                    qd[switch] = 0.0
                elif bank_rows is not None:
                    # the walking policy's own state at the controller's switch (touchdown after the stop request)
                    B = self.switch_bank
                    bp = B["pose"][bank_rows].clone()
                    bp[:, :3] += self.scene.env_origins[s_ids]
                    pose[switch] = bp
                    vel[switch] = B["vel"][bank_rows]
                    q[switch], qd[switch] = B["q"][bank_rows], B["qd"][bank_rows]
                else:
                    # mid-stride walking reference state (random phase, random walking input)
                    gw = self.walk_lo + torch.rand(len(s_ids), 3, device=dev) * (self.walk_hi - self.walk_lo)
                    idx, w = self._walk_blend(gw)
                    gw = (w[..., None] * self.walk_keys[idx]).sum(1)
                    rows = self._walk_rows(gw, torch.rand(len(s_ids), device=dev))
                    wq, wqd, _, wpos, wquat, wv, ww = self._resolve_walk(rows, self.path[s_ids], self.scene.env_origins[s_ids, 2])
                    pose[switch] = torch.cat((wpos, wquat), -1)
                    vel[switch] = torch.cat((wv, ww), -1)
                    q[switch], qd[switch] = wq, wqd
            q = torch.clamp(q, self._joint_pos_lower[env_ids], self._joint_pos_upper[env_ids])
            self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            self.robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)  # ref v = link origin
            self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
            if self.cfg.actuator_model and hasattr(self, "actuator_buffer"):
                self.actuator_buffer[:, env_ids] = q.unsqueeze(0)
            # targets / actions live in the encoder frame (_apply_action sends q_target - joint_obs_offset): the spawned
            # pose there is q + offset, so the first targets hold it instead of stepping by the zero error
            q_enc = q + self.joint_obs_offset[env_ids] if hasattr(self, "joint_obs_offset") else q
            self.q_target[env_ids] = q_enc
            dq = q_enc - self.default_q[env_ids]
            scale = torch.where(dq >= 0.0, self.action_positive_scale, self.action_negative_scale)
            act = torch.clamp(dq / scale, -1.0, 1.0)
            self.actions[env_ids] = act
            self.previous_actions[env_ids] = act
            self.previous_actions2[env_ids] = act
        self.ref_q_prev[env_ids] = self.ref_q[env_ids]
        self.ref_pos_prev[env_ids] = self.ref_pos[env_ids]
        self.ref_quat_prev[env_ids] = self.ref_quat[env_ids]
        self.est_init[env_ids] = False
        lo, hi = cfg.accel_bias_range
        self.accel_bias[env_ids] = torch.empty(k, 3, device=dev).uniform_(lo, hi)
        lo, hi = cfg.accel_impact_scale_range
        self.accel_impact_scale[env_ids] = torch.empty(k, 1, device=dev).uniform_(lo, hi)
        self.accel_contact_prev[env_ids] = True
        self.imu_vel_prev[env_ids] = self._imu_point_velocity()[env_ids]
        self.q_des[env_ids] = self.q_target[env_ids]
        self.q_des_prev[env_ids] = self.q_target[env_ids]
        self.q_filt[env_ids] = self.q_target[env_ids]
        self.foot_vz_prev[env_ids] = 0.0
        self.impact_skip[env_ids] = True  # first physics step after a reset: no previous foot velocity
        self.impact_acc[env_ids] = 0.0
        self.tau_sq_acc[env_ids] = 0.0
