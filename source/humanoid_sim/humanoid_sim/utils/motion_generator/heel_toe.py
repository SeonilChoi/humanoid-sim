"""PlaCo walk with a heel-toe roll (Olaf-style stylised gait, arXiv 2512.16705 Sec. V-A).

The plain PlaCo walk keeps the feet flat. HeelToeEngine adds, on the generic_frame foot targets (walk frame, sole origin):
  * late double support: the next swing foot peels off about its toe (heel-off, toes down to -toe_off),
  * swing: the foot pitch goes -toe_off -> +heel_strike while the pivot moves toe -> heel (continuous at both ends),
  * early double support: the landed foot rolls down about its heel (heel strike -> flat).
Angles scale with the forward stride (dx / walk_max_dx_forward); backward / sideways / turning steps stay flat.
Gait keys: heel_strike_deg, toe_off_deg, heel_toe_half_length (m), heel_toe_window_s (s), walk_max_dx_forward.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from . import ROBOTS_DIR


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    return T


def _trans(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


def make_engine(gait: dict, robot_folder: str | None = None, quiet: bool = True):
    """HeelToeEngine for a gait dict (config/<robot>/*.json). robot_folder defaults to gait["robot_folder"], else
    robots/<gait["robot"]> inside this package. With heel_strike_deg = toe_off_deg = 0 it is the plain PlaCo walk."""
    from .motion_engine import MotionEngine

    class HeelToeEngine(MotionEngine):
        """MotionEngine with a heel-toe roll on the generic_frame foot targets (walk frame, sole origin)."""

        def _foot_pitch(self, t: float) -> dict:
            out = {"left": (0.0, 0.0), "right": (0.0, 0.0)}  # side -> (pitch, toes up +, pivot x in the walk frame)
            k = float(np.clip(self._d_x / max(self._ht_dx_full, 1e-6), 0.0, 1.0))
            if k <= 0.0:
                return out
            hs, to, half, W = k * self._ht_heel, k * self._ht_toe, self._ht_half, self._ht_window
            # A pure function of the support-aligned gait phase. Not the trajectory parts (a replan restarts the current
            # part at the replan time) and not lift-off / touchdown state (MotionEngine.tick evaluates the trajectory up
            # to one planner dt ahead and back again within a tick): both made the swing progress jump.
            if t < self.periodic_start_time - W:
                return out
            P, ss = self.period, self._robot_parameters.single_support_duration
            phi = self.get_gait_phase(t)  # 0 = left single support start (right foot lifts), 0.5 = right support start
            for side, start in (("right", 0.0), ("left", 0.5)):
                tau = ((phi - start) % 1.0) * P  # time since this foot's lift-off
                if tau < ss:
                    s = smoothstep(tau / ss)
                    out[side] = (-to + (to + hs) * s, half * (1.0 - 2.0 * s))  # toe-off -> heel strike, pivot toe -> heel
                elif tau - ss < W:
                    out[side] = (hs * (1.0 - smoothstep((tau - ss) / W)), -half)  # heel strike -> flat about the heel
                elif P - tau < W:
                    out[side] = (-to * smoothstep(1.0 - (P - tau) / W), half)  # heel-off about the toe
            return out

        def _update_generic_tasks_from_trajectory(self, t: float) -> None:
            super()._update_generic_tasks_from_trajectory(t)
            p = self._foot_pitch(t)
            for side in ("left", "right"):
                pitch, px = p[side]
                if pitch == 0.0:
                    continue
                T_walk = np.asarray(getattr(self._trajectory, f"get_T_world_{side}")(t))
                T_walk = T_walk @ _trans(px) @ _rot_y(-pitch) @ _trans(-px)
                T_model = T_walk @ getattr(self, f"_T_model_from_walk_{side}")
                self._generic_tasks[f"{side}_foot"].T_world_frame = T_model

    # class attributes: MotionEngine.__init__ already calls _update_generic_tasks_from_trajectory
    HeelToeEngine._ht_heel = np.deg2rad(gait.get("heel_strike_deg", 0.0))
    HeelToeEngine._ht_toe = np.deg2rad(gait.get("toe_off_deg", 0.0))
    HeelToeEngine._ht_half = float(gait.get("heel_toe_half_length", 0.057))
    HeelToeEngine._ht_dx_full = float(gait["walk_max_dx_forward"])
    HeelToeEngine._ht_window = float(gait.get("heel_toe_window_s", 0.2))
    folder = robot_folder or gait.get("robot_folder") or os.path.join(ROBOTS_DIR, gait["robot"])
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        me = HeelToeEngine(folder, gait)
    return me
