# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Play an ARTI-H1 skrl checkpoint through the fixed command sequence with scheduled pushes.

Isaac Lab 3.x port of the 2.x script, following ``isaaclab_rl.entrypoints.backends.play_skrl``. Arguments are those of
``isaaclab play --rl_library skrl`` plus --no_push / --push_scale / --video_width / --video_height, for example::

    python scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --checkpoint latest --num_envs 4 \
        physics=newton_mjwarp --viz newton_gl
    python scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --checkpoint <run>/checkpoints/agent_N.pt \
        --num_envs 4 --video --video_length 2400

Like the 2.x script it plays the training env config (no play_mode() caps), with balance support and the random
push event off, and a 40 s episode.
"""

import argparse
import contextlib
import logging
import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_skrl  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_seed,
    run_playback,
    set_hydra_args,
)
from isaaclab_rl.skrl import (  # noqa: E402
    SkrlVecEnvWrapper,
    check_skrl_version,
    import_skrl_runner,
    resolve_skrl_agent_cfg_entry_point,
    resolve_skrl_algorithm,
)

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli  # noqa: E402

import humanoid_sim.tasks  # noqa: F401, E402

logger = logging.getLogger(__name__)

PLAY_COMMAND_DURATION_S = 5.0
VIDEO_CAMERA_EYE = (0.3, -2.8, 0.82)
VIDEO_CAMERA_LOOKAT = (0.3, 0.0, 0.08)
# Commands inside the training range (x .046 y .017 yaw .046) plus standing (zero command) segments.
PLAY_COMMAND_SEQUENCE = (
    ("forward", (0.040, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.015, 0.0)),
    ("backward", (-0.035, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("diagonal_right", (0.035, -0.012, 0.0)),
    ("turn_left", (0.0, 0.0, 0.040)),
    ("stand", (0.0, 0.0, 0.0)),
)

# Pushes to watch the recovery: (episode time s, delta root velocity vx, vy m/s, yaw rate rad/s), one in the middle of
# every 5 s command segment. Training pushes are only +-0.04 m/s / +-0.15 rad/s, so these are 2-4x stronger.
PLAY_PUSH_SCHEDULE = (
    (2.5, 0.10, 0.0, 0.0),    # forward: from behind
    (7.5, 0.0, -0.10, 0.0),   # side_left: from the left
    (12.5, 0.0, 0.10, 0.0),   # backward: from the right
    (17.5, -0.10, 0.0, 0.0),  # diagonal_right: from the front
    (22.5, 0.0, 0.0, 0.5),    # turn_left: yaw kick
    (27.5, 0.0, 0.15, 0.0),   # forward: stronger, from the right
)


# Template-Arti-H1-Velocity-*: the same 8 x 5 s layout at the velocity task's command ranges (x +-0.15, y +-0.08,
# yaw +-0.4) and pushes at its training push size (+-0.3 / +-0.2 m/s).
VELOCITY_PLAY_COMMAND_SEQUENCE = (
    ("forward", (0.15, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.08, 0.0)),
    ("backward", (-0.15, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("diagonal_right", (0.10, -0.06, 0.0)),
    ("turn_left", (0.0, 0.0, 0.40)),
    ("stand", (0.0, 0.0, 0.0)),
)
VELOCITY_PLAY_PUSH_SCHEDULE = (
    (2.5, 0.30, 0.0, 0.0),    # forward: from behind
    (7.5, 0.0, -0.20, 0.0),   # stand: from the left
    (12.5, 0.0, 0.20, 0.0),   # side_left: from the right
    (17.5, -0.30, 0.0, 0.0),  # backward: from the front
    (22.5, 0.0, 0.0, 0.40),   # stand: yaw kick
    (27.5, 0.0, 0.20, 0.0),   # diagonal_right: from the right
)


# HumanoidSim-Olaf-*: the control input g (walk: vx m/s, vy m/s, wz rad/s; stand: dz m, roll, pitch, yaw rad) inside the
# training ranges (olaf_env_cfg walk / stand command ranges). The policy observes g itself (the last obs entries), so g
# is set in the env (walk: projected onto the reference coverage like the env samples it) and the obs is left alone.
OLAF_WALK_SEQUENCE = (
    ("forward", (0.12, 0.0, 0.0)),
    ("stop", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.05, 0.0)),
    ("backward", (-0.10, 0.0, 0.0)),
    ("turn_left", (0.0, 0.0, 0.30)),
    ("forward_turn_right", (0.10, 0.0, -0.25)),
    ("stop", (0.0, 0.0, 0.0)),
)
OLAF_STAND_SEQUENCE = (
    ("nominal", (0.0, 0.0, 0.0, 0.0)),
    ("crouch", (-0.03, 0.0, 0.0, 0.0)),
    ("high", (0.015, 0.0, 0.0, 0.0)),
    ("roll", (0.0, 0.08, 0.0, 0.0)),
    ("pitch", (0.0, 0.0, 0.12, 0.0)),
    ("yaw", (0.0, 0.0, 0.0, 0.25)),
    ("nominal", (0.0, 0.0, 0.0, 0.0)),
)
# pushes at about half the final training push range (x +-0.5, y +-0.3 m/s)
OLAF_PUSH_SCHEDULE = (
    (2.5, 0.25, 0.0, 0.0),
    (12.5, 0.0, -0.15, 0.0),
    (22.5, -0.20, 0.0, 0.0),
    (32.5, 0.0, 0.15, 0.0),
)


def _apply_olaf_sequence(env):
    seq = OLAF_WALK_SEQUENCE if env.walk else OLAF_STAND_SEQUENCE
    t = env.episode_length_buf.float() * env.step_dt
    ids = torch.clamp((t / PLAY_COMMAND_DURATION_S).long(), max=len(seq) - 1)
    g = torch.tensor([c for _, c in seq], device=env.device, dtype=torch.float32).index_select(0, ids)
    env.command_timer[:] = 1.0e9
    if env.walk:
        idx, w = env._walk_blend(g)
        env.g[:] = (w[..., None] * env.walk_keys[idx]).sum(1)
        env.commands[:] = env.g
    else:
        env.g_target[:] = g
    cur = int(ids[0].item())
    if getattr(env, "_last_play_command_id", None) != cur:
        env._last_play_command_id = cur
        print(f"[INFO] t={float(t[0]):.1f} s g: {seq[cur][0]} {seq[cur][1]} (walk g after projection: "
              f"{[round(x, 3) for x in env.g[0].tolist()]})" if env.walk else f"[INFO] t={float(t[0]):.1f} s g: {seq[cur][0]} {seq[cur][1]}", flush=True)


def _apply_play_pushes(env, scale: float):
    """Add the scheduled velocity kick to every env's root when env 0's episode time crosses a push time."""
    t = float(env.episode_length_buf[0].item()) * env.step_dt
    last = getattr(env, "_last_play_push_t", -1.0)
    env._last_play_push_t = t
    for push_t, vx, vy, wz in PLAY_PUSH_SCHEDULE:
        if last < push_t <= t:
            vel = env.robot.data.root_vel_w.torch.clone()
            vel[:, 0] += scale * vx
            vel[:, 1] += scale * vy
            vel[:, 5] += scale * wz
            env.robot.write_root_velocity_to_sim_index(root_velocity=vel)
            env._push_arrow = (scale * vx, scale * vy, t + PUSH_ARROW_SHOW_S)
            print(f"[INFO] push at t={t:.2f} s: dvx {scale * vx:+.2f} dvy {scale * vy:+.2f} m/s, dwz {scale * wz:+.2f} rad/s", flush=True)


PUSH_ARROW_SHOW_S = 1.0     # the push arrow stays on screen this long after the kick
PUSH_ARROW_M_PER_MPS = 1.2  # arrow length per m/s of the velocity kick


def _log_push_frame(env, cam):
    """One row per policy step (= one video frame): env 0 pelvis, the camera eye / lookat and the active push, for
    scripts/overlay_push_arrows.py (markers / late USD prims are not rendered in a headless video run)."""
    if not hasattr(env, "_push_rows"):
        env._push_rows = []
    arrow = getattr(env, "_push_arrow", None)
    t = float(env.episode_length_buf[0].item()) * env.step_dt
    on = arrow is not None and arrow[2] - PUSH_ARROW_SHOW_S - 1e-6 <= t <= arrow[2]
    # the pelvis link origin sits near the soles; the kick moves the pelvis body, so aim at its centre of mass
    p = env.robot.data.root_com_pos_w.torch[0].tolist()
    eye, look = cam
    qx, qy, qz, qw = env.robot.data.root_quat_w.torch[0].tolist()   # xyzw (Isaac Lab 3.x)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    env._push_rows.append((t, *p, *eye, *look, arrow[0] if on else 0.0, arrow[1] if on else 0.0, yaw))


def _log_leg_poses(env):
    """One row per policy step (= one video frame): env 0 leg link poses in the pelvis frame, for the shell clearance
    overlay (scripts/eval/exterior_overlay.py)."""
    from isaaclab.utils.math import matrix_from_quat

    from humanoid_sim.utils.exterior import LEG_LINKS

    if not hasattr(env, "_leg_rows"):
        names = list(env.robot.data.body_names)
        env._leg_ids = ([names.index(n) for n in LEG_LINKS], names.index("pelvis"))
        env._leg_rows = []
    ids, ip = env._leg_ids
    pos = env.robot.data.body_link_pos_w.torch[0]
    Rw = matrix_from_quat(env.robot.data.body_link_quat_w.torch[0])
    RpT = Rw[ip].T
    env._leg_rows.append((float(env.episode_length_buf[0].item()) * env.step_dt, (RpT @ Rw[ids]).cpu().numpy(),
                          ((pos[ids] - pos[ip]) @ RpT.T).cpu().numpy(), env.g[0].cpu().numpy().copy()))


def _overwrite_observation_commands(obs, command_obs: torch.Tensor):
    """Keep policy observations consistent after overriding environment commands."""
    if isinstance(obs, torch.Tensor):
        obs = obs.clone()
        obs[..., -command_obs.shape[-1] :] = command_obs
        return obs
    if isinstance(obs, dict):
        return {key: _overwrite_observation_commands(value, command_obs) for key, value in obs.items()}
    return obs


def _apply_play_command_sequence(env, obs):
    required_attrs = ("commands", "episode_length_buf", "max_episode_length")
    if not all(hasattr(env, attr) for attr in required_attrs):
        return obs
    if hasattr(env, "command_timer"):
        env.command_timer[:] = 1.0e9  # velocity task: keep its own command resampling from overriding the sequence

    command_values = torch.tensor(
        [command for _, command in PLAY_COMMAND_SEQUENCE],
        dtype=torch.float32,
        device=env.device,
    )
    episode_progress = env.episode_length_buf.float() / max(float(env.max_episode_length), 1.0)
    command_ids = torch.clamp((episode_progress * len(PLAY_COMMAND_SEQUENCE)).long(), max=len(PLAY_COMMAND_SEQUENCE) - 1)
    env.commands[:] = command_values.index_select(0, command_ids)
    current_command_id = int(command_ids[0].item())
    if getattr(env, "_last_play_command_id", None) != current_command_id:
        command_name, command = PLAY_COMMAND_SEQUENCE[current_command_id]
        print(
            f"[INFO] Current command: {command_name} "
            f"(x={command[0]:.4f}, y={command[1]:.4f}, yaw={command[2]:.4f})",
            flush=True,
        )
        env._last_play_command_id = current_command_id
    return _overwrite_observation_commands(obs, env.commands * getattr(env, "command_scale", 1.0))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an ARTI-H1 skrl checkpoint with the play command sequence.")
    add_common_play_args(parser, agent_default=None, agent_help="Agent configuration entry point.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
    parser.add_argument("--video_width", type=int, default=1920, help="Recorded video width in pixels.")
    parser.add_argument("--video_height", type=int, default=1080, help="Recorded video height in pixels.")
    parser.add_argument("--no_push", action="store_true", default=False, help="Disable the scheduled play pushes.")
    parser.add_argument("--camera_follow", action="store_true", default=False, help="Move the video camera with the robot.")
    parser.add_argument("--exterior", action="store_true", default=False,
                        help="Show the Togepi exterior shell on the pelvis (visual only; humanoid_sim/utils/exterior.py).")
    parser.add_argument("--camera_fixed", action="store_true", default=False,
                        help="Olaf: fixed wide camera at the start position (does not follow the robot).")
    parser.add_argument("--exterior_cut", action="store_true", default=False,
                        help="With --exterior: cut away the half of the shell facing the camera (legs visible), turn the "
                             "camera with the robot yaw and log env 0 leg poses (exterior_frames.npz, for "
                             "scripts/eval/exterior_overlay.py).")
    parser.add_argument("--camera", choices=("side", "front"), default="side", help="Video camera: side or front (velocity task).")
    parser.add_argument("--zero_action", action="store_true", default=False, help="Send zero actions (PD hold of the default pose) with a zero command; shows the standing pose.")
    parser.add_argument("--dr_step", type=int, default=-1, help="Olaf: play at the training DR level of this step (default: the final DR of the Play cfg).")
    parser.add_argument("--push_scale", type=float, default=1.0, help="Scale of the scheduled play pushes.")
    cli_args_skrl.add_skrl_args(parser)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(args_cli, log_root_path: str, algorithm: str, metadata: dict[str, str]) -> str:
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="skrl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r".*",
            other_dirs=["checkpoints"],
            latest_checkpoint_pattern=r"agent_.*",
            metadata=metadata,
        )
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])


def main(argv: list[str] | None = None) -> None:
    global PLAY_COMMAND_SEQUENCE, PLAY_PUSH_SCHEDULE
    args_cli = _parse_args(sys.argv[1:] if argv is None else argv)
    check_skrl_version()
    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)
    # the 2.x script played the training config (no play_mode() overrides); keep that
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point, play_mode=False)
    algorithm = resolve_skrl_algorithm(agent_cfg, args_cli.algorithm)

    olaf = "Olaf" in args_cli.task
    if olaf:
        PLAY_COMMAND_SEQUENCE = OLAF_WALK_SEQUENCE if "Walk" in args_cli.task else OLAF_STAND_SEQUENCE
        PLAY_PUSH_SCHEDULE = OLAF_PUSH_SCHEDULE
        env_cfg.reset_yaw_range = (0.0, 0.0)  # face +x for the side camera
        if args_cli.dr_step >= 0:
            env_cfg.dr_widen_steps = (24_000, 1_200_000)
            env_cfg.actuator_delay_extra_curriculum_steps = (24_000, 1_200_000)
            env_cfg.curriculum_step_offset = args_cli.dr_step
    if "Velocity" in args_cli.task:
        PLAY_COMMAND_SEQUENCE, PLAY_PUSH_SCHEDULE = VELOCITY_PLAY_COMMAND_SEQUENCE, VELOCITY_PLAY_PUSH_SCHEDULE
        # the velocity task resets with a random yaw; face +x so the commands read the same in the side-view video
        if hasattr(env_cfg, "reset_yaw_range"):
            env_cfg.reset_yaw_range = (0.0, 0.0)
    env_cfg.episode_length_s = PLAY_COMMAND_DURATION_S * len(PLAY_COMMAND_SEQUENCE)
    # Training-only balance support torque must never act during play.
    if hasattr(env_cfg, "balance_support_kp"):
        env_cfg.balance_support_kp = 0.0
    # The random training push event is off; play uses the fixed PLAY_PUSH_SCHEDULE instead (--no_push to disable).
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args_cli.video:
        # ViewerCfg is deprecated in 3.x but DirectRLEnv still forwards it to the visualizers (incl. env-origin tracking)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = None
        velocity = "Velocity" in args_cli.task or olaf
        if velocity:
            # the velocity task walks ~1 m per segment: follow the robot root from the side, closer than the default
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
        if velocity and args_cli.camera == "front":
            # fixed camera in front of the start point (robot walks toward +x), far enough to keep the whole sequence
            # (asset_root: placed at the robot's start position, it does not move afterwards)
            # with --camera_follow the offset rides along with the robot root, so it can sit closer
            env_cfg.viewer.eye = (1.1, 0.0, 0.4) if args_cli.camera_follow else (2.6, 0.0, 0.7)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.18)
            if olaf and args_cli.exterior and args_cli.camera_follow:  # close and low, as the side exterior view
                env_cfg.viewer.eye = (0.75, 0.15, 0.22)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.14)
            if olaf and args_cli.camera_fixed:  # fixed in front of the start position (the robot first walks toward it)
                env_cfg.viewer.eye = (2.0, 0.0, 0.6)
                env_cfg.viewer.lookat = (0.3, 0.0, 0.2)
        else:
            env_cfg.viewer.eye = (0.0, -1.6, 0.55) if velocity else VIDEO_CAMERA_EYE
            env_cfg.viewer.lookat = (0.0, 0.0, 0.15) if velocity else VIDEO_CAMERA_LOOKAT
            if olaf:  # follow from the side, close (the KB robot is ~0.6 m tall)
                env_cfg.viewer.eye = (0.0, -1.0, 0.45)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.28)
            if olaf and args_cli.exterior:  # closer and lower: the feet and the shell's skirt
                env_cfg.viewer.eye = (0.15, -0.75, 0.22)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.14)
            if olaf and args_cli.camera_fixed:  # wide, fixed at the start position: the whole play sequence stays in view
                env_cfg.viewer.eye = (0.3, -1.7, 0.6)
                env_cfg.viewer.lookat = (0.3, 0.0, 0.2)
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
    pre_launch_video_config(env_cfg, args_cli)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        if args_cli.exterior:  # robot asset with the shell under the pelvis (visual only)
            from humanoid_sim.utils.exterior import build_exterior_robot_usd

            cut = ("front" if args_cli.camera == "front" else "side") if args_cli.exterior_cut else None
            env_cfg.scene.robot.spawn = env_cfg.scene.robot.spawn.replace(usd_path=build_exterior_robot_usd(cut=cut))
            print(f"[exterior] robot spawn: {env_cfg.scene.robot.spawn.usd_path}")
        runner_cls = import_skrl_runner(args_cli.ml_framework)
        apply_env_overrides(args_cli, env_cfg)
        args_cli.seed = resolve_seed(args_cli.seed)
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = agent_cfg["seed"]

        experiment_cfg = agent_cfg["agent"]["experiment"]
        log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["directory"]))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        metadata = {"agent": agent_cfg_entry_point, "algorithm": algorithm, "ml_framework": args_cli.ml_framework}
        resume_path = _resolve_checkpoint(args_cli, log_root_path, algorithm, metadata)
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        command_env = env.unwrapped
        dt = command_env.step_dt
        print("[INFO] Play command sequence:")
        for index, (name, command) in enumerate(PLAY_COMMAND_SEQUENCE, start=1):
            print(f"  {index}. {name}: {command}")

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["write_interval"] = 0  # don't log to TensorBoard
        experiment_cfg["checkpoint_interval"] = 0  # don't generate checkpoints
        runner = runner_cls(env, agent_cfg)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        obs, _ = env.reset()
        if olaf:
            _apply_olaf_sequence(command_env)
        else:
            obs = _apply_play_command_sequence(command_env, obs)
        states = env.state()

        def step() -> None:
            nonlocal obs, states
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            if args_cli.zero_action:
                actions = torch.zeros_like(actions)
            obs, _, terminated, _, _ = env.step(actions)
            if bool(terminated.flatten()[0]):
                print(f"[INFO] env 0 fell (episode restarts) at sim step {command_env.common_step_counter}", flush=True)
            if olaf:
                _apply_olaf_sequence(command_env)
            else:
                obs = _apply_play_command_sequence(command_env, obs)
            if not args_cli.no_push:
                _apply_play_pushes(command_env, args_cli.push_scale)
            states = env.state()
            if args_cli.exterior:
                _log_leg_poses(command_env)
            if follow is not None:
                # asset_root viewer tracking does not move the recording camera in 3.x: follow the root by hand
                root = command_env.robot.data.root_pos_w.torch[0].tolist()
                eye, look = follow
                if args_cli.exterior_cut:  # the cut is in the pelvis frame: turn the camera offset with the yaw
                    qx, qy, qz, qw = command_env.robot.data.root_quat_w.torch[0].tolist()
                    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    eye = (cy * eye[0] - sy * eye[1], sy * eye[0] + cy * eye[1], eye[2])
                    look = (cy * look[0] - sy * look[1], sy * look[0] + cy * look[1], look[2])
                cam = ([root[0] + eye[0], root[1] + eye[1], eye[2]], [root[0] + look[0], root[1] + look[1], look[2]])
                command_env.sim.set_camera_view(*cam)
                if olaf:
                    _log_push_frame(command_env, cam)

        follow = None
        if args_cli.video and ("Velocity" in args_cli.task or olaf) and (args_cli.camera_follow or olaf) \
                and not args_cli.camera_fixed:
            follow = (tuple(env_cfg.viewer.eye), tuple(env_cfg.viewer.lookat))
        run_playback(step, dt=dt, args_cli=args_cli, env_cfg=env_cfg)
        legs = getattr(command_env, "_leg_rows", None)
        if legs:
            path = os.path.join(log_dir, "videos", "play", "exterior_frames.npz")
            np.savez_compressed(path, t=np.array([r[0] for r in legs]), R_rel=np.stack([r[1] for r in legs]),
                                t_rel=np.stack([r[2] for r in legs]), g=np.stack([r[3] for r in legs]))
            print(f"[INFO] leg poses: {path} (overlay with scripts/eval/exterior_overlay.py)", flush=True)
        rows = getattr(command_env, "_push_rows", None)
        if rows:
            import csv
            path = os.path.join(log_dir, "videos", "play", "push_frames.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "px", "py", "pz", "ex", "ey", "ez", "lx", "ly", "lz", "push_vx", "push_vy", "yaw"])
                w.writerows(rows)
            print(f"[INFO] push / camera log: {path} (draw with scripts/overlay_push_arrows.py)", flush=True)


if __name__ == "__main__":
    main()
