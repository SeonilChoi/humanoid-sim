"""Gait / motion generation with PlaCo (ported from the motion_generator project).

  * MotionEngine (motion_engine.py): PlaCo walk pattern generator (footstep planner, LIPM / ZMP CoM plan, swing
    trajectories) + whole-body IK, ticked at a fixed dt; the original motion_generator engine, unchanged.
  * make_engine (heel_toe.py): MotionEngine with the Olaf heel-toe roll, used by reference/gen_walk_clips.py and
    reference/gen_stand_grid.py.
  * gait_generator.py: the motion_generator episode recorder / Meshcat viewer, as a module:
        python -m humanoid_sim.utils.motion_generator.gait_generator --index 0 --robot arti_h1 \
            --dx 0.05 --dy 0 --dth 0 --stand false --visual false --out_dir data/motions/arti_h1
  * config/<robot>/*.json: gait parameters, robots/<robot>/: URDF + limits.json (+ collisions.json, meshes).

PlaCo is imported lazily (only when an engine is built), so importing this package needs no PlaCo.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PACKAGE_DIR, "config")
ROBOTS_DIR = os.path.join(PACKAGE_DIR, "robots")


def gait_config_path(robot: str = "arti_h1", name: str = "gait_olaf_kb.json") -> str:
    return os.path.join(CONFIG_DIR, robot, name)


def robot_folder(robot: str = "arti_h1") -> str:
    return os.path.join(ROBOTS_DIR, robot)


def __getattr__(name):  # lazy: MotionEngine / make_engine pull in PlaCo
    if name == "MotionEngine":
        from .motion_engine import MotionEngine

        return MotionEngine
    if name == "make_engine":
        from .heel_toe import make_engine

        return make_engine
    raise AttributeError(name)


__all__ = ["CONFIG_DIR", "ROBOTS_DIR", "MotionEngine", "gait_config_path", "make_engine", "robot_folder"]
