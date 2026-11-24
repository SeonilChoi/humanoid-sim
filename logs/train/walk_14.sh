#!/bin/bash
# walk_14: fine-tune walk_13 with the deployed observation path (ekf_freeze_gyro_bias_z=True, cfg default since 11-24).
# Recipe = CLAUDE.md section 5 walking (as walk_13), local RTX 5070 Ti.
set -e
cd /opt/isaac/projects/humanoid_sim
source /opt/isaac/IsaacLab-main/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=yes
python scripts/skrl/train.py --task HumanoidSim-Olaf-Walk-v0 physics=isaacsim_physx --num_envs 4096 \
  agent.agent.experiment.directory=olaf_walk \
  --checkpoint logs/skrl/olaf_walk/2026-11-23_14-51-56_ppo_torch/checkpoints/agent_480000.pt \
  env.curriculum_step_offset=2400000 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
  env.walk_zero_command_fraction=0.3 env.w_yaw_drift=50.0 env.episode_length_s=40.0 \
  env.ekf_freeze_gyro_bias_z=True agent.trainer.timesteps=480000
