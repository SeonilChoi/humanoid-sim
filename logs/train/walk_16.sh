#!/bin/bash
# walk_16 (11-27): walk_15 on the robot (11-26) shook more and stomped harder than walk_10 (operator: left foot harder).
# Real gyro: roll 1-2 Hz 0.48 vs 0.28-0.31 rad/s, yaw 2-8 Hz 0.42-0.51 vs 0.17-0.22; knee speed lost at touchdown L 2.0 /
# R 1.6 vs 1.7 / 0.9 (scripts/eval/stomp_compare.py). Sim replay (logs/eval/1127): walk_15 lands left-biased
# (IMU peak L/R 1.26, walk_10 1.01), the shake gap is mostly real-only.
# From walk_15 agent_480000, walk_15 recipe + high-passed torso roll / yaw rate error penalty (w_ang_vel_hf 3, 0.5 s) +
# impact weight x3 (2.5e-3 -> 7.5e-3).
set -e
cd /opt/isaac/projects/humanoid_sim
source /opt/isaac/IsaacLab-main/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=yes
python scripts/skrl/train.py --task HumanoidSim-Olaf-Walk-v0 physics=isaacsim_physx --num_envs ${N:-4096} \
  agent.agent.experiment.directory=olaf_walk \
  --checkpoint logs/skrl/olaf_walk/2026-11-24_19-51-43_ppo_torch/checkpoints/agent_480000.pt \
  env.curriculum_step_offset=2880000 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
  env.walk_zero_command_fraction=0.3 env.w_yaw_drift=50.0 env.episode_length_s=40.0 \
  env.ekf_freeze_gyro_bias_z=True "env.joint_series_compliance_scale_range=[0.0,0.0]" \
  env.w_ang_vel_hf=${W_HF:-3.0} env.w_impact=${W_IMP:-0.0075} \
  agent.trainer.timesteps=${STEPS:-240000}
