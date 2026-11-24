#!/bin/bash
# walk_15 (11-24): compliance-DR-off comparison. Robot 11-24: walk_10 (trained without series-compliance DR) walked
# 130 s with the frozen b_g,z, walk_13 (compliance DR) spun and fell in 3.5 s; stand stage 4/5 (compliance DR) shook.
# From walk_10 agent_480000, walk_13 recipe (rate 15 / acc 3, zero command 0.3, yaw_drift 50, 40 s episodes), b_g,z
# frozen, series compliance off (scale 0). Everything else as the current real2sim DR.
set -e
cd /opt/isaac/projects/humanoid_sim
source /opt/isaac/IsaacLab-main/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=yes
python scripts/skrl/train.py --task HumanoidSim-Olaf-Walk-v0 physics=isaacsim_physx --num_envs ${N:-4096} \
  agent.agent.experiment.directory=olaf_walk \
  --checkpoint logs/skrl/olaf_walk/2026-11-20_15-40-44_ppo_torch/checkpoints/agent_480000.pt \
  env.curriculum_step_offset=2400000 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
  env.walk_zero_command_fraction=0.3 env.w_yaw_drift=50.0 env.episode_length_s=40.0 \
  env.ekf_freeze_gyro_bias_z=True "env.joint_series_compliance_scale_range=[0.0,0.0]" \
  agent.trainer.timesteps=${STEPS:-480000}
