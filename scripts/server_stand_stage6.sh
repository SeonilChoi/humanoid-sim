#!/usr/bin/env bash
# Server stand run stage 6 (11-24): compliance-DR-off comparison. On the robot (11-24) stage 5 shook worst (gyro_y
# 6-10 Hz 0.39 rad/s, ankle q_des 1.2 deg, coherence 0.99), stage 4 too; stage 3 and stand_07 (both without
# series-compliance DR) did not. The sim tremor proxy ranked stage 5 above stage 4, so it cannot gate this run.
#  - from stage 3 agent_480000 (logs/skrl/olaf_stand/server_stage3), stage 3 weights (rate 15 / acc 3)
#  - series compliance off (scale 0), b_g,z frozen
#  - action low-pass 20 Hz = the robot's stand_action_lowpass_hz (training used 37.5 until now)
#  - switch bank from walk_10 without compliance (the deployed walk policy)
#   bash scripts/server_stand_stage6.sh [num_envs=8192]      # runs in tmux session "olaf_stand"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/isaac/IsaacLab-main/.venv}
N=${1:-8192}
mkdir -p "$ROOT/logs/train"
CMD="source $VENV/bin/activate && cd $ROOT && export OMNI_KIT_ACCEPT_EULA=yes && python scripts/skrl/train.py \
 --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs $N agent.agent.experiment.directory=olaf_stand \
 --checkpoint logs/skrl/olaf_stand/server_stage3/checkpoints/agent_480000.pt env.curriculum_step_offset=1200000 \
 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 env.action_lowpass_hz=20.0 \
 env.ekf_freeze_gyro_bias_z=True 'env.joint_series_compliance_scale_range=[0.0,0.0]' \
 env.switch_bank_file=olaf_switch_bank_walk10_480000.npz env.reset_walk_pose_fraction=0.6 \
 agent.trainer.timesteps=480000 2>&1 | tee logs/train/stand_s6_\$(date +%m%d_%H%M).log"
tmux new-session -d -s olaf_stand "bash -lc \"$CMD\""
echo "started: tmux attach -t olaf_stand"
