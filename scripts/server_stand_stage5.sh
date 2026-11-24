#!/usr/bin/env bash
# Server stand run (2026-11-23, stand_s5): stage 4 (agent_264000, robot test 20261123_132540) stood with an ~8.8 Hz
# policy-loop tremor (gyro_y power >5 Hz 0.79 vs 0.22 for stage 3; sim showed it only as action power >5 Hz 0.35 vs 0.05).
#  - start again from stage 3 (agent_stand_stage3_480000.pt, repo root; tremor-free on the robot)
#  - compliance model + compliance switch bank as stage 4, action acceleration penalty 3 -> 10 (high-frequency actions)
#  - gate: scripts/real2sim/stand_tremor.py action power >5 Hz <= 0.1 (stage 3: 0.05, stage 4: 0.35), plus eval_switch
#   bash scripts/server_stand_stage5.sh [num_envs=8192]      # runs in tmux session "olaf_stand"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/isaac/IsaacLab-main/.venv}
N=${1:-8192}
mkdir -p "$ROOT/logs/train"
CMD="source $VENV/bin/activate && cd $ROOT && export OMNI_KIT_ACCEPT_EULA=yes && python scripts/skrl/train.py \
 --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs $N agent.agent.experiment.directory=olaf_stand \
 --checkpoint agent_stand_stage3_480000.pt env.curriculum_step_offset=1200000 \
 env.w_leg_action_rate=15.0 env.w_leg_action_acc=10.0 \
 env.switch_bank_file=olaf_switch_bank_walk10_480000_compl.npz env.reset_walk_pose_fraction=0.6 \
 agent.trainer.timesteps=480000 2>&1 | tee logs/train/stand_s5_$(date +%m%d_%H%M).log"
tmux new-session -d -s olaf_stand "bash -lc '$CMD'"
echo "started: tmux attach -t olaf_stand"
