#!/usr/bin/env bash
# Server stand run (2026-11-22, stand_s3c): real2sim v2 (series joint compliance, estimator height drift, no torso
# noise when standing) on top of the stage-3 server stand.
#  - start: agent_stand_stage3_480000.pt (repo root; server stage 3 agent_480000, on the robot since controller ec72b25)
#  - 60 % of standing episodes start from walk_10 agent_480000 touchdown states generated WITH the compliance model
#  - action rate 15 / acc 3, final DR (offset 1.2M), 480k env steps
#   bash scripts/server_stand_stage3.sh [num_envs=8192]      # runs in tmux session "olaf_stand"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/isaac/IsaacLab-main/.venv}
N=${1:-8192}
mkdir -p "$ROOT/logs/train"
CMD="source $VENV/bin/activate && cd $ROOT && export OMNI_KIT_ACCEPT_EULA=yes && python scripts/skrl/train.py \
 --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs $N agent.agent.experiment.directory=olaf_stand \
 --checkpoint agent_stand_stage3_480000.pt env.curriculum_step_offset=1200000 \
 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
 env.switch_bank_file=olaf_switch_bank_walk10_480000_compl.npz env.reset_walk_pose_fraction=0.6 \
 agent.trainer.timesteps=480000 2>&1 | tee logs/train/stand_s3c_$(date +%m%d_%H%M).log"
tmux new-session -d -s olaf_stand "bash -lc '$CMD'"
echo "started: tmux attach -t olaf_stand"
