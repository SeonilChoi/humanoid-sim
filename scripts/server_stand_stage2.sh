#!/usr/bin/env bash
# Server stand run (2026-11-21, stand_s2b): stage-2 switch training with the real2sim cfg of this commit.
#  - start: agent_stand_smooth_96000.pt (repo root; local stand_07 smoothness fine-tune, robot-proven stand)
#  - 60 % of standing episodes start from walk_10 agent_480000 touchdown states (the walk policy on the robot)
#  - action rate 15 / acc 3 (stage 2 at 10 / 3 doubled the jitter at 20 Hz LPF: |da| 0.045 vs 0.027 -> not deployed)
#  - final DR from the start (offset 1.2M); 480k env steps
#   bash scripts/server_stand_stage2.sh [num_envs=8192]      # runs in tmux session "olaf_stand"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/isaac/IsaacLab-main/.venv}
N=${1:-8192}
mkdir -p "$ROOT/logs/train"
CMD="source $VENV/bin/activate && cd $ROOT && export OMNI_KIT_ACCEPT_EULA=yes && python scripts/skrl/train.py \
 --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs $N agent.agent.experiment.directory=olaf_stand \
 --checkpoint agent_stand_smooth_96000.pt env.curriculum_step_offset=1200000 \
 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
 env.switch_bank_file=olaf_switch_bank_walk10_480000.npz env.reset_walk_pose_fraction=0.6 \
 agent.trainer.timesteps=480000 2>&1 | tee logs/train/stand_s2b_$(date +%m%d_%H%M).log"
tmux new-session -d -s olaf_stand "bash -lc '$CMD'"
echo "started: tmux attach -t olaf_stand"
