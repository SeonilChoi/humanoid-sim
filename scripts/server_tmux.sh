#!/usr/bin/env bash
# Start Olaf training in one tmux session "olaf" with three windows: walk, stand, tensorboard.
#
#   bash scripts/server_tmux.sh                 # 8192 envs each (paper)
#   bash scripts/server_tmux.sh 4096 4096       # walk / stand env counts (one GPU for both)
#   WALK_CKPT=logs/skrl/olaf_walk/<run>/checkpoints/agent_N.pt bash scripts/server_tmux.sh   # resume walk (STAND_CKPT too)
#   ONLY=walk bash scripts/server_tmux.sh       # only one policy (walk | stand)
#
# Attach: tmux attach -t olaf   | switch windows: Ctrl+b n / Ctrl+b 0..2   | detach: Ctrl+b d
# Stop one run: go to its window, Ctrl+c (skrl saves the last checkpoint). Stop all: tmux kill-session -t olaf
set -euo pipefail

SESSION=${SESSION:-olaf}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/isaac/IsaacLab-main/.venv}
NW=${1:-8192}
NS=${2:-8192}
ONLY=${ONLY:-}
PORT=${TB_PORT:-6006}

command -v tmux >/dev/null || { echo "tmux is not installed: sudo apt install -y tmux"; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION (or tmux kill-session -t $SESSION)"
    exit 1
fi

# install the package once (editable) if this venv does not have it yet
if ! "$VENV/bin/python" -c "import humanoid_sim" 2>/dev/null; then
    echo "installing humanoid_sim into $VENV"
    if command -v uv >/dev/null; then
        uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT/source/humanoid_sim"
    else
        "$VENV/bin/python" -m pip install --no-deps -e "$ROOT/source/humanoid_sim"
    fi
fi
mkdir -p "$ROOT/logs/train"

PRE="source $VENV/bin/activate && cd $ROOT && export OMNI_KIT_ACCEPT_EULA=yes"
STAMP=$(date +%m%d_%H%M)
train_cmd() {  # $1 task, $2 envs, $3 dir, $4 ckpt
    local resume=""
    [ -n "$4" ] && resume="--checkpoint $4"
    echo "$PRE && python scripts/skrl/train.py --task $1 physics=isaacsim_physx --num_envs $2 agent.agent.experiment.directory=$3 $resume 2>&1 | tee logs/train/${3}_${STAMP}.log"
}

tmux new-session -d -s "$SESSION" -n tensorboard
tmux send-keys -t "$SESSION:tensorboard" "$PRE && tensorboard --logdir logs/skrl --port $PORT --bind_all" C-m
if [ "$ONLY" != "stand" ]; then
    tmux new-window -t "$SESSION" -n walk
    tmux send-keys -t "$SESSION:walk" "$(train_cmd HumanoidSim-Olaf-Walk-v0 "$NW" olaf_walk "${WALK_CKPT:-}")" C-m
fi
if [ "$ONLY" != "walk" ]; then
    if [ "$ONLY" != "stand" ]; then
        echo "walk started in tmux '$SESSION' (tmux attach -t $SESSION to watch); waiting 60 s before starting stand ..."
        sleep 60  # let the first Kit instance finish starting
    fi
    tmux new-window -t "$SESSION" -n stand
    tmux send-keys -t "$SESSION:stand" "$(train_cmd HumanoidSim-Olaf-Stand-v0 "$NS" olaf_stand "${STAND_CKPT:-}")" C-m
fi
echo "started tmux session '$SESSION' (windows: tensorboard, ${ONLY:-walk, stand}); logs in logs/train/*_${STAMP}.log"
echo "attach: tmux attach -t $SESSION"
