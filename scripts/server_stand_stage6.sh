#!/usr/bin/env bash
# Server stand run stage 6 (11-24): compliance-DR-off comparison. On the robot (11-24) stage 5 shook worst (gyro_y
# 6-10 Hz 0.39 rad/s, ankle q_des 1.2 deg, coherence 0.99), stage 4 too; stage 3 and stand_07 (both without
# series-compliance DR) did not. The sim tremor proxy ranked stage 5 above stage 4, so it cannot gate this run.
#  - from stage 3 agent_480000 (logs/skrl/olaf_stand/server_stage3), stage 3 weights (rate 15 / acc 3)
#  - series compliance off (scale 0), b_g,z frozen
#  - action low-pass 20 Hz = the robot's stand_action_lowpass_hz (training used 37.5 until now)
#  - switch bank from walk_10 without compliance (the deployed walk policy)
#
#   bash scripts/server_stand_stage6.sh [num_envs=8192]      # runs in tmux session "olaf_stand"
#
# Python: the interpreter active in the calling shell (conda env / venv), or PY=/path/to/python. It must have Isaac Lab
# and this repo installed (pip install -e source/humanoid_sim); the script checks that before starting. The tmux
# session stays open after the run ends or fails, so the error can be read with tmux attach.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=${1:-8192}
PY=${PY:-$(command -v python || true)}
CKPT=logs/skrl/olaf_stand/server_stage3/checkpoints/agent_480000.pt
BANK=source/humanoid_sim/humanoid_sim/assets/reference/olaf_switch_bank_walk10_480000.npz
cd "$ROOT"

[ -x "$PY" ] || { echo "no python found: activate the Isaac Lab env (conda / venv) or set PY=..."; exit 1; }
for f in "$CKPT" "$BANK"; do [ -f "$f" ] || { echo "missing $f (git pull?)"; exit 1; }; done
"$PY" - <<'EOF' || { echo "-> install this repo into that env: $PY -m pip install -e source/humanoid_sim"; exit 1; }
import importlib.util, sys
missing = [m for m in ("isaaclab", "humanoid_sim") if importlib.util.find_spec(m) is None]
if missing:
    print(f"{sys.executable}: cannot import {missing}"); sys.exit(1)
import humanoid_sim, os
here = os.path.realpath(os.path.dirname(humanoid_sim.__file__))
print("humanoid_sim from", here)
if not here.startswith(os.path.realpath(os.getcwd())):
    print("WARNING: humanoid_sim is imported from another checkout, not this repo")
EOF
tmux has-session -t olaf_stand 2>/dev/null && { echo "tmux session olaf_stand already exists (tmux attach -t olaf_stand)"; exit 1; }

mkdir -p logs/train
LOG=logs/train/stand_s6_$(date +%m%d_%H%M).log
cat > logs/train/stand_s6_run.sh <<EOF
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=yes
"$PY" scripts/skrl/train.py --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs $N \\
  agent.agent.experiment.directory=olaf_stand --checkpoint $CKPT env.curriculum_step_offset=1200000 \\
  env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 env.action_lowpass_hz=20.0 \\
  env.ekf_freeze_gyro_bias_z=True "env.joint_series_compliance_scale_range=[0.0,0.0]" \\
  env.switch_bank_file=olaf_switch_bank_walk10_480000.npz env.reset_walk_pose_fraction=0.6 \\
  agent.trainer.timesteps=480000 2>&1 | tee $LOG
echo "[stand_s6] exited with status \${PIPESTATUS[0]} (log $LOG)"
exec bash
EOF
tmux new-session -d -s olaf_stand "bash logs/train/stand_s6_run.sh"
echo "started with $PY, log $LOG"
echo "  tmux attach -t olaf_stand"
