# walk_15 (compliance-DR off) vs walk_10 (deployed), CLAUDE.md section 4, same code and settings for both.
source /opt/isaac/IsaacLab-main/.venv/bin/activate; export OMNI_KIT_ACCEPT_EULA=yes; cd /opt/isaac/projects/humanoid_sim
O=logs/eval/1125
declare -A CK=(
  [walk15]=logs/skrl/olaf_walk/2026-11-24_19-51-43_ppo_torch/checkpoints/agent_480000.pt
  [walk10]=logs/skrl/olaf_walk/2026-11-20_15-40-44_ppo_torch/checkpoints/agent_480000.pt
)
R=data/real_logs/arti_h1_olaf_run_
for n in walk15 walk10; do
  c=${CK[$n]}
  # 1. 11-23 log replay (falls, in-place yaw, turn tracking)
  RAW=1 R2S='cfg.est_z_offset_range=(0.0,0.0); cfg.est_z_drift_range=(0.0,0.0)' python scripts/real2sim/walk_replay/walk_replay.py \
    --checkpoint $c --num_envs 32 --tag e1125_$n --logs ${R}20261123_132540.csv --headless > $O/replay1123_$n.log 2>&1
  python scripts/eval/replay_yaw.py scripts/real2sim/walk_replay/res_e1125_${n}_raw.npz >> $O/replay1123_$n.log 2>&1
  # 2. 11-21 hard-floor logs, compliance model s = 0 / 0.75 / 1.0
  for s in 0.0 0.75 1.0; do
    R2S="cfg.joint_series_compliance_scale_range=($s,$s); cfg.torso_torque_noise_std=(1.5,2.0,1.0)" python scripts/real2sim/walk_replay/walk_replay.py \
      --checkpoint $c --num_envs 32 --tag e1125_${n}_hf$s --logs ${R}20261121_095329.csv ${R}20261121_095641.csv ${R}20261121_100031.csv \
      --headless > $O/replay1121_s${s}_$n.log 2>&1
  done
  # 3. starts from real pre-fall states
  python scripts/real2sim/walk_replay/state_start.py --checkpoint $c --tag e1125_$n --headless > $O/state_start_$n.log 2>&1
  # 4. stability / gait quality at the final DR (cfg default compliance DR, and compliance off)
  python scripts/eval/gait_check.py --checkpoint $c --steps 750 --num_envs 256 --headless > $O/gait_$n.log 2>&1
  NO_PUSH=1 python scripts/eval/gait_check.py --checkpoint $c --steps 750 --num_envs 256 --headless > $O/gait_nopush_$n.log 2>&1
  NO_COMPLIANCE=1 NO_PUSH=1 python scripts/eval/gait_check.py --checkpoint $c --steps 750 --num_envs 256 --headless \
    > $O/gait_nopush_nocompl_$n.log 2>&1
  # 5. knee saturation
  python scripts/eval/walk_sat.py --checkpoint $c --headless > $O/sat_$n.log 2>&1
  # 6. Togepi shell clearance (+3 cm)
  NO_PUSH=1 python scripts/eval/exterior_clearance.py --checkpoint $c --num_envs 64 --steps 750 --headless \
    --out $O/exterior_$n > $O/exterior_$n.log 2>&1
  echo DONE_$n
done
echo ALLDONE
