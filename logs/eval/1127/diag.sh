# 11-27 stage 0 diagnostics: walk_10 vs walk_15 sim replay of all 11-26 walk logs with touchdown / shake recording (RAW)
source /opt/isaac/IsaacLab-main/.venv/bin/activate; export OMNI_KIT_ACCEPT_EULA=yes; cd /opt/isaac/projects/humanoid_sim
R=data/real_logs/arti_h1_olaf_run_
declare -A CK=([walk10]=logs/skrl/olaf_walk/2026-11-20_15-40-44_ppo_torch/checkpoints/agent_480000.pt [walk15]=logs/skrl/olaf_walk/2026-11-24_19-51-43_ppo_torch/checkpoints/agent_480000.pt)
for n in walk10 walk15; do
  RAW=1 R2S='cfg.est_z_offset_range=(0.0,0.0); cfg.est_z_drift_range=(0.0,0.0)' python scripts/real2sim/walk_replay/walk_replay.py \
    --checkpoint ${CK[$n]} --num_envs 32 --tag d1127_$n --logs ${R}20261126_174517.csv ${R}20261126_174832.csv ${R}20261126_175413.csv \
    ${R}20261126_181056.csv ${R}20261126_181502.csv --headless > logs/eval/1127/diag_$n.log 2>&1
  echo DONE_$n
done
