# walk_16 agent_120000 mid-training check: same replay as diag.sh (11-26 logs, 32 envs, RAW)
source /opt/isaac/IsaacLab-main/.venv/bin/activate; export OMNI_KIT_ACCEPT_EULA=yes; cd /opt/isaac/projects/humanoid_sim
R=data/real_logs/arti_h1_olaf_run_
RAW=1 R2S='cfg.est_z_offset_range=(0.0,0.0); cfg.est_z_drift_range=(0.0,0.0)' python scripts/real2sim/walk_replay/walk_replay.py \
  --checkpoint logs/skrl/olaf_walk/2026-11-27_08-50-54_ppo_torch/checkpoints/agent_120000.pt --num_envs 32 --tag d1127_walk16_120k \
  --logs ${R}20261126_174517.csv ${R}20261126_174832.csv ${R}20261126_175413.csv ${R}20261126_181056.csv ${R}20261126_181502.csv \
  --headless > logs/eval/1127/diag_walk16_120k.log 2>&1
echo DONE_walk16_120k
