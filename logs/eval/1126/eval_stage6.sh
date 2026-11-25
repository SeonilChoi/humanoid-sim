# stand stage 6 (server, compliance DR off) vs stand_07 (deployed) and stage 3, same code. Walk = walk_15 (model_file).
source /opt/isaac/IsaacLab-main/.venv/bin/activate; export OMNI_KIT_ACCEPT_EULA=yes; cd /opt/isaac/projects/humanoid_sim
O=logs/eval/1126
W15=logs/skrl/olaf_walk/2026-11-24_19-51-43_ppo_torch/checkpoints/agent_480000.pt
declare -A CK=(
  [s6_480]=logs/skrl/olaf_stand/server_stage6/checkpoints/agent_480000.pt
  [s6_240]=logs/skrl/olaf_stand/server_stage6/checkpoints/agent_240000.pt
  [stand07]=logs/skrl/olaf_stand/2026-11-20_13-25-38_ppo_torch/checkpoints/agent_72000.pt
  [s3]=logs/skrl/olaf_stand/server_stage3/checkpoints/agent_480000.pt
)
for n in s6_480 stand07 s3 s6_240; do
  c=${CK[$n]}
  python scripts/eval/stand_check.py --checkpoint $c --steps 750 --num_envs 256 --headless > $O/stand_$n.log 2>&1
  NO_PUSH=1 python scripts/eval/stand_check.py --checkpoint $c --steps 750 --num_envs 256 --headless > $O/stand_nopush_$n.log 2>&1
  NO_COMPLIANCE=1 python scripts/eval/stand_check.py --checkpoint $c --steps 750 --num_envs 256 --headless > $O/stand_nocompl_$n.log 2>&1
  NO_COMPLIANCE=1 NO_PUSH=1 python scripts/eval/stand_check.py --checkpoint $c --steps 750 --num_envs 256 --headless > $O/stand_nopush_nocompl_$n.log 2>&1
  python scripts/eval_switch.py --walk_ckpt $W15 --stand_ckpt $c --out $O/switch_$n --headless > $O/switch_$n.log 2>&1
  NO_COMPLIANCE=1 python scripts/eval_switch.py --walk_ckpt $W15 --stand_ckpt $c --out $O/switch_nocompl_$n --headless > $O/switch_nocompl_$n.log 2>&1
  NO_PUSH=1 LPF_HZ=20 python scripts/real2sim/stand_tremor.py --checkpoint $c --num_envs 64 --steps 1000 --headless > $O/tremor_$n.log 2>&1
  echo DONE_$n
done
echo ALLDONE
