# walk_13 + Togepi shell (+3 cm), mid DR (--dr_step 612000 = halfway through dr_widen_steps), no pushes.
#   cut_track : cutaway, camera follows (turns with the robot)  -> + clearance overlay
#   cut_fixed : cutaway, wide fixed camera at the start          -> + clearance overlay
#   full_fixed: whole shell, wide fixed camera
#   full_front_fixed: whole shell, fixed camera in front of the start
#   cut_front_track: front half cut away, camera in front follows  -> + clearance overlay
source /opt/isaac/IsaacLab-main/.venv/bin/activate; export OMNI_KIT_ACCEPT_EULA=yes; cd /opt/isaac/projects/humanoid_sim
W13=logs/skrl/olaf_walk/2026-11-23_14-51-56_ppo_torch/checkpoints/agent_480000.pt
V=logs/skrl/olaf_walk/2026-11-23_14-51-56_ppo_torch/videos/play
for kind in ${KINDS:-cut_track cut_fixed full_fixed}; do
  case $kind in
    cut_track) extra="--exterior_cut --camera_follow" ;;
    cut_fixed) extra="--exterior_cut --camera_fixed" ;;
    full_fixed) extra="--camera_fixed" ;;
    full_front_fixed) extra="--camera front --camera_fixed" ;;
    cut_front_track) extra="--exterior_cut --camera front --camera_follow" ;;
  esac
  L=logs/eval/1124/exterior_${kind}_drmid_nopush.log
  python scripts/skrl/play.py --task HumanoidSim-Olaf-Walk-Play-v0 --checkpoint $W13 --num_envs 1 --video --video_length 1750 \
    --exterior $extra --no_push --dr_step 612000 --seed 2 physics=isaacsim_physx > $L 2>&1
  mv $V/clip_0000.mp4 $V/exterior_${kind}_drmid_nopush.mp4; mv $V/exterior_frames.npz $V/exterior_frames_${kind}_drmid_nopush.npz
  case $kind in cut_*)
    python scripts/eval/exterior_overlay.py $V/exterior_${kind}_drmid_nopush.mp4 $V/exterior_frames_${kind}_drmid_nopush.npz \
      $V/exterior_${kind}_drmid_nopush_clearance.mp4 >> $L 2>&1 ;;
  esac
  echo DONE_$kind
done
