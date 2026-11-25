# humanoid_sim: working notes for Claude

The ARTI-H1 robot is trained with the imitation RL of the Olaf paper (arXiv 2512.16705), with **separate
walking and standing policies**. The goal is to deploy them on the real robot (humanoid_controller, branch
`arti-h1-olaf`) so that it **stands and walks without falling**. This project continues `arti_h1_olaf` (commit 2c756fe,
2026-11-24) and carries the PlaCo reference generator as `humanoid_sim.utils.motion_generator`. Everything below comes
from the work of 2026-11-17 to 11-24.

## 1. Goals and decision rules

- **Stability over motion quality.** Accept a round on falls, push robustness, tilt and drift. "It walks" is not a pass.
- **Do not adopt on sim numbers alone.** The sim does not reproduce the real 8.8 Hz standing tremor. Use the criteria
  confirmed on the robot together with the proxies in section 4.
- **Judge straight walking per env.** Means cancel across envs. Look at the per-env distribution (p10/p90) of |vy|,
  |lean| and yaw rate. Straight but slow beats fast but crabbing.
- **Reproduce a real problem before fixing it.** Replay the real log's g sequence in sim (`walk_replay`) or start the sim
  from real states (`state_start`) and check that the same failure appears. If it does not, keep looking for the cause.
- **Series-compliance DR made the policies worse on the robot (11-24).** Every policy trained with it (walk_12, walk_13,
  stand stage 4 / 5) misbehaved on the robot (7-9 Hz standing limit cycle at the ankle, spinning / falling at walk
  start or after turns); every policy trained without it (walk_10, stand_07, stage 3) did not, even though the sim
  ranked the compliance ones far better. Working hypothesis: the compliance values (fitted on walking single support)
  overstate the standing ankle compliance, so the policies learned too much feedback gain. Train new rounds without it
  (`joint_series_compliance_scale_range=[0.0,0.0]`) until a robot test says otherwise.
- **Do not look for the cause only in training.** The 11-23 in-place yaw drift was not fixed by a reward (walk_13
  yaw_drift); the cause was the estimator's b_g,z. Suspect the observation path too (estimator, controller resets, filters).

## 2. Environment and working rules

- **venv:** `source /opt/isaac/IsaacLab-main/.venv/bin/activate` and `export OMNI_KIT_ACCEPT_EULA=yes`.
  `/opt/isaac/.venv` is an empty venv without `activate`; do not use it.
- **Install:** `uv pip install --python /opt/isaac/IsaacLab-main/.venv/bin/python --no-deps -e source/humanoid_sim`.
  Do this on every new machine or clone (the server too) before training; without it `train.py` fails with
  `No module named 'humanoid_sim'` (11-25 on the server). Check with
  `python -c "import humanoid_sim; print(humanoid_sim.__file__)"`: the path must be inside this repo (editable install).
  `arti_h1_olaf` is installed in the same venv; the task IDs differ (`HumanoidSim-Olaf-*` vs `Arti-H1-Olaf-*`).
- **Tasks:** `HumanoidSim-Olaf-Walk-v0` and `HumanoidSim-Olaf-Stand-v0`. Evaluate with the `-Play-v0` variants, which use
  the final DR.
- **PlaCo** (only needed to regenerate references): placo 0.12.0 is installed in the Isaac venv (pin 4.1.0 unchanged).
  It reproduces the old PlaCo 0.10.2 clips and stand grid to 1e-15.
- **Where training runs:** walking locally (RTX 5070 Ti, 4096 envs, about 11-14 it/s, about 10-12 h per 480k steps),
  standing on the server (8192 envs). The user runs the server and sends checkpoints through git (`agent_*.pt` at the
  repo root, see section 3). Run standing locally only if the user asks (then 4096 envs, as stand_07 did); one training
  run at a time on the local GPU.
- **Launching a training run:**
  - Name rounds `walk_NN` / stand `stage N` (next: walk_15, stage 6). The run directory name is the skrl timestamp; record
    the name ↔ directory mapping in the section 3 checkpoint table and the section 6 history.
  - Write a launcher `logs/train/<name>.sh` (venv, EULA, `cd` to the repo, the full train command with every override,
    a comment saying why) and start it detached from the repo root:
    `nohup bash logs/train/<name>.sh > logs/train/<name>.log 2>&1 &` (or a tmux session).
  - `train.py` creates the run directory at once but prints no progress for about 2-3 minutes while Isaac Sim starts.
    Confirm the start: the skrl progress bar in the log, the new `logs/skrl/<dir>/<timestamp>_ppo_torch/`, and the
    overrides (for example `ekf_freeze_gyro_bias_z`) in its `params/env.yaml`. Report the measured it/s and the ETA.
  - Pass every recipe override explicitly, including ones equal to the current cfg default (for example
    `env.ekf_freeze_gyro_bias_z=True`), so the launcher states the intent even if the default changes later.
- **Evaluation outputs:** write logs to `logs/eval/<MMDD>/<metric>_<name>.log` (for example `logs/eval/1125/gait_walk14.log`);
  `eval_switch.py` writes to `logs/eval_switch/<tag>`, `walk_replay.py` to `scripts/real2sim/walk_replay/res_<tag>*`.
  These are not committed. Summarise the numbers in the section 6 history and in commit messages.
- **git:** the shell has no credentials. The user pushes, or gives a token in a message. Use the token only in that push
  command and never store it (no remote URL, no credential helper). Commit and push only when asked.
- **Running things:**
  - `scripts/skrl/train.py` and `play.py` do not accept `--headless` (pass `physics=isaacsim_physx`).
    `scripts/eval/*`, `eval_switch.py` and `walk_replay.py` do accept it.
  - In play.py output, grep falls with "fell", not "fall". Use seed 2 for standing videos; seed 42 gives env 0 a bad DR
    sample.
  - Isaac Lab 3.x quaternions are xyzw. In scratch scripts, take the yaw with `olaf_env._yaw_of`.
  - Do not use `pkill -f` or `pgrep -f <own name>` in background scripts; they match their own shell.

## 3. Paths

| What | Where |
|---|---|
| Past runs (selected checkpoints, params, TensorBoard) | `logs/skrl/olaf_{walk,stand}/` in this repo (force-added: `logs/` is in .gitignore, so `git add -f`). All checkpoints of these runs are still in `/opt/isaac/projects/arti_h1_olaf/logs/skrl/` on the dev PC |
| Repo-root `agent_*.pt` | the server's hand-off spot. Now `agent_240000.pt` / `agent_480000.pt` = **stand stage 5** (not walk). Copy each new one to `logs/skrl/olaf_stand/server_stageN/checkpoints/` and identify it by the observation preprocessor count (stage start count + N × 8192) |
| Controller checkout | `/opt/isaac/projects/humanoid_controller_olaf` (branch `arti-h1-olaf`, remote SeonilChoi/humanoid_controller) |
| Controller config / ONNX | `config/robot/ARTI-H1/robot_olaf.yaml`, `agent_olaf_*.onnx`, `olaf_walk_keys.yaml` |
| Robot PC | Jetson Orin `~/humanoid_controller_olaf`. The C++ build works only there; the dev PC lacks yaml-cpp and onnxruntime |
| Real logs | `data/real_logs/arti_h1_olaf_run_<YYYYMMDD_HHMMSS>.csv`: all 12 Olaf robot runs (11-20 x7, 11-21 x4, 11-23 x1), taken from the controller history (removed there by commit 6af2da5). Hard floor: 11-21 095329 / 095641 / 100031, 11-23 132540; 11-21 095050 is carpet. Run the replay scripts from the repo root (their defaults are `data/real_logs/...`) |
| Log columns | `t, phase, obs_0..79, q_des_*, action_*, pos_/vel_/tau_/temp_<joint>, imu_q*, imu_g*, imu_a*, olaf_walk, est_x..est_vz, g0..g2`. g0, g1, g2 = path vx, vy, yaw rate. obs_3..8 = rot6 of the estimated attitude relative to the path frame; heading relative to the path = atan2(obs_5, obs_3) |

Checkpoints:

All checkpoints listed here except walk_14 were trained with a **free** b_g,z (their `params/env.yaml` has no
`ekf_freeze_gyro_bias_z` key); the controller runs them with b_g,z frozen since 11-24.

| Name | Path (under logs/skrl/) |
|---|---|
| walk_09 | `olaf_walk/2026-11-18_09-03-50_ppo_torch/checkpoints/agent_1944000.pt` |
| **walk_10 (deployed since 11-24)** | `olaf_walk/2026-11-20_15-40-44_ppo_torch/checkpoints/agent_480000.pt` |
| walk_11 | `olaf_walk/2026-11-21_12-07-56_ppo_torch/checkpoints/agent_480000.pt` |
| walk_12 | `olaf_walk/2026-11-22_17-06-20_ppo_torch/checkpoints/agent_480000.pt` |
| walk_13 | `olaf_walk/2026-11-23_14-51-56_ppo_torch/checkpoints/agent_480000.pt` (robot 11-24: spun and fell in 3.5 s) |
| walk_14 (stopped at 96k on 11-24 for walk_15) | `olaf_walk/2026-11-24_13-06-19_ppo_torch/checkpoints/agent_96000.pt` (local only; launcher `logs/train/walk_14.sh`) |
| walk_15 (training, 11-24 19:51, about 12.5 it/s, ETA about 06:30 11-25) | `olaf_walk/2026-11-24_19-51-43_ppo_torch/` (local only until it is committed; launcher `logs/train/walk_15.sh`) |
| **stand_07 (smooth, deployed since 11-24)** | `olaf_stand/2026-11-20_13-25-38_ppo_torch/checkpoints/agent_{72000,96000}.pt` (72000 = `agent_olaf_stand_smooth_72000.onnx`) |
| stand stage 3 | `olaf_stand/server_stage3/checkpoints/agent_480000.pt` (no tremor on the robot) |
| stand stage 4 | `olaf_stand/server_stage4/checkpoints/agent_264000.pt` (8.8 Hz tremor on the robot) |
| stand stage 5 | `olaf_stand/server_stage5/checkpoints/agent_{240000,480000}.pt` (robot 11-24: worst tremor so far). Its `params/` is a copy of stage 4's, for export only (NOTE.txt). Server stages have no TensorBoard files here |

## 4. Evaluation metrics and acceptance criteria

Always evaluate at the **final DR** (`-Play-` tasks), and rerun the baseline (the deployed policy) with **the same code
and settings**. Fall counts carry large DR-sampling noise (the same checkpoint has given 3 vs 24), so do not draw
conclusions from small differences.

### Walking

| Metric | Tool / command | Criterion | Reference values |
|---|---|---|---|
| Real-log replay falls | `RAW=1 R2S='cfg.est_z_offset_range=(0.0,0.0); cfg.est_z_drift_range=(0.0,0.0)' python scripts/real2sim/walk_replay/walk_replay.py --checkpoint <pt> --num_envs 32 --tag <t> --logs <20261123_132540.csv> --headless` | 0-2 / 32 | walk_13 0/32, walk_12 2/32, walk_10 22/32 |
| In-place yaw rate (g = 0, log 132540 t 43-103 s) | `python scripts/eval/replay_yaw.py` on that run's `res_<t>_raw.npz` | mean \|rate\| < 1.5 deg/s | b_g,z frozen: walk_13 ±0.2; before: walk_12 +7.4 |
| Turn tracking (t ≈ 123 s, g2 −0.39 rad/s = −22 deg/s) | same output | ≥ 90 % of the command | frozen: −21 deg/s; before: −13 |
| 11-21 hard-floor log replay (compliance model) | `R2S="cfg.joint_series_compliance_scale_range=(s,s); cfg.torso_torque_noise_std=(1.5,2.0,1.0)"`, s = 0.75 / 1.0, logs 095329 095641 100031 | fewest falls | walk_12 0/128, 2/128; walk_10 42/128, 86/128 |
| Starts from real pre-fall states | `scripts/real2sim/walk_replay/state_start.py --checkpoint <pt> --tag <t> --headless` | falls ≤ 2 % | walk_12 1.7 %, walk_10 33 % |
| Walking stability | `python scripts/eval/gait_check.py --checkpoint <pt> --steps 750 --num_envs 256 --headless` (and `NO_PUSH=1`) | ≤ the deployed policy | walk_13 62/21, walk_12 55/26 (push / no push) |
| Gait quality | gait_check output | contact match ≥ 0.91, swing apex about 2.6 cm (reference 3.0), tilt about 6.5 deg | walk_13 0.910 / 2.57 cm / 6.65 deg |
| Knee saturation, action change | `python scripts/eval/walk_sat.py` | lower is better | knee \|a\| > 0.95 fraction: walk_09 0.81, walk_10 0.55 |
| Falls in the walk stage before a switch | walk stage of `scripts/eval_switch.py` | ≤ 1 % | walk_12, walk_13 0.4 % |

### Standing

| Metric | Tool / command | Criterion | Reference values |
|---|---|---|---|
| Sim tremor proxy: share of action power above 5 Hz | `NO_PUSH=1 LPF_HZ=20 python scripts/real2sim/stand_tremor.py --checkpoint <pt> --num_envs 64 --steps 1000 --headless` | **not a gate any more** (report only) | stage 3 0.05-0.09 (fine), stage 4 0.35 (shook), stage 5 0.21 (shook **worse** than stage 4 on 11-24): it does not rank policies. The sim does not reproduce the tremor at all, so a stand policy is accepted only after a robot test |
| Tremor-band amplitude | same output: gyro 7.5-11 Hz rms, q_des 7.5-11 Hz rms | for reference | stage 5 0.027/0.029, stage 4 0.036/0.035 |
| Standing stability | `python scripts/eval/stand_check.py --checkpoint <pt> --steps 750 --num_envs 256 --headless` (and `NO_PUSH=1`) | ≤ the deployed policy | stage 5 34/3, stage 4 82/13, stage 3 306/161 |
| Walk → stand switch | `python scripts/eval_switch.py --walk_ckpt <deployed walk> --stand_ckpt <pt> --out logs/eval_switch/<tag> --headless` | ≤ 10 % | with walk_12: stage 5 5.5 %, stage 4 9.0 %, stage 3 35 % |
| Action change \|da\|/step (20 Hz LPF) | stand_tremor output | ≤ about 0.03 | stage 5 0.014 |

`stand_check` defaults to 1000 steps; pass `--steps 750` when comparing with the numbers above.

### Metrics on real logs (after a robot test)

- Standing tremor (longest standing stretch, skip the first 3 s): gyro_y rms in 6-10 Hz, ankle_pitch q_des rms in
  6-10 Hz, and the coherence between them. stand_07: 0.07 rad/s / 0.26 deg (11-24), stage 4: 0.13 / 0.62 deg
  (coherence 0.97), stage 5: 0.39 / 1.17 deg (coherence 0.99 = policy-loop limit cycle). Also |da|/step, tilt ≤ 4 deg.
- Walking: IMU yaw rate at g = 0 (from `imu_q*`); whether the heading relative to the path (obs_3, obs_5) runs out to
  the leash (20 deg); knee saturation (about 0.55 on the robot and in sim); stride pitch range right after a turn
  (7 deg growing to 15-20 deg precedes a fall).
- Falls: look at g and the pose 1-2 s before the fall. Every 11-21 real fall came about 2 s after a turn ended.

## 5. Current training recipe

The cfg defaults (walk rate 5 / acc 1, stand rate 2 / acc 0.5) are **not the deployed recipe**. Pass these CLI overrides.
Since 11-24 add `env.ekf_freeze_gyro_bias_z=True "env.joint_series_compliance_scale_range=[0.0,0.0]"` (section 1), and
for standing `env.action_lowpass_hz=20.0` (the robot's stand low-pass). Current launchers: `logs/train/walk_15.sh`
(local, from walk_10) and `scripts/server_stand_stage6.sh` (server, from stage 3, switch bank
`olaf_switch_bank_walk10_480000.npz` = walk_10 without compliance).

Walking (as walk_13, fine-tuned from the previous checkpoint):
```bash
python scripts/skrl/train.py --task HumanoidSim-Olaf-Walk-v0 physics=isaacsim_physx --num_envs 4096 \
  agent.agent.experiment.directory=olaf_walk --checkpoint <walk pt> \
  env.curriculum_step_offset=2400000 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
  env.walk_zero_command_fraction=0.3 env.w_yaw_drift=50.0 env.episode_length_s=40.0 agent.trainer.timesteps=480000
```

Standing (as stage 5, server):
```bash
python scripts/skrl/train.py --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs 8192 \
  agent.agent.experiment.directory=olaf_stand --checkpoint <stand pt> env.curriculum_step_offset=1200000 \
  env.w_leg_action_rate=15.0 env.w_leg_action_acc=10.0 \
  env.switch_bank_file=olaf_switch_bank_walk10_480000_compl.npz env.reset_walk_pose_fraction=0.6 agent.trainer.timesteps=480000
```

- `curriculum_step_offset`: for fine-tuning, set it at or above the end of `dr_widen_steps` (24k, 1.2M) so the DR is at
  its final range from the start.
- A switch bank (`switch_bank`) belongs to the walk policy that produced it. The current file comes from walk_10.
  The deployed walk is walk_10 again, whose non-compliance bank is `olaf_switch_bank_walk10_480000.npz`. When the
  deployed walk policy changes, rebuild it:
  `python scripts/eval_switch.py --walk_ckpt <walk> --stand_ckpt - --num_envs 4096 --bank <name>.npz --headless`.
- `scripts/server_stand_stage5.sh` starts from `agent_stand_stage3_480000.pt` at the repo root, which was removed on
  11-24. Fix that path before reusing the script.

## 6. Training history

| Round | Change | Result |
|---|---|---|
| walk_01-02 | no estimator | invalid |
| walk_03-08 | robot-side estimator (InEKF), 600 Hz / 50 Hz runtime, impact scaling, contact-flag fixes | restarted after each audit |
| walk_09 | contact flags binarised at 0.5, 2.4M cumulative | robot (11-20): knee bang-bang (\|a\| > 0.95 80 % of the time), fell within seconds |
| walk_10 | smoothness fine-tune (rate 15, acc 3) | knee saturation 0.81 → 0.55. Robot: 53 s on a hard floor, fell right after a turn |
| walk_11 | real2sim DR v1 (delay +15 ms, kp, torso torque noise), zero command 0.3 | marginal gain, not adopted |
| walk_12 | real2sim v2 (series joint compliance, est_z DR) | replay falls 86 → 2/128. Robot (11-23): in-place yaw drift |
| walk_13 | yaw_drift reward (w 50, tau 1 s), 40 s episodes; trained with b_g,z **free** | the reward alone did not fix the yaw. Deployed on 11-24 with the controller's b_g,z **frozen** (sim replay: yaw fixed, 0/32 falls) |
| walk_14 | walk_13 + the same recipe, trained with b_g,z **frozen** (training = deployment observation path) | stopped at 96k (11-24) to free the GPU for walk_15; still has compliance DR, so low priority |
| walk_10 on the robot with b_g,z frozen (11-24) | — | 130 s walking without a fall, in-place yaw +0.4 deg/s (free bias 11-21: −1.5), right turns only; **deployed** with stand_07 |
| walk_15 | from walk_10, walk_13 recipe, b_g,z frozen, **no compliance DR** | running (11-24 19:51, `2026-11-24_19-51-43_ppo_torch`) |
| stand_01-06 | base standing | robot (11-20): 9.6 Hz shaking |
| stand_07 | smoothness (rate 10, acc 3) | robot standing fine (\|da\| 0.0013-0.0027) |
| stage 2 | starts from the walk_09 switch bank | better switches but more jitter, not adopted |
| stage 3 | walk_10 compliance switch bank | adopted; no tremor on the robot |
| stage 4 | compliance DR | much more stable in sim, but an 8.8 Hz tremor on the robot |
| stage 5 | restarted from stage 3, acc 3 → 10 | most stable in sim, but on the robot (11-24) the worst tremor so far (7.4 Hz, gyro 0.39 rad/s) |
| stage 6 | from stage 3, **no compliance DR**, b_g,z frozen, 20 Hz low-pass, walk_10 bank | running on the server since 11-25 01:12 (`logs/train/stand_s6_1125_0112.log` there) |

## 7. real2sim (all in the env cfg)

- **Runtime:** physics 600 Hz, policy 50 Hz (decimation 12). Per physics step: first-order hold → 5.24 rad/s slew limit
  → 37.5 Hz low-pass. The controller runs the same chain in a 600 Hz servo thread. On the robot the low-pass is per
  mode: 20 Hz standing, 37.5 Hz walking (`stand_action_lowpass_hz`).
- **Actuators:** delay 19 ms (±6, step jitter 1), extra delay hip_roll 10 ms / ankle_pitch 8 ms, Coulomb friction,
  armature, kp DR (0.7, 1.2). Inherited from the earlier arti_h1_local line's closed-loop replay of the 11-13 logs. The 125 Hz Unitree bus
  command update is not modelled (risk of double counting the fitted 4 ms delay).
- **Thermal model:** fitted on 28 real logs (alpha fixed at 0.001, beta per joint type). T_max 80 °C, observation clip
  [70, 85].
- **IMU (MTi-3):** gyro noise, bias and delay. The accelerometer uses the 10 ms mean of `body_link_lin_vel_w` (not the
  COM velocity). Touchdown impact noise (3.8, 5.2, 16) × a per-episode scale, bias ±0.15.
- **Estimator:** contact-aided InEKF (`olaf/inekf.py`); its C++ twin `robot::OlafInEKF` agrees to 3e-13. Noise:
  accel 2.0, contact 0.05, fk 0.004. **`ekf_freeze_gyro_bias_z = True` (11-24):** a free b_g,z absorbed the true yaw
  rate, the estimated heading stood still, and on the robot the walk policy turned at up to +13 deg/s with g = 0.
- **Series joint compliance (11-22):** link deflection the encoders do not see. link = enc − c·tau, with c [rad/Nm] =
  hip_roll 0.021, ankle_pitch 0.016, hip_pitch 0.006, knee 0.006, ankle_roll 0.003, DR ×0.6-1.2. It raises the
  correlation between single-support torso attitude and encoder kinematics from 0.27 to 0.89. It was the main cause of
  the falls right after turns.
- **Torso torque noise:** (1.5, 2, 1) Nm, bands (3-12, 3-12, 0.5-5 Hz), scale 0.5-1.75. Walking only; 0 when standing.
- **est_z:** offset (−0.12, 0.01) m and drift (−0.002, 0.0005) m/s DR, for the real estimator's height drift (about
  −1.4 mm/s).
- **DR widening curriculum:** `dr_widen_steps` (24k, 1.2M), 10 levels.

## 8. sim2real (controller)

- `robot_olaf.yaml` (`policy.observation: olaf`) runs both policies. Pad: cross = walk, circle = stand (switches at the
  next touchdown, g = 0 meanwhile, 1 s timeout).
- **STAND stick shaping (controller, 11-24):** continuous deadzone, expo 0.6, a 0.4 s low-pass on the stick target, and
  half the training follow rate (`olaf.stand_stick_expo`, `stand_target_tau`, `stand_command_rate`). Slower than
  training, so it adds nothing out of distribution; the operator found the old direct mapping too abrupt.
- **Mode-switch reset:** on a policy switch, a(t) and a(t−1) are reset to clamp((q − default) / scale)
  (`olaf_switch_reset`, `olaf.action_scale`). This must mirror the env's `_reset_idx`. On 11-20 the robot fell because
  the walk policy's ±1 actions were fed to the stand policy.
- **Deployment:**
  1. Export with `python scripts/export_onnx.py --checkpoint <run>/checkpoints/agent_N.pt --out <ctrl>/config/robot/ARTI-H1/agent_olaf_<name>.onnx [--keys_out ...]`
     and check that ONNX and torch agree to 1e-6.
  2. The run directory needs `params/env.yaml`. If the server sent only a checkpoint, copy the params of another stage
     whose export fields are identical.
  3. Update `olaf_walk_keys.yaml` if the walk keys changed, then `model_file` / `stand_model_file`.
  4. Put the sim evaluation numbers in the commit message.
- When the sim observation path changes (estimator, filters, resets), change the C++ side in the same piece of work and
  test that Python and C++ agree. Parity harness: feed the same 400 random input steps to the Python InEKF and to a
  small C++ program compiled against the header, and compare.
- Test on a **hard floor** (on carpet the robot fell after 2.9 s; the sim ground is rigid). Walk slowly on the hanger
  first.

## 9. Lessons from robot tests

- 11-20: the 9.6 Hz standing shake was the policy's own jitter (adding sim delay did not reproduce it). Fixed with the
  smoothness rewards (stand_07).
- 11-20: the walking knee bang-bang was the same in sim. The policy used the 5.24 rad/s slew limit as its actuator.
  Fixed with a smoothness fine-tune (walk_10).
- 11-21: falls came about 2 s after a turn ended, with growing pitch / yaw oscillation. Cause: series compliance
  (fixed in walk_12).
- 11-23: stand stage 4 showed an 8.8 Hz policy-loop limit cycle. The sim does not reproduce it; the only proxy that
  ranks policies is action power above 5 Hz.
- 11-23: the in-place yaw drift came from the InEKF's b_g,z (frozen on 11-24). A short spike in the estimated yaw over
  the first ~4 s of walking is still open.
- 11-20: at 71 s all 8 Unitree motors went stale at once. Check the bus / power. The hold worked.
- 11-24: freezing b_g,z worked on the robot (walk_10: in-place yaw −1.5 → +0.4 deg/s). walk_13 turned −18 deg in the
  first 2 s of walking and fell after 3.5 s; stand stage 5 shook at 7.4 Hz, worse than stage 4. walk_10 + stand_07
  (no compliance DR) walked 130 s; it fell once in STAND at the end, during quick roll / pitch stick changes.

## 10. Open issues and next steps (in priority order)

1. **Compliance-off comparison:** walk_15 (local, running since 11-24 19:51, `2026-11-24_19-51-43_ppo_torch`) and stand stage 6 (server, `scripts/server_stand_stage6.sh`). Evaluate with
   section 4 against walk_10 / stand_07 rerun with the same code; the deciding test is the robot (stand tremor metrics,
   in-place yaw, left and right turns).
2. **Robot test of the deployed pair** (walk_10 + stand_07, b_g,z frozen): **left turns** (only right turns were
   tested on 11-24; the 11-21 falls came after left turns) and STAND stick changes with the new stick shaping.
3. Re-fit or drop the series compliance: check it on standing data before using it again.
4. The yaw transient over the first ~2 s of walking (walk_13 −18 deg, walk_10 −5 to −7 deg; always negative).
5. walk_14 (compliance DR, b_g,z frozen): finish only if the GPU is free; lower priority than walk_15.
6. Not modelled (no action planned): encoder zero errors (sim FK uses the true joint angles), the 125 Hz bus, the mass
   properties of the new pelvis (`arti-h1-new-pelvis`), the Togepi exterior's mass.
