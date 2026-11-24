# Olaf walk real2sim (2026-11-21)

Replays the real walk segments of humanoid_controller `arti_h1_olaf_run_*.csv` logs in sim, with the same g sequence from the
stand -> walk switch, and compares per-segment metrics (in place / forward / turn / 2.5 s after a turn): gyro std and band
power, roll / pitch, |q_des - q| and torque rms per joint type, knee saturation, double support.

    python scripts/real2sim/walk_replay/walk_replay.py --checkpoint <walk.pt> --num_envs 32 --tag nominal --logs <csv...> --headless
    R2S='cfg.dr_start_kp=(0.8,0.9)' python .../walk_replay.py ... --tag kp   # cfg overrides
    python scripts/real2sim/walk_replay/compare.py nominal kp

Result (walk_10 agent_480000, hard-floor logs 095329 / 095641 / 100031, 4 walk segments, 3 real falls):
- old cfg: 0 / 128 sim falls. Real |q_des - q| 1.3-1.8x sim at 0.6-0.9x torque -> +15 ms actuator delay and kp x0.8-0.9.
- real gyro 2-3x sim at 3-12 Hz (roll / pitch rate) and 0-6 Hz (yaw rate); no delay / gain / Coulomb / friction / hip-yaw
  stiffness change reproduces it -> torso_torque_noise_std (2.0, 2.5, 2.0) Nm, bands (3-12, 3-12, 0.5-5 Hz) matches the std.
- every real fall came ~2 s after a left turn ended, with yaw-rate and pitch oscillation growing (gz std 0.3 -> 1.0).

## 2026-11-22: why the real robot falls (real2sim v2)

Tools: `fall_analysis.py` / `turn_end.py` (per-stride precursors, real vs sim replay `RAW=1`), `state_start.py` (multiple
shooting: start the sim from the REAL state at many log instants, policy closed loop or `SS_OPENLOOP=1` real actions),
`foot_flat.py` (kinematic consistency of the logged pose in double support), `compliance_fit.py` (series-compliance
signature, same statistic for real and sim).

- Every real fall: pitch range per stride jumps 7 -> 15-20 deg right after a turn ends, then gz / stance-foot yaw slip / hip-yaw
  action grow for 2-3 strides. Steady walking is CALMER on the robot than in sim (pitch range p99 11.5 vs 19 deg), yet it falls.
- Sim started from the real state 0.5-3 s before each real fall: 1.7 % sim falls (rigid model) -> the sim recovers from
  states the robot cannot. Friction (ground mu 0-0.5), estimator height drift (real est_z -1.5 mm/s, -0.12 m after 75 s,
  not reset at stand<->walk) and torso noise do not explain it (5 -> 5..9 / 128 replay falls).
- In single support the torso pitch / roll change is uncorrelated with the encoder leg kinematics on the robot
  (corr 0.27 / -0.33; rigid sim 0.98 / 0.95). link = encoder - c tau explains it (corr 0.83-0.89 / 0.91), same c for both
  legs: hip_roll 0.021, ankle_pitch 0.016, hip_pitch 0.006, knee 0.006, ankle_roll 0.003 rad/Nm (Ks 47 / 61 / 160 / 180
  / 360 Nm/rad vs PD kp 72 / 30 / 72 / 100 / 27). This is what the old extra actuator lag on hip_roll / ankle_pitch stood in for.
- cfg.joint_series_compliance (motor PD on the encoder, link sees kp / (1 + kp c), the policy sees the encoder):
  replay of the real g 0 / 128 falls (rigid) -> 29 / 128 (c x1, no noise), 86 / 128 (c x1 + noise 1.5/2/1 Nm), 42 / 128
  (c x0.75 + noise); real: 3 of 4 segments. From the real pre-fall states 1.7 % -> 33 %. The same regression on the sim
  gives back the c it was run with (real ~ 0.85 x cfg) -> training range x0.6-1.2. Gyro std with c + noise 0.45/0.55/0.33
  (real 0.42/0.51/0.36). Remaining gaps: steady pitch range p50 9.3 vs 7.4 deg, two-feet-down fraction higher in sim.
- Ankle joint zero errors on the robot (standing, feet flat): ankle pitch +2.6 deg both legs (true = encoder + 2.6 deg).
