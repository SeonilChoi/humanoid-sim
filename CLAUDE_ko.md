# humanoid_sim: Claude 작업 지침

ARTI-H1 KB(무릎 굽힌 자세) 로봇에 Olaf 논문(arXiv 2512.16705) 방식의 모방 RL로 **보행 정책과 서기 정책을 따로**
학습한다. 실물(humanoid_controller, 브랜치 `arti-h1-olaf`)에 올려 **넘어지지 않고 서고 걷는 것**이 목표다.
이 프로젝트는 `arti_h1_olaf`(커밋 2c756fe, 2026-11-24)를 이어받았고, PlaCo 참조 생성기를
`humanoid_sim.utils.motion_generator`로 옮겨 왔다. 아래는 2026-11-17 ~ 11-24 작업에서 얻은 내용이다.

## 1. 학습 목표와 판단 원칙

- **안정성 > 동작 품질.** 라운드 채택은 넘어짐, 밀림 내성, 기울기, 드리프트로 판정하고 "걷긴 걷는다"로 통과시키지 않는다.
- **sim 수치만으로 채택하지 않는다.** 시뮬레이션은 실물의 8.8 Hz 서기 떨림을 재현하지 못한다.
  실물에서 확인된 기준과 프록시(아래 4절)를 함께 본다.
- **보행 직진성은 env별로 본다.** 평균은 env 사이에서 상쇄되므로 env별 |vy|, |lean|, yaw rate의 분포(p10/p90)를 확인한다.
  느려도 곧게 가는 정책이 빠르지만 옆으로 새는 정책보다 낫다.
- **실물 문제는 리플레이로 재현한 뒤에 고친다.** 실물 로그의 g 시퀀스를 sim에서 재생(`walk_replay`)하거나 실물
  상태에서 시작(`state_start`)해 같은 현상이 나는지 먼저 본다. 재현되지 않으면 원인부터 다시 찾는다.
- **원인을 학습 쪽에서만 찾지 않는다.** 11-23의 제자리 yaw 회전은 보상(walk_13 yaw_drift)으로 못 고쳤고, 실제 원인은
  추정기 bg_z였다. 관측 경로(추정기, 컨트롤러 리셋, 필터)를 함께 의심한다.

## 2. 환경과 작업 규칙

- **venv:** `source /opt/isaac/IsaacLab-main/.venv/bin/activate`, `export OMNI_KIT_ACCEPT_EULA=yes`.
  `/opt/isaac/.venv`는 activate가 없는 빈 venv이니 쓰지 않는다.
- **설치:** `uv pip install --python /opt/isaac/IsaacLab-main/.venv/bin/python --no-deps -e source/humanoid_sim`.
  `arti_h1_olaf`도 같은 venv에 설치돼 있다. 태스크 ID가 다르다(`HumanoidSim-Olaf-*` / `Arti-H1-Olaf-*`).
- **태스크:** `HumanoidSim-Olaf-Walk-v0`, `HumanoidSim-Olaf-Stand-v0`. 평가용은 `-Play-v0`이고 최종 DR이 적용된다.
- **PlaCo** (참조를 다시 만들 때만 필요): Isaac venv에 placo 0.12.0이 설치돼 있다(pin 4.1.0 유지).
  0.10.2로 만든 기존 클립 및 그리드와 1e-15까지 같다.
- **학습 장소:** 보행은 로컬(RTX 5070 Ti, 4096 envs, 약 14 it/s, 480k 스텝에 약 10.5 h). 서기는 서버(8192 envs).
  사용자가 서버를 직접 돌리고, 체크포인트는 git으로 받는다(저장소 루트의 `agent_*.pt`).
- **git:** 셸에 자격 증명이 없다. 사용자가 직접 push하거나 토큰을 메시지에 준다. 토큰은 그 push 명령에서만 쓰고
  저장하지 않는다(remote URL, credential helper 포함). 커밋과 push는 요청받았을 때만 한다.
- **실행 관련 주의:**
  - `scripts/skrl/train.py`와 `play.py`는 `--headless`를 받지 않는다(`physics=isaacsim_physx`를 넘긴다). `scripts/eval/*`, `eval_switch.py`, `walk_replay.py`는 `--headless`를 받는다.
  - play.py 로그에서 넘어짐은 "fell"로 grep한다("fall"이 아니다). 서기 영상에는 seed 2를 쓴다. seed 42는 env 0의 DR 샘플이 나쁘다.
  - Isaac Lab 3.x 쿼터니언은 xyzw다. 임시 스크립트에서 yaw는 `olaf_env._yaw_of`를 쓴다.
  - 백그라운드 스크립트에서 `pkill -f`나 `pgrep -f <자기 이름>`을 쓰지 않는다. 자기 셸에 매칭된다.

## 3. 관련 경로

| 항목 | 위치 |
|---|---|
| 지난 학습 결과 (체크포인트, TB) | `/opt/isaac/projects/arti_h1_olaf/logs/skrl/olaf_{walk,stand}/` (humanoid_sim에는 없다) |
| 컨트롤러 worktree | `/opt/isaac/projects/humanoid_controller_olaf` (브랜치 `arti-h1-olaf`, 원격 SeonilChoi/humanoid_controller) |
| 컨트롤러 설정 / ONNX | `config/robot/ARTI-H1/robot_olaf.yaml`, `agent_olaf_*.onnx`, `olaf_walk_keys.yaml` |
| 로봇 PC | Jetson Orin `~/humanoid_controller_olaf` (C++ 빌드는 여기서만 된다. 개발 PC에는 yaml-cpp와 onnxruntime이 없다) |
| 실물 로그 | `arti_h1_olaf_run_<YYYYMMDD_HHMMSS>.csv`. 11-24 커밋 6af2da5에서 working tree에서 지워졌다. `git -C /opt/isaac/projects/humanoid_controller_olaf show 42ad823:arti_h1_olaf_run_20261123_132540.csv > /tmp/...`로 꺼낸다(11-21 로그도 같은 커밋). 11-20 로그는 61fd3ad 근처 커밋에 있다 |
| 로그 열 | 로그 열: `t, phase, obs_0..79, q_des_*, action_*, pos_/vel_/tau_/temp_<joint>, imu_q*, imu_g*, imu_a*, olaf_walk, est_x..est_vz, g0..g2`. g0, g1, g2 = 경로 vx, vy, yaw rate. obs_3..8 = rot6 (추정 자세의 경로 기준 상대값), 경로 기준 상대 yaw = atan2(obs_5, obs_3) |

체크포인트 목록:

| 이름 | 경로 (arti_h1_olaf/logs/skrl/...) |
|---|---|
| walk_09 | `olaf_walk/2026-11-18_09-03-50_ppo_torch/checkpoints/agent_1944000.pt` |
| walk_10 | `olaf_walk/2026-11-20_15-40-44_ppo_torch/checkpoints/agent_480000.pt` |
| walk_11 | `olaf_walk/2026-11-21_12-07-56_ppo_torch/checkpoints/agent_480000.pt` |
| walk_12 | `olaf_walk/2026-11-22_17-06-20_ppo_torch/checkpoints/agent_480000.pt` |
| **walk_13 (배포 중)** | `olaf_walk/2026-11-23_14-51-56_ppo_torch/checkpoints/agent_480000.pt` |
| stand_07 (smooth) | `olaf_stand/2026-11-20_13-25-38_ppo_torch/checkpoints/agent_{72000,96000}.pt` |
| stand stage3 | `olaf_stand/server_stage3/checkpoints/agent_480000.pt` (실물에서 떨림 없음) |
| stand stage4 | `olaf_stand/server_stage4/checkpoints/agent_264000.pt` (실물 8.8 Hz 떨림) |
| **stand stage5 (배포 중)** | `olaf_stand/server_stage5/checkpoints/agent_480000.pt`. `params/`는 stage4 것을 복사했다(export용, NOTE.txt) |

## 4. 평가 지표와 합격 기준

평가는 항상 **최종 DR**(`-Play-` 태스크)에서 하고, 비교 대상(지금 배포 중인 정책)과 **같은 코드, 같은 설정**으로 다시 돌린다.
넘어짐 수는 DR 추첨 노이즈가 크다(같은 체크포인트에서 3 대 24가 나온 적 있다). 작은 차이로 결론 내지 않는다.

### 보행

| 지표 | 도구 / 명령 | 기준 | 참고 값 |
|---|---|---|---|
| 실물 로그 리플레이 넘어짐 | `RAW=1 R2S='cfg.est_z_offset_range=(0.0,0.0); cfg.est_z_drift_range=(0.0,0.0)' python scripts/real2sim/walk_replay/walk_replay.py --checkpoint <pt> --num_envs 32 --tag <t> --logs <20261123_132540.csv> --headless` | 0~2 / 32 | walk_13 0/32, walk_12 2/32, walk_10 22/32 |
| 제자리 yaw rate (g=0, 로그 132540의 t 43~103 s) | 위 결과의 `res_<t>_raw.npz`에 `python scripts/eval/replay_yaw.py ...` | 평균 \|rate\| < 1.5°/s | bg_z 고정 후 walk_13 ±0.2, 이전에는 walk_12 +7.4 |
| 회전 추종 (t≈123 s, g2 −0.39 rad/s = −22°/s) | 같은 출력 | 명령의 90% 이상 | 고정 후 −21°/s, 이전에는 −13 |
| 11-21 하드 바닥 로그 리플레이 (컴플라이언스 모델) | `R2S="cfg.joint_series_compliance_scale_range=(s,s); cfg.torso_torque_noise_std=(1.5,2.0,1.0)"`, s = 0.75 / 1.0, 로그 095329 095641 100031 | 넘어짐 최소 | walk_12 0/128, 2/128; walk_10 42/128, 86/128 |
| 실물 넘어지기 직전 상태에서 시작 | `scripts/real2sim/walk_replay/state_start.py --checkpoint <pt> --tag <t> --headless` | 넘어짐 ≤ 2% | walk_12 1.7%, walk_10 33% |
| 보행 안정성 | `python scripts/eval/gait_check.py --checkpoint <pt> --steps 750 --num_envs 256 --headless` (+ `NO_PUSH=1`) | 배포 정책 이하 | walk_13 62/21, walk_12 55/26 (push / no-push) |
| 보행 품질 | gait_check 출력 | contact match ≥ 0.91, swing apex 약 2.6 cm (참조 3.0), tilt 약 6.5° | walk_13 0.910 / 2.57 cm / 6.65° |
| 무릎 포화, 출력 변화 | `python scripts/eval/walk_sat.py` | 낮을수록 좋다 | 무릎 \|a\|>0.95 비율: walk_09 0.81, walk_10 0.55 |
| 전환 직전 보행 단계 넘어짐 | `scripts/eval_switch.py` 출력의 walk stage | ≤ 1% | walk_12, walk_13 0.4% |

### 서기

| 지표 | 도구 / 명령 | 기준 | 참고 값 |
|---|---|---|---|
| **떨림 프록시**: 출력 파워 중 5 Hz 이상 비율 | `NO_PUSH=1 LPF_HZ=20 python scripts/real2sim/stand_tremor.py --checkpoint <pt> --num_envs 64 --steps 1000 --headless` | **≤ 0.1** | stage3 0.05~0.09 (실물 OK), stage4 0.35 (실물 떨림), stage5 0.21 (실물 미확인) |
| 떨림 대역 진폭 | 같은 출력의 gyro 7.5~11 Hz rms, q_des 7.5~11 Hz rms | 참고용 | stage5 0.027/0.029, stage4 0.036/0.035 |
| 서기 안정성 | `python scripts/eval/stand_check.py --checkpoint <pt> --steps 750 --num_envs 256 --headless` (+ `NO_PUSH=1`) | 배포 정책 이하 | stage5 34/3, stage4 82/13, stage3 306/161 |
| 보행→서기 전환 | `python scripts/eval_switch.py --walk_ckpt <배포 walk> --stand_ckpt <pt> --out logs/eval_switch/<tag> --headless` | ≤ 10% | walk_12 기준 stage5 5.5%, stage4 9.0%, stage3 35% |
| 출력 변화 \|da\|/step (20 Hz LPF) | stand_tremor 출력 | ≤ 약 0.03 | stage5 0.014 |

`stand_check`의 기본 스텝은 1000이다. 비교할 때는 `--steps 750`으로 맞춘다.

### 실물 로그에서 보는 지표 (테스트 후)

- 서기: |da|/step(정상 0.0013~0.0027), gyro_y std(정상 약 0.09), gyro 파워 중 5 Hz 이상 비율(stage3 22%, 떨린 stage4 79%), tilt ≤ 4°.
- 보행: g=0일 때 IMU yaw rate(`imu_q*`로 계산), 경로 기준 상대 yaw(obs_3, obs_5)가 리시(20°)까지 벌어지는지, 무릎 포화 비율(실물과 sim 모두 약 0.55), 회전 직후 pitch 범위(7° → 15~20°가 되면 넘어짐 전조).
- 넘어짐: 넘어지기 1~2초 전의 g와 자세를 확인한다. 11-21 실물 넘어짐은 모두 회전이 끝나고 약 2초 뒤에 났다.

## 5. 현재 학습 레시피

cfg 기본값(walk rate 5 / acc 1, stand rate 2 / acc 0.5)은 **배포 레시피가 아니다.** 아래 CLI 오버라이드를 함께 넘긴다.

보행 (walk_13 기준, 이전 체크포인트에서 미세조정):
```bash
python scripts/skrl/train.py --task HumanoidSim-Olaf-Walk-v0 physics=isaacsim_physx --num_envs 4096 \
  agent.agent.experiment.directory=olaf_walk --checkpoint <walk pt> \
  env.curriculum_step_offset=2400000 env.w_leg_action_rate=15.0 env.w_leg_action_acc=3.0 \
  env.walk_zero_command_fraction=0.3 env.w_yaw_drift=50.0 env.episode_length_s=40.0 agent.trainer.timesteps=480000
```

서기 (stage5 기준, 서버):
```bash
python scripts/skrl/train.py --task HumanoidSim-Olaf-Stand-v0 physics=isaacsim_physx --num_envs 8192 \
  agent.agent.experiment.directory=olaf_stand --checkpoint <stand pt> env.curriculum_step_offset=1200000 \
  env.w_leg_action_rate=15.0 env.w_leg_action_acc=10.0 \
  env.switch_bank_file=olaf_switch_bank_walk10_480000_compl.npz env.reset_walk_pose_fraction=0.6 agent.trainer.timesteps=480000
```

- `curriculum_step_offset`: 미세조정에서는 DR이 처음부터 최종 범위가 되도록 dr_widen_steps (24k, 1.2M) 이상으로 준다.
- 전환 상태 모음(`switch_bank`)은 그것을 만든 보행 정책에 묶여 있다. 지금 파일은 walk_10으로 만든 것이다. 다음 서기 학습 전에
  배포 중인 보행 정책(walk_13)으로 다시 만든다: `python scripts/eval_switch.py --walk_ckpt <walk> --stand_ckpt - --num_envs 4096 --bank <name>.npz --headless`.
- 서버 스크립트 `scripts/server_stand_stage5.sh`는 저장소 루트의 `agent_stand_stage3_480000.pt`를 쓰는데, 이 파일은 11-24에 지워졌다. 쓰려면 경로를 고친다.

## 6. 학습 내역

| 라운드 | 내용 | 결과 |
|---|---|---|
| walk_01~02 | 추정기 없음 | 무효 |
| walk_03~08 | 실물 추정기(InEKF), 600 Hz/50 Hz 런타임, 임팩트 스케일, 접촉 플래그 수정 등 | 감사 후 재시작 반복 |
| walk_09 | 접촉 플래그 0.5 이진화, 2.4M 누적 | 실물(11-20)에서 무릎 bang-bang(\|a\|>0.95가 80%), 몇 초 만에 넘어짐 |
| walk_10 | 매끄러움 미세조정(rate 15, acc 3) | 무릎 포화 0.81→0.55. 실물 하드 바닥 53 s 보행, 회전 직후 넘어짐 |
| walk_11 | 1차 real2sim DR(지연 +15 ms, kp, 몸통 토크 노이즈), zero command 0.3 | 이득 미미, 채택 안 함 |
| walk_12 | 2차 real2sim(직렬 관절 컴플라이언스, est_z DR) | 리플레이 넘어짐 86→2/128. 실물(11-23)에서 제자리 yaw 회전 |
| walk_13 | yaw_drift 보상(w 50, τ 1 s), 40 s 에피소드 | 보상만으로는 yaw를 못 고침. **bg_z 고정과 함께 배포** (11-24) |
| stand_01~06 | 기본 서기 | 실물(11-20) 9.6 Hz 떨림 |
| stand_07 | 매끄러움(rate 10, acc 3) | 실물 서기 정상(\|da\| 0.0013~0.0027) |
| stage2 | walk_09 전환 상태 모음에서 시작 | 전환은 좋아졌지만 떨림 커져 채택 안 함 |
| stage3 | walk_10 컴플라이언스 전환 상태 모음 | 채택. 실물 떨림 없음 |
| stage4 | 컴플라이언스 DR | sim 안정성은 크게 좋아졌지만 실물 8.8 Hz 떨림 |
| stage5 | stage3에서 다시 시작, acc 3→10 | 가장 안정적이지만 떨림 프록시 0.21. **배포, 실물 확인 대기** (11-24) |

## 7. real2sim 적용 내역 (모두 env cfg에 있다)

- **런타임:** 물리 600 Hz, 정책 50 Hz(decimation 12). 물리 스텝마다 FOH → 기울기 제한 5.24 rad/s → 37.5 Hz LPF.
  컨트롤러는 같은 체인을 600 Hz 서보 스레드로 돌린다. 실물은 서기 20 Hz, 보행 37.5 Hz로 모드별 LPF를 쓴다(`stand_action_lowpass_hz`).
- **액추에이터:** 지연 19 ms(±6, step jitter 1), 추가 지연 hip_roll 10 ms / ankle_pitch 8 ms, Coulomb 마찰, armature, kp DR (0.7, 1.2).
  KB 라인의 11-13 로그 폐루프 리플레이 값을 이어받았다.
  Unitree 버스의 125 Hz 지령 갱신은 모델링하지 않았다(4 ms 지연 피팅과 이중으로 계산될 위험이 있어서).
- **열 모델:** 28개 실물 로그로 피팅했다(α 0.001 고정, β는 관절 종류별). T_max 80 °C, 관측 clip [70, 85].
- **IMU (MTi-3):** 자이로 노이즈, 바이어스, 지연. 가속도계는 `body_link_lin_vel_w`의 10 ms 평균을 쓴다(COM 속도가 아니다).
  착지 충격 노이즈 (3.8, 5.2, 16) × 에피소드별 스케일, 바이어스 ±0.15.
- **추정기:** 접촉 보조 InEKF(`olaf/inekf.py`)이고 C++ 쌍둥이 `robot::OlafInEKF`와 3e-13까지 일치한다.
  노이즈는 accel 2.0, contact 0.05, fk 0.004. **`ekf_freeze_gyro_bias_z = True`(11-24)**: bg_z를 자유롭게 두면 실제 yaw
  회전을 흡수해서 추정 heading이 멈췄고, 그 결과 실물에서 g=0인데 최대 +13°/s로 돌았다.
- **직렬 관절 컴플라이언스(11-22):** 엔코더에 안 보이는 링크 처짐이다. link = enc − c·τ, c [rad/Nm] = hip_roll 0.021,
  ankle_pitch 0.016, hip_pitch 0.006, knee 0.006, ankle_roll 0.003, DR ×0.6~1.2. 단일 지지기 몸통 자세와 엔코더 기구학의 상관이
  0.27 → 0.89로 올라간다. 회전 직후 넘어짐의 주원인이었다.
- **몸통 토크 노이즈:** (1.5, 2, 1) Nm, 대역 (3~12, 3~12, 0.5~5 Hz), 스케일 0.5~1.75. 보행에만 쓰고 서기는 0이다.
- **est_z:** 실물 추정기의 높이 드리프트(−1.4 mm/s 수준)를 반영해 offset (−0.12, 0.01) m, drift (−0.002, 0.0005) m/s DR.
- **DR 확대 커리큘럼:** `dr_widen_steps` (24k, 1.2M), 10단계.

## 8. sim2real 적용 내역 (컨트롤러)

- `robot_olaf.yaml`(`policy.observation: olaf`)이 두 정책을 돌린다. 패드: cross = 보행, circle = 서기(다음 착지 때 전환, 그동안 g=0, 타임아웃 1 s).
- **모드 전환 리셋:** 정책이 바뀔 때 a(t), a(t−1)을 clamp((q − default)/scale)로 다시 설정한다(`olaf_switch_reset`, `olaf.action_scale`).
  env `_reset_idx`와 똑같아야 한다. 보행의 ±1 출력이 서기 정책 입력으로 넘어가서 넘어진 적이 있다(11-20).
- **배포 절차:**
  1. `python scripts/export_onnx.py --checkpoint <run>/checkpoints/agent_N.pt --out <ctrl>/config/robot/ARTI-H1/agent_olaf_<name>.onnx [--keys_out ...]` 로 변환한다. ONNX와 torch 차이가 1e-6 이하인지 확인한다.
  2. 실행 폴더에 `params/env.yaml`이 있어야 한다. 서버에서 체크포인트만 왔으면, export에 쓰는 항목이 같은 다른 stage의 params를 복사한다.
  3. 보행 키 파일이 바뀌었으면 `olaf_walk_keys.yaml`을 갱신하고, `model_file`과 `stand_model_file`을 고친다.
  4. 커밋 메시지에 sim 평가 수치를 남긴다.
- 추정기, 필터, 리셋처럼 sim 쪽 관측 경로를 바꾸면 C++도 같은 커밋 흐름에서 바꾸고 Python과의 일치를 테스트한다.
  일치 테스트 하니스: 같은 난수 입력 400 스텝을 Python InEKF와 헤더를 컴파일한 C++에 넣어 비교한다.
- 테스트는 **하드 바닥**에서 한다(카펫에서는 2.9 s 만에 넘어졌다. sim 지면은 강체다). 처음에는 행거에서 천천히 걷는다.

## 9. 실물 테스트 교훈

- 11-20: 서기 9.6 Hz 떨림은 정책 자체의 jitter였다(sim 지연을 늘려도 재현되지 않음). 매끄러움 보상으로 해결했다(stand_07).
- 11-20: 보행 무릎 bang-bang은 sim에도 똑같이 있었다. 정책이 기울기 제한 5.24 rad/s를 액추에이터처럼 쓴 것이고, 매끄러움 미세조정으로 해결했다(walk_10).
- 11-21: 넘어짐은 회전이 끝나고 약 2초 뒤 pitch/yaw 진동이 커지면서 났다. 원인은 직렬 컴플라이언스였다(walk_12에서 해결).
- 11-23: stage4 서기는 8.8 Hz 정책 루프 limit cycle을 보였다. sim은 이 떨림을 재현하지 못하고, 순위만 매기는 프록시가 5 Hz 이상 출력 파워다.
- 11-23: 제자리 yaw 회전은 InEKF bg_z가 원인이었다(11-24 고정). 보행을 시작한 첫 약 4 s 동안 추정 yaw가 잠깐 튀는 것은 아직 열린 문제다.
- 11-20: 71 s에 Unitree 모터 8개가 동시에 stale이 됐다. 버스 또는 전원 문제를 확인해야 한다. hold는 정상 동작했다.

## 10. 열린 문제와 다음 후보

- stage5 서기의 실물 떨림 여부. 떨리면 `agent_olaf_stand_stage3_480000.onnx`로 되돌리고, 매끄러움을 더 높인 stage6(acc 30 등, bg_z 고정)을 학습한다.
- walk_13의 gait_check와 eval_switch를 bg_z 고정 코드로 다시 돌리지 않았다.
- 전환 상태 모음을 walk_13으로 다시 만들기.
- 보행 시작 직후의 추정 yaw 과도 현상.
- 모델링하지 않은 것: 엔코더 영점 오차(sim FK는 참 관절각을 쓴다), 125 Hz 버스, 새 골반(`arti-h1-new-pelvis`) 질량 특성.
