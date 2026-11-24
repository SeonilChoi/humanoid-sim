"""Per-tag table of walk_replay.py results vs the real logs: python compare.py <tag> [<tag> ...]  (res_<tag>.json next to it)."""
import json, sys, numpy as np
tags = sys.argv[1:]; R = {t: json.load(open(f"{__file__.rsplit('/',1)[0]}/res_{t}.json")) for t in tags}
base = R[tags[0]]
keys = ["gyro_std_x", "gyro_std_y", "gyro_std_z", "wz_mean", "roll_std", "pitch_std", "pitch_mean", "err_hip_yaw", "err_hip_roll", "err_hip_pitch", "err_knee", "err_ankle_pitch", "err_ankle_roll",
        "tau_hip_yaw", "tau_hip_roll", "tau_hip_pitch", "tau_knee", "tau_ankle_pitch", "tau_ankle_roll", "sat_knee", "contact_both"] + [f"gb_{n}_{a}_{b}" for n in "xyz" for a, b in ((0, 3), (3, 6), (6, 12), (12, 25))]
for s in ("inplace", "fwd", "turn", "postturn"):
    print(f"--- {s}")
    print(f"{'':16}{'real':>8}" + "".join(f"{t[:9]:>10}" for t in tags))
    for k in keys:
        rv = [x[s][k] for x in base["real"] if k in x.get(s, {})]
        if not rv: continue
        row = f"{k:16}{np.mean(rv):8.3f}"
        for t in tags:
            sv = [x[s][k] for x in R[t]["sim"] if x and k in x.get(s, {})]
            row += f"{np.mean(sv):10.3f}" if sv else f"{'-':>10}"
        print(row)
for t in tags: print(t, "falls", [(f["log"], f["falls"], f["envs"]) for f in R[t]["falls"]])
