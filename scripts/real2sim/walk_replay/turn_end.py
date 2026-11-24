"""Stride features aligned to the end of each turn command (g2 back to ~0), real vs sim replay (RAW npz)."""
import sys, numpy as np
from fall_analysis import *
def aligned(series, feat):
    rows = []
    for s in series:
        cut = s["fall"]; g2 = s["g"][:, 2]; T = len(s["ph"]) if cut is None else cut
        st = strides(s["ph"], s["yaw"], s["P"], s["R"], s["gz"], s["gx"], s["hyL"], s["hyR"], s["ahyL"], s["ahyR"], s["aknL"], s["aknR"], s.get("fyL"), s.get("fyR"), cut)
        if not st: continue
        starts = np.array([a for a, _ in st])
        turn = np.abs(g2) > 0.15
        ends = np.flatnonzero(turn[:-1] & ~turn[1:]) + 1
        for e in ends:
            if e >= T: continue
            k0 = np.searchsorted(starts, e)
            row = [st[k][1][feat] if 0 <= k < len(st) else np.nan for k in range(k0 - 2, k0 + 5)]
            fell = cut is not None and cut - e < 6 / 0.02
            rows.append((int(e), fell, row))
    return rows
feat = sys.argv[1]
rs = [s for lg in ["095329", "095641", "100031"] for s in real_series(lg)]
for name, ser in [("real", rs)] + [(p.split("res_")[-1][:-8], [s for i in range(len({k.split('_')[0] for k in np.load(p).files})) for s in sim_series(np.load(p, allow_pickle=True), i)]) for p in sys.argv[2:]]:
    R = aligned(ser, feat)
    M = np.array([r[2] for r in R], float); F = np.array([r[1] for r in R])
    print(f"{name}: {len(R)} turn ends, fall within 6 s: {F.sum()} | {feat} per stride (-2..+4 rel. to turn end), mean / p90")
    print("   mean", " ".join(f"{x:6.2f}" for x in np.nanmean(M, 0)), "  p90", " ".join(f"{x:6.2f}" for x in np.nanpercentile(M, 90, 0)))
    if name == "real":
        for e, f, r in R: print("   ", e, f, " ".join(f"{x:6.2f}" for x in r))
