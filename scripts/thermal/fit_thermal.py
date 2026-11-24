"""beta per joint type with alpha fixed (time constant prior) and T_amb_s = session start temperature: the leg motors heat
only 0-6 C per session, so alpha / T_amb are not identifiable from these logs (thermal_fit2 gave negative / 200 C)."""
import glob, json, sys, numpy as np, pandas as pd
J = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle_pitch', 'ankle_roll']
alpha = float(sys.argv[3])
data = [pd.read_csv(f, usecols=lambda c: c == 't' or c.startswith(('temp_', 'tau_'))) for f in sorted(glob.glob(sys.argv[1] + '/*.csv'))]
data = [d for d in data if d.t.iloc[-1] - d.t.iloc[0] > 30]
res = {}
for j in J:
    # rollout of the heat input alone: x(t) = T0 + int e^{-a(t-s)} b tau^2 ds -> linear in b; fit b on all samples
    G, Y, T0s = [], [], []
    for d in data:
        t = d.t.values; dt = np.diff(t, prepend=t[0])
        for s in ('left', 'right'):
            T, tau2 = d[f'temp_{s}_{j}'].values.astype(float), d[f'tau_{s}_{j}'].values ** 2
            T0 = T[:60].mean(); g = np.zeros_like(T)
            for k in range(1, len(T)): g[k] = g[k-1] + dt[k] * (-alpha * g[k-1] + tau2[k-1])
            G.append(g); Y.append(T - T0)
    g, yv = np.concatenate(G), np.concatenate(Y)
    b = max(0.0, float(g @ yv / (g @ g)))
    mae = np.mean([np.abs(b * gi - yi).mean() for gi, yi in zip(G, Y)])
    mae0 = np.mean([np.abs(yi).mean() for yi in Y])
    pk = max(np.max(b * gi) for gi in G)
    res[j] = {'alpha': alpha, 'beta': b, 'rollout_mae': float(mae), 'constant_mae': float(mae0)}
    print(f'{j:12s} beta {b:.5f}  MAE {mae:.2f} C (constant-T baseline {mae0:.2f})  max predicted rise {pk:.1f} C')
json.dump(res, open(sys.argv[2], 'w'), indent=1)
