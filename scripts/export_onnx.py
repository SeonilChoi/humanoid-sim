"""Export an Olaf walk / stand checkpoint to ONNX for humanoid_controller (branch arti-h1-olaf, policy.observation olaf).

    python scripts/export_onnx.py --checkpoint logs/skrl/olaf_walk/<run>/checkpoints/agent_N.pt --out agent_olaf_walk_N.onnx
    python scripts/export_onnx.py ... --keys_out olaf_walk_keys.yaml   # walk reference keys for the controller

Reads <run>/params/env.yaml (the training cfg). ONNX: input "obs" [1, 80 walk / 79 stand] (ArtiH1OlafEnv policy
observation), outputs "actions" (clamped to [-1, 1], fed back as a(t)) and "q_des" (rad, policy joint order:
clamp(default + scale * actions, real joint limits, hip_roll +-hip_roll_limit)), exactly as the env's
_pre_physics_step before the controller-side low-pass / rate limit. The script checks ONNX == torch on random inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda l, s, n: None)


class Policy(nn.Module):
    def __init__(self, sd, pre, default, neg, pos, lower, upper):
        super().__init__()
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net_container.") and k.endswith(".weight")})
        layers = []
        for j, i in enumerate(idx):
            w, b = sd[f"net_container.{i}.weight"], sd[f"net_container.{i}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if j < len(idx) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", pre["running_mean"].float())
        self.register_buffer("std", torch.sqrt(pre["running_variance"].float()) + 1e-8)
        for name, v in (("default", default), ("neg", neg), ("pos", pos), ("lower", lower), ("upper", upper)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))

    def forward(self, obs):
        x = torch.clamp((obs - self.mean) / self.std, -5.0, 5.0)  # skrl RunningStandardScaler (clip 5)
        a = torch.clamp(self.net(x), -1.0, 1.0)
        q = self.default + torch.where(a >= 0.0, self.pos, self.neg) * a
        return a, torch.clamp(q, self.lower, self.upper)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keys_out", default="")
    a = p.parse_args()
    ckpt = Path(a.checkpoint)
    run = ckpt.parent.parent
    cfg = yaml.load(open(run / "params" / "env.yaml"), Loader=_Loader)
    names = list(cfg["policy_joint_names"])
    jp = cfg["scene"]["robot"]["init_state"]["joint_pos"]
    default = [float(jp[n]) for n in names]
    neg = [float(cfg["action_negative_scale"][n]) for n in names]
    pos = [float(cfg["action_positive_scale"][n]) for n in names]
    tab = cfg["real_joint_limit_table"]
    lower = [float(tab[n.split("_", 1)[1]][0]) for n in names]
    upper = [float(tab[n.split("_", 1)[1]][1]) for n in names]
    hr = float(cfg["hip_roll_limit"])
    for i, n in enumerate(names):
        if n.endswith("hip_roll") and hr > 0.0:
            lower[i], upper[i] = -hr, hr
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Policy(c["policy"], c["observation_preprocessor"], default, neg, pos, lower, upper).eval()
    n_obs = int(cfg["observation_space"])
    dummy = torch.zeros(1, n_obs)
    torch.onnx.export(model, (dummy,), a.out, input_names=["obs"], output_names=["actions", "q_des"], opset_version=17,
                      dynamo=False)
    import onnx

    m = onnx.load(a.out)  # single file (no external data), as the controller loads it
    onnx.save(m, a.out, save_as_external_data=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(a.out)
    x = (torch.randn(256, n_obs) * model.std + model.mean).float()
    err = 0.0
    for i in range(0, 256, 1):
        ao, qo = sess.run(None, {"obs": x[i:i + 1].numpy()})
        at, qt = model(x[i:i + 1])
        err = max(err, float(np.abs(ao - at.detach().numpy()).max()), float(np.abs(qo - qt.detach().numpy()).max()))
    print(f"{a.out}: mode {cfg['mode']}, obs {n_obs}, max |onnx - torch| {err:.2e}")
    if a.keys_out:
        ref = np.load(cfg["reference_file"])
        keys = ref["walk_keys"].astype(float).round(6).tolist()
        with open(a.keys_out, "w") as f:
            f.write("# humanoid_sim reference/build_reference.py walk clip keys (path velocity vx m/s, vy m/s, wz rad/s);\n"
                    "# the controller projects the stick command onto them (ArtiH1OlafEnv._walk_blend)\n")
            yaml.safe_dump({"walk_keys": keys}, f, default_flow_style=None, width=200)
        print(f"{a.keys_out}: {len(keys)} keys")


if __name__ == "__main__":
    main()
