"""Olaf-style (arXiv 2512.16705) walking and standing policies for the ARTI-H1 knee-bent robot."""

import gymnasium as gym

from . import agents

_KW = {"default_agent": "skrl", "skrl_cfg_entry_point": f"{agents.__name__}:skrl_olaf_ppo_cfg.yaml"}

for _name, _cfg in (
    ("HumanoidSim-Olaf-Walk-v0", "ArtiH1OlafWalkEnvCfg"),
    ("HumanoidSim-Olaf-Walk-Play-v0", "ArtiH1OlafWalkEnvCfg_PLAY"),
    ("HumanoidSim-Olaf-Stand-v0", "ArtiH1OlafStandEnvCfg"),
    ("HumanoidSim-Olaf-Stand-Play-v0", "ArtiH1OlafStandEnvCfg_PLAY"),
):
    gym.register(
        id=_name,
        entry_point=f"{__name__}.olaf_env:ArtiH1OlafEnv",
        disable_env_checker=True,
        kwargs={**_KW, "env_cfg_entry_point": f"{__name__}.olaf_env_cfg:{_cfg}"},
    )
