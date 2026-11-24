"""Installation script for the 'humanoid_sim' python package."""

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="humanoid_sim",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=["psutil"],
    # PlaCo (utils/motion_generator, reference generation only): uv pip install placo
    extras_require={"motion": ["placo"]},
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "humanoid_sim": [
            "assets/meshes/arti_h1/*.STL",
            "assets/models/arti_h1/*.urdf",
            "assets/models/arti_h1/arti_h1/*.usd",
            "assets/models/arti_h1/arti_h1/configuration/*.usd",
            "assets/reference/*.npz",
            "assets/thermal/*.json",
            "utils/motion_generator/config/*/*.json",
            "utils/motion_generator/robots/*/*.urdf",
            "utils/motion_generator/robots/*/*.json",
            "utils/motion_generator/robots/*/meshes/*.STL",
        ]
    },
    # `isaaclab train|play|list_envs` import every package registered under this group
    entry_points={"isaaclab.tasks": ["humanoid_sim = humanoid_sim.tasks"]},
    python_requires=">=3.12",
    zip_safe=False,
)
