"""Togepi exterior shell (assets/meshes/arti_h1/togepi_exterior.STL) fixed to the pelvis.

The STL is in mm, already in the pelvis frame (11-24 version): the axis of its bottom opening (a circle at z = 69 mm,
inner radius 135 mm, wall about 8 mm) is at x = y = 0, and it is mirror-symmetric about y = 0. Mounted 30 mm above the
pelvis origin (fixed joint at z = +0.03 m, user 11-24), so the skirt ends 99 mm above the pelvis origin, i.e. about
7 cm above the ground in the standing pose.
URDF with the shell: assets/models/arti_h1/arti_h1_togepi.urdf (same placement).

Simulation: the training USD is not changed. `build_exterior_robot_usd` writes a wrapper asset next to the robot USD
(models/arti_h1/arti_h1/arti_h1_togepi.usda: references arti_h1.usd and adds the shell mesh under /arti_h1/pelvis), used
as the robot spawn for videos (`scripts/skrl/play.py --exterior`). The shell is visual only (no collision, no mass), so
the policy runs as trained; contact is judged geometrically (scripts/eval/exterior_clearance.py). Shell prims added to
the stage after the scene is cloned do not show in rendered videos, hence the asset-level wrapper.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from humanoid_sim.robots import ISAACLAB_ASSETS_DIR

EXTERIOR_STL = os.path.join(ISAACLAB_ASSETS_DIR, "meshes", "arti_h1", "togepi_exterior.STL")
# STL (mm) -> pelvis frame (m): p_pelvis = 0.001 * p_stl + EXTERIOR_OFFSET
EXTERIOR_OFFSET = (0.0, 0.0, 0.03)  # 3 cm above the pelvis origin
EXTERIOR_RIM_Z = 0.069 + EXTERIOR_OFFSET[2]  # m, bottom edge of the skirt in the pelvis frame


def read_binary_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V, 3), faces (F, 3)) of a binary STL, vertices deduplicated, in the file's units."""
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    tri = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"]
    verts, inv = np.unique(tri.reshape(-1, 3).astype(np.float64), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


def exterior_mesh_pelvis() -> tuple[np.ndarray, np.ndarray]:
    """Shell vertices (m, pelvis frame) and faces."""
    v, f = read_binary_stl(EXTERIOR_STL)
    return v * 1e-3 + np.asarray(EXTERIOR_OFFSET), f


ROBOT_USD_DIR = os.path.join(ISAACLAB_ASSETS_DIR, "models", "arti_h1", "arti_h1")


CUTS = {  # cutaway: keep the triangles whose centroid is on the far side from the camera (pelvis frame)
    "side": lambda c: c[:, 1] >= 0.0,   # side camera at -y: drop the y < 0 half
    "front": lambda c: c[:, 0] <= 0.0,  # front camera at +x: drop the x > 0 half
}


def build_exterior_robot_usd(color=(0.98, 0.93, 0.82), cut: str | None = None) -> str:
    """Write togepi_exterior[_cut].usd (the shell mesh in the pelvis frame, metres) and the wrapper
    arti_h1_togepi[_cut].usda next to the robot USD; returns the wrapper path (use it as the robot spawn usd_path).
    cut = "side" / "front" removes the half of the shell facing that camera so the legs inside are visible.
    Regenerated on every call, so it follows EXTERIOR_OFFSET and the STL."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    v, f = exterior_mesh_pelvis()
    if cut:
        f = f[CUTS[cut](v[f].mean(1))]
    sfx = f"_{cut}" if cut else ""
    mesh_path = os.path.join(ROBOT_USD_DIR, f"togepi_exterior{sfx}.usd")
    st = Usd.Stage.CreateNew(mesh_path) if not os.path.exists(mesh_path) else Usd.Stage.Open(mesh_path)
    st.GetRootLayer().Clear()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/togepi_exterior")
    st.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(st, "/togepi_exterior/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)  # the inside of the wall is seen through the cut
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mat = UsdShade.Material.Define(st, "/togepi_exterior/Looks/Shell")
    sh = UsdShade.Shader.Define(st, "/togepi_exterior/Looks/Shell/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    st.GetRootLayer().Save()
    wrapper = os.path.join(ROBOT_USD_DIR, f"arti_h1_togepi{sfx}.usda")
    open(wrapper, "w").write(
        '#usda 1.0\n(\n    defaultPrim = "arti_h1"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "arti_h1" (\n    prepend references = @./arti_h1.usd@\n)\n{\n'
        '    over "pelvis"\n    {\n'
        '        def Xform "togepi_exterior" (\n            prepend references = @./togepi_exterior' + sfx + '.usd@\n        )\n'
        '        {\n        }\n    }\n}\n')
    return wrapper


LEG_LINKS = [f"{s}_{l}" for s in ("left", "right") for l in ("hip_rotation", "hip", "thigh", "shin", "foot_rotation", "foot_link")]


class ShellClearance:
    """Leg-to-shell clearance from leg link poses in the pelvis frame (used by scripts/eval/exterior_clearance.py and
    the per-frame overlay of the contact video).

      * leg-link surface points (URDF visual meshes: 600 even surface samples + convex-hull vertices per link) vs the
        shell: 3-D distance to a dense surface sample,
      * above the skirt rim the shell wall is tabulated as R_in(z, angle) .. R_out(z, angle) from horizontal mesh
        sections. A link is in contact (negative clearance) when one of its points lies inside the wall band, or when
        it has points both inside (r < R_in) and outside (r > R_out) above the rim, i.e. it passes through the wall.
        Points outside the shell below its bulge are free air (the egg widens above the rim).
    """

    def __init__(self, links=LEG_LINKS):
        import xml.etree.ElementTree as ET

        import trimesh
        from scipy.spatial import cKDTree

        self.links = list(links)
        ev, ef = exterior_mesh_pelvis()
        shell = trimesh.Trimesh(ev, ef, process=False)
        self.tree = cKDTree(np.vstack([ev, trimesh.sample.sample_surface_even(shell, 400_000, seed=0)[0]]))
        # wall band: smallest / largest radius of the loops around the axis along each ray
        self.zs = np.arange(EXTERIOR_RIM_Z, 0.40, 0.002)
        self.ang = np.linspace(-np.pi, np.pi, 181)
        self.R_in = np.full((len(self.zs), len(self.ang)), np.inf)
        self.R_out = np.full((len(self.zs), len(self.ang)), -np.inf)
        for i, z in enumerate(self.zs):
            s = shell.section(plane_origin=[0, 0, z + 1e-4], plane_normal=[0, 0, 1])
            if s is None:
                continue
            for loop in s.discrete:
                th = np.arctan2(loop[:, 1], loop[:, 0])
                r = np.hypot(loop[:, 0], loop[:, 1])
                if np.ptp(np.unwrap(th)) < 1.9 * np.pi:  # does not go around the axis (side bodies)
                    continue
                o = np.argsort(th)
                ri = np.interp(self.ang, th[o], r[o], period=2 * np.pi)
                self.R_in[i], self.R_out[i] = np.minimum(self.R_in[i], ri), np.maximum(self.R_out[i], ri)
        urdf = os.path.join(ISAACLAB_ASSETS_DIR, "models", "arti_h1", "arti_h1.urdf")
        self.link_pts = {}
        for l in ET.parse(urdf).getroot().findall("link"):
            if l.get("name") not in self.links:
                continue
            v = l.find("visual")
            m = trimesh.load(os.path.join(os.path.dirname(urdf), v.find("geometry/mesh").get("filename")))
            o = v.find("origin")
            xyz = [float(x) for x in o.get("xyz").split()] if o is not None else [0.0] * 3
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
            m.apply_scale(0.001)
            m.apply_transform(trimesh.transformations.compose_matrix(angles=rpy, translate=xyz))
            # on-surface points only (bounding-box corners are not on the link and gave false contacts)
            self.link_pts[l.get("name")] = np.vstack([trimesh.sample.sample_surface_even(m, 600, seed=1)[0],
                                                      m.convex_hull.vertices])

    def __call__(self, R_rel: np.ndarray, t_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """R_rel (..., L, 3, 3), t_rel (..., L, 3): link poses in the pelvis frame (link order = self.links).
        Returns clearance (..., L) in m (negative = contact) and through (..., L): contact by passing through the wall
        without a sampled point inside it."""
        clear = np.full(R_rel.shape[:-2], np.inf, dtype=np.float32)
        through = np.zeros(R_rel.shape[:-2], dtype=bool)
        for j, name in enumerate(self.links):
            p = np.einsum("...ab,pb->...pa", R_rel[..., j, :, :], self.link_pts[name]) + t_rel[..., j, None, :]
            d = self.tree.query(p.reshape(-1, 3), workers=-1)[0].reshape(p.shape[:-1])
            z = p[..., 2]
            above = z > EXTERIOR_RIM_Z
            zi = np.clip(np.round((z - self.zs[0]) / 0.002).astype(int), 0, len(self.zs) - 1)
            ai = np.clip(np.round((np.arctan2(p[..., 1], p[..., 0]) + np.pi) / (2 * np.pi) * (len(self.ang) - 1)).astype(int),
                         0, len(self.ang) - 1)
            r = np.hypot(p[..., 0], p[..., 1])
            wall = above & (r >= self.R_in[zi, ai]) & (r <= self.R_out[zi, ai])
            straddle = (above & (r < self.R_in[zi, ai])).any(-1) & (above & (r > self.R_out[zi, ai])).any(-1)
            th = straddle & ~wall.any(-1)
            c = np.where(wall.any(-1), -np.where(wall, d, 0.0).max(-1), d.min(-1))
            clear[..., j] = np.where(th, -d.min(-1) - 1e-4, c)
            through[..., j] = th
        return clear, through
