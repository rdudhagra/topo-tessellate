#!/usr/bin/env python3
"""
Base Generator

Creates a rectangular prism base (length × width × height in millimeters, Z-up)
and subtracts a provided cutout mesh on each side of the prism's bottom face.

Because the correct cutout orientation is unknown, this module generates a set
of variants by trying multiple axis-up interpretations for the cutout (Z-up,
Y-up, X-up) combined with yaw rotations around Z (0°, 90°, 180°, 270°).

Outputs are saved as OBJ files for visual inspection to decide which variant is
correct.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Tuple

import numpy as np

import meshlib.mrmeshpy as mr  # type: ignore
import meshlib.mrmeshnumpy as mn  # type: ignore

from .console import output


AxisUp = Literal["z", "y", "x"]


@dataclass
class Variant:
    axis_up: AxisUp
    yaw_deg: int

    @property
    def name(self) -> str:
        return f"up-{self.axis_up}_yaw-{self.yaw_deg}"


class BaseGenerator:
    def __init__(self) -> None:
        pass

    def create_base_box(self, length_mm: float, width_mm: float, height_mm: float = 20.0) -> mr.Mesh:
        """Create a rectangular prism aligned to axes with bottom at y=0 (Y-up).

        Coordinate system (Y-up): X=length, Z=width, Y=height.
        """
        L = float(length_mm)  # X
        W = float(width_mm)   # Z
        H = float(height_mm)  # Y (up)

        # Vertices (bottom y=0, top y=H)
        verts = np.array(
            [
                [0.0, 0.0, 0.0],   # 0: (x=0, y=0, z=0)
                [L, 0.0, 0.0],     # 1: (x=L, y=0, z=0)
                [L, 0.0, W],       # 2: (x=L, y=0, z=W)
                [0.0, 0.0, W],     # 3: (x=0, y=0, z=W)
                [0.0, H, 0.0],     # 4: (x=0, y=H, z=0)
                [L, H, 0.0],       # 5: (x=L, y=H, z=0)
                [L, H, W],         # 6: (x=L, y=H, z=W)
                [0.0, H, W],       # 7: (x=0, y=H, z=W)
            ],
            dtype=np.float32,
        )

        # Triangles (12) with outward-facing normals (right-hand rule)
        faces = np.array(
            [
                # Bottom (y=0) outward = -Y
                [0, 1, 2],
                [0, 2, 3],
                # Top (y=H) outward = +Y
                [4, 6, 5],
                [4, 7, 6],
                # +X side (x=L) outward = +X
                [1, 5, 6],
                [1, 6, 2],
                # -X side (x=0) outward = -X
                [0, 3, 7],
                [0, 7, 4],
                # +Z side (z=W) outward = +Z
                [2, 6, 7],
                [2, 7, 3],
                # -Z side (z=0) outward = -Z
                [0, 4, 5],
                [0, 5, 1],
            ],
            dtype=np.int32,
        )

        return mn.meshFromFacesVerts(faces, verts)

    def load_cutout_mesh(self, cutout_path: Path) -> mr.Mesh:
        """Load the cutout mesh from OBJ (or any supported) file."""
        if not cutout_path.exists():
            raise FileNotFoundError(f"Cutout not found: {cutout_path}")
        return mr.loadMesh(str(cutout_path))

    # ---------- math helpers ----------
    @staticmethod
    def _rot_x(rad: float) -> np.ndarray:
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

    @staticmethod
    def _rot_y(rad: float) -> np.ndarray:
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    @staticmethod
    def _rot_z(rad: float) -> np.ndarray:
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def _apply_rotation_inplace(verts: np.ndarray, R: np.ndarray) -> None:
        # verts: (N,3), R: (3,3) — in-place transform
        verts[:] = verts @ R.T

    @staticmethod
    def _apply_translation_inplace(verts: np.ndarray, t: Tuple[float, float, float]) -> None:
        verts[:] = verts + np.array(t, dtype=np.float32)[None, :]

    @staticmethod
    def _bbox_from_numpy(verts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        return vmin, vmax

    # ---------- positioning logic ----------
    def _orient_cutout_inplace(self, cutout: mr.Mesh, axis_up: AxisUp, yaw_deg: int, y_penetration_mm: float = 2.0, flip_cutout_degrees: float = 0.0) -> None:
        """Rotate cutout so that its 'up' axis maps to world Y (Y-up), then yaw around Y."""
        
        # Map source up axis to world Y
        if axis_up == "y":
            R_up = mr.Matrix3f()  # identity
        elif axis_up == "z":
            # Rotate -90° about X: Z -> Y
            R_up = mr.Matrix3f(
                mr.Vector3f(1, 0, 0),
                mr.Vector3f(0, 0, 1),
                mr.Vector3f(0, -1, 0)
            )
        elif axis_up == "x":
            # Rotate +90° about Z: X -> Y
            R_up = mr.Matrix3f(
                mr.Vector3f(0, 1, 0),
                mr.Vector3f(-1, 0, 0),
                mr.Vector3f(0, 0, 1)
            )
        else:
            raise ValueError(f"Unknown axis_up: {axis_up}")

        cutout.transform(mr.AffineXf3f(R_up, mr.Vector3f(0, 0, 0)))

        # Yaw around Y (vertical) - will be set per side in positioning
        yaw_rad = math.radians(float(yaw_deg))
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        Ry = mr.Matrix3f(
            mr.Vector3f(c, 0, s),
            mr.Vector3f(0, 1, 0),
            mr.Vector3f(-s, 0, c)
        )
        cutout.transform(mr.AffineXf3f(Ry, mr.Vector3f(0, 0, 0)))

        # Apply additional flip rotation around X if requested
        if abs(float(flip_cutout_degrees)) % 360.0 > 1e-6:
            flip_rad = math.radians(float(flip_cutout_degrees))
            c_flip, s_flip = math.cos(flip_rad), math.sin(flip_rad)
            R_flip = mr.Matrix3f(
                mr.Vector3f(1, 0, 0),
                mr.Vector3f(0, c_flip, -s_flip),
                mr.Vector3f(0, s_flip, c_flip)
            )
            cutout.transform(mr.AffineXf3f(R_flip, mr.Vector3f(0, 0, 0)))

        # Additional 180° rotation around Y-axis before cutting
        R_y180 = mr.Matrix3f(
            mr.Vector3f(-1, 0, 0),
            mr.Vector3f(0, 1, 0),
            mr.Vector3f(0, 0, -1)
        )
        cutout.transform(mr.AffineXf3f(R_y180, mr.Vector3f(0, 0, 0)))

        # Place bottom of cutout slightly below y=0 to ensure volumetric intersection (boolean robustness)
        bbox = cutout.computeBoundingBox()
        y_offset = -float(bbox.min.y) - float(y_penetration_mm)
        cutout.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(0, y_offset, 0)))

    def _position_cutout_on_side_inplace(
        self,
        cutout: mr.Mesh,
        side: Literal["left", "right", "front", "back"],
        length_mm: float,
        width_mm: float,
        embed_inside_mm: float = 15.0,
    ) -> None:
        """Position cutout on the specified side with appropriate rotation and embedding.
        
        The cutout will be rotated so the same edge always touches the base, then positioned
        with proper inward embedding by the cutout's own size.
        """
        
        # Apply additional rotation based on which side we're cutting
        if side == "left":
            # No additional rotation needed - cutout faces +X direction
            additional_yaw = 0
        elif side == "right":
            # Rotate 180° so cutout faces -X direction  
            additional_yaw = 180
        elif side == "front":
            # Rotate -90° so cutout faces +Z direction
            additional_yaw = -90
        elif side == "back":
            # Rotate 90° so cutout faces -Z direction
            additional_yaw = 90
        else:
            raise ValueError(f"Unknown side: {side}")
            
        if additional_yaw != 0:
            yaw_rad = math.radians(float(additional_yaw))
            c, s = math.cos(yaw_rad), math.sin(yaw_rad)
            Ry_additional = mr.Matrix3f(
                mr.Vector3f(c, 0, s),
                mr.Vector3f(0, 1, 0),
                mr.Vector3f(-s, 0, c)
            )
            cutout.transform(mr.AffineXf3f(Ry_additional, mr.Vector3f(0, 0, 0)))

        # Get updated bbox after rotation
        bbox = cutout.computeBoundingBox()
        center_x = 0.5 * (float(bbox.min.x) + float(bbox.max.x))  # X center
        center_z = 0.5 * (float(bbox.min.z) + float(bbox.max.z))  # Z center

        embed = float(embed_inside_mm)
        # Use small epsilon to ensure face gets cut without significantly changing cutout size
        face_cut_epsilon = -0.5  # mm - small extension to ensure boolean intersection
        
        if side == "left":
            # Position on left side (x=0), ensure cutout extends into base
            dx = (embed + face_cut_epsilon) - float(bbox.min.x)
            dz = (width_mm * 0.5) - center_z
        elif side == "right":
            # Position on right side (x=length), ensure cutout extends into base
            dx = (length_mm - embed - face_cut_epsilon) - float(bbox.max.x)
            dz = (width_mm * 0.5) - center_z
        elif side == "front":
            # Position on front side (z=0), ensure cutout extends into base
            dx = (length_mm * 0.5) - center_x
            dz = (embed + face_cut_epsilon) - float(bbox.min.z)
        elif side == "back":
            # Position on back side (z=width), ensure cutout extends into base
            dx = (length_mm * 0.5) - center_x
            dz = (width_mm - embed - face_cut_epsilon) - float(bbox.max.z)

        cutout.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(dx, 0.0, dz)))

    def _position_cutout_touch_edge_center_inplace(
        self,
        cutout: mr.Mesh,
        side: Literal["left", "right", "front", "back"],
        length_mm: float,
        width_mm: float,
    ) -> None:
        """Translate oriented cutout so it touches the given edge (on bottom face) and is centered on that edge, Y-up.

        - left:  move so min_x = 0, center_z = width/2
        - right: move so max_x = length, center_z = width/2
        - front: move so min_z = 0, center_x = length/2
        - back:  move so max_z = width, center_x = length/2
        """
        verts = mn.getNumpyVerts(cutout)
        vmin, vmax = self._bbox_from_numpy(verts)
        center_x = 0.5 * float(vmin[0] + vmax[0])
        center_z = 0.5 * float(vmin[2] + vmax[2])

        if side == "left":
            dx = -float(vmin[0])  # min_x -> 0
            dz = (float(width_mm) * 0.5) - center_z
        elif side == "right":
            dx = float(length_mm) - float(vmax[0])  # max_x -> length
            dz = (float(width_mm) * 0.5) - center_z
        elif side == "front":
            dx = (float(length_mm) * 0.5) - center_x
            dz = -float(vmin[2])  # min_z -> 0
        elif side == "back":
            dx = (float(length_mm) * 0.5) - center_x
            dz = float(width_mm) - float(vmax[2])  # max_z -> width
        else:
            raise ValueError(f"Unknown side: {side}")

        self._apply_translation_inplace(verts, (dx, 0.0, dz))

    @staticmethod
    def _scale_uniform_inplace(mesh: mr.Mesh, scale: float) -> None:
        if scale and scale != 1.0:
            verts = mn.getNumpyVerts(mesh)
            vmin = verts.min(axis=0)
            vmax = verts.max(axis=0)
            center = 0.5 * (vmin + vmax)
            verts[:] = (verts - center) * float(scale) + center

    def _apply_inward_offset_inplace(
        self,
        mesh: mr.Mesh,
        side: Literal["left", "right", "front", "back"],
        inward_mm: float,
    ) -> None:
        if not inward_mm:
            return
        dx = dz = 0.0
        if side == "left":
            dx = float(inward_mm)  # inside +X
        elif side == "right":
            dx = -float(inward_mm)  # inside -X
        elif side == "front":
            dz = float(inward_mm)  # inside +Z
        elif side == "back":
            dz = -float(inward_mm)  # inside -Z
        verts = mn.getNumpyVerts(mesh)
        self._apply_translation_inplace(verts, (dx, 0.0, dz))

    def _repair_cutout_inplace(self, cutout: mr.Mesh) -> None:
        """Repair the cutout mesh to be watertight and robust for boolean."""
        try:
            holes = cutout.topology.findHoleRepresentiveEdges()
            if not holes.empty():
                mr.fillHoles(cutout, holes)
        except Exception:
            pass
        try:
            bbox = cutout.computeBoundingBox()
            diag = float(bbox.diagonal()) or 1.0
            mr.uniteCloseVertices(cutout, float(diag) * 1e-6, True)
            params = mr.FixMeshDegeneraciesParams()
            mr.fixMeshDegeneracies(cutout, params)
        except Exception:
            pass

    # ---------- boolean helpers ----------
    def _boolean_difference(self, base: mr.Mesh, cutter: mr.Mesh) -> mr.Mesh:
        """Boolean difference base - cutter using MRMesh DifferenceAB only.

        Returns a new Mesh instance, or raises if boolean fails or produces no change.
        """
        op = mr.BooleanOperation.DifferenceAB
        res = mr.boolean(base, cutter, op, mr.AffineXf3f())
        if not (hasattr(res, 'valid') and callable(res.valid) and res.valid() and getattr(res, 'mesh', None) is not None):
            raise RuntimeError("Boolean difference failed")
        mesh = res.mesh
        # simple no-change check: face count different or bbox changed
        af = base.topology.numValidFaces(); bf = mesh.topology.numValidFaces()
        if bf == af:
            ab = base.computeBoundingBox(); bb = mesh.computeBoundingBox()
            eps = 1e-6
            if (
                abs(float(ab.min.x) - float(bb.min.x)) < eps and
                abs(float(ab.min.y) - float(bb.min.y)) < eps and
                abs(float(ab.min.z) - float(bb.min.z)) < eps and
                abs(float(ab.max.x) - float(bb.max.x)) < eps and
                abs(float(ab.max.y) - float(bb.max.y)) < eps and
                abs(float(ab.max.z) - float(bb.max.z)) < eps
            ):
                raise RuntimeError("Boolean produced no change")
        return mesh

    # ---------- main entry ----------
    def generate_variants(
        self,
        length_mm: float,
        width_mm: float,
        height_mm: float = 20.0,
        cutout_path: Path | None = None,
        output_dir: Path | None = None,
        yaw_deg_options: Iterable[int] = (0, 90, 180, 270),
        embed_mm: float = 0.0,
        cutout_up_axis: AxisUp = "z",
        flip_model_degrees: float = 180.0,
        flush_epsilon_mm: float = 0.05,
    ) -> List[Tuple[Variant, Path]]:
        """Generate Y-up variants: cutout centered on each side plane with given yaw.

        - X = length, Y = height (up), Z = width
        - Places cutout at each side (left/right/front/back) centered, with embed_mm inward
        - Uses pure geometric boolean (DifferenceAB)
        """
        root = Path(__file__).resolve().parents[1]
        if cutout_path is None:
            # prefer STL if present
            stl = root / "joint_cutout.stl"
            cutout_path = stl if stl.exists() else (root / "joint_cutout.obj")
        if output_dir is None:
            output_dir = root / "analysis" / "base_variants"
        output_dir.mkdir(parents=True, exist_ok=True)

        output.header("Generating base with joint cutout variants (Y-up)")
        output.info(f"Base size: {length_mm} × {width_mm} × {height_mm} mm")
        output.info(f"Cutout: {cutout_path}")
        output.info(f"Output dir: {output_dir}")

        results: List[Tuple[Variant, Path]] = []
        # Primary simple flow only: embed on each side, try yaws
        edges = ("left", "right", "front", "back")
        for side in edges:
            for yaw in yaw_deg_options:
                variant = Variant(axis_up=cutout_up_axis, yaw_deg=int(yaw))
                output.subheader(f"{side} edge, yaw {yaw} (cutout flip {flip_model_degrees}°)")
                base = self.create_base_box(length_mm, width_mm, height_mm)
                cut = self.load_cutout_mesh(cutout_path)
                # Repair and orient
                self._repair_cutout_inplace(cut)
                
                self._orient_cutout_inplace(cut, axis_up=cutout_up_axis, yaw_deg=int(yaw), flip_cutout_degrees=float(flip_model_degrees))
                # Place flush (faces aligned) at side center; use tiny inward epsilon to ensure boolean overlap
                embed_to_use = float(embed_mm) if float(embed_mm) > 0 else float(flush_epsilon_mm)
                self._position_cutout_on_side_inplace(cut, side, length_mm, width_mm, embed_inside_mm=embed_to_use)
                base = self._boolean_difference(base, cut)
                out_path = output_dir / f"base_yup_h{int(height_mm)}_embed{int(embed_mm)}_{side}_yaw{int(yaw)}.obj"
                mr.saveMesh(base, str(out_path))
                results.append((variant, out_path))
                output.file_saved(str(out_path), "mesh")

        output.success(f"Generated {len(results)} variant meshes")
        return results


def _parse_args(argv: List[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate base prism with joint cutout variants")
    parser.add_argument("--length-mm", type=float, required=True, help="Base length in millimeters (X)")
    parser.add_argument("--width-mm", type=float, required=True, help="Base width in millimeters (Y)")
    parser.add_argument("--height-mm", type=float, default=20.0, help="Base height in millimeters (Z, default 20)")
    parser.add_argument(
        "--cutout",
        type=str,
        default=None,
        help="Path to cutout mesh (OBJ). Defaults to repository root joint_cutout.obj",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to write variant OBJ files (default: analysis/base_variants)",
    )
    parser.add_argument(
        "--axis-up",
        type=str,
        default="z,y,x",
        help="Comma-separated axis-up options to try from {z,y,x} (default: z,y,x)",
    )
    parser.add_argument(
        "--yaws",
        type=str,
        default="0,90,180,270",
        help="Comma-separated yaw degrees to try (default: 0,90,180,270)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    axis_up_opts: List[AxisUp] = [a.strip() for a in str(args.axis_up).split(",") if a.strip()]  # type: ignore[assignment]
    yaw_opts: List[int] = [int(x.strip()) for x in str(args.yaws).split(",") if x.strip()]

    gen = BaseGenerator()
    gen.generate_variants(
        float(args.length_mm),
        float(args.width_mm),
        float(args.height_mm),
        Path(args.cutout) if args.cutout else None,
        Path(args.outdir) if args.outdir else None,
        axis_up_options=axis_up_opts,  # type: ignore[arg-type]
        yaw_deg_options=yaw_opts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


