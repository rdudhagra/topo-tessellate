#!/usr/bin/env python3
"""
Base Generator

Creates a rectangular prism base (length × width × height in millimeters, Y-up)
and selectively subtracts a provided cutout mesh on specified sides of the prism.

The cutout orientation and positioning parameters have been optimized through testing
and are hardcoded for reliable results.

Outputs a single OBJ file with the requested cutouts applied.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

import meshlib.mrmeshpy as mr  # type: ignore
import meshlib.mrmeshnumpy as mn  # type: ignore

from .console import output


AxisUp = Literal["z", "y", "x"]


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

    # ---------- positioning logic ----------
    def _orient_cutout_inplace(self, cutout: mr.Mesh, axis_up: AxisUp, yaw_deg: int, flip_cutout_degrees: float) -> None:
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

        # Place bottom of cutout at y=0
        bbox = cutout.computeBoundingBox()
        y_offset = -float(bbox.min.y)
        cutout.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(0, y_offset, 0)))

    def _position_cutout_on_side_inplace(
        self,
        cutout: mr.Mesh,
        side: Literal["left", "right", "front", "back"],
        length_mm: float,
        width_mm: float,
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

        # Hardcoded optimal values determined through testing
        face_cut_epsilon = -0.5  # mm - optimal epsilon for boolean intersection
        
        if side == "left":
            # Position on left side (x=0), ensure cutout extends into base
            dx = face_cut_epsilon - float(bbox.min.x)
            dz = (width_mm * 0.5) - center_z
        elif side == "right":
            # Position on right side (x=length), ensure cutout extends into base
            dx = (length_mm - face_cut_epsilon) - float(bbox.max.x)
            dz = (width_mm * 0.5) - center_z
        elif side == "front":
            # Position on front side (z=0), ensure cutout extends into base
            dx = (length_mm * 0.5) - center_x
            dz = face_cut_epsilon - float(bbox.min.z)
        elif side == "back":
            # Position on back side (z=width), ensure cutout extends into base
            dx = (length_mm * 0.5) - center_x
            dz = (width_mm - face_cut_epsilon) - float(bbox.max.z)

        cutout.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(dx, 0.0, dz)))









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
    def generate_base_with_cutouts(
        self,
        length_mm: float,
        width_mm: float,
        height_mm: float = 20.0,
        cutout_left: bool = False,
        cutout_right: bool = False,
        cutout_front: bool = False,
        cutout_back: bool = False,
        cutout_path: Path | None = None,
    ) -> mr.Mesh:
        """Generate a base with selective joint cutouts on specified sides.

        Args:
            length_mm: Base length in millimeters (X-axis)
            width_mm: Base width in millimeters (Z-axis)
            height_mm: Base height in millimeters (Y-axis, default 20)
            cutout_left: Whether to add cutout on left side (x=0)
            cutout_right: Whether to add cutout on right side (x=length)
            cutout_front: Whether to add cutout on front side (z=0)
            cutout_back: Whether to add cutout on back side (z=width)
            cutout_path: Path to cutout STL file (defaults to joint_cutout.stl)

        Returns:
            mr.Mesh: The generated mesh with requested cutouts applied
        """
        root = Path(__file__).resolve().parents[1]
        if cutout_path is None:
            cutout_path = root / "joint_cutout.stl"

        output.header("Generating base with selective joint cutouts (Y-up)")
        output.info(f"Base size: {length_mm} × {width_mm} × {height_mm} mm")
        output.info(f"Cutout: {cutout_path}")
        
        # Create base box
        base = self.create_base_box(length_mm, width_mm, height_mm)
        
        # Hardcoded optimal parameters determined through testing
        cutout_up_axis = "z"  # Source cutout up axis
        yaw_deg = 0  # Working yaw rotation
        flip_cutout_degrees = 180.0  # X-axis flip
        
        # Apply cutouts for each requested side
        sides_to_cut = []
        if cutout_left:
            sides_to_cut.append("left")
        if cutout_right:
            sides_to_cut.append("right")
        if cutout_front:
            sides_to_cut.append("front")
        if cutout_back:
            sides_to_cut.append("back")
        
        if sides_to_cut:
            output.info(f"Applying cutouts to sides: {', '.join(sides_to_cut)}")
            
            for side in sides_to_cut:
                output.subheader(f"Cutting {side} side")
                
                # Load and prepare cutout mesh
                cut = self.load_cutout_mesh(cutout_path)
                self._orient_cutout_inplace(cut, cutout_up_axis, yaw_deg, flip_cutout_degrees)
                self._position_cutout_on_side_inplace(cut, side, length_mm, width_mm)
                
                # Perform boolean subtraction
                base = self._boolean_difference(base, cut)
                output.info(f"✓ {side} cutout applied")
        else:
            output.info("No cutouts requested - generating plain base")
        
        output.success("Base generation complete")
        return base


if __name__ == "__main__":
    # Example usage - run from project root with: python -m terrain_generator.basegenerator
    gen = BaseGenerator()
    base_mesh = gen.generate_base_with_cutouts(
        length_mm=250,
        width_mm=175,
        height_mm=20,
        cutout_left=True,
        cutout_right=True,
        cutout_front=True,
        cutout_back=True,
    )
    
    # Caller handles saving the mesh
    output_path = Path("analysis/example_base.obj")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mr.saveMesh(base_mesh, str(output_path))
    print(f"Generated base mesh and saved to: {output_path}")


