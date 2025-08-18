#!/usr/bin/env python3
"""
Base Generator

Creates a rectangular prism base (length × width × height in millimeters, Z-up)
and selectively subtracts a provided cutout mesh on specified sides of the prism.

The cutout orientation and positioning parameters have been optimized through testing
and are hardcoded for reliable results.

Coordinate system: X=length, Y=width, Z=height (Z-up).
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
        """Create a rectangular prism aligned to axes with bottom at z=0 (Z-up).

        Coordinate system (Z-up): X=length, Y=width, Z=height.
        """
        L = float(length_mm)  # X
        W = float(width_mm)   # Y
        H = float(height_mm)  # Z (up)

        # Vertices (bottom z=0, top z=H)
        verts = np.array(
            [
                [0.0, 0.0, 0.0],   # 0: (x=0, y=0, z=0)
                [L, 0.0, 0.0],     # 1: (x=L, y=0, z=0)
                [L, W, 0.0],       # 2: (x=L, y=W, z=0)
                [0.0, W, 0.0],     # 3: (x=0, y=W, z=0)
                [0.0, 0.0, H],     # 4: (x=0, y=0, z=H)
                [L, 0.0, H],       # 5: (x=L, y=0, z=H)
                [L, W, H],         # 6: (x=L, y=W, z=H)
                [0.0, W, H],       # 7: (x=0, y=W, z=H)
            ],
            dtype=np.float32,
        )

        # Triangles (12) with outward-facing normals (right-hand rule)
        faces = np.array(
            [
                # Bottom (z=0) outward = -Z
                [0, 2, 1],
                [0, 3, 2],
                # Top (z=H) outward = +Z
                [4, 5, 6],
                [4, 6, 7],
                # +X side (x=L) outward = +X
                [1, 2, 6],
                [1, 6, 5],
                # -X side (x=0) outward = -X
                [0, 4, 7],
                [0, 7, 3],
                # +Y side (y=W) outward = +Y
                [2, 3, 7],
                [2, 7, 6],
                # -Y side (y=0) outward = -Y
                [0, 1, 5],
                [0, 5, 4],
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
        """Rotate cutout so that its 'up' axis maps to world Z (Z-up), then yaw around Z."""
        
        # Map source up axis to world Z
        if axis_up == "z":
            R_up = mr.Matrix3f()  # identity - Z already maps to Z
        elif axis_up == "y":
            # Rotate +90° about X: Y -> Z
            R_up = mr.Matrix3f(
                mr.Vector3f(1, 0, 0),
                mr.Vector3f(0, 0, -1),
                mr.Vector3f(0, 1, 0)
            )
        elif axis_up == "x":
            # Rotate -90° about Y: X -> Z
            R_up = mr.Matrix3f(
                mr.Vector3f(0, 0, 1),
                mr.Vector3f(0, 1, 0),
                mr.Vector3f(-1, 0, 0)
            )
        else:
            raise ValueError(f"Unknown axis_up: {axis_up}")

        cutout.transform(mr.AffineXf3f(R_up, mr.Vector3f(0, 0, 0)))

        # Yaw around Z (vertical) - will be set per side in positioning
        yaw_rad = math.radians(float(yaw_deg))
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        Rz = mr.Matrix3f(
            mr.Vector3f(c, -s, 0),
            mr.Vector3f(s, c, 0),
            mr.Vector3f(0, 0, 1)
        )
        cutout.transform(mr.AffineXf3f(Rz, mr.Vector3f(0, 0, 0)))

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

        # Additional 180° rotation around Z-axis before cutting (Z-up coordinate system)
        R_z180 = mr.Matrix3f(
            mr.Vector3f(-1, 0, 0),
            mr.Vector3f(0, -1, 0),
            mr.Vector3f(0, 0, 1)
        )
        cutout.transform(mr.AffineXf3f(R_z180, mr.Vector3f(0, 0, 0)))

        # Place bottom of cutout at z=0 (Z-up coordinate system)
        bbox = cutout.computeBoundingBox()
        z_offset = -float(bbox.min.z)
        cutout.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(0, 0, z_offset)))

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
        
        # Apply additional rotation based on which side we're cutting (Z-up coordinate system)
        if side == "left":
            # No additional rotation needed - cutout faces +X direction
            additional_yaw = 0
        elif side == "right":
            # Rotate 180° so cutout faces -X direction  
            additional_yaw = 180
        elif side == "front":
            # Rotate -90° so cutout faces +Y direction
            additional_yaw = 90
        elif side == "back":
            # Rotate 90° so cutout faces -Y direction
            additional_yaw = -90
        else:
            raise ValueError(f"Unknown side: {side}")
            
        if additional_yaw != 0:
            yaw_rad = math.radians(float(additional_yaw))
            c, s = math.cos(yaw_rad), math.sin(yaw_rad)
            # Rotation around Z-axis (up axis) for Z-up coordinate system
            Rz_additional = mr.Matrix3f(
                mr.Vector3f(c, -s, 0),
                mr.Vector3f(s, c, 0),
                mr.Vector3f(0, 0, 1)
            )
            cutout.transform(mr.AffineXf3f(Rz_additional, mr.Vector3f(0, 0, 0)))

        # Get updated bbox after rotation
        bbox = cutout.computeBoundingBox()
        center_x = 0.5 * (float(bbox.min.x) + float(bbox.max.x))  # X center
        center_y = 0.5 * (float(bbox.min.y) + float(bbox.max.y))  # Y center (width in Z-up)

        # Hardcoded optimal values determined through testing
        face_cut_epsilon = -0.5  # mm - optimal epsilon for boolean intersection
        
        if side == "left":
            # Position on left side (x=0), ensure cutout extends into base
            dx = face_cut_epsilon - float(bbox.min.x)
            dy = (width_mm * 0.5) - center_y
        elif side == "right":
            # Position on right side (x=length), ensure cutout extends into base
            dx = (length_mm - face_cut_epsilon) - float(bbox.max.x)
            dy = (width_mm * 0.5) - center_y
        elif side == "front":
            # Position on front side (y=0), ensure cutout extends into base
            dx = (length_mm * 0.5) - center_x
            dy = face_cut_epsilon - float(bbox.min.y)
        elif side == "back":
            # Position on back side (y=width), ensure cutout extends into base
            dx = (length_mm * 0.5) - center_x
            dy = (width_mm - face_cut_epsilon) - float(bbox.max.y)

        cutout.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(dx, dy, 0.0)))









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

    def _apply_cleat_cutout(
        self, 
        base: mr.Mesh, 
        cleat_cut: mr.Mesh, 
        length_mm: float, 
        width_mm: float,
        rotation_z_deg: float = 0.0,
        offset_x_mm: float = 0.0,
        offset_y_mm: float = 0.0,
        offset_z_mm: float = -0.5,
        flip_y_deg: float = 180.0
    ) -> mr.Mesh:
        """Apply cleat cutout to the bottom face of the base.
        
        Args:
            base: Base mesh to cut
            cleat_cut: Cleat cutout mesh
            length_mm: Base length for centering
            width_mm: Base width for centering  
            rotation_z_deg: Rotation around Z-axis in degrees (default 0)
            offset_x_mm: X offset from center in mm (default 0)
            offset_y_mm: Y offset from center in mm (default 0)
            offset_z_mm: Z offset to ensure cutting (default -0.5mm)
            flip_y_deg: Rotation around Y-axis in degrees (default 180)
        """
        # Apply Y-axis flip FIRST (180deg around Y-axis for optimal orientation)
        if abs(flip_y_deg) > 1e-6:
            rad_y = math.radians(flip_y_deg)
            cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
            R_y = mr.Matrix3f()
            R_y.x.x, R_y.x.y, R_y.x.z = cos_y, 0.0, sin_y
            R_y.y.x, R_y.y.y, R_y.y.z = 0.0, 1.0, 0.0
            R_y.z.x, R_y.z.y, R_y.z.z = -sin_y, 0.0, cos_y
            cleat_cut.transform(mr.AffineXf3f(R_y, mr.Vector3f()))
        
        # Apply rotation around Z-axis SECOND (around cleat's center after Y-flip)
        if abs(rotation_z_deg) > 1e-6:
            rad = math.radians(rotation_z_deg)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            R = mr.Matrix3f()
            R.x.x, R.x.y, R.x.z = cos_r, -sin_r, 0.0
            R.y.x, R.y.y, R.y.z = sin_r, cos_r, 0.0
            R.z.x, R.z.y, R.z.z = 0.0, 0.0, 1.0
            cleat_cut.transform(mr.AffineXf3f(R, mr.Vector3f()))
        
        # Now get bbox after rotation and calculate translation
        cleat_bbox = cleat_cut.computeBoundingBox()
        
        # Center X,Y and align bottom face with base bottom (z=0 + small offset)
        center_x = length_mm / 2.0 + offset_x_mm
        center_y = width_mm / 2.0 + offset_y_mm
        center_z = offset_z_mm  # Small negative offset to ensure cutting
        
        # Calculate translation to center the cleat
        cleat_center_x = (cleat_bbox.min.x + cleat_bbox.max.x) / 2.0
        cleat_center_y = (cleat_bbox.min.y + cleat_bbox.max.y) / 2.0
        cleat_bottom_z = cleat_bbox.min.z
        
        translate_x = center_x - cleat_center_x
        translate_y = center_y - cleat_center_y
        translate_z = center_z - cleat_bottom_z
        
        # Apply translation
        cleat_cut.transform(mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(translate_x, translate_y, translate_z)))
        
        # Perform boolean difference
        return self._boolean_difference(base, cleat_cut)

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
        cutout_cleat: bool = True,
        cutout_path: Path | None = None,
        cleat_cutout_path: Path | None = None,
    ) -> mr.Mesh:
        """Generate a base with selective joint cutouts on specified sides and optional cleat cutout on bottom.

        Args:
            length_mm: Base length in millimeters (X-axis)
            width_mm: Base width in millimeters (Y-axis)
            height_mm: Base height in millimeters (Z-axis, default 20)
            cutout_left: Whether to add cutout on left side (x=0)
            cutout_right: Whether to add cutout on right side (x=length)
            cutout_front: Whether to add cutout on front side (y=0)
            cutout_back: Whether to add cutout on back side (y=width)
            cutout_cleat: Whether to add cleat cutout on bottom face (z=0, default True)
            cutout_path: Path to joint cutout STL file (defaults to joint_cutout.stl)
            cleat_cutout_path: Path to cleat cutout STL file (defaults to cleat_cutout.stl)

        Returns:
            mr.Mesh: The generated mesh with requested cutouts applied
        """
        root = Path(__file__).resolve().parents[1]
        if cutout_path is None:
            cutout_path = root / "joint_cutout.stl"
        if cleat_cutout_path is None:
            cleat_cutout_path = root / "cleat_cutout.stl"

        output.header("Generating base with selective joint cutouts (Z-up)")
        output.info(f"Base size: {length_mm} × {width_mm} × {height_mm} mm")
        output.info(f"Joint cutout: {cutout_path}")
        if cutout_cleat:
            output.info(f"Cleat cutout: {cleat_cutout_path}")
        
        # Create base box
        base = self.create_base_box(length_mm, width_mm, height_mm)
        
        # Hardcoded optimal parameters determined through testing (Z-up coordinate system)
        cutout_up_axis = "z"  # Source cutout up axis (matches our Z-up system)
        yaw_deg = 0  # Working yaw rotation around Z-axis
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
            output.info("No side cutouts requested")
        
        # Apply cleat cutout on bottom face if requested
        if cutout_cleat:
            output.subheader("Cutting cleat on bottom face")
            
            # Load cleat cutout mesh
            cleat_cut = self.load_cutout_mesh(cleat_cutout_path)
            
            # Apply cleat cutout with optimal parameters (center 0deg + 180deg Y-axis flip)
            base = self._apply_cleat_cutout(
                base, cleat_cut, length_mm, width_mm,
                rotation_z_deg=0.0,  # Center 0deg orientation
                offset_x_mm=0.0,     # Centered X
                offset_y_mm=0.0,     # Centered Y
                flip_y_deg=180.0     # 180deg rotation around Y-axis
            )
            output.info("✓ cleat cutout applied")
        
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


