#!/usr/bin/env python3
"""
Model Generator for creating 3D meshes from elevation data

This module provides the ModelGenerator class that creates 3D models from elevation data
using the meshlib library. It integrates with the ElevationMap class to use real SRTM data.
"""

import numpy as np
import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mn
from geopy.distance import geodesic
import pickle
import gzip
import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

# Import the new console output system
from .console import output

# Handle both direct execution and module import
try:
    from .srtm import SRTM
except ImportError:
    from terrain_generator.srtm import SRTM

import os


class ModelGenerator:
    """
    A class for generating 3D models from elevation data using meshlib.

    This class integrates with the Elevation to create realistic 3D meshes
    from SRTM elevation data with proper surface normals and base structures.
    """

    CACHE_DIR = "terrain_cache"

    def __init__(self, elevation=None, use_cache=True, cache_max_age_days=30):
        """
        Initialize the ModelGenerator with an elevation source.

        Args:
            elevation (Elevation, optional): Elevation instance.
                                                   If None, creates a new one.
            use_cache (bool): Whether to use local caching for terrain generation
            cache_max_age_days (int): Maximum age of cached data in days
        """
        self.elevation = elevation or SRTM()
        self.use_cache = use_cache
        self.cache_max_age_days = cache_max_age_days
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            Path(self.CACHE_DIR).mkdir(exist_ok=True)

    def _get_cache_filename(self, bounds: Tuple[float, float, float, float], 
                           topo_dir: str, elevation_multiplier: float,
                           downsample_factor: int, water_threshold: Optional[float],
                           adaptive_tolerance_z: float,
                           adaptive_max_gap_fraction: float,
                           adaptive_max_sampled_rows: int,
                           adaptive_max_sampled_cols: int,
                           split_at_water_level: bool) -> str:
        """Generate a cache filename based on all input parameters.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box
            topo_dir: Directory containing SRTM data files
            elevation_multiplier: Multiplier for elevation scaling
            downsample_factor: Factor to downsample elevation data
            water_threshold: Elevation below which areas are considered water

        Returns:
            Cache filename
        """
        # Create a string from all parameters that affect the result
        params_str = f"{bounds[0]:.6f}_{bounds[1]:.6f}_{bounds[2]:.6f}_{bounds[3]:.6f}"
        params_str += f"_{topo_dir}_{elevation_multiplier:.2f}"
        params_str += f"_{downsample_factor}_{water_threshold}"
        # Include adaptive meshing parameters
        params_str += f"_adtol{adaptive_tolerance_z}"
        params_str += f"_adgap{adaptive_max_gap_fraction}"
        params_str += f"_adr{adaptive_max_sampled_rows}_adc{adaptive_max_sampled_cols}"
        params_str += f"_split{int(bool(split_at_water_level))}"
        
        # Create hash of parameters for shorter filename
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        # Create readable filename with key parameters
        bounds_str = f"{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        filename = (
            f"terrain_{params_hash}_{bounds_str}_ds{downsample_factor}_wt{water_threshold}"
            f"_ad{adaptive_tolerance_z}"
            f"_sp{int(bool(split_at_water_level))}"
            f".pkl.gz"
        )
        
        return os.path.join(self.CACHE_DIR, filename)

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if the cache file is still valid.

        Args:
            cache_file: Path to cache file

        Returns:
            True if cache file exists and is within max age
        """
        if not os.path.exists(cache_file):
            return False

        # Check age
        file_age = time.time() - os.path.getmtime(cache_file)
        max_age_seconds = self.cache_max_age_days * 24 * 3600
        
        return file_age < max_age_seconds

    def _save_to_cache(self, bounds: Tuple[float, float, float, float], 
                       topo_dir: str, elevation_multiplier: float,
                       downsample_factor: int, water_threshold: Optional[float],
                       adaptive_tolerance_z: float,
                       adaptive_max_gap_fraction: float,
                       adaptive_max_sampled_rows: int,
                       adaptive_max_sampled_cols: int,
                       split_at_water_level: bool,
                       result: Dict) -> None:
        """Save terrain generation result to cache.

        Args:
            bounds: Bounding box used for the query
            topo_dir: Directory containing SRTM data files
            elevation_multiplier: Multiplier for elevation scaling
            downsample_factor: Factor to downsample elevation data
            water_threshold: Elevation below which areas are considered water
            result: Dictionary containing the terrain generation result
        """
        if not self.use_cache:
            return

        cache_file = self._get_cache_filename(
            bounds,
            topo_dir,
            elevation_multiplier,
            downsample_factor,
            water_threshold,
            adaptive_tolerance_z,
            adaptive_max_gap_fraction,
            adaptive_max_sampled_rows,
            adaptive_max_sampled_cols,
            split_at_water_level,
        )

        # Create base filename without extension for mesh files
        cache_base = cache_file.replace('.pkl.gz', '')
        land_mesh_file = f"{cache_base}_land.obj"

        try:
            # Save meshes to separate files
            if 'land_mesh' in result and result['land_mesh'] is not None:
                mr.saveMesh(result['land_mesh'], land_mesh_file)
                output.progress_info(f"Saved land mesh to {land_mesh_file}")
            
            # base mesh removed

            # Prepare cache data with mesh file paths instead of mesh objects
            cache_data = {
                "bounds": bounds,
                "topo_dir": topo_dir,
                "elevation_multiplier": elevation_multiplier,
                "downsample_factor": downsample_factor,
                "water_threshold": water_threshold,
                "adaptive_tolerance_z": adaptive_tolerance_z,
                "adaptive_max_gap_fraction": adaptive_max_gap_fraction,
                "adaptive_max_sampled_rows": adaptive_max_sampled_rows,
                "adaptive_max_sampled_cols": adaptive_max_sampled_cols,
                "split_at_water_level": split_at_water_level,
                "elevation_data": result.get('elevation_data'),
                "land_mesh_file": land_mesh_file if 'land_mesh' in result else None,
                "timestamp": time.time(),
                "version": "1.0"
            }

            # Save serializable data to cache
            with gzip.open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            output.cache_info("Saved terrain generation result", is_hit=False)
            
        except Exception as e:
            output.warning(f"Could not save terrain cache: {e}")
            # Clean up partial mesh files if they were created
            for mesh_file in [land_mesh_file]:
                if os.path.exists(mesh_file):
                    try:
                        os.remove(mesh_file)
                    except:
                        pass

    def _load_from_cache(self, bounds: Tuple[float, float, float, float], 
                         topo_dir: str, elevation_multiplier: float,
                         downsample_factor: int, water_threshold: Optional[float],
                         adaptive_tolerance_z: float,
                         adaptive_max_gap_fraction: float,
                         adaptive_max_sampled_rows: int,
                         adaptive_max_sampled_cols: int,
                         split_at_water_level: bool) -> Optional[Dict]:
        """Load terrain generation result from cache.

        Args:
            bounds: Bounding box for the query
            topo_dir: Directory containing SRTM data files
            elevation_multiplier: Multiplier for elevation scaling
            downsample_factor: Factor to downsample elevation data
            water_threshold: Elevation below which areas are considered water

        Returns:
            Cached result dictionary if cache hit, None otherwise
        """
        if not self.use_cache:
            return None

        cache_file = self._get_cache_filename(
            bounds,
            topo_dir,
            elevation_multiplier,
            downsample_factor,
            water_threshold,
            adaptive_tolerance_z,
            adaptive_max_gap_fraction,
            adaptive_max_sampled_rows,
            adaptive_max_sampled_cols,
            split_at_water_level,
        )

        if not self._is_cache_valid(cache_file):
            return None

        try:
            with gzip.open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify the parameters match
            if (
                cache_data.get("bounds") != bounds
                or cache_data.get("topo_dir") != topo_dir
                or cache_data.get("elevation_multiplier") != elevation_multiplier
                or cache_data.get("downsample_factor") != downsample_factor
                or cache_data.get("water_threshold") != water_threshold
                or cache_data.get("adaptive_tolerance_z") != adaptive_tolerance_z
                or cache_data.get("adaptive_max_gap_fraction") != adaptive_max_gap_fraction
                or cache_data.get("adaptive_max_sampled_rows") != adaptive_max_sampled_rows
                or cache_data.get("adaptive_max_sampled_cols") != adaptive_max_sampled_cols
                or cache_data.get("split_at_water_level") != split_at_water_level
            ):
                output.warning("Cache parameter mismatch, ignoring cache")
                return None

            # Load meshes from separate files
            result = {
                'elevation_data': cache_data.get('elevation_data')
            }

            # Load land mesh if it exists
            land_mesh_file = cache_data.get('land_mesh_file')
            if land_mesh_file and os.path.exists(land_mesh_file):
                try:
                    land_mesh = mr.loadMesh(land_mesh_file)
                    result['land_mesh'] = land_mesh
                    output.progress_info(f"Loaded land mesh from {land_mesh_file}")
                except Exception as e:
                    output.warning(f"Could not load land mesh from cache: {e}")
                    return None

            # base mesh removed

            cache_age_hours = (time.time() - cache_data["timestamp"]) / 3600
            output.cache_info(f"Loaded terrain generation result (age: {cache_age_hours:.1f} hours)")
            
            return result

        except Exception as e:
            output.warning(f"Could not load terrain cache: {e}")
            return None

    def clear_cache(self, bounds: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Clear cached terrain data.

        Args:
            bounds: If provided, clear only cache for these bounds. Otherwise clear all cache.
        """
        if bounds is not None:
            # This is more complex since we don't know all the other parameters
            # For now, clear all cache files that match the bounds
            cache_dir = Path(self.CACHE_DIR)
            if cache_dir.exists():
                bounds_str = f"{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
                
                # Clear pickle cache files
                cache_files = list(cache_dir.glob(f"terrain_*_{bounds_str}_*.pkl.gz"))
                
                # Clear associated mesh files  
                mesh_files = list(cache_dir.glob(f"terrain_*_{bounds_str}_*_land.obj"))
                
                total_files = cache_files + mesh_files
                for cache_file in total_files:
                    cache_file.unlink()
                    
                output.success(f"Cleared {len(total_files)} cache files ({len(cache_files)} pickle, {len(mesh_files)} mesh) for bounds {bounds}")
        else:
            # Clear all cache files
            cache_dir = Path(self.CACHE_DIR)
            if cache_dir.exists():
                # Clear pickle cache files
                cache_files = list(cache_dir.glob("terrain_*.pkl.gz"))
                
                # Clear mesh files
                mesh_files = list(cache_dir.glob("terrain_*_land.obj"))
                
                total_files = cache_files + mesh_files
                for cache_file in total_files:
                    cache_file.unlink()
                    
                output.success(f"Cleared {len(total_files)} terrain cache files ({len(cache_files)} pickle, {len(mesh_files)} mesh)")

    def _downsample_elevation_data(self, elevation_data, downsample_factor=10):
        """
        Downsample elevation data for faster processing.

        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            downsample_factor (int): Factor by which to downsample (e.g., 10 means take every 10th point)

        Returns:
            numpy.ndarray: Downsampled elevation data
        """
        if downsample_factor <= 1:
            return elevation_data

        # Use slicing to downsample - take every nth point
        downsampled = elevation_data[::downsample_factor, ::downsample_factor]

        output.progress_info(
            f"Downsampled elevation data from {elevation_data.shape} to {downsampled.shape}"
        )

        return downsampled

    def _split_mesh_at_water_level(self, terrain_mesh, water_level, elevation_multiplier=1.0):
        """
        Split the terrain mesh at the water level and translate it so the water plane is at z=0.

        Args:
            terrain_mesh (meshlib.mrmeshpy.Mesh): The terrain mesh to split
            water_level (float): Elevation level at which to split the mesh
            elevation_multiplier (float): Elevation scaling multiplier

        Returns:
            meshlib.mrmeshpy.Mesh: Above-water terrain translated so water plane is z=0
        """
        output.subheader("Splitting terrain mesh at water level")
        
        # Convert water level to model units
        plane_z = water_level * elevation_multiplier
        
        output.progress_info(f"Cutting plane at z = {plane_z:.2f}")
        
        # Create cutting plane at water level (z = plane_z)
        cutting_plane = mr.Plane3f(mr.Vector3f(0, 0, 1), plane_z)
        
        # Setup trim parameters for above-water mesh (keep everything above plane)
        trim_params_above = mr.TrimWithPlaneParams()
        trim_params_above.plane = cutting_plane
        # Use a small epsilon based on mesh size
        bbox = terrain_mesh.computeBoundingBox()
        trim_params_above.eps = 1e-6 * bbox.diagonal()
        
        # Trim the above-water mesh (this will keep the upper part)
        output.progress_info("Trimming above-water mesh...")
        
        # Perform the trim operation and collect cut contours
        mr.trimWithPlane(terrain_mesh, trim_params_above)
        
        output.progress_info(f"Trimming complete.")

        # Seal the newly-opened hole on the *kept* half
        output.progress_info("Sealing holes in above-water mesh...")
        # Pre-clean hole boundaries to avoid self-intersections during triangulation
        try:
            # Duplicate vertices that appear on a hole boundary more than twice
            dup_count = mr.duplicateMultiHoleVertices(terrain_mesh)
            if dup_count:
                output.info(f"Duplicated {dup_count} multi-hole boundary vertices")

            # Remove faces that complicate holes (e.g., holes passing the same vertex multiple times)
            complicating_faces = mr.findHoleComplicatingFaces(terrain_mesh)
            has_faces = False
            try:
                if hasattr(complicating_faces, 'count') and callable(getattr(complicating_faces, 'count')):
                    has_faces = complicating_faces.count() > 0
                else:
                    has_faces = len(complicating_faces) > 0  # type: ignore[arg-type]
            except Exception:
                # if BitSet does not support len(), try 'any()'
                if hasattr(complicating_faces, 'any') and callable(getattr(complicating_faces, 'any')):
                    has_faces = bool(complicating_faces.any())
            if has_faces:
                terrain_mesh.deleteFaces(complicating_faces)
                output.info("Removed faces complicating holes")

            # Weld near-duplicate boundary vertices to avoid tiny edges causing self-intersections
            bbox_after_trim = terrain_mesh.computeBoundingBox()
            close_dist = 1e-6 * bbox_after_trim.diagonal()
            welded = mr.uniteCloseVertices(terrain_mesh, float(close_dist), True)
            if welded:
                output.info(f"Welded {welded} close boundary vertices")
        except Exception as e:
            output.warning(f"Hole pre-clean failed (continuing): {e}")

        # Find holes and run fast planar fill
        hole_edges = terrain_mesh.topology.findHoleRepresentiveEdges()
        if not hole_edges.empty():  # C++ .empty() is exposed in Py
            mr.fillContours2D(terrain_mesh, hole_edges)
        else:
            output.warning("No holes found in above-water mesh")

        # Translate the above-water mesh down so the cut plane aligns to z=0
        try:
            verts = mn.getNumpyVerts(terrain_mesh)
            verts[:, 2] -= float(plane_z)
            output.info(f"Translated above-water mesh by -{plane_z:.2f} along Z to align water level to z=0")
        except Exception as e:
            output.warning(f"Failed to translate mesh to z=0: {e}")

        output.success("Mesh split and alignment to z=0 complete!")
        return terrain_mesh
    
    def _flatten_water_level(self, elevation_data, water_threshold):
        """
        Flatten the water level to the lowest elevation.
        """
        
        # Flatten the water level to the threshold elevation
        water_area = elevation_data < water_threshold
        elevation_data[water_area] = water_threshold

        # Raise the land level to the threshold elevation plus a bit
        land_area = elevation_data > water_threshold
        elevation_data[land_area] += 10
        
        return elevation_data

    def _create_mesh_from_elevation(
        self,
        elevation_data,
        bounds,
        elevation_multiplier: float = 1.0,
        wall_bottom_z: float = 0.0,
        adaptive_tolerance_z: float = 1.0,
        adaptive_max_gap_fraction: float = 1/256,
        adaptive_max_sampled_rows: int = 400,
        adaptive_max_sampled_cols: int = 400,
    ):
        """
        Create a 3D mesh from elevation data with a flat base using an adaptive grid.

        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            elevation_multiplier (float): Multiplier for realistic elevation scaling (1.0 = realistic scale)

        Returns:
            meshlib.mrmeshpy.Mesh: The generated 3D mesh
        """
        height, width = elevation_data.shape

        # Calculate real-world dimensions of the bounds
        width_meters, height_meters = self.elevation.calculate_bounds_dimensions_meters(bounds)

        # Scale factors (meters per grid step)
        scale_x = width_meters / width
        scale_y = height_meters / height
        elevation_scale = float(elevation_multiplier)

        # Report source grid
        output.progress_info(f"Creating adaptive mesh from {width}x{height} elevation grid")
        output.info(f"Real-world size: {width_meters/1000:.2f} x {height_meters/1000:.2f} km")
        output.info(f"Elevation range: {elevation_data.min():.1f} to {elevation_data.max():.1f} meters")

        # Choose tolerance (meters in Z) for adaptive sampling from parameters
        adaptive_tol = float(adaptive_tolerance_z)

        # Helper: RDP for 1D sequence to select indices
        def rdp_indices_1d(coord: np.ndarray, values: np.ndarray, eps: float) -> np.ndarray:
            n = coord.shape[0]
            if n <= 2:
                return np.array([0, n - 1], dtype=np.int32)
            keep = np.zeros(n, dtype=bool)
            keep[0] = True
            keep[-1] = True
            stack = [(0, n - 1)]
            while stack:
                s, e = stack.pop()
                if e - s <= 1:
                    continue
                x1, y1 = coord[s], values[s]
                x2, y2 = coord[e], values[e]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    # All collinear zero-length, skip
                    continue
                # Perpendicular distance of points to line segment (x1,y1)-(x2,y2)
                idx = np.arange(s + 1, e, dtype=np.int32)
                x = coord[idx]
                y = values[idx]
                # Distance formula: |dy*x - dx*y + (x2*y1 - y2*x1)| / sqrt(dx^2 + dy^2)
                num = np.abs(dy * x - dx * y + (x2 * y1 - y2 * x1))
                den = np.hypot(dx, dy)
                d = num / den
                imax_local = int(idx[np.argmax(d)]) if idx.size > 0 else -1
                dmax = float(d.max()) if idx.size > 0 else 0.0
                if dmax > eps and imax_local >= 0:
                    keep[imax_local] = True
                    stack.append((s, imax_local))
                    stack.append((imax_local, e))
            return np.flatnonzero(keep).astype(np.int32)

        # Build selected columns via sampling several rows
        xs = (np.arange(width, dtype=np.float32) * np.float32(scale_x))
        ys = (np.arange(height, dtype=np.float32) * np.float32(scale_y))
        selected_cols: set[int] = set([0, width - 1])
        selected_rows: set[int] = set([0, height - 1])

        # Choose sampling steps to bound runtime
        # Determine sampling step sizes based on caps
        row_cap = max(1, int(adaptive_max_sampled_rows))
        col_cap = max(1, int(adaptive_max_sampled_cols))
        row_step = max(1, height // row_cap)
        col_step = max(1, width // col_cap)

        # Sample rows to pick important x positions
        tol = float(adaptive_tol)
        for i in range(0, height, row_step):
            z_row = elevation_data[i, :].astype(np.float32) * np.float32(elevation_scale)
            keep_idx = rdp_indices_1d(xs, z_row, tol)
            for j in keep_idx:
                selected_cols.add(int(j))

        # Sample columns to pick important y positions
        for j in range(0, width, col_step):
            z_col = elevation_data[:, j].astype(np.float32) * np.float32(elevation_scale)
            keep_idx = rdp_indices_1d(ys, z_col, tol)
            for i in keep_idx:
                selected_rows.add(int(i))

        # Sort unique indices
        sel_cols = np.array(sorted(selected_cols), dtype=np.int32)
        sel_rows = np.array(sorted(selected_rows), dtype=np.int32)

        # Optional: cap maximum spacing to avoid over-large faces (insert midpoints if huge gaps)
        def enforce_max_gap(indices: np.ndarray, max_gap: int) -> np.ndarray:
            out = [int(indices[0])]
            for k in range(1, len(indices)):
                prev = int(out[-1])
                cur = int(indices[k])
                while cur - prev > max_gap:
                    prev = prev + max_gap
                    out.append(prev)
                out.append(cur)
            return np.array(out, dtype=np.int32)

        # Avoid cells larger than a fraction of original resolution
        max_gap_cols = max(2, int(round(width * float(adaptive_max_gap_fraction))))
        max_gap_rows = max(2, int(round(height * float(adaptive_max_gap_fraction))))
        sel_cols = enforce_max_gap(sel_cols, max_gap=max_gap_cols)
        sel_rows = enforce_max_gap(sel_rows, max_gap=max_gap_rows)

        nr = int(sel_rows.shape[0])
        nc = int(sel_cols.shape[0])
        output.info(f"Adaptive grid: {nc} x {nr} samples (from {width} x {height})")

        # Build top vertices for adaptive grid
        Xg = (sel_cols.astype(np.float32) * np.float32(scale_x))[None, :].repeat(nr, axis=0)
        Yg = (sel_rows.astype(np.float32) * np.float32(scale_y))[:, None].repeat(nc, axis=1)
        Zg = (elevation_data[sel_rows[:, None], sel_cols[None, :]].astype(np.float32) * np.float32(elevation_scale))
        top_vertices = np.stack((Xg, Yg, Zg), axis=-1).reshape(-1, 3)

        # Walls: use perimeter of the adaptive grid
        z_front = np.full((nc,), np.float32(wall_bottom_z), dtype=np.float32)
        z_back = np.full((nc,), np.float32(wall_bottom_z), dtype=np.float32)
        z_left = np.full((nr,), np.float32(wall_bottom_z), dtype=np.float32)
        z_right = np.full((nr,), np.float32(wall_bottom_z), dtype=np.float32)
        front_bottom = np.stack((Xg[0, :], Yg[0, :], z_front), axis=-1)
        back_bottom = np.stack((Xg[-1, :], Yg[-1, :], z_back), axis=-1)
        left_bottom = np.stack((Xg[:, 0], Yg[:, 0], z_left), axis=-1)
        right_bottom = np.stack((Xg[:, -1], Yg[:, -1], z_right), axis=-1)
        perimeter_bottom = np.concatenate((front_bottom, back_bottom, left_bottom, right_bottom), axis=0)
        vertices_array = np.concatenate((top_vertices, perimeter_bottom), axis=0).astype(np.float32, copy=False)

        # Faces for top surface over adaptive grid
        i = np.arange(nr - 1, dtype=np.int32)[:, None]
        j = np.arange(nc - 1, dtype=np.int32)[None, :]
        tl = i * np.int32(nc) + j
        tr = tl + 1
        bl = tl + np.int32(nc)
        br = bl + 1
        top_faces = np.stack([tl, tr, bl, tr, br, bl], axis=-1).reshape(-1, 3)

        faces_list = [top_faces]

        # Walls faces using perimeter indices
        offset = np.int32(nr * nc)
        j1 = np.arange(nc - 1, dtype=np.int32)
        i1 = np.arange(nr - 1, dtype=np.int32)

        # front edge (row 0)
        front_bottom_offset = offset
        front = np.stack([
            j1, front_bottom_offset + j1, j1 + 1,
            j1 + 1, front_bottom_offset + j1, front_bottom_offset + j1 + 1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(front)

        # back edge (last row)
        back_row = np.int32((nr - 1) * nc)
        back_bottom_offset = offset + np.int32(nc)
        back = np.stack([
            back_row + j1, back_row + j1 + 1, back_bottom_offset + j1,
            back_row + j1 + 1, back_bottom_offset + j1 + 1, back_bottom_offset + j1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(back)

        # left edge (col 0)
        left_bottom_offset = offset + np.int32(2 * nc)
        left = np.stack([
            i1 * np.int32(nc), (i1 + 1) * np.int32(nc), left_bottom_offset + i1,
            (i1 + 1) * np.int32(nc), left_bottom_offset + i1 + 1, left_bottom_offset + i1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(left)

        # right edge (last col)
        right_bottom_offset = offset + np.int32(2 * nc + nr)
        right = np.stack([
            i1 * np.int32(nc) + (nc - 1), right_bottom_offset + i1, (i1 + 1) * np.int32(nc) + (nc - 1),
            (i1 + 1) * np.int32(nc) + (nc - 1), right_bottom_offset + i1, right_bottom_offset + i1 + 1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(right)

        faces_array = np.concatenate(faces_list, axis=0).astype(np.int32, copy=False)

        output.success(f"Generated adaptive mesh with {vertices_array.shape[0]} vertices and {faces_array.shape[0]} faces")

        mesh = mn.meshFromFacesVerts(faces_array, vertices_array)
        return mesh

    def _scale_mesh_to_max_length(self, mesh : mr.Mesh, max_length_mm : float, bounds : Tuple[float, float, float, float]) -> mr.Mesh:
        """
        Scale the mesh to the max length.
        """
        width_meters, height_meters = self.elevation.calculate_bounds_dimensions_meters(bounds)
        max_dim = max(width_meters, height_meters)
        scale_factor = max_length_mm / max_dim

        # Scale the mesh to the max length
        output.progress_info(f"Scaling mesh to max length {max_length_mm} mm with scale factor {scale_factor}")
        mesh.transform(mr.AffineXf3f(mr.Matrix3f.scale(scale_factor), mr.Vector3f(0, 0, 0)))
        return mesh
    
    def save_mesh(self, mesh : mr.Mesh, filename : str) -> None:
        """
        Save the mesh to a file.

        Args:
            mesh (meshlib.mrmeshpy.Mesh): The mesh to save
            filename (str): Output filename
        """
        try:
            mr.saveMesh(mesh, filename)
            output.file_saved(filename, "mesh")
        except Exception as e:
            output.error(f"Failed to save {filename}: {e}")

    def generate_terrain_model(
        self,
        bounds,
        topo_dir="topo",
        elevation_multiplier=1.0,
        downsample_factor=10,
        water_threshold=None,
        force_refresh=False,
        adaptive_tolerance_z: float = 1.0,
        adaptive_max_gap_fraction: float = 1/256,
        adaptive_max_sampled_rows: int = 400,
        adaptive_max_sampled_cols: int = 400,
        split_at_water_level: bool = True,
        max_length_mm: float = 200.0,
    ):
        """
        Generate a 3D terrain model from SRTM elevation data.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files
            elevation_multiplier (float): Multiplier for realistic elevation scaling (1.0 = realistic scale)
            downsample_factor (int): Factor to downsample elevation data (default: 10)
            water_threshold (float, optional): Elevation below which areas are considered water
            force_refresh (bool): If True, ignore cache and generate fresh data

        Returns:
            dict: Dictionary containing generated meshes and data:
                - 'elevation_data': The elevation data used
                - 'land_mesh': Above-water terrain mesh translated so water plane is at z=0 (if split)
        """
        output.header("Terrain Model Generation", f"Bounds: {bounds}")

        # Try to load from cache first
        if not force_refresh:
            cached_result = self._load_from_cache(
                bounds,
                topo_dir,
                elevation_multiplier,
                downsample_factor,
                water_threshold,
                float(adaptive_tolerance_z),
                float(adaptive_max_gap_fraction),
                int(adaptive_max_sampled_rows),
                int(adaptive_max_sampled_cols),
                split_at_water_level,
            )
            if cached_result is not None:
                return cached_result

        # Get elevation data from SRTM
        output.subheader("Loading elevation data")
        elevation_data = self.elevation.get_elevation(bounds, topo_dir)

        output.info(
            f"Elevation data: {elevation_data.shape}, range: {elevation_data.min():.1f} to {elevation_data.max():.1f} m"
        )

        # Downsample elevation data for faster processing
        if downsample_factor > 1:
            elevation_data = self._downsample_elevation_data(
                elevation_data, downsample_factor
            )

        # Prepare elevation data for meshing (use a copy to avoid altering the original data)
        output.subheader("Flattening water level for meshing")
        elevation_data_for_mesh = elevation_data.copy()
        if water_threshold is not None:
            elevation_data_for_mesh = self._flatten_water_level(elevation_data_for_mesh, water_threshold)

        # Elevation quantization removed per user preference

        # Create terrain mesh
        output.subheader("Creating terrain mesh")
        # Determine wall bottom height so after split and translation walls reach z=0
        wall_bottom_z = float(water_threshold) * float(elevation_multiplier) if (split_at_water_level and water_threshold is not None) else 0.0
        terrain = self._create_mesh_from_elevation(
            elevation_data_for_mesh,
            bounds,
            elevation_multiplier,
            wall_bottom_z=wall_bottom_z,
            adaptive_tolerance_z=float(adaptive_tolerance_z),
            adaptive_max_gap_fraction=float(adaptive_max_gap_fraction),
            adaptive_max_sampled_rows=int(adaptive_max_sampled_rows),
            adaptive_max_sampled_cols=int(adaptive_max_sampled_cols),
        )

        # Optionally split the terrain at water level and translate to z=0
        if split_at_water_level:
            output.subheader("Splitting terrain mesh at water level")
            if water_threshold is None:
                output.warning("Water threshold is None; skipping split")
                land = terrain
            else:
                output.progress_info(
                    f"Splitting at water level: {water_threshold:.2f} meters"
                )
                land = self._split_mesh_at_water_level(
                    terrain,
                    water_threshold,
                    elevation_multiplier,
                )
        else:
            land = terrain

        # Scale the land mesh to the max length
        if max_length_mm is not None:
            output.subheader("Scaling land mesh to max length")
            land = self._scale_mesh_to_max_length(land, max_length_mm, bounds)

        # Create result dictionary
        result = {
            'elevation_data': elevation_data,  # keep original (non-flattened) for downstream consumers
            'land_mesh': land,
        }

        # Save to cache
        self._save_to_cache(
            bounds,
            topo_dir,
            elevation_multiplier,
            downsample_factor,
            water_threshold,
            float(adaptive_tolerance_z),
            float(adaptive_max_gap_fraction),
            int(adaptive_max_sampled_rows),
            int(adaptive_max_sampled_cols),
            split_at_water_level,
            result,
        )

        output.success("Terrain model generation complete!")

        return result
