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
                           topo_dir: str, base_height: float, elevation_multiplier: float,
                           downsample_factor: int, water_threshold: Optional[float],
                           decimate: bool, decimate_max_error: Optional[float],
                           decimate_target_face_count: Optional[int],
                           split_at_water_level: bool) -> str:
        """Generate a cache filename based on all input parameters.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box
            topo_dir: Directory containing SRTM data files
            base_height: Height of the flat base
            elevation_multiplier: Multiplier for elevation scaling
            downsample_factor: Factor to downsample elevation data
            water_threshold: Elevation below which areas are considered water

        Returns:
            Cache filename
        """
        # Create a string from all parameters that affect the result
        params_str = f"{bounds[0]:.6f}_{bounds[1]:.6f}_{bounds[2]:.6f}_{bounds[3]:.6f}"
        params_str += f"_{topo_dir}_{base_height:.2f}_{elevation_multiplier:.2f}"
        params_str += f"_{downsample_factor}_{water_threshold}"
        # Include new mesh generation options in cache key
        params_str += f"_dec{int(bool(decimate))}"
        params_str += f"_split{int(bool(split_at_water_level))}"
        params_str += f"_decmerr{decimate_max_error if decimate_max_error is not None else 'None'}"
        params_str += f"_decfaces{decimate_target_face_count if decimate_target_face_count is not None else 'None'}"
        
        # Create hash of parameters for shorter filename
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        # Create readable filename with key parameters
        bounds_str = f"{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        filename = (
            f"terrain_{params_hash}_{bounds_str}_ds{downsample_factor}_wt{water_threshold}"
            f"_dec{int(bool(decimate))}"
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
                       topo_dir: str, base_height: float, elevation_multiplier: float,
                       downsample_factor: int, water_threshold: Optional[float],
                       decimate: bool, decimate_max_error: Optional[float],
                       decimate_target_face_count: Optional[int],
                       split_at_water_level: bool,
                       result: Dict) -> None:
        """Save terrain generation result to cache.

        Args:
            bounds: Bounding box used for the query
            topo_dir: Directory containing SRTM data files
            base_height: Height of the flat base
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
            base_height,
            elevation_multiplier,
            downsample_factor,
            water_threshold,
            decimate,
            decimate_max_error,
            decimate_target_face_count,
            split_at_water_level,
        )

        # Create base filename without extension for mesh files
        cache_base = cache_file.replace('.pkl.gz', '')
        land_mesh_file = f"{cache_base}_land.obj"
        base_mesh_file = f"{cache_base}_base.obj"

        try:
            # Save meshes to separate files
            if 'land_mesh' in result and result['land_mesh'] is not None:
                mr.saveMesh(result['land_mesh'], land_mesh_file)
                output.progress_info(f"Saved land mesh to {land_mesh_file}")
            
            if 'base_mesh' in result and result['base_mesh'] is not None:
                mr.saveMesh(result['base_mesh'], base_mesh_file)
                output.progress_info(f"Saved base mesh to {base_mesh_file}")

            # Prepare cache data with mesh file paths instead of mesh objects
            cache_data = {
                "bounds": bounds,
                "topo_dir": topo_dir,
                "base_height": base_height,
                "elevation_multiplier": elevation_multiplier,
                "downsample_factor": downsample_factor,
                "water_threshold": water_threshold,
                "decimate": decimate,
                "decimate_max_error": decimate_max_error,
                "decimate_target_face_count": decimate_target_face_count,
                "split_at_water_level": split_at_water_level,
                "elevation_data": result.get('elevation_data'),
                "land_mesh_file": land_mesh_file if 'land_mesh' in result else None,
                "base_mesh_file": base_mesh_file if 'base_mesh' in result else None,
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
            for mesh_file in [land_mesh_file, base_mesh_file]:
                if os.path.exists(mesh_file):
                    try:
                        os.remove(mesh_file)
                    except:
                        pass

    def _load_from_cache(self, bounds: Tuple[float, float, float, float], 
                         topo_dir: str, base_height: float, elevation_multiplier: float,
                         downsample_factor: int, water_threshold: Optional[float],
                         decimate: bool, decimate_max_error: Optional[float],
                         decimate_target_face_count: Optional[int],
                         split_at_water_level: bool) -> Optional[Dict]:
        """Load terrain generation result from cache.

        Args:
            bounds: Bounding box for the query
            topo_dir: Directory containing SRTM data files
            base_height: Height of the flat base
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
            base_height,
            elevation_multiplier,
            downsample_factor,
            water_threshold,
            decimate,
            decimate_max_error,
            decimate_target_face_count,
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
                or cache_data.get("base_height") != base_height
                or cache_data.get("elevation_multiplier") != elevation_multiplier
                or cache_data.get("downsample_factor") != downsample_factor
                or cache_data.get("water_threshold") != water_threshold
                or cache_data.get("decimate") != decimate
                or cache_data.get("decimate_max_error") != decimate_max_error
                or cache_data.get("decimate_target_face_count") != decimate_target_face_count
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

            # Load base mesh if it exists
            base_mesh_file = cache_data.get('base_mesh_file')
            if base_mesh_file and os.path.exists(base_mesh_file):
                try:
                    base_mesh = mr.loadMesh(base_mesh_file)
                    result['base_mesh'] = base_mesh
                    output.progress_info(f"Loaded base mesh from {base_mesh_file}")
                except Exception as e:
                    output.warning(f"Could not load base mesh from cache: {e}")
                    return None

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
                mesh_files.extend(list(cache_dir.glob(f"terrain_*_{bounds_str}_*_base.obj")))
                
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
                mesh_files.extend(list(cache_dir.glob("terrain_*_base.obj")))
                
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

    def _split_mesh_at_water_level(self, terrain_mesh, water_level, bounds, base_height=1.0, elevation_multiplier=1.0):
        """
        Split the terrain mesh into two parts at the water level using efficient cutting plane operations.

        Args:
            terrain_mesh (meshlib.mrmeshpy.Mesh): The terrain mesh to split
            water_level (float): Elevation level at which to split the mesh
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            base_height (float): Height of the base
            elevation_multiplier (float): Elevation scaling multiplier

        Returns:
            tuple: (above_water_mesh, below_water_mesh)
                - above_water_mesh: Part of the terrain above water level with sealed base
                - below_water_mesh: Rectangular prism representing the underwater base
        """
        output.subheader("Splitting terrain mesh at water level")
        
        # Convert water level to model units
        plane_z = water_level * elevation_multiplier + base_height
        
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
        hole_edges = terrain_mesh.topology.findHoleRepresentiveEdges()
        if not hole_edges.empty():                     # C++ .empty() is exposed in Py
            mr.fillContours2D(terrain_mesh, hole_edges)

        output.progress_info("Successfully sealed holes in above-water mesh")

        # Create below-water mesh as a simple rectangular prism
        output.progress_info("Creating below-water rectangular prism...")
        below_water_mesh = self._create_rectangular_prism(bounds, base_height)
        
        output.success("Mesh splitting at water level complete!")
        return terrain_mesh, below_water_mesh
            
    def _create_rectangular_prism(self, bounds, height):
        """
        Create a rectangular prism mesh for the below-water base.
        
        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            height (float): Height of the prism (base_height)
            
        Returns:
            meshlib.mrmeshpy.Mesh: Rectangular prism mesh
        """
        # Calculate real-world dimensions
        width_meters, height_meters = self.elevation.calculate_bounds_dimensions_meters(bounds)
        
        # Create vertices for a rectangular prism
        # Bottom face (z = 0)
        vertices = np.array([
            [0, 0, 0],                              # bottom-front-left
            [width_meters, 0, 0],                   # bottom-front-right
            [width_meters, height_meters, 0],       # bottom-back-right
            [0, height_meters, 0],                  # bottom-back-left
            # Top face (z = height)
            [0, 0, height],                         # top-front-left
            [width_meters, 0, height],              # top-front-right
            [width_meters, height_meters, height],  # top-back-right
            [0, height_meters, height]              # top-back-left
        ], dtype=np.float32)
        
        # Create faces for the rectangular prism
        faces = np.array([
            # Bottom face (facing down)
            [0, 2, 1], [0, 3, 2],
            # Top face (facing up)  
            [4, 5, 6], [4, 6, 7],
            # Front face (y = 0)
            [0, 1, 5], [0, 5, 4],
            # Back face (y = height_meters)
            [2, 3, 7], [2, 7, 6],
            # Left face (x = 0)
            [3, 0, 4], [3, 4, 7],
            # Right face (x = width_meters)
            [1, 2, 6], [1, 6, 5]
        ], dtype=np.int32)
        
        # Create mesh using meshlib
        prism_mesh = mn.meshFromFacesVerts(faces, vertices)
        
        output.progress_info(f"Created rectangular prism: {width_meters/1000:.2f} x {height_meters/1000:.2f} km x {height:.1f} m")
        
        return prism_mesh
    
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
        base_height: float = 1.0,
        elevation_multiplier: float = 1.0,
        wall_bottom_z: float = 0.0,
    ):
        """
        Create a 3D mesh from elevation data with a flat base using meshlib.

        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            base_height (float): Height of the flat base
            elevation_multiplier (float): Multiplier for realistic elevation scaling (1.0 = realistic scale)

        Returns:
            meshlib.mrmeshpy.Mesh: The generated 3D mesh
        """
        height, width = elevation_data.shape

        # Calculate real-world dimensions of the bounds
        width_meters, height_meters = self.elevation.calculate_bounds_dimensions_meters(bounds)

        # Calculate scale factor to fit real-world dimensions to grid
        # This makes 1 grid unit = actual meters in the real world
        scale_x = width_meters / width
        scale_y = height_meters / height

        # For realistic elevation scaling: 1 meter elevation = 1 meter in model
        elevation_scale = float(elevation_multiplier)

        output.progress_info(f"Creating mesh from {width}x{height} elevation grid")
        output.info(f"Real-world size: {width_meters/1000:.2f} x {height_meters/1000:.2f} km")
        output.info(
            f"Elevation range: {elevation_data.min():.1f} to {elevation_data.max():.1f} meters"
        )

        # Vectorized vertex generation
        xs = (np.arange(width, dtype=np.float32) * np.float32(scale_x))
        ys = (np.arange(height, dtype=np.float32) * np.float32(scale_y))
        X, Y = np.meshgrid(xs, ys)  # (H, W)
        Z = elevation_data.astype(np.float32, copy=False) * np.float32(elevation_scale) + np.float32(base_height)

        top_vertices = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        # Always use light base: include only perimeter bottom vertices to create outer walls
        z_front = np.full_like(X[0, :], np.float32(wall_bottom_z), dtype=np.float32)
        z_back = np.full_like(X[-1, :], np.float32(wall_bottom_z), dtype=np.float32)
        z_left = np.full_like(X[:, 0], np.float32(wall_bottom_z), dtype=np.float32)
        z_right = np.full_like(X[:, -1], np.float32(wall_bottom_z), dtype=np.float32)
        front_bottom = np.stack((X[0, :], Y[0, :], z_front), axis=-1)
        back_bottom = np.stack((X[-1, :], Y[-1, :], z_back), axis=-1)
        left_bottom = np.stack((X[:, 0], Y[:, 0], z_left), axis=-1)
        right_bottom = np.stack((X[:, -1], Y[:, -1], z_right), axis=-1)
        perimeter_bottom = np.concatenate((front_bottom, back_bottom, left_bottom, right_bottom), axis=0)
        vertices_array = np.concatenate((top_vertices, perimeter_bottom), axis=0).astype(np.float32, copy=False)

        # Vectorized face generation
        i = np.arange(height - 1, dtype=np.int32)[:, None]  # (H-1, 1)
        j = np.arange(width - 1, dtype=np.int32)[None, :]   # (1, W-1)
        tl = i * width + j
        tr = tl + 1
        bl = tl + width
        br = bl + 1

        top_faces = np.stack([tl, tr, bl, tr, br, bl], axis=-1).reshape(-1, 3)

        faces_list = [top_faces]

        # Always light base: construct walls using perimeter bottom vertices
        offset = np.int32(height * width)
        j1 = np.arange(width - 1, dtype=np.int32)
        i1 = np.arange(height - 1, dtype=np.int32)

        # Index mapping for perimeter bottom vertices
        # front: [0 .. width-1] -> offset + 0 .. offset + (width-1)
        # back:  next width vertices -> offset + width .. offset + 2*width-1
        # left:  next height vertices -> offset + 2*width .. offset + 2*width + (height-1)
        # right: next height vertices -> offset + 2*width + height .. offset + 2*width + 2*height - 1

        # Front edge (y=0)
        front_bottom_offset = offset
        front = np.stack([
            j1, front_bottom_offset + j1, j1 + 1,
            j1 + 1, front_bottom_offset + j1, front_bottom_offset + j1 + 1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(front)

        # Back edge (y=height-1)
        back_row = np.int32((height - 1) * width)
        back_bottom_offset = offset + np.int32(width)
        back = np.stack([
            back_row + j1, back_row + j1 + 1, back_bottom_offset + j1,
            back_row + j1 + 1, back_bottom_offset + j1 + 1, back_bottom_offset + j1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(back)

        # Left edge (x=0)
        left_bottom_offset = offset + np.int32(2 * width)
        left = np.stack([
            i1 * width, (i1 + 1) * width, left_bottom_offset + i1,
            (i1 + 1) * width, left_bottom_offset + i1 + 1, left_bottom_offset + i1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(left)

        # Right edge (x=width-1)
        right_bottom_offset = offset + np.int32(2 * width + height)
        right = np.stack([
            i1 * width + (width - 1), right_bottom_offset + i1, (i1 + 1) * width + (width - 1),
            (i1 + 1) * width + (width - 1), right_bottom_offset + i1, right_bottom_offset + i1 + 1
        ], axis=-1).reshape(-1, 3)
        faces_list.append(right)

        faces_array = np.concatenate(faces_list, axis=0).astype(np.int32, copy=False)

        output.success(f"Generated mesh with {vertices_array.shape[0]} vertices and {faces_array.shape[0]} faces")

        # Create the mesh using meshlib mrmeshnumpy
        mesh = mn.meshFromFacesVerts(faces_array, vertices_array)

        return mesh

    def _decimate_mesh(
        self,
        mesh,
        max_error: Optional[float] = None,
        target_face_count: Optional[int] = None,
        preserve_boundary: bool = True,
        max_edge_length: Optional[float] = None,
    ) -> None:
        """
        Optionally decimate the mesh if supported by meshlib, preserving boundaries.

        Args:
            mesh: meshlib.mrmeshpy.Mesh instance
            max_error: Maximum geometric error in world units (meters)
            target_face_count: Target number of faces
            preserve_boundary: Preserve boundary edges if supported
            max_edge_length: Optional edge length cap
        """
        try:
            if hasattr(mr, "DecimateSettings") and hasattr(mr, "decimateMesh"):
                settings = mr.DecimateSettings()
                if max_error is not None and hasattr(settings, "maxError"):
                    settings.maxError = float(max_error)
                if target_face_count is not None and hasattr(settings, "targetFaceCount"):
                    settings.targetFaceCount = int(target_face_count)
                if hasattr(settings, "preserveBoundary"):
                    settings.preserveBoundary = bool(preserve_boundary)
                if max_edge_length is not None and hasattr(settings, "maxEdgeLen"):
                    settings.maxEdgeLen = float(max_edge_length)

                mr.decimateMesh(mesh, settings)
                if hasattr(mr, "optimizeTopology"):
                    mr.optimizeTopology(mesh)
                output.success("Applied mesh decimation")
            else:
                output.warning("Mesh decimation not available in this meshlib build; skipping")
        except Exception as e:
            output.warning(f"Mesh decimation failed: {e}")
    
    def save_mesh(self, mesh, filename):
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
        base_height=1.0,
        elevation_multiplier=1.0,
        downsample_factor=10,
        water_threshold=None,
        force_refresh=False,
        decimate: bool = False,
        decimate_max_error: Optional[float] = None,
        decimate_target_face_count: Optional[int] = None,
        decimate_preserve_boundary: bool = True,
        split_at_water_level: bool = True,
        merge_land_and_base: bool = False,
    ):
        """
        Generate a 3D terrain model from SRTM elevation data.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files
            base_height (float): Height of the flat base
            elevation_multiplier (float): Multiplier for realistic elevation scaling (1.0 = realistic scale)
            downsample_factor (int): Factor to downsample elevation data (default: 10)
            water_threshold (float, optional): Elevation below which areas are considered water
            force_refresh (bool): If True, ignore cache and generate fresh data

        Returns:
            dict: Dictionary containing generated meshes and data:
                - 'terrain_mesh': The main terrain mesh (or above-water part if split)
                - 'water_mesh': Water surface mesh (if extract_water=True)
                - 'water_mask': Boolean array indicating water locations (if extract_water=True)
                - 'elevation_data': The elevation data used
                - 'below_water_mesh': Part of terrain below water level (if split_at_water_level=True)
                - 'above_water_mesh': Part of terrain above water level (if split_at_water_level=True)
        """
        output.header("Terrain Model Generation", f"Bounds: {bounds}")

        # Try to load from cache first
        if not force_refresh:
            cached_result = self._load_from_cache(
                bounds,
                topo_dir,
                base_height,
                elevation_multiplier,
                downsample_factor,
                water_threshold,
                decimate,
                decimate_max_error,
                decimate_target_face_count,
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
        # Determine wall bottom height for light_base to keep walls after split
        if split_at_water_level and water_threshold is not None:
            wall_bottom_z = float(water_threshold) * float(elevation_multiplier) + float(base_height)
        else:
            wall_bottom_z = 0.0
        terrain = self._create_mesh_from_elevation(
            elevation_data_for_mesh, bounds, base_height, elevation_multiplier, wall_bottom_z=wall_bottom_z
        )

        # Optionally split the terrain at water level and create base prism
        if split_at_water_level:
            output.subheader("Splitting terrain mesh at water level")
            if water_threshold is None:
                output.warning("Water threshold is None; skipping split")
                land, base = terrain, None
            else:
                output.progress_info(
                    f"Splitting at water level: {water_threshold:.2f} meters"
                )
                land, base = self._split_mesh_at_water_level(
                    terrain,
                    water_threshold,
                    bounds,
                    base_height,
                    elevation_multiplier,
                )
        else:
            land, base = terrain, None

        # Optional decimation of the land mesh
        if decimate and land is not None:
            output.subheader("Decimating land mesh")
            self._decimate_mesh(
                land,
                max_error=decimate_max_error,
                target_face_count=decimate_target_face_count,
                preserve_boundary=decimate_preserve_boundary,
            )

        # Optionally merge land and base into a single mesh for export
        merged_mesh = None
        if merge_land_and_base:
            try:
                meshes = mr.std_vector_std_shared_ptr_Mesh()
                if land is not None:
                    meshes.append(land)
                if base is not None:
                    meshes.append(base)
                if len(meshes) > 0:
                    merged_mesh = mr.mergeMeshes(meshes)
                    output.success("Merged land and base into a single mesh")
            except Exception as e:
                output.warning(f"Failed to merge meshes: {e}")

        # Create result dictionary
        result = {
            'elevation_data': elevation_data,  # keep original (non-flattened) for downstream consumers
            'land_mesh': land,
            'base_mesh': base,
            'merged_mesh': merged_mesh,
        }

        # Save to cache
        self._save_to_cache(
            bounds,
            topo_dir,
            base_height,
            elevation_multiplier,
            downsample_factor,
            water_threshold,
            decimate,
            decimate_max_error,
            decimate_target_face_count,
            split_at_water_level,
            result,
        )

        output.success("Terrain model generation complete!")

        return result
