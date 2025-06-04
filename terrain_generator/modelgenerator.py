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
                           downsample_factor: int, water_threshold: Optional[float]) -> str:
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
        
        # Create hash of parameters for shorter filename
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        # Create readable filename with key parameters
        bounds_str = f"{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        filename = f"terrain_{params_hash}_{bounds_str}_ds{downsample_factor}_wt{water_threshold}.pkl.gz"
        
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

        cache_file = self._get_cache_filename(bounds, topo_dir, base_height, 
                                            elevation_multiplier, downsample_factor, 
                                            water_threshold)

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
                         downsample_factor: int, water_threshold: Optional[float]) -> Optional[Dict]:
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

        cache_file = self._get_cache_filename(bounds, topo_dir, base_height, 
                                            elevation_multiplier, downsample_factor, 
                                            water_threshold)

        if not self._is_cache_valid(cache_file):
            return None

        try:
            with gzip.open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify the parameters match
            if (cache_data.get("bounds") != bounds or 
                cache_data.get("topo_dir") != topo_dir or
                cache_data.get("base_height") != base_height or
                cache_data.get("elevation_multiplier") != elevation_multiplier or
                cache_data.get("downsample_factor") != downsample_factor or
                cache_data.get("water_threshold") != water_threshold):
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

    def _calculate_bounds_dimensions_meters(self, bounds):
        """
        Calculate the real-world dimensions of geographic bounds in meters using geopy.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat)

        Returns:
            tuple: (width_meters, height_meters)
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Calculate width (longitude difference) in meters
        # Use center latitude for calculation
        center_lat = (min_lat + max_lat) / 2
        width_meters = geodesic((center_lat, min_lon), (center_lat, max_lon)).meters

        # Calculate height (latitude difference) in meters
        # Use center longitude for calculation
        center_lon = (min_lon + max_lon) / 2
        height_meters = geodesic((min_lat, center_lon), (max_lat, center_lon)).meters

        return width_meters, height_meters

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
        width_meters, height_meters = self._calculate_bounds_dimensions_meters(bounds)
        
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
        self, elevation_data, bounds, base_height=1.0, elevation_multiplier=1.0
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
        width_meters, height_meters = self._calculate_bounds_dimensions_meters(bounds)

        # Calculate scale factor to fit real-world dimensions to grid
        # This makes 1 grid unit = actual meters in the real world
        scale_x = width_meters / width
        scale_y = height_meters / height

        # For realistic elevation scaling: 1 meter elevation = 1 meter in model
        # The elevation_multiplier allows scaling from this realistic baseline
        elevation_scale = elevation_multiplier  # Direct multiplier of realistic scale

        output.progress_info(f"Creating mesh from {width}x{height} elevation grid")
        output.info(f"Real-world size: {width_meters/1000:.2f} x {height_meters/1000:.2f} km")
        output.info(
            f"Elevation range: {elevation_data.min():.1f} to {elevation_data.max():.1f} meters"
        )

        # Create vertices in a structured way
        vertices = []

        # Add top surface vertices (elevation surface)
        for y in range(height):
            for x in range(width):
                point_x = x * scale_x
                point_y = y * scale_y
                # Scale elevation with realistic scaling multiplied by user multiplier
                point_z = elevation_data[y, x] * elevation_scale + base_height
                vertices.append([point_x, point_y, point_z])

        # Add bottom surface vertices (flat base)
        for y in range(height):
            for x in range(width):
                point_x = x * scale_x
                point_y = y * scale_y
                point_z = 0.0  # Flat base at z=0
                vertices.append([point_x, point_y, point_z])

        # Create faces manually for structured grid
        faces = []

        # Top surface faces (elevation surface)
        for y in range(height - 1):
            for x in range(width - 1):
                # Get vertex indices for current quad on top surface
                top_left = y * width + x
                top_right = y * width + (x + 1)
                bottom_left = (y + 1) * width + x
                bottom_right = (y + 1) * width + (x + 1)

                # Create two triangles for each quad (counter-clockwise for upward normals)
                # Triangle 1: top_left -> top_right -> bottom_left
                faces.append([top_left, top_right, bottom_left])
                # Triangle 2: top_right -> bottom_right -> bottom_left
                faces.append([top_right, bottom_right, bottom_left])

        # Bottom surface faces (flat base) - facing downward
        offset = width * height  # Offset to bottom vertices
        for y in range(height - 1):
            for x in range(width - 1):
                # Get vertex indices for current quad on bottom surface
                top_left = offset + y * width + x
                top_right = offset + y * width + (x + 1)
                bottom_left = offset + (y + 1) * width + x
                bottom_right = offset + (y + 1) * width + (x + 1)

                # Create two triangles for each quad (clockwise for downward normals)
                # Triangle 1: top_left -> bottom_left -> top_right
                faces.append([top_left, bottom_left, top_right])
                # Triangle 2: top_right -> bottom_left -> bottom_right
                faces.append([top_right, bottom_left, bottom_right])

        # Side walls connecting top and bottom surfaces

        # Front edge (y=0)
        for x in range(width - 1):
            top_left = x  # Top surface, front edge
            top_right = x + 1  # Top surface, front edge
            bottom_left = offset + x  # Bottom surface, front edge
            bottom_right = offset + x + 1  # Bottom surface, front edge

            # Two triangles connecting top front edge to bottom front edge
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])

        # Back edge (y=height-1)
        back_row_offset = (height - 1) * width
        for x in range(width - 1):
            top_left = back_row_offset + x  # Top surface, back edge
            top_right = back_row_offset + x + 1  # Top surface, back edge
            bottom_left = offset + back_row_offset + x  # Bottom surface, back edge
            bottom_right = offset + back_row_offset + x + 1  # Bottom surface, back edge

            # Two triangles connecting top back edge to bottom back edge
            faces.append([top_left, top_right, bottom_left])
            faces.append([top_right, bottom_right, bottom_left])

        # Left edge (x=0)
        for y in range(height - 1):
            top_left = y * width  # Top surface, left edge
            top_right = (y + 1) * width  # Top surface, left edge
            bottom_left = offset + y * width  # Bottom surface, left edge
            bottom_right = offset + (y + 1) * width  # Bottom surface, left edge

            # Two triangles connecting top left edge to bottom left edge
            faces.append([top_left, top_right, bottom_left])
            faces.append([top_right, bottom_right, bottom_left])

        # Right edge (x=width-1)
        for y in range(height - 1):
            top_left = y * width + (width - 1)  # Top surface, right edge
            top_right = (y + 1) * width + (width - 1)  # Top surface, right edge
            bottom_left = offset + y * width + (width - 1)  # Bottom surface, right edge
            bottom_right = (
                offset + (y + 1) * width + (width - 1)
            )  # Bottom surface, right edge

            # Two triangles connecting top right edge to bottom right edge
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])

        output.success(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")

        # Convert to numpy arrays
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32)

        # Create the mesh using meshlib mrmeshnumpy
        mesh = mn.meshFromFacesVerts(faces_array, vertices_array)

        return mesh
    
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
            cached_result = self._load_from_cache(bounds, topo_dir, base_height, 
                                                elevation_multiplier, downsample_factor, 
                                                water_threshold)
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

        # Flatten the water level to the lowest elevation
        output.subheader("Flattening water level")
        terrain = self._flatten_water_level(elevation_data, water_threshold)

        # Create terrain mesh
        output.subheader("Creating terrain mesh")
        terrain = self._create_mesh_from_elevation(
            elevation_data, bounds, base_height, elevation_multiplier
        )

        # Split the terrain mesh at water level
        output.subheader("Splitting terrain mesh at water level")

        output.progress_info(f"Splitting at water level: {water_threshold:.2f} meters")

        land, base = self._split_mesh_at_water_level(
            terrain, water_threshold, bounds, base_height, elevation_multiplier
        )

        # Create result dictionary
        result = {
            'elevation_data': elevation_data,
            'land_mesh': land,
            'base_mesh': base
        }

        # Save to cache
        self._save_to_cache(bounds, topo_dir, base_height, elevation_multiplier,
                           downsample_factor, water_threshold, result)

        output.success("Terrain model generation complete!")

        return result
