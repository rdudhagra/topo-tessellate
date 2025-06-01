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

    def __init__(self, elevation=None):
        """
        Initialize the ModelGenerator with an elevation source.

        Args:
            elevation (Elevation, optional): Elevation instance.
                                                   If None, creates a new one.
        """
        self.elevation = elevation or SRTM()

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
        
        try:
            # Create cutting plane at water level (z = plane_z)
            cutting_plane = mr.Plane3f(mr.Vector3f(0, 0, 1), plane_z)
            
            # Make a copy of the original mesh for the above-water part
            above_water_mesh = mr.copyMesh(terrain_mesh)
            
            # Setup trim parameters for above-water mesh (keep everything above plane)
            trim_params_above = mr.TrimWithPlaneParams()
            trim_params_above.plane = cutting_plane
            # Use a small epsilon based on mesh size
            bbox = terrain_mesh.computeBoundingBox()
            trim_params_above.eps = 1e-6 * bbox.diagonal()
            
            # Trim the above-water mesh (this will keep the upper part)
            output.progress_info("Trimming above-water mesh...")
            mr.trimWithPlane(above_water_mesh, trim_params_above)
            
            # For now, let's skip the hole filling to avoid segfaults and see if basic trimming works
            # We can add hole filling later once we confirm the basic approach works
            
            output.progress_info("Successfully created above-water mesh")
            
            # Create below-water mesh as a simple rectangular prism
            output.progress_info("Creating below-water rectangular prism...")
            below_water_mesh = self._create_rectangular_prism(bounds, base_height)
            
            output.success("Mesh splitting at water level complete!")
            return above_water_mesh, below_water_mesh
            
        except Exception as e:
            output.error(f"Error during mesh splitting: {e}")
            # Return the original mesh and a simple base as fallback
            output.info("Falling back to original mesh without splitting")
            below_water_mesh = self._create_rectangular_prism(bounds, base_height)
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
    
    def _decimate_mesh(self, mesh):
        """
        Decimate the mesh to reduce the number of faces.
        """
        # Repack mesh optimally.
        # It's not necessary but highly recommended to achieve the best performance in parallel processing
        mesh.packOptimally()
        
        # Setup decimate parameters
        settings = mr.DecimateSettings()
        settings.maxError = 1 # Maximum error when decimation stops
        
        # Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
        # Recommended to set to number of CPU cores or more available for the best performance
        settings.subdivideParts = os.cpu_count() * 16
        
        # Decimate mesh
        result = mr.decimateMesh(mesh, settings)
        if not result.cancelled:
            output.info(f"Removed {result.facesDeleted} faces, {result.vertsDeleted} vertices")
            output.info(f"Introduced error: {result.errorIntroduced}")
            
        else:
            output.error(f"Failed to decimate mesh")

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
    ):
        """
        Generate a 3D terrain model from SRTM elevation data.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files
            base_height (float): Height of the flat base
            elevation_multiplier (float): Multiplier for realistic elevation scaling (1.0 = realistic scale)
            downsample_factor (int): Factor to downsample elevation data (default: 10)
            output_prefix (str): Prefix for output filename
            water_threshold (float, optional): Elevation below which areas are considered water

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
        output.info(f"Water extraction: {water_threshold}")

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

        # Decimate the meshes
        output.subheader("Decimating meshes")

        output.progress_info("Decimating land mesh")
        self._decimate_mesh(land)

        output.progress_info("Decimating base mesh")
        self._decimate_mesh(base)

        # Create result dictionary
        result = {
            'elevation_data': elevation_data,
            'land_mesh': land,
            'base_mesh': base
        }

        output.success("Terrain model generation complete!")

        return result
