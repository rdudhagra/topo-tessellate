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

# Handle both direct execution and module import
try:
    from .elevation import ElevationMap
except ImportError:
    from elevation import ElevationMap


class ModelGenerator:
    """
    A class for generating 3D models from elevation data using meshlib.
    
    This class integrates with the ElevationMap to create realistic 3D meshes
    from SRTM elevation data with proper surface normals and base structures.
    """
    
    def __init__(self, elevation_map=None):
        """
        Initialize the ModelGenerator.
        
        Args:
            elevation_map (ElevationMap, optional): ElevationMap instance. 
                                                   If None, creates a new one.
        """
        self.elevation_map = elevation_map or ElevationMap()
    
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
        
        print(f"Downsampled elevation data from {elevation_data.shape} to {downsampled.shape}")
        
        return downsampled
    
    def _detect_water_areas(self, elevation_data, water_threshold=None, min_water_area=100):
        """
        Detect water areas in elevation data based on elevation thresholds and area constraints.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            water_threshold (float, optional): Elevation below which areas are considered water.
                                             If None, uses percentile-based approach.
            min_water_area (int): Minimum number of connected pixels to be considered a water body
            
        Returns:
            numpy.ndarray: Boolean mask where True indicates water areas
        """
        from scipy import ndimage
        from skimage.measure import label
        
        # Determine water threshold
        if water_threshold is None:
            # Use a percentile-based approach: areas in the lowest 10% of elevations
            water_threshold = np.percentile(elevation_data, 10)
        
        # Initial water mask based on elevation
        water_mask = elevation_data <= water_threshold
        
        # Filter small water bodies based on connected component analysis
        labeled_water = label(water_mask, connectivity=2)
        
        # Count pixels in each component
        component_sizes = np.bincount(labeled_water.ravel())
        
        # Throw out `0` since this is the land
        component_sizes = component_sizes[1:]

        # Keep only components larger than minimum area
        large_components = np.where(component_sizes >= min_water_area)[0]

        # Increment the large components by 1 to account for the land being removed
        large_components += 1
        
        # Create final water mask
        final_water_mask = np.isin(labeled_water, large_components)
        
        water_pixel_count = np.sum(final_water_mask)
        total_pixels = elevation_data.size
        water_percentage = (water_pixel_count / total_pixels) * 100
        
        print(f"Detected {water_pixel_count} water pixels ({water_percentage:.1f}% of area)")
        
        return final_water_mask
    
    def _create_flat_water_surface(self, water_mask, bounds, water_level, base_height=1.0, elevation_multiplier=1.0):
        """
        Create a flat surface mesh for water areas.
        
        Args:
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            water_level (float): Elevation level for water surface
            base_height (float): Height of the base
            elevation_multiplier (float): Elevation scaling multiplier
            
        Returns:
            meshlib.mrmeshpy.Mesh: Flat water surface mesh
        """
        height, width = water_mask.shape
        
        # Calculate real-world dimensions
        width_meters, height_meters = self._calculate_bounds_dimensions_meters(bounds)
        scale_x = width_meters / width
        scale_y = height_meters / height
        
        # Convert water level to model units
        surface_z = water_level * elevation_multiplier + base_height
        
        vertices = []
        faces = []
        
        # Create vertex index map for water pixels
        vertex_indices = {}
        vertex_count = 0
        
        # Step 1: Create vertices for all water pixels
        for y in range(height):
            for x in range(width):
                if water_mask[y, x]:
                    px = x * scale_x
                    py = y * scale_y
                    
                    vertices.append([px, py, surface_z])
                    vertex_indices[(y, x)] = vertex_count
                    vertex_count += 1
        
        # Step 2: Create faces for water surface
        for y in range(height - 1):
            for x in range(width - 1):
                # Check all 4 corners of this quad
                corners = [(y, x), (y, x+1), (y+1, x), (y+1, x+1)]
                water_corners = [water_mask[cy, cx] for cy, cx in corners]
                
                if all(water_corners):
                    # All 4 corners are water - create 2 triangles
                    v0 = vertex_indices[(y, x)]      # bottom-left
                    v1 = vertex_indices[(y, x+1)]    # bottom-right
                    v2 = vertex_indices[(y+1, x)]    # top-left
                    v3 = vertex_indices[(y+1, x+1)]  # top-right
                    
                    # Counter-clockwise winding for upward normal
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        
        if not vertices:
            # Return empty mesh if no water found
            vertices = np.array([[0, 0, 0]], dtype=np.float32)
            faces = np.array([[0, 0, 0]], dtype=np.int32)
        else:
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
        
        # Create mesh using meshlib
        water_surface = mn.meshFromFacesVerts(faces, vertices)
        
        return water_surface
    
    def _extrude_water_surface(self, water_surface, water_depth, elevation_multiplier=1.0):
        """
        Extrude the water surface to create a volume with thickness.
        """
        # Create a new mesh for the water volume
        water_volume = mr.copyMesh(water_surface)

        # Extrude the water surface by the specified depth
        mr.addBaseToPlanarMesh(water_volume, water_depth * elevation_multiplier)

        return water_volume

    def _modify_elevation_for_water(self, elevation_data, water_mask, water_depth=2.0):
        """
        Modify elevation data directly by lowering water areas.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            water_depth (float): How much to lower water areas (meters)
            
        Returns:
            numpy.ndarray: Modified elevation data with lowered water areas
        """
        modified_elevation = elevation_data.copy()
        
        if np.any(water_mask):
            # Lower the elevation in water areas
            modified_elevation[water_mask] -= water_depth
            
            # Ensure water areas don't go below the minimum terrain elevation
            min_terrain_elevation = np.min(elevation_data[~water_mask]) if np.any(~water_mask) else elevation_data.min()
            modified_elevation[water_mask] = np.maximum(
                modified_elevation[water_mask], 
                min_terrain_elevation - water_depth
            )
            
            water_pixel_count = np.sum(water_mask)
            print(f"Lowered {water_pixel_count} water pixels by {water_depth}m")
        
        return modified_elevation

    def create_water_features(self, elevation_data, bounds, 
                            water_threshold=None, water_depth=2.0, 
                            base_height=1.0, elevation_multiplier=1.0,
                            min_water_area=100):
        """
        Process elevation data to extract water features and modify terrain elevation.
        This method detects water areas and returns modified elevation data and water information.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            water_threshold (float, optional): Elevation below which areas are considered water.
                                             If None, uses automatic detection (10th percentile)
            water_depth (float): How much to lower water areas below original elevation (meters)
            base_height (float): Height of the flat base
            elevation_multiplier (float): Elevation scaling multiplier
            min_water_area (int): Minimum number of connected pixels to be considered a water body
            output_prefix (str): Prefix for output filenames
            connect_water_regions (bool): Whether to connect nearby water regions
            max_connection_gap (int): Maximum gap to bridge between water regions
            
        Returns:
            tuple: (modified_elevation_data, water_surface_mesh, water_mask)
                - modified_elevation_data: Elevation data with lowered water areas
                - water_surface_mesh: Separate mesh representing water surfaces
                - water_mask: Boolean array indicating water locations
        """
        print("Processing water features from elevation data...")
        
        # Step 1: Detect water areas from elevation data
        print("Step 1: Detecting water areas...")
        water_mask = self._detect_water_areas(elevation_data, water_threshold, min_water_area)
        
        if not np.any(water_mask):
            print("No significant water areas detected.")
            return elevation_data, None, water_mask
        
        # Step 2: Modify elevation data directly by lowering water areas
        print("Step 2: Modifying elevation data for water areas...")
        modified_elevation = self._modify_elevation_for_water(elevation_data, water_mask, water_depth)
        
        # Step 3: Create water surface mesh for water areas (at original water level)
        print("Step 3: Creating water surface mesh...")
        water_level = np.mean(elevation_data[water_mask]) if np.any(water_mask) else 0
        water_surface = self._create_flat_water_surface(
            water_mask, bounds, water_level, base_height, elevation_multiplier
        )

        # Step 4: Thicken the water surface
        print("Step 4: Thicken the water surface...")
        water_volume = self._extrude_water_surface(water_surface, water_depth, elevation_multiplier)
        
        print("Water feature processing complete!")
        
        return modified_elevation, water_volume, water_mask
    
    def create_mesh_from_elevation(self, elevation_data, bounds, base_height=1.0, 
                                 elevation_multiplier=1.0):
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
        
        print(f"Creating mesh from {width}x{height} elevation grid")
        print(f"Real-world size: {width_meters/1000:.2f} x {height_meters/1000:.2f} km")
        print(f"Elevation range: {elevation_data.min():.1f} to {elevation_data.max():.1f} meters")
        
        # Create vertices in a structured way
        vertices = []
        
        # Add top surface vertices (elevation surface)
        for y in range(height):
            for x in range(width):
                point_x = x * scale_x
                point_y = y * scale_y
                # Scale elevation with realistic scaling multiplied by user multiplier
                point_z = (elevation_data[y, x] - elevation_data.min()) * elevation_scale + base_height
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
            top_left = x                    # Top surface, front edge
            top_right = x + 1               # Top surface, front edge  
            bottom_left = offset + x        # Bottom surface, front edge
            bottom_right = offset + x + 1   # Bottom surface, front edge
            
            # Two triangles connecting top front edge to bottom front edge
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])
        
        # Back edge (y=height-1)
        back_row_offset = (height - 1) * width
        for x in range(width - 1):
            top_left = back_row_offset + x              # Top surface, back edge
            top_right = back_row_offset + x + 1         # Top surface, back edge
            bottom_left = offset + back_row_offset + x  # Bottom surface, back edge  
            bottom_right = offset + back_row_offset + x + 1  # Bottom surface, back edge
            
            # Two triangles connecting top back edge to bottom back edge
            faces.append([top_left, top_right, bottom_left])
            faces.append([top_right, bottom_right, bottom_left])
        
        # Left edge (x=0)  
        for y in range(height - 1):
            top_left = y * width                    # Top surface, left edge
            top_right = (y + 1) * width             # Top surface, left edge
            bottom_left = offset + y * width        # Bottom surface, left edge
            bottom_right = offset + (y + 1) * width # Bottom surface, left edge
            
            # Two triangles connecting top left edge to bottom left edge
            faces.append([top_left, top_right, bottom_left])
            faces.append([top_right, bottom_right, bottom_left])
        
        # Right edge (x=width-1)
        for y in range(height - 1):
            top_left = y * width + (width - 1)              # Top surface, right edge
            top_right = (y + 1) * width + (width - 1)       # Top surface, right edge  
            bottom_left = offset + y * width + (width - 1)  # Bottom surface, right edge
            bottom_right = offset + (y + 1) * width + (width - 1)  # Bottom surface, right edge
            
            # Two triangles connecting top right edge to bottom right edge
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])
        
        print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
        
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
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")
    
    def generate_terrain_model(self, bounds, topo_dir="topo", base_height=1.0, 
                             elevation_multiplier=1.0,
                             downsample_factor=10, output_prefix="terrain_model", water_threshold=None, 
                             water_depth=2.0, min_water_area=100):
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
            water_depth (float): How much to lower water areas below original elevation (meters)
            min_water_area (int): Minimum number of connected pixels to be considered a water body
            connect_water_regions (bool): Whether to connect nearby water regions
            max_connection_gap (int): Maximum gap to bridge between water regions
            remove_small_components (bool): Whether to remove small disconnected components
            min_component_size (int): Minimum number of faces for a component to be kept
            
        Returns:
            dict: Dictionary containing generated meshes and data:
                - 'terrain_mesh': The main terrain mesh
                - 'water_mesh': Water surface mesh (if extract_water=True)
                - 'water_mask': Boolean array indicating water locations (if extract_water=True)
                - 'elevation_data': The elevation data used
        """
        print("=== Terrain Model Generation ===")
        print(f"Bounds: {bounds}")
        print(f"Water extraction: {water_threshold}")
        
        # Get elevation data from SRTM
        print("Loading elevation data...")
        elevation_data = self.elevation_map.get_elevation_data(bounds, topo_dir)
        
        print(f"Elevation data: {elevation_data.shape}, range: {elevation_data.min():.1f} to {elevation_data.max():.1f} m")
        
        # Downsample elevation data for faster processing
        if downsample_factor > 1:
            elevation_data = self._downsample_elevation_data(elevation_data, downsample_factor)
        
        # Extract water features
        print("Extracting water features...")
        modified_elevation, water_mesh, water_mask = self.create_water_features(
            elevation_data, bounds,
            water_threshold=water_threshold,
            water_depth=water_depth,
            base_height=base_height,
            elevation_multiplier=elevation_multiplier,
            min_water_area=min_water_area,
        )
        
        # Update result with water features
        print("Creating terrain mesh...")
        terrain = self.create_mesh_from_elevation(modified_elevation, bounds, base_height, elevation_multiplier)
        
        print("Terrain model generation complete!")

        return terrain, water_mesh


    def _connect_water_regions(self, water_mask, max_gap=2):
        """
        Connect nearby water regions to reduce disconnected components.
        
        Args:
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            max_gap (int): Maximum gap to bridge between water regions
            
        Returns:
            numpy.ndarray: Modified water mask with connected regions
        """
        from scipy.ndimage import binary_dilation
        from skimage.measure import label
        
        # Create a copy to work with
        connected_mask = water_mask.copy()
        
        # Dilate the water mask to bridge small gaps
        if max_gap > 0:
            # Use multiple iterations of dilation to bridge gaps
            dilated = binary_dilation(water_mask, iterations=max_gap)
            
            # Fill the dilated area but only connect existing water regions
            connected_mask = dilated
            
            # Count connected components before and after
            original_components = len(np.unique(label(water_mask))) - 1  # -1 for background
            new_components = len(np.unique(label(connected_mask))) - 1
            
            print(f"Connected water regions: {original_components} → {new_components} components")
        
        return connected_mask

    def save_mesh(self, mesh : mr.Mesh, filename : str):
        """
        Save the mesh to a file.
        """
        mr.saveMesh(mesh, filename)
