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
        
        print(f"Downsampled from {elevation_data.shape} to {downsampled.shape} "
              f"(factor: {downsample_factor})")
        
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
        
        print(f"Water detection threshold: {water_threshold:.2f} meters")
        
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
        
        print(f"Detected {water_pixel_count} water pixels ({water_percentage:.1f}% of total area)")
        print(f"Found {len(large_components)-1} significant water bodies")  # -1 for background
        
        return final_water_mask
    
    def _create_water_mesh(self, water_mask, bounds, water_level, water_depth, base_height=1.0, elevation_multiplier=1.0):
        """
        Create a thickened mesh for water areas with proper volume.
        
        Args:
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            water_level (float): Elevation level for water surface
            water_depth (float): Thickness/depth of water volume
            base_height (float): Height of the base
            elevation_multiplier (float): Elevation scaling multiplier
            
        Returns:
            meshlib.mrmeshpy.Mesh: Thickened water mesh with volume
        """
        height, width = water_mask.shape
        
        # Calculate real-world dimensions
        width_meters, height_meters = self._calculate_bounds_dimensions_meters(bounds)
        scale_x = width_meters / width
        scale_y = height_meters / height
        
        # Convert water levels to model units
        top_z = water_level * elevation_multiplier + base_height
        bottom_z = (water_level - water_depth) * elevation_multiplier + base_height
        
        vertices = []
        faces = []
        
        # Create vertex index maps for top and bottom surfaces
        top_vertex_indices = {}
        bottom_vertex_indices = {}
        
        # Step 1: Create vertices for all water pixels
        vertex_count = 0
        for y in range(height):
            for x in range(width):
                if water_mask[y, x]:
                    px = x * scale_x
                    py = y * scale_y
                    
                    # Top surface vertex
                    vertices.append([px, py, top_z])
                    top_vertex_indices[(y, x)] = vertex_count
                    vertex_count += 1
                    
                    # Bottom surface vertex  
                    vertices.append([px, py, bottom_z])
                    bottom_vertex_indices[(y, x)] = vertex_count
                    vertex_count += 1
        
        # Step 2: Create top surface faces
        for y in range(height - 1):
            for x in range(width - 1):
                # Check all 4 corners of this quad
                corners = [(y, x), (y, x+1), (y+1, x), (y+1, x+1)]
                water_corners = [water_mask[cy, cx] for cy, cx in corners]
                
                if all(water_corners):
                    # All 4 corners are water - create 2 triangles
                    v0 = top_vertex_indices[(y, x)]      # bottom-left
                    v1 = top_vertex_indices[(y, x+1)]    # bottom-right
                    v2 = top_vertex_indices[(y+1, x)]    # top-left
                    v3 = top_vertex_indices[(y+1, x+1)]  # top-right
                    
                    # Counter-clockwise winding for upward normal
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        
        # Step 3: Create bottom surface faces (flipped normals)
        for y in range(height - 1):
            for x in range(width - 1):
                corners = [(y, x), (y, x+1), (y+1, x), (y+1, x+1)]
                water_corners = [water_mask[cy, cx] for cy, cx in corners]
                
                if all(water_corners):
                    v0 = bottom_vertex_indices[(y, x)]      # bottom-left
                    v1 = bottom_vertex_indices[(y, x+1)]    # bottom-right
                    v2 = bottom_vertex_indices[(y+1, x)]    # top-left
                    v3 = bottom_vertex_indices[(y+1, x+1)]  # top-right
                    
                    # Clockwise winding for downward normal (when viewed from above)
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
        
        # Step 4: Create side walls by finding boundary edges
        # For each water pixel, check its 4 neighbors and create side walls where needed
        for y in range(height):
            for x in range(width):
                if water_mask[y, x]:
                    # Check each of the 4 directions for boundary edges
                    
                    # North edge (top of pixel)
                    if (y == 0 or not water_mask[y-1, x]) and x < width - 1 and water_mask[y, x+1]:
                        # This pixel and its right neighbor both have water, and there's a boundary to the north
                        tl = top_vertex_indices[(y, x)]
                        tr = top_vertex_indices[(y, x+1)]
                        bl = bottom_vertex_indices[(y, x)]
                        br = bottom_vertex_indices[(y, x+1)]
                        
                        # Create outward-facing quad (normal points north)
                        faces.append([tl, bl, tr])
                        faces.append([bl, br, tr])
                    
                    # South edge (bottom of pixel)
                    if (y == height - 1 or not water_mask[y+1, x]) and x < width - 1 and water_mask[y, x+1]:
                        # This pixel and its right neighbor both have water, and there's a boundary to the south
                        tl = top_vertex_indices[(y, x)]
                        tr = top_vertex_indices[(y, x+1)]
                        bl = bottom_vertex_indices[(y, x)]
                        br = bottom_vertex_indices[(y, x+1)]
                        
                        # Create outward-facing quad (normal points south)
                        faces.append([tl, tr, bl])
                        faces.append([bl, tr, br])
                    
                    # West edge (left of pixel)
                    if (x == 0 or not water_mask[y, x-1]) and y < height - 1 and water_mask[y+1, x]:
                        # This pixel and its bottom neighbor both have water, and there's a boundary to the west
                        tb = top_vertex_indices[(y, x)]
                        tt = top_vertex_indices[(y+1, x)]
                        bb = bottom_vertex_indices[(y, x)]
                        bt = bottom_vertex_indices[(y+1, x)]
                        
                        # Create outward-facing quad (normal points west)
                        faces.append([tb, tt, bb])
                        faces.append([bb, tt, bt])
                    
                    # East edge (right of pixel)
                    if (x == width - 1 or not water_mask[y, x+1]) and y < height - 1 and water_mask[y+1, x]:
                        # This pixel and its bottom neighbor both have water, and there's a boundary to the east
                        tb = top_vertex_indices[(y, x)]
                        tt = top_vertex_indices[(y+1, x)]
                        bb = bottom_vertex_indices[(y, x)]
                        bt = bottom_vertex_indices[(y+1, x)]
                        
                        # Create outward-facing quad (normal points east)
                        faces.append([tb, bb, tt])
                        faces.append([bb, bt, tt])
        
        if not vertices:
            print("Warning: No water vertices found, creating empty water mesh")
            # Return empty mesh
            vertices = np.array([[0, 0, 0]], dtype=np.float32)
            faces = np.array([[0, 0, 0]], dtype=np.int32)
        else:
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
        
        print(f"Thickened water mesh: {len(vertices)} vertices, {len(faces)} faces")
        print(f"Water depth: {water_depth:.2f} meters")
        print(f"Water volume: top at {top_z:.2f}, bottom at {bottom_z:.2f}")
        
        # Create mesh using meshlib
        water_mesh = mn.meshFromFacesVerts(faces, vertices)
        
        return water_mesh
    
    def _lower_terrain_at_water(self, mesh, elevation_data, water_mask, bounds, 
                               water_depth=2.0, base_height=1.0, elevation_multiplier=1.0):
        """
        Lower the terrain mesh at water locations using meshlib functions.
        
        Args:
            mesh (meshlib.mrmeshpy.Mesh): The terrain mesh to modify
            elevation_data (numpy.ndarray): Original elevation data
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            bounds (tuple): Geographic bounds
            water_depth (float): How much to lower water areas (in meters)
            base_height (float): Base height of the mesh
            elevation_multiplier (float): Elevation scaling multiplier
            
        Returns:
            meshlib.mrmeshpy.Mesh: Modified mesh with lowered water areas
        """
        height, width = water_mask.shape
        
        # Calculate scaling factors
        width_meters, height_meters = self._calculate_bounds_dimensions_meters(bounds)
        scale_x = width_meters / width
        scale_y = height_meters / height
        
        # Get mesh points as numpy array
        points = mn.getNumpyVerts(mesh)
        
        print(f"Modifying {len(points)} mesh vertices for water areas")
        
        # Modify vertices that fall within water areas
        modified_count = 0
        for i, point in enumerate(points):
            # Convert mesh coordinates back to grid coordinates
            grid_x = int(point[0] / scale_x)
            grid_y = int(point[1] / scale_y)
            
            # Check bounds
            if 0 <= grid_x < width and 0 <= grid_y < height:
                if water_mask[grid_y, grid_x]:
                    # Lower this vertex
                    current_elevation = (point[2] - base_height) / elevation_multiplier
                    new_elevation = current_elevation - water_depth
                    points[i][2] = new_elevation * elevation_multiplier + base_height
                    modified_count += 1
        
        print(f"Modified {modified_count} vertices for water depression")
        
        # Create new mesh with modified vertices
        faces = mn.getNumpyFaces(mesh.topology)
        modified_mesh = mn.meshFromFacesVerts(faces, points)
        
        return modified_mesh
    
    def create_water_features(self, mesh, elevation_data, bounds, 
                            water_threshold=None, water_depth=2.0, 
                            base_height=1.0, elevation_multiplier=1.0,
                            min_water_area=100, output_prefix="terrain"):
        """
        Extract water regions from the mesh and create separate water objects.
        
        This function identifies water areas from elevation data, lowers those areas 
        in the terrain mesh, and creates a separate water mesh to fill the cavities.
        
        Args:
            mesh (meshlib.mrmeshpy.Mesh): The terrain mesh to process
            elevation_data (numpy.ndarray): 2D array of elevation values used to create the mesh
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            water_threshold (float, optional): Elevation below which areas are considered water.
                                             If None, uses automatic detection (10th percentile)
            water_depth (float): How much to lower water areas below original elevation (meters)
            base_height (float): Height of the flat base
            elevation_multiplier (float): Elevation scaling multiplier used in original mesh
            min_water_area (int): Minimum number of connected pixels to be considered a water body
            output_prefix (str): Prefix for output filenames
            
        Returns:
            tuple: (modified_terrain_mesh, water_mesh, water_mask)
                - modified_terrain_mesh: Terrain mesh with lowered water areas
                - water_mesh: Separate mesh representing water surfaces
                - water_mask: Boolean array indicating water locations
        """
        print("\n=== Water Feature Extraction ===")
        
        # Step 1: Detect water areas from elevation data
        print("Step 1: Detecting water areas...")
        water_mask = self._detect_water_areas(elevation_data, water_threshold, min_water_area)
        
        if not np.any(water_mask):
            print("No significant water areas detected.")
            return mesh, None, water_mask
        
        # Step 2: Lower terrain at water locations
        print("Step 2: Lowering terrain at water locations...")
        modified_terrain = self._lower_terrain_at_water(
            mesh, elevation_data, water_mask, bounds, 
            water_depth, base_height, elevation_multiplier
        )
        
        # Step 3: Create water surface mesh
        print("Step 3: Creating water surface mesh...")
        water_level = np.mean(elevation_data[water_mask]) if np.any(water_mask) else 0
        water_mesh = self._create_water_mesh(
            water_mask, bounds, water_level, water_depth, 
            base_height, elevation_multiplier
        )
        
        # Step 4: Save results
        print("Step 4: Saving water feature results...")
        
        # Save modified terrain
        terrain_file = f"{output_prefix}_with_water.obj"
        try:
            self.save_mesh(modified_terrain, terrain_file)
        except Exception as e:
            print(f"Failed to save terrain with water: {e}")
        
        # Save water mesh if it exists and has vertices
        if water_mesh is not None and water_mesh.points.size() > 1:
            water_file = f"{output_prefix}_water.obj"
            try:
                self.save_mesh(water_mesh, water_file)
            except Exception as e:
                print(f"Failed to save water mesh: {e}")
        
        print(f"\n✓ Water feature extraction complete!")
        print(f"Generated files:")
        print(f"  - Modified terrain: {terrain_file}")
        if water_mesh is not None and water_mesh.points.size() > 1:
            print(f"  - Water surface: {water_file}")
        
        return modified_terrain, water_mesh, water_mask
    
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
        elevation_range = elevation_data.max() - elevation_data.min()
        elevation_scale = elevation_multiplier  # Direct multiplier of realistic scale
            
        print(f"Grid dimensions: {width}x{height}")
        print(f"Real-world dimensions: {width_meters/1000:.2f} km x {height_meters/1000:.2f} km")
        print(f"Grid scale: {scale_x:.2f} m/unit (x), {scale_y:.2f} m/unit (y)")
        print(f"Elevation range: {elevation_data.min():.1f} to {elevation_data.max():.1f} meters")
        print(f"Elevation scaling: {elevation_scale:.6f} (multiplier: {elevation_multiplier})")
        print(f"Max elevation in model: {elevation_range * elevation_scale:.1f} units")
        
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
        
        print(f"Created {len(vertices)} vertices")
        
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
        
        print(f"Created {len(faces)} faces")
        
        # Convert to numpy arrays
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32)
        
        print(f"Vertices array shape: {vertices_array.shape}")
        print(f"Faces array shape: {faces_array.shape}")
        
        # Create the mesh using meshlib mrmeshnumpy
        mesh = mn.meshFromFacesVerts(faces_array, vertices_array)
        
        print(f"Final mesh has {mesh.points.size()} vertices and {mesh.topology.numValidFaces()} faces")
        
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
            print(f"Mesh saved to: {filename}")
        except Exception as e:
            print(f"Failed to save mesh to: {filename} - Error: {e}")
    
    def generate_terrain_model(self, bounds, topo_dir="topo", base_height=1.0, 
                             elevation_multiplier=1.0,
                             downsample_factor=10, output_prefix="terrain_model",
                             extract_water=False, water_threshold=None, 
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
            extract_water (bool): Whether to extract water features from the terrain
            water_threshold (float, optional): Elevation below which areas are considered water
            water_depth (float): How much to lower water areas below original elevation (meters)
            min_water_area (int): Minimum number of connected pixels to be considered a water body
            
        Returns:
            dict: Dictionary containing generated meshes and data:
                - 'terrain_mesh': The main terrain mesh
                - 'water_mesh': Water surface mesh (if extract_water=True)
                - 'water_mask': Boolean array indicating water locations (if extract_water=True)
                - 'elevation_data': The elevation data used
        """
        print("=== Terrain Model Generation ===")
        print(f"Bounds: {bounds}")
        print(f"Base height: {base_height}")
        print(f"Elevation multiplier: {elevation_multiplier}")
        print(f"Downsample factor: {downsample_factor}")
        print(f"Water extraction: {extract_water}")
        
        # Get elevation data from SRTM
        print("Loading elevation data from SRTM...")
        elevation_data = self.elevation_map.get_elevation_data(bounds, topo_dir)
        
        print(f"Original elevation data shape: {elevation_data.shape}")
        print(f"Original elevation range: {elevation_data.min():.3f} to {elevation_data.max():.3f} meters")
        
        # Downsample elevation data for faster processing
        if downsample_factor > 1:
            elevation_data = self._downsample_elevation_data(elevation_data, downsample_factor)
        
        print(f"Final elevation data shape: {elevation_data.shape}")
        print(f"Final elevation range: {elevation_data.min():.3f} to {elevation_data.max():.3f} meters")
        
        # Create 3D mesh from elevation data
        print("Creating 3D mesh from elevation data...")
        mesh = self.create_mesh_from_elevation(elevation_data, bounds, base_height, elevation_multiplier)
        
        # Initialize result dictionary
        result = {
            'terrain_mesh': mesh,
            'elevation_data': elevation_data,
            'water_mesh': None,
            'water_mask': None
        }
        
        # Extract water features if requested
        if extract_water:
            print("Extracting water features...")
            modified_terrain, water_mesh, water_mask = self.create_water_features(
                mesh, elevation_data, bounds,
                water_threshold=water_threshold,
                water_depth=water_depth,
                base_height=base_height,
                elevation_multiplier=elevation_multiplier,
                min_water_area=min_water_area,
                output_prefix=output_prefix
            )
            
            # Update result with water features
            result['terrain_mesh'] = modified_terrain
            result['water_mesh'] = water_mesh
            result['water_mask'] = water_mask
        else:
            # Save the standard mesh
            output_file = f"{output_prefix}.obj"
            print("Saving mesh file...")
            try:
                self.save_mesh(mesh, output_file)
            except Exception as e:
                print(f"Failed to save {output_file}: {e}")
        
        print(f"\n✓ Terrain model generation complete!")
        if not extract_water:
            print(f"Generated file: {output_prefix}.obj")
        
        return result


def main():
    """
    Example usage of the ModelGenerator class.
    """
    # Create model generator
    generator = ModelGenerator()
    
    # Example 1: Generate standard terrain model
    print("=== Example 1: Standard Terrain Generation ===")
    bounds = (-122.5, 37.7, -122.4, 37.8)  # San Francisco area
    standard_result = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir="topo",
        base_height=10.0,           # 10 unit base height
        elevation_multiplier=1.0,   # Default elevation multiplier
        downsample_factor=10,       # Default downsampling
        output_prefix="sf_terrain_standard",
        extract_water=False         # Standard generation without water
    )
    
    # Example 2: Generate terrain with water features
    print("\n=== Example 2: Terrain with Water Feature Extraction ===")
    # Use a coastal area that's likely to have water features
    bounds_coastal = (-122.6, 37.6, -122.3, 37.9)  # Larger SF Bay area
    water_result = generator.generate_terrain_model(
        bounds=bounds_coastal,
        topo_dir="topo",
        base_height=5.0,
        elevation_multiplier=2.0,   # Exaggerate elevation for better visualization
        downsample_factor=8,        # Slightly higher resolution for water detection
        output_prefix="sf_bay_area_water",
        extract_water=True,         # Enable water extraction
        water_threshold=5.0,        # Areas below 5 meters considered water
        water_depth=3.0,            # Lower water areas by 3 meters
        min_water_area=50           # Minimum water body size
    )
    
    # Example 3: Automatic water detection (no threshold specified)
    print("\n=== Example 3: Automatic Water Detection ===")
    # Example for a lake area or river delta
    bounds_lake = (-122.55, 37.75, -122.45, 37.85)  # Central SF area
    auto_water_result = generator.generate_terrain_model(
        bounds=bounds_lake,
        topo_dir="topo",
        base_height=2.0,
        elevation_multiplier=1.5,
        downsample_factor=12,
        output_prefix="sf_auto_water",
        extract_water=True,
        water_threshold=None,       # Automatic detection using percentiles
        water_depth=2.0,
        min_water_area=25
    )
    
    # Report results
    print("\n=== Generation Summary ===")
    print(f"Standard terrain mesh: {standard_result['terrain_mesh'].points.size()} vertices")
    
    if water_result['water_mesh'] is not None:
        print(f"Bay area terrain mesh: {water_result['terrain_mesh'].points.size()} vertices")
        print(f"Bay area water mesh: {water_result['water_mesh'].points.size()} vertices")
        water_pixels = np.sum(water_result['water_mask'])
        total_pixels = water_result['water_mask'].size
        print(f"Water coverage: {water_pixels}/{total_pixels} pixels ({100*water_pixels/total_pixels:.1f}%)")
    
    if auto_water_result['water_mesh'] is not None:
        print(f"Auto-detect terrain mesh: {auto_water_result['terrain_mesh'].points.size()} vertices")
        print(f"Auto-detect water mesh: {auto_water_result['water_mesh'].points.size()} vertices")
        water_pixels_auto = np.sum(auto_water_result['water_mask'])
        total_pixels_auto = auto_water_result['water_mask'].size
        print(f"Auto water coverage: {water_pixels_auto}/{total_pixels_auto} pixels ({100*water_pixels_auto/total_pixels_auto:.1f}%)")
    
    print("\nAll examples completed successfully!")
    print("\nGenerated files:")
    print("- sf_terrain_standard.obj (standard terrain)")
    print("- sf_bay_area_water_with_water.obj (modified terrain)")  
    print("- sf_bay_area_water_water.obj (water surface)")
    print("- sf_bay_area_water_water_mask.npy (water mask)")
    print("- sf_auto_water_with_water.obj (auto-detect terrain)")
    print("- sf_auto_water_water.obj (auto-detect water)")
    print("- sf_auto_water_water_mask.npy (auto-detect water mask)")


if __name__ == "__main__":
    main() 