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
        Extrude the flat water surface to create a volume with thickness.
        
        Args:
            water_surface (meshlib.mrmeshpy.Mesh): Flat water surface mesh
            water_depth (float): Thickness/depth of water volume
            elevation_multiplier (float): Elevation scaling multiplier
            
        Returns:
            meshlib.mrmeshpy.Mesh: Extruded water volume mesh
        """
        if water_surface.points.size() <= 1:
            return water_surface
        
        # Get vertices and faces from the surface mesh
        surface_vertices = mn.getNumpyVerts(water_surface)
        surface_faces = mn.getNumpyFaces(water_surface.topology)
        
        num_surface_verts = len(surface_vertices)
        
        # Create vertices for the extruded volume
        vertices = []
        faces = []
        
        # Add top surface vertices (original)
        for vert in surface_vertices:
            vertices.append([vert[0], vert[1], vert[2]])
        
        # Add bottom surface vertices (extruded down)
        for vert in surface_vertices:
            bottom_z = vert[2] - (water_depth * elevation_multiplier)
            vertices.append([vert[0], vert[1], bottom_z])
        
        # Add top surface faces (original orientation)
        for face in surface_faces:
            faces.append([face[0], face[1], face[2]])
        
        # Add bottom surface faces (flipped orientation)
        for face in surface_faces:
            bottom_face = [
                face[0] + num_surface_verts,
                face[2] + num_surface_verts,  # Flipped
                face[1] + num_surface_verts   # Flipped
            ]
            faces.append(bottom_face)
        
        # Create side walls by finding boundary edges
        boundary_edges = self._find_boundary_edges(surface_faces, num_surface_verts)
        
        # Create side wall faces for each boundary edge
        for edge in boundary_edges:
            v1_top, v2_top = edge
            v1_bottom = v1_top + num_surface_verts
            v2_bottom = v2_top + num_surface_verts
            
            # Create two triangles for the side wall
            faces.append([v1_top, v1_bottom, v2_top])
            faces.append([v2_top, v1_bottom, v2_bottom])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        # Create extruded water volume mesh
        water_volume = mn.meshFromFacesVerts(faces, vertices)
        
        return water_volume

    def _find_boundary_edges(self, faces, num_verts):
        """
        Find boundary edges of a mesh (edges that belong to only one face).
        
        Args:
            faces (numpy.ndarray): Face indices
            num_verts (int): Number of vertices
            
        Returns:
            list: List of boundary edge tuples (v1, v2)
        """
        # Count occurrences of each edge
        edge_count = {}
        
        for face in faces:
            edges = [
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[2], face[0]), max(face[2], face[0]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Find edges that appear only once (boundary edges)
        boundary_edges = []
        for edge, count in edge_count.items():
            if count == 1:
                boundary_edges.append(edge)
        
        return boundary_edges

    def _subtract_water_from_terrain(self, terrain_mesh, water_volume):
        """
        Subtract the water volume from the terrain mesh using boolean operations.
        
        Args:
            terrain_mesh (meshlib.mrmeshpy.Mesh): The terrain mesh
            water_volume (meshlib.mrmeshpy.Mesh): The water volume mesh
            
        Returns:
            meshlib.mrmeshpy.Mesh: Modified terrain mesh with water subtracted
        """
        if water_volume.points.size() <= 1:
            return terrain_mesh
        
        try:
            # Perform boolean difference operation (terrain - water)
            boolean_result = mr.boolean(
                terrain_mesh, 
                water_volume, 
                mr.BooleanOperation.DifferenceAB
            )
            
            if boolean_result.valid():
                print("Successfully subtracted water volume from terrain")
                return boolean_result.mesh
            else:
                print(f"Boolean operation failed: {boolean_result.errorString}")
                return terrain_mesh
                
        except Exception as e:
            print(f"Failed to perform boolean subtraction: {e}")
            return terrain_mesh

    def _create_water_mesh_new_process(self, water_mask, bounds, water_level, water_depth, 
                                     base_height=1.0, elevation_multiplier=1.0):
        """
        Create water mesh using the new 4-step process:
        1. Use detect_water_areas (already done)
        2. Create flat surface for water areas
        3. Extrude this surface to have thickness
        4. The extruded volume will be used for boolean subtraction
        
        Args:
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for real-world scaling
            water_level (float): Elevation level for water surface
            water_depth (float): Thickness/depth of water volume
            base_height (float): Height of the base
            elevation_multiplier (float): Elevation scaling multiplier
            
        Returns:
            tuple: (water_surface_mesh, water_volume_mesh)
                - water_surface_mesh: The flat water surface
                - water_volume_mesh: The extruded water volume for boolean ops
        """
        print("Creating water mesh using new 4-step process...")
        
        # Step 2: Create flat surface for water areas
        print("Step 2: Creating flat water surface...")
        water_surface = self._create_flat_water_surface(
            water_mask, bounds, water_level, base_height, elevation_multiplier
        )
        
        # Step 3: Extrude this surface to have thickness
        print("Step 3: Extruding water surface to create volume...")
        water_volume = self._extrude_water_surface(
            water_surface, water_depth, elevation_multiplier
        )
        
        print("Water mesh creation complete")
        return water_surface, water_volume

    def _remove_small_components(self, mesh, min_component_size=100):
        """
        Remove disconnected components that are smaller than the specified threshold.
        This helps eliminate free-floating faces and small isolated pieces.
        
        Args:
            mesh (meshlib.mrmeshpy.Mesh): The mesh to clean
            min_component_size (int): Minimum number of faces for a component to be kept
            
        Returns:
            meshlib.mrmeshpy.Mesh: Cleaned mesh with small components removed
        """
        try:
            # Get all connected components
            components = mr.getAllComponents(mesh)
            
            if len(components) <= 1:
                return mesh  # Single component, nothing to remove
            
            # Find the largest component (main terrain)
            component_sizes = [comp.count() for comp in components]
            largest_idx = np.argmax(component_sizes)
            
            # Keep components that are either the largest or above the minimum size
            faces_to_keep = mr.FaceBitSet()
            components_kept = 0
            
            for i, component in enumerate(components):
                component_size = component.count()
                if i == largest_idx or component_size >= min_component_size:
                    faces_to_keep |= component
                    components_kept += 1
            
            # If we're removing components, create a new mesh with only the kept faces
            if components_kept < len(components):
                print(f"Removing {len(components) - components_kept} small disconnected components")
                
                # Extract submesh with only the faces we want to keep
                cleaned_mesh = mesh.clone()
                faces_to_delete = mr.FaceBitSet()
                faces_to_delete.resize(mesh.topology.lastValidFace() + 1, True)
                faces_to_delete ^= faces_to_keep  # XOR to get faces NOT in faces_to_keep
                
                mr.deleteFaces(cleaned_mesh, faces_to_delete)
                
                # Pack the mesh to remove unused vertices
                mr.packMesh(cleaned_mesh)
                
                return cleaned_mesh
            else:
                return mesh
                
        except Exception as e:
            print(f"Warning: Failed to remove small components: {e}")
            return mesh
    
    def create_water_features(self, mesh, elevation_data, bounds, 
                            water_threshold=None, water_depth=2.0, 
                            base_height=1.0, elevation_multiplier=1.0,
                            min_water_area=100, output_prefix="terrain",
                            connect_water_regions=False, max_connection_gap=2,
                            remove_small_components=True, min_component_size=100):
        """
        Extract water regions from the mesh using the new 4-step process:
        1. Use detect_water_areas to find water areas
        2. Create a flat surface for where the water should be
        3. Extrude this water surface to have thickness
        4. Subtract this mesh from the land mesh using boolean operations
        
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
            connect_water_regions (bool): Whether to connect nearby water regions
            max_connection_gap (int): Maximum gap to bridge between water regions
            remove_small_components (bool): Whether to remove small disconnected components
            min_component_size (int): Minimum number of faces for a component to be kept
            
        Returns:
            tuple: (modified_terrain_mesh, water_surface_mesh, water_mask)
                - modified_terrain_mesh: Terrain mesh with water volume subtracted
                - water_surface_mesh: Separate mesh representing water surfaces
                - water_mask: Boolean array indicating water locations
        """
        print("Extracting water features using new 4-step process...")
        
        # Step 1: Detect water areas from elevation data
        print("Step 1: Detecting water areas...")
        water_mask = self._detect_water_areas(elevation_data, water_threshold, min_water_area)
        
        if not np.any(water_mask):
            print("No significant water areas detected.")
            return mesh, None, water_mask
        
        # Step 1a: Optionally connect nearby water regions
        if connect_water_regions:
            water_mask = self._connect_water_regions(water_mask, max_connection_gap)
        
        # Steps 2-3: Create flat surface and extrude to create water volume
        water_level = np.mean(elevation_data[water_mask]) if np.any(water_mask) else 0
        water_surface, water_volume = self._create_water_mesh_new_process(
            water_mask, bounds, water_level, water_depth, 
            base_height, elevation_multiplier
        )
        # # Repair water mesh
        # water_surface = mr.rebuildMesh(water_surface, mr.RebuildMeshSettings(
        #     preSubdivide=True,
        #     voxelSize=20,
        #     signMode=mr.SignDetectionModeShort.Auto,
        #     closeHolesInHoleWindingNumber=True,
        #     offsetMode=mr.OffsetMode.Smooth,
        #     outSharpEdges = mr.UndirectedEdgeBitSet(),
        #     windingNumberThreshold=0.5,
        #     windingNumberBeta=1,
        #     fwn=mr.FastWindingNumber(water_surface),
        #     decimate=False,
        #     tinyEdgeLength=10,
        #     progress=mr.func_bool_from_float(lambda _: True),
        #     onSignDetectionModeSelected=mr.func_void_from_SignDetectionMode(lambda x: print(f"Sign detection mode selected: {x}")),
        # ))
        
        # Step 4: Subtract water volume from terrain mesh
        print("Step 4: Subtracting water volume from terrain...")
        modified_terrain = self._subtract_water_from_terrain(mesh, water_volume)
        
        # Remove small disconnected components from terrain if requested
        if remove_small_components:
            modified_terrain = self._remove_small_components(modified_terrain, min_component_size)
        
        # Save results
        terrain_file = f"{output_prefix}_with_water.obj"
        try:
            self.save_mesh(modified_terrain, terrain_file)
        except Exception as e:
            print(f"Failed to save terrain: {e}")
        
        # Save water mesh if it exists and has vertices
        if water_surface is not None and water_surface.points.size() > 1:
            water_file = f"{output_prefix}_water.obj"
            try:
                self.save_mesh(water_surface, water_file)
            except Exception as e:
                print(f"Failed to save water mesh: {e}")
        
        # Optionally save water volume for debugging
        if water_volume is not None and water_volume.points.size() > 1:
            volume_file = f"{output_prefix}_water_volume.obj"
            try:
                self.save_mesh(water_volume, volume_file)
                print(f"Saved water volume for debugging: {volume_file}")
            except Exception as e:
                print(f"Failed to save water volume: {e}")
        
        print("Water feature extraction complete using new process!")
        
        return modified_terrain, water_surface, water_mask
    
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
                             downsample_factor=10, output_prefix="terrain_model",
                             extract_water=False, water_threshold=None, 
                             water_depth=2.0, min_water_area=100,
                             connect_water_regions=False, max_connection_gap=2,
                             remove_small_components=True, min_component_size=100):
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
        print(f"Water extraction: {extract_water}")
        
        # Get elevation data from SRTM
        print("Loading elevation data...")
        elevation_data = self.elevation_map.get_elevation_data(bounds, topo_dir)
        
        print(f"Elevation data: {elevation_data.shape}, range: {elevation_data.min():.1f} to {elevation_data.max():.1f} m")
        
        # Downsample elevation data for faster processing
        if downsample_factor > 1:
            elevation_data = self._downsample_elevation_data(elevation_data, downsample_factor)
        
        # Create 3D mesh from elevation data
        print("Creating 3D mesh...")
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
            modified_terrain, water_mesh, water_mask = self.create_water_features(
                mesh, elevation_data, bounds,
                water_threshold=water_threshold,
                water_depth=water_depth,
                base_height=base_height,
                elevation_multiplier=elevation_multiplier,
                min_water_area=min_water_area,
                output_prefix=output_prefix,
                connect_water_regions=connect_water_regions,
                max_connection_gap=max_connection_gap,
                remove_small_components=remove_small_components,
                min_component_size=min_component_size
            )
            
            # Update result with water features
            result['terrain_mesh'] = modified_terrain
            result['water_mesh'] = water_mesh
            result['water_mask'] = water_mask
        else:
            # Save the standard mesh
            output_file = f"{output_prefix}.obj"
            print("Saving mesh...")
            try:
                self.save_mesh(mesh, output_file)
            except Exception as e:
                print(f"Failed to save {output_file}: {e}")
        
        print("Terrain model generation complete!")
        
        return result

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


def main():
    """
    Example usage of the ModelGenerator class.
    """
    # Create model generator
    generator = ModelGenerator()
    
    # Example 1: Generate standard terrain model
    print("=== Example 1: Standard Terrain ===")
    bounds = (-122.5, 37.7, -122.4, 37.8)  # San Francisco area
    standard_result = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir="topo",
        base_height=10.0,
        elevation_multiplier=1.0,
        downsample_factor=10,
        output_prefix="sf_terrain_standard",
        extract_water=False
    )
    
    # Example 2: Generate terrain with water features
    print("\n=== Example 2: Terrain with Water ===")
    bounds_coastal = (-122.6, 37.6, -122.3, 37.9)  # Larger SF Bay area
    water_result = generator.generate_terrain_model(
        bounds=bounds_coastal,
        topo_dir="topo",
        base_height=5.0,
        elevation_multiplier=2.0,
        downsample_factor=8,
        output_prefix="sf_bay_area_water",
        extract_water=True,
        water_threshold=5.0,
        water_depth=3.0,
        min_water_area=50,
        connect_water_regions=True,
        max_connection_gap=2,
        remove_small_components=True,
        min_component_size=100
    )
    
    # Example 3: Automatic water detection
    print("\n=== Example 3: Auto Water Detection ===")
    bounds_lake = (-122.55, 37.75, -122.45, 37.85)  # Central SF area
    auto_water_result = generator.generate_terrain_model(
        bounds=bounds_lake,
        topo_dir="topo",
        base_height=2.0,
        elevation_multiplier=1.5,
        downsample_factor=12,
        output_prefix="sf_auto_water",
        extract_water=True,
        water_threshold=None,  # Automatic detection
        water_depth=2.0,
        min_water_area=25,
        connect_water_regions=True,
        max_connection_gap=2,
        remove_small_components=True,
        min_component_size=50
    )
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main() 