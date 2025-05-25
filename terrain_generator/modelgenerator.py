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
                             downsample_factor=10, output_prefix="terrain_model"):
        """
        Generate a 3D terrain model from SRTM elevation data.
        
        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files
            base_height (float): Height of the flat base
            elevation_multiplier (float): Multiplier for realistic elevation scaling (1.0 = realistic scale)
            downsample_factor (int): Factor to downsample elevation data (default: 10)
            output_prefix (str): Prefix for output filename
            
        Returns:
            meshlib.mrmeshpy.Mesh: The generated 3D mesh
        """
        print("=== Terrain Model Generation ===")
        print(f"Bounds: {bounds}")
        print(f"Base height: {base_height}")
        print(f"Elevation multiplier: {elevation_multiplier}")
        print(f"Downsample factor: {downsample_factor}")
        
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
        
        # Save the mesh as OBJ file only
        output_file = f"{output_prefix}.obj"
        
        print("Saving mesh file...")
        try:
            self.save_mesh(mesh, output_file)
        except Exception as e:
            print(f"Failed to save {output_file}: {e}")
        
        print(f"\n✓ Terrain model generation complete!")
        print(f"Generated file: {output_file}")
        
        return mesh


def main():
    """
    Example usage of the ModelGenerator class.
    """
    # Create model generator
    generator = ModelGenerator()
    
    # Example: Generate real terrain model
    print("Generating terrain model from SRTM data...")
    # Example bounds for a small area (adjust as needed)
    bounds = (-122.5, 37.7, -122.4, 37.8)  # San Francisco area
    real_mesh = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir="topo",
        base_height=10.0,           # 10 unit base height
        elevation_multiplier=1.0,   # Default elevation multiplier
        downsample_factor=10,       # Default downsampling
        output_prefix="example_terrain"
    )


if __name__ == "__main__":
    main() 