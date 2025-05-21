import os
import numpy as np
import rasterio
from osgeo import gdal
import cadquery as cq
from pyproj import Transformer
from tqdm import tqdm
import glob
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation, zoom
import concurrent.futures
import multiprocessing
from functools import partial
import time

class CQTerrainGenerator:
    def __init__(self):
        """Initialize the CadQuery TerrainGenerator with coordinate transformer."""
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def find_required_tiles(self, bounds):
        """
        Find all SRTM tiles needed to cover the given bounds.
        
        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            list: List of required tile coordinates [(lat, lon), ...]
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Floor to get the southwest corner of each tile
        min_lat_tile = int(np.floor(min_lat))
        max_lat_tile = int(np.floor(max_lat))
        min_lon_tile = int(np.floor(min_lon))
        max_lon_tile = int(np.floor(max_lon))
        
        required_tiles = []
        for lat in range(min_lat_tile, max_lat_tile + 1):
            for lon in range(min_lon_tile, max_lon_tile + 1):
                # For western hemisphere (negative longitude), we use negative values
                required_tiles.append((lat, lon))
                
        return required_tiles
    
    def find_tile_files(self, required_tiles, topo_dir="topo"):
        """Find available SRTM files for required tiles."""
        tile_files = {}
        for lat, lon in required_tiles:
            # SRTM naming convention: Nx.Wx.hgt.zip where x=latitude, longitude digits
            pattern = f"N{lat:02d}W{abs(lon):03d}*.hgt.zip"
            matches = glob.glob(os.path.join(topo_dir, pattern))
            if matches:
                tile_files[(lat, lon)] = matches[0]
        return tile_files
    
    def read_hgt_file(self, hgt_path):
        """
        Read SRTM .hgt file and return elevation data.
        
        Args:
            hgt_path (str): Path to .hgt file (can be zipped)
            
        Returns:
            numpy.ndarray: 2D array of elevation values
            float: pixel size in degrees
        """
        if not os.path.exists(hgt_path):
            raise ValueError(f"HGT file not found: {hgt_path}")
        
        import zipfile
        import tempfile
            
        try:
            # If it's a zip file, extract the .hgt file
            if hgt_path.endswith('.zip'):
                with zipfile.ZipFile(hgt_path, 'r') as zip_ref:
                    # Find the .hgt file in the zip
                    hgt_files = [f for f in zip_ref.namelist() if f.endswith('.hgt')]
                    if not hgt_files:
                        raise ValueError("No .hgt file found in zip archive")
                    
                    # Extract to a temporary directory
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_ref.extract(hgt_files[0], tmpdir)
                        hgt_file_path = os.path.join(tmpdir, hgt_files[0])
                        
                        # Use rasterio for faster reading
                        with rasterio.open(hgt_file_path) as ds:
                            # Read the data
                            elevation_data = ds.read(1)
                            
                            # Get pixel size
                            pixel_size = abs(ds.transform[0])
                        
                        # Handle no-data values
                            if ds.nodata is not None:
                                elevation_data = np.where(elevation_data == ds.nodata, 0, elevation_data)
                        
                        return elevation_data, pixel_size
            else:
                # Handle non-zipped .hgt files
                with rasterio.open(hgt_path) as ds:
                    # Read the data
                    elevation_data = ds.read(1)
                    
                    # Get pixel size
                    pixel_size = abs(ds.transform[0])
                
                # Handle no-data values
                    if ds.nodata is not None:
                        elevation_data = np.where(elevation_data == ds.nodata, 0, elevation_data)
                    
            return elevation_data, pixel_size
        except Exception as e:
            raise ValueError(f"Error reading HGT file: {str(e)}")

    def _read_tile(self, coords_path):
        """
        Read a single SRTM tile.
        
        Args:
            coords_path (tuple): (coords, file_path) tuple
            
        Returns:
            tuple: (coords, tile_info) tuple
        """
        coords, file_path = coords_path
        elevation_data, _ = self.read_hgt_file(file_path)
        return coords, {
            'data': elevation_data,
            'north': coords[0],
            'south': coords[0] - 1,
            'west': coords[1],
            'east': coords[1] + 1
        }
    
    def _stitch_srtm_tiles(self, tile_files):
        """
        Stitch SRTM tiles with proper north-up orientation using parallel processing.
        
        Args:
            tile_files (dict): Dictionary mapping (lat, lon) to file paths
            
        Returns:
            numpy.ndarray: Combined elevation data with proper orientation
        """
        print("Reading and arranging SRTM tiles in parallel...")
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
        start_time = time.time()
        num_cores = multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(lambda item: self._read_tile(item), tile_files.items()))
        
        # Convert results to a dictionary
        tile_data = {coords: info for coords, info in results}
        print(f"Tile reading completed in {time.time() - start_time:.2f} seconds using {num_cores} cores")
        
        # Find the dimensions of the tile grid
        all_lats = set(lat for lat, _ in tile_data.keys())
        all_lons = set(lon for _, lon in tile_data.keys())
        
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        
        # Calculate grid dimensions
        lat_range = max_lat - min_lat + 1
        lon_range = max_lon - min_lon + 1
        
        # Find our tile dimensions
        sample_data = next(iter(tile_data.values()))['data']
        tile_height, tile_width = sample_data.shape
        
        # Initialize the merged grid
        merged_height = tile_height * lat_range
        merged_width = tile_width * lon_range
        merged_data = np.zeros((merged_height, merged_width), dtype=np.float32)
        
        # Print info about available tiles
        print("\nProcessing tiles:")
        for (lat, lon), info in tile_data.items():
            print(f"Tile N{lat}W{abs(lon)}: elevation {info['data'].min():.0f}m to {info['data'].max():.0f}m")
        
        # Place each tile in its correct position
        for (lat, lon), info in tile_data.items():
            # Calculate row and column in the grid
            # Higher latitudes go at the top (row 0)
            row = max_lat - lat
            # Higher longitudes go to the right
            col = lon - min_lon
            
            print(f"  Placing tile N{lat}W{abs(lon)} at position ({row}, {col})")
            
            # Calculate the starting position in the merged grid
            start_row = row * tile_height
            start_col = col * tile_width
            
            # Place the data
            merged_data[start_row:start_row + tile_height, start_col:start_col + tile_width] = info['data']
        
        # Flip the entire grid vertically to get north-up orientation
        # This step makes North at the top and South at the bottom
        print("  Applying vertical flip for north-up orientation")
        merged_data = np.flipud(merged_data)
        
        return merged_data

    def _downsample_elevation(self, elevation_data, detail_level, target_points=10000):
        """
        Downsample elevation data based on detail level.
        """
        # Calculate the downsampling factor based on detail level and target points
        current_points = elevation_data.shape[0] * elevation_data.shape[1]
        
        # Skip downsampling for small datasets or high detail level
        if current_points <= target_points or detail_level >= 0.9:
            return elevation_data
            
        # Calculate factor to get close to target_points
        factor = np.sqrt(target_points / current_points) * detail_level
        factor = min(max(factor, 0.05), 1.0)  # Keep factor between 0.05 and 1.0
        
        # Calculate new dimensions
        new_shape = (int(elevation_data.shape[0] * factor), int(elevation_data.shape[1] * factor))
        
        print(f"Downsampling elevation data from {elevation_data.shape} to {new_shape}")
        
        # Use zoom for clean downsampling (order=1 for bilinear interpolation)
        downsampled = zoom(elevation_data, (factor, factor), order=1)
        
        return downsampled

    def generate_terrain(self, bounds, topo_dir="topo", detail_level=0.2, output_prefix="terrain",
                       water_level=0.0, height_scale=0.05, export_format="step"):
        """
        Generate terrain model using CadQuery.
        
        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files
            detail_level (float): Detail level (1.0 = highest detail, lower values reduce detail)
            output_prefix (str): Prefix for output files
            water_level (float): Elevation value for water areas (typically negative)
            height_scale (float): Scale factor for height relative to horizontal dimensions
            export_format (str): Format to export ('step', 'stl', etc.)
            
        Returns:
            cadquery.Assembly: The generated terrain assembly
        """
        total_start_time = time.time()
        print(f"Generating CAD terrain model for bounds: {bounds}")
        
        # Find required tiles
        required_tiles = self.find_required_tiles(bounds)
        tile_files = self.find_tile_files(required_tiles, topo_dir)
        
        if not tile_files:
            raise ValueError(f"No SRTM tiles found in {topo_dir} for the specified bounds")
            
        print(f"Using {len(tile_files)} SRTM tiles:")
        for coords in tile_files.keys():
            print(f"  N{coords[0]}W{-coords[1]}")
        
        # Stitch tiles
        elevation_data = self._stitch_srtm_tiles(tile_files)
        
        # Downsample for performance
        elevation_data = self._downsample_elevation(elevation_data, detail_level)
        
        # Scale height
        min_height = elevation_data.min()
        max_height = elevation_data.max()
        print(f"Elevation range: {min_height} to {max_height} meters")
        
        # Create water mask
        water_mask = elevation_data <= water_level
        print(f"Water coverage: {np.mean(water_mask) * 100:.1f}% of region")
        
        # Extract land and water regions
        land_elevation = np.where(water_mask, water_level, elevation_data)
        
        # Convert geographic coordinates to normalized model space
        min_lon, min_lat, max_lon, max_lat = bounds
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        # Use CadQuery to create terrain model
        print("Creating CadQuery terrain model...")
        cq_start_time = time.time()
        
        # Get dimensions
        elev_rows, elev_cols = elevation_data.shape
        x_size = 100  # Model width in CAD units
        y_size = 100 * (height / width)  # Maintain aspect ratio
        z_size = max(20, 100 * height_scale)  # Scaled height
        
        # Try different terrain methods in order of detail/complexity
        try:
            # First try: contoured terrain approach (most detailed)
            print("Attempting contoured terrain approach...")
            # Calculate number of contours based on detail level
            num_contours = int(10 * detail_level) + 5  # 5-15 contours based on detail level
            terrain_solid = self._create_contoured_terrain_surface(
                land_elevation, 
                water_mask,
                x_size, 
                y_size, 
                z_size, 
                water_level, 
                min_height, 
                max_height,
                num_contours=num_contours
            )
        except Exception as e:
            print(f"Contoured terrain approach failed: {str(e)}")
            
            try:
                # Second try: Shell/face-based approach
                print("Attempting face-based terrain approach...")
                terrain_solid = self._create_terrain_shell(
                    land_elevation, 
                    water_mask,
                    x_size, 
                    y_size, 
                    z_size, 
                    water_level, 
                    min_height, 
                    max_height
                )
            except Exception as e2:
                print(f"Face-based approach also failed: {str(e2)}")
                
                # Third try: Multi-level heightmap approach (simpler but still has features)
                print("Falling back to multi-level heightmap approach...")
                terrain_solid = self._create_terrain_from_heightmap(
                    land_elevation,
                    water_mask,
                    x_size, 
                    y_size, 
                    z_size, 
                    water_level, 
                    min_height, 
                    max_height
                )
        
        # Create water body
        water_body = self._create_water_body(
            water_mask, 
            x_size, 
            y_size, 
            z_size, 
            water_level, 
            min_height, 
            max_height
        )
        
        # Create assembly
        assembly = cq.Assembly()
        
        # Add terrain with tan color
        assembly.add(terrain_solid, name="terrain", color=cq.Color("tan"))
        
        # Add water with blue color if present
        if water_body:
            assembly.add(water_body, name="water", color=cq.Color("royalblue", alpha=0.8))
            
        print(f"CadQuery model creation completed in {time.time() - cq_start_time:.2f} seconds")
        
        # Export the model
        self._export_model(assembly, output_prefix, export_format)
        
        print(f"Total terrain generation time: {time.time() - total_start_time:.2f} seconds")
        
        return assembly
        
    def _create_terrain_shell(self, elevation_data, water_mask, x_size, y_size, z_size, water_level, min_height, max_height):
        """
        Create a 3D terrain shell using CadQuery.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            x_size, y_size (float): Size of model in x,y directions
            z_size (float): Maximum height of model
            water_level (float): Elevation for water surface
            min_height, max_height (float): Min/max elevation values
            
        Returns:
            cadquery.Solid: The terrain shell
        """
        # Get grid dimensions
        rows, cols = elevation_data.shape
        
        # Scale factor for vertical exaggeration
        vertical_scale = z_size / (max_height - min_height) if max_height > min_height else 1.0
        
        print(f"Creating terrain with dimensions {x_size} x {y_size} x {z_size}")
        print(f"Grid resolution: {rows} x {cols}")
        
        # For larger terrains, use heightmap approach instead of lofting
        if rows * cols > 10000:
            return self._create_terrain_from_heightmap(
                elevation_data, water_mask, x_size, y_size, z_size, 
                water_level, min_height, max_height
            )
            
        # Calculate the grid spacing
        dx = x_size / (cols - 1)
        dy = y_size / (rows - 1)
        
        # Create a face-based representation by using a grid of vertices
        vertices = []
        # Use fewer rows and columns for better performance
        step_size = max(1, min(rows, cols) // 40)
        
        print(f"Creating terrain mesh with step size {step_size}...")
        
        # Create grid of vertices
        for i in range(0, rows, step_size):
            row_vertices = []
            for j in range(0, cols, step_size):
                # Calculate x,y,z position
                x = j * dx - x_size/2
                y = i * dy - y_size/2
                
                # Get height (non-water areas only)
                height = elevation_data[i, j]
                z = (height - min_height) * vertical_scale
                
                # Add vertex
                row_vertices.append((x, y, z))
            vertices.append(row_vertices)
        
        # Create a box for the base
        base_height = 5  # Thickness of the base
        base = cq.Workplane("XY").box(x_size, y_size, base_height, centered=(True, True, False))
        
        try:
            # Try creating a terrain shell using polyhedron
            # First, create faces for the terrain surface
            faces = []
            for i in range(len(vertices) - 1):
                for j in range(len(vertices[i]) - 1):
                    # Get the four corners of each grid cell
                    v1 = vertices[i][j]
                    v2 = vertices[i][j + 1]
                    v3 = vertices[i + 1][j]
                    v4 = vertices[i + 1][j + 1]
                    
                    # Create two triangular faces for each grid cell
                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])
            
            print(f"Created {len(faces)} terrain faces")
            
            # Create the terrain solid
            try:
                # First attempt: shell based approach
                terrain_shell = cq.Solid.makeShell(faces)
                terrain_solid = cq.Workplane().add(terrain_shell)
                
                # Add a base by extruding down
                result = base.union(terrain_solid)
                print("Created terrain using shell method")
                return result
            except Exception as e1:
                print(f"Shell method failed: {str(e1)}")
                
                # Second attempt: create a compound and then convert to solid
                try:
                    # Create a series of faces
                    compound = cq.Compound.makeCompound([cq.Face.makeFromWires(cq.Wire.makePolygon([v1, v2, v3, v1])) 
                                                      for v1, v2, v3 in faces])
                    
                    # Create a solid from the compound
                    terrain_solid = cq.Workplane().add(compound)
                    result = base.union(terrain_solid)
                    print("Created terrain using compound method")
                    return result
                except Exception as e2:
                    print(f"Compound method failed: {str(e2)}")
                    
                    # Third attempt: simplified heightmap mesh approach
                    return self._create_terrain_from_heightmap(
                        elevation_data, water_mask, x_size, y_size, z_size, 
                        water_level, min_height, max_height
                    )
        
        except Exception as e:
            print(f"Error creating terrain shell: {str(e)}")
            return self._create_terrain_from_heightmap(
                elevation_data, water_mask, x_size, y_size, z_size, 
                water_level, min_height, max_height
            )
            
    def _create_terrain_from_heightmap(self, elevation_data, water_mask, x_size, y_size, z_size, 
                                      water_level, min_height, max_height):
        """
        Create a terrain model using a heightmap approach with multiple elevation levels.
        """
        print("Creating terrain using multi-level heightmap approach")
        
        # Convert numpy types to Python float
        min_height = float(min_height)
        max_height = float(max_height)
        vertical_scale = float(z_size / (max_height - min_height)) if max_height > min_height else 1.0
        
        # Create base
        base_thickness = 5.0
        base = cq.Workplane("XY").box(x_size, y_size, base_thickness, centered=(True, True, False))
        terrain_solid = base
        
        # Get dimensions and calculate grid spacing
        rows, cols = elevation_data.shape
        dx = x_size / (cols - 1)
        dy = y_size / (rows - 1)
        
        # Determine elevation breakpoints for terrain levels (5-10 levels works well)
        land_mask = ~water_mask
        land_elevation = elevation_data[land_mask] if np.any(land_mask) else elevation_data
        
        # If we have water, make sure water level is one of our breakpoints
        if np.any(water_mask):
            # Get min land elevation (anything below water_level will be water)
            min_land_elev = float(np.min(land_elevation))
            max_land_elev = float(np.max(land_elevation))
            
            # Create breakpoints for elevation levels (more levels for more detail)
            num_levels = 7
            
            # Make sure water level is one of our breakpoints
            breakpoints = [water_level]
            
            # Add levels above water
            if max_land_elev > water_level:
                step_size = (max_land_elev - water_level) / (num_levels-1)
                for i in range(1, num_levels):
                    breakpoints.append(water_level + i * step_size)
            
            # Sort breakpoints
            breakpoints.sort()
        else:
            # No water, just divide the elevation range evenly
            min_land_elev = float(np.min(land_elevation))
            max_land_elev = float(np.max(land_elevation))
            
            num_levels = 8
            step_size = (max_land_elev - min_land_elev) / num_levels
            breakpoints = [min_land_elev + i * step_size for i in range(num_levels + 1)]
        
        print(f"Creating terrain with {len(breakpoints)} elevation levels")
        
        # Create a series of progressively smaller and higher blocks for each elevation level
        for i, elevation_breakpoint in enumerate(breakpoints):
            if i == 0:
                continue  # Skip the first level (it's just the base)
                
            # Determine which areas are at or above this elevation
            level_mask = elevation_data >= elevation_breakpoint
            
            # Skip if no points at this elevation
            if not np.any(level_mask):
                continue
                
            # Calculate the scaled height for this level
            level_height = float((elevation_breakpoint - min_height) * vertical_scale)
            
            # Extract regions at this elevation level
            # Simplify by sampling the mask at lower resolution
            sample_step = max(1, min(rows, cols) // 40)
            sampled_mask = level_mask[::sample_step, ::sample_step]
            sampled_rows, sampled_cols = sampled_mask.shape
            
            # Create polygons for this elevation level
            try:
                level_workplane = cq.Workplane("XY").transformed(offset=(0, 0, base_thickness))
                
                # Create a grid of points that correspond to our elevation mask
                for r in range(sampled_rows):
                    for c in range(sampled_cols):
                        if sampled_mask[r, c]:
                            # Convert grid position to model coordinates
                            x = (c * sample_step * dx) - x_size/2 + dx/2
                            y = (r * sample_step * dy) - y_size/2 + dy/2
                            level_workplane = level_workplane.moveTo(x, y).circle(dx*1.2).close()
                
                # Extrude this level to its proper height
                level_height_relative = level_height - base_thickness
                if level_height_relative > 0:
                    level_shape = level_workplane.extrude(level_height_relative)
                    
                    # Union with the terrain solid
                    try:
                        terrain_solid = terrain_solid.union(level_shape)
                    except Exception as e:
                        print(f"Failed to union level {i}: {str(e)}")
            except Exception as e:
                print(f"Failed to create level {i}: {str(e)}")
        
        # Ensure we have at least some terrain features by adding a peak
        if terrain_solid == base:
            print("Adding fallback terrain peak")
            # Find highest point location
            max_idx = np.unravel_index(elevation_data.argmax(), elevation_data.shape)
            max_row, max_col = int(max_idx[0]), int(max_idx[1])
            
            # Calculate position in model coordinates
            max_x = float((max_col / cols) * x_size - x_size/2)
            max_y = float((max_row / rows) * y_size - y_size/2)
            
            # Create a peak feature
            max_point_radius = float(min(x_size, y_size) * 0.2)  # 20% of the model size
            peak_height = float(z_size * 0.8)  # 80% of z_size
            
            peak = (cq.Workplane("XY")
                   .circle(max_point_radius)
                   .extrude(peak_height)
                   .translate((max_x, max_y, base_thickness)))
            
            terrain_solid = terrain_solid.union(peak)
            
        return terrain_solid

    def _create_water_body(self, water_mask, x_size, y_size, z_size, water_level, min_height, max_height):
        """
        Create a water body that follows terrain contours.
        
        Args:
            water_mask (numpy.ndarray): Boolean mask indicating water areas
            x_size, y_size (float): Size of model in x,y directions
            z_size (float): Maximum height of model
            water_level (float): Elevation for water surface
            min_height, max_height (float): Min/max elevation values
            
        Returns:
            cadquery.Solid or None: The water body (if water exists)
        """
        # Skip if there's no water
        if not np.any(water_mask):
            print("No water areas detected in elevation data")
            return None
        
        print(f"Creating water body with {np.mean(water_mask) * 100:.1f}% coverage")
        
        # Convert to Python float types to prevent CadQuery issues
        water_level = float(water_level)
        min_height = float(min_height)
        max_height = float(max_height)
        x_size = float(x_size)
        y_size = float(y_size)
        z_size = float(z_size)
            
        # Compute normalized water height
        vertical_scale = float(z_size / (max_height - min_height)) if max_height > min_height else 1.0
        water_height = float((water_level - min_height) * vertical_scale)
        
        # Add a small offset to ensure water is visible above terrain
        water_offset = 0.5
        
        # Calculate water position
        water_thickness = 3.0  # Make water thick enough to be visible
        water_z_position = float(max(water_height, 0) + water_offset)
        
        # Get dimensions of the water mask
        rows, cols = water_mask.shape
        
        # Sample the elevation at lower resolution for performance
        sample_step = max(1, min(rows, cols) // 30)  # Use about 30 points per dimension
        sampled_mask = water_mask[::sample_step, ::sample_step]
        sampled_rows, sampled_cols = sampled_mask.shape
        
        # Try to create a contoured water body that follows shorelines
        try:
            # First, create water surface shape based on water mask
            if np.any(sampled_mask):
                # Calculate the grid spacing
                dx = x_size / (cols - 1)
                dy = y_size / (rows - 1)
                
                # Create a polygon from water mask outline
                water_points = []
                
                # Find boundary points of water mask
                boundary_found = False
                
                # Create a padded mask to find the boundary
                padded_mask = np.pad(sampled_mask, 1, mode='constant', constant_values=False)
                
                # For each water point, check if it's at the boundary
                for r in range(1, padded_mask.shape[0]-1):
                    for c in range(1, padded_mask.shape[1]-1):
                        if padded_mask[r, c]:
                            # Check if it's at the boundary (has at least one non-water neighbor)
                            neighbors = [
                                padded_mask[r-1, c],  # North
                                padded_mask[r+1, c],  # South
                                padded_mask[r, c-1],  # West
                                padded_mask[r, c+1],  # East
                            ]
                            if not all(neighbors):
                                # This is a boundary point
                                boundary_found = True
                                # Convert to model coordinates
                                x = ((c-1) * sample_step * dx) - x_size/2 + dx/2
                                y = ((r-1) * sample_step * dy) - y_size/2 + dy/2
                                water_points.append((x, y))
                
                if boundary_found and len(water_points) > 2:
                    # Create a simplified convex hull of these points to make a cleaner shape
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(water_points)
                        hull_points = [water_points[i] for i in hull.vertices]
                        
                        # Create the water surface from the hull points
                        water_surface = (cq.Workplane("XY")
                                        .moveTo(hull_points[0][0], hull_points[0][1])
                                        .polyline([p for p in hull_points[1:]])
                                        .close()
                                        .extrude(water_thickness)
                                        .translate((0, 0, water_z_position)))
                        
                        # Create sides that extend down to base
                        water_sides = (cq.Workplane("XY")
                                      .moveTo(hull_points[0][0], hull_points[0][1])
                                      .polyline([p for p in hull_points[1:]])
                                      .close()
                                      .extrude(water_z_position)
                                      .translate((0, 0, 0)))
                        
                        # Combine water surface and sides
                        combined_water = water_surface.union(water_sides)
                        print("Created contoured water body following shorelines")
                        return combined_water
                    except Exception as e:
                        print(f"Failed to create water hull: {str(e)}")
        except Exception as e:
            print(f"Contoured water creation failed: {str(e)}")
            
        # Fallback: Use simpler water representation
        print("Using simple water body representation")
        
        # Create water surface that extends slightly beyond the terrain
        water_extension = float(min(x_size, y_size) * 0.01)  # Extend water 1% beyond terrain
        water_width = float(x_size + 2 * water_extension)
        water_length = float(y_size + 2 * water_extension)
        
        # Create water body as an extruded box
        try:
            print(f"Creating water surface at height {water_z_position:.2f}")
            water_body = (
                cq.Workplane("XY")
                .box(water_width, water_length, water_thickness, centered=(True, True, False))
                .translate((0, 0, water_z_position))
            )
            
            # Create sides for the water to make it more visually interesting
            # Only add sides if the water is above the base
            if water_z_position > 0.1:
                # Front edge
                front_wall = (
                    cq.Workplane("XY")
                    .box(water_width, water_thickness, water_z_position, centered=(True, True, False))
                    .translate((0, -water_length/2, 0))
                )
                
                # Back edge
                back_wall = (
                    cq.Workplane("XY")
                    .box(water_width, water_thickness, water_z_position, centered=(True, True, False))
                    .translate((0, water_length/2, 0))
                )
                
                # Left edge
                left_wall = (
                    cq.Workplane("XY")
                    .box(water_thickness, water_length, water_z_position, centered=(True, True, False))
                    .translate((-water_width/2, 0, 0))
                )
                
                # Right edge
                right_wall = (
                    cq.Workplane("XY")
                    .box(water_thickness, water_length, water_z_position, centered=(True, True, False))
                    .translate((water_width/2, 0, 0))
                )
                
                # Combine all water parts
                try:
                    # Union one at a time with error checking
                    combined_water = water_body.union(front_wall)
                    combined_water = combined_water.union(back_wall)
                    combined_water = combined_water.union(left_wall)
                    combined_water = combined_water.union(right_wall)
                    print("Created water body with sides")
                    return combined_water
                except Exception as e:
                    print(f"Failed to create complex water body: {str(e)}")
                    print("Falling back to simple water body")
                    return water_body
            else:
                print("Water position too low for side walls, using simple water body")
                return water_body
                
        except Exception as e:
            print(f"Error creating water body: {str(e)}")
            # Create a very simple water representation as fallback
            try:
                simple_water = (
                    cq.Workplane("XY")
                    .box(x_size * 0.9, y_size * 0.9, 1.0, centered=(True, True, False))
                    .translate((0, 0, 1.0))
                )
                print("Created simplified water body")
                return simple_water
            except:
                print("Could not create any water body")
                return None
        
    def _export_model(self, assembly, output_prefix, export_format):
        """
        Export the CadQuery model.
        
        Args:
            assembly (cadquery.Assembly): The model to export
            output_prefix (str): Prefix for output files
            export_format (str): Format to export ('step', 'stl', etc.)
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Determine output path
        output_path = f"{output_prefix}.{export_format.lower()}"
        
        print(f"Exporting model to {output_path}...")
        
        # For STL export, we need to use a different approach
        if export_format.lower() == 'stl':
            try:
                # Get all the solids from the assembly
                all_solids = []
                for _, part in assembly.traverse():
                    if hasattr(part, 'obj') and part.obj:
                        all_solids.append(part.obj)
                
                # Export directly using the exporter
                if all_solids:
                    # If only one solid, export it directly
                    if len(all_solids) == 1:
                        cq.exporters.export(all_solids[0], output_path)
                    else:
                        # Combine all solids into a compound
                        compound = cq.Compound.makeCompound(all_solids)
                        cq.exporters.export(compound, output_path)
                    print(f"Model exported successfully to {output_path}")
                    return
            except Exception as e:
                print(f"STL export using exporters failed: {str(e)}")
        
        try:
            # Try standard assembly export
            if export_format.lower() == 'step':
                assembly.save(output_path, 'STEP')
            elif export_format.lower() == 'stl':
                assembly.save(output_path, 'STL')
            elif export_format.lower() == '3mf':
                assembly.save(output_path, '3MF')
            else:
                print(f"Unsupported export format: {export_format}. Exporting as STEP instead.")
                assembly.save(f"{output_prefix}.step", 'STEP')
                
            print(f"Model exported successfully to {output_path}")
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            # Try a simpler export if the assembly export fails
            try:
                print("Attempting simplified export...")
                # Convert assembly to a compound and export
                compound = assembly.toCompound()
                cq.exporters.export(compound, output_path)
                print(f"Simplified export completed to {output_path}")
            except Exception as e2:
                print(f"Simplified export also failed: {str(e2)}")

    def _create_contoured_terrain_surface(self, elevation_data, water_mask, x_size, y_size, z_size, 
                                         water_level, min_height, max_height, num_contours=10):
        """
        Create a more accurate terrain surface using contour lines
        """
        # Convert numpy types to Python float
        min_height = float(min_height)
        max_height = float(max_height)
        water_level = float(water_level)
        
        # Keep only land elevation data (we'll handle water separately)
        land_mask = ~water_mask
        
        if not np.any(land_mask):
            # No land, return a flat surface
            base = cq.Workplane("XY").box(x_size, y_size, 1.0, centered=(True, True, False))
            return base
            
        rows, cols = elevation_data.shape
        dx = x_size / (cols - 1)
        dy = y_size / (rows - 1)
        
        # Calculate scaled elevation range
        elev_range = max_height - min_height
        vertical_scale = z_size / elev_range if elev_range > 0 else 1.0
        
        # Create a base
        base_thickness = 3.0
        base = cq.Workplane("XY").box(x_size, y_size, base_thickness, centered=(True, True, False))
        
        # Create contour levels - focus more detail near water level if there's water
        if np.any(water_mask):
            # Make sure water level is included
            levels = [water_level]
            
            # Add levels above water to max height
            above_range = max_height - water_level
            if above_range > 0:
                levels_above = np.linspace(water_level, max_height, num_contours // 2 + 1)[1:]
                levels.extend(levels_above)
                
            # Add a few levels below water level if needed
            below_range = water_level - min_height
            if below_range > 0 and below_range < above_range:
                levels_below = np.linspace(min_height, water_level, 3)[:-1]  # exclude water_level
                levels = list(levels_below) + levels
        else:
            # No water, just create evenly spaced contours
            levels = np.linspace(min_height, max_height, num_contours)
            
        # Convert levels to floats
        levels = [float(level) for level in levels]
        
        # Sample the elevation at lower resolution for performance
        sample_step = max(1, min(rows, cols) // 40)  # Don't use more than ~40 points in each dimension
        
        # Create a list to hold all the terrain parts
        terrain_parts = [base]
        
        # Create each contour level
        for i, level in enumerate(levels):
            if i > 0:  # Skip the first level which is just the base
                # Determine mask of points at or above this elevation level
                level_mask = elevation_data >= level
                
                # Sample the mask at lower resolution
                sampled_mask = level_mask[::sample_step, ::sample_step]
                sampled_rows, sampled_cols = sampled_mask.shape
                
                if not np.any(sampled_mask):
                    continue  # Skip empty levels
                
                # Calculate height for this level
                level_height = float(base_thickness + (level - min_height) * vertical_scale)
                
                # Create a workplane at this height
                try:
                    # Create a polygon for this contour level
                    level_points = []
                    
                    # Add points from the elevation mask
                    for r in range(sampled_rows):
                        for c in range(sampled_cols):
                            if sampled_mask[r, c]:
                                # Convert grid coordinates to model coordinates
                                x = (c * sample_step) * dx - x_size/2 + dx/2
                                y = (r * sample_step) * dy - y_size/2 + dy/2
                                level_points.append((x, y))
                    
                    if len(level_points) > 2:  # Need at least 3 points to make a polygon
                        # Create a simplified convex hull of these points to make a cleaner shape
                        from scipy.spatial import ConvexHull
                        try:
                            hull = ConvexHull(level_points)
                            hull_points = [level_points[i] for i in hull.vertices]
                            
                            # Create the contour solid
                            if hull_points and len(hull_points) >= 3:
                                # Create a polygon from the hull points
                                level_solid = (cq.Workplane("XY")
                                              .moveTo(hull_points[0][0], hull_points[0][1])
                                              .polyline([p for p in hull_points[1:]])
                                              .close()
                                              .extrude(level_height - (levels[i-1] - min_height) * vertical_scale)
                                              .translate((0, 0, base_thickness + (levels[i-1] - min_height) * vertical_scale)))
                                
                                terrain_parts.append(level_solid)
                        except Exception as e:
                            print(f"Failed to create convex hull for level {i}: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Failed to create contour level {i}: {str(e)}")
                    continue
        
        # Combine all terrain parts
        terrain_solid = terrain_parts[0]
        for part in terrain_parts[1:]:
            try:
                terrain_solid = terrain_solid.union(part)
            except Exception as e:
                print(f"Failed to union terrain part: {str(e)}")
        
        return terrain_solid 