import os
import numpy as np
import rasterio
from osgeo import gdal
import trimesh
from pyproj import Transformer
from tqdm import tqdm
import glob
from pathlib import Path
import re  # Add this for regex pattern matching
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation, zoom
import concurrent.futures
import multiprocessing
from functools import partial
import time

class TerrainGenerator:
    def __init__(self):
        """Initialize the TerrainGenerator with coordinate transformer."""
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
            
        Raises:
            ValueError: If file is not found or invalid
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

    def merge_elevation_data(self, tile_data, bounds):
        """Merge elevation data from multiple tiles with boundary blending."""
        if not tile_data:
            raise ValueError("No tile data provided for merging")

        points_per_degree = 1201  # SRTM data points per degree
        min_lon, min_lat, max_lon, max_lat = bounds
        
        print("\nMerging elevation data...")
        print(f"Region bounds: lon({min_lon:.4f}, {max_lon:.4f}), lat({min_lat:.4f}, {max_lat:.4f})")
        
        # Validate bounds
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bounds: min values must be less than max values")

        # Calculate output dimensions for the entire region
        total_lat_points = int((max_lat - min_lat) * (points_per_degree - 1))
        total_lon_points = int((max_lon - min_lon) * (points_per_degree - 1))
        
        if total_lat_points <= 0 or total_lon_points <= 0:
            raise ValueError("Invalid dimensions calculated from bounds")

        print(f"Output grid: {total_lat_points}x{total_lon_points} points")

        # Initialize output arrays
        merged = np.zeros((total_lat_points, total_lon_points), dtype=np.float32)
        weights = np.zeros((total_lat_points, total_lon_points), dtype=np.float32)

        # Print info about available tiles
        print("\nProcessing tiles:")
        for (lat, lon), (data, _) in tile_data.items():
            print(f"Tile N{lat}W{abs(lon)}: elevation {data.min():.0f}m to {data.max():.0f}m")
        
        # Process each tile
        blend_width = 10  # pixels for edge blending
        for (tile_lat, tile_lon), (data, pixel_size) in tile_data.items():
            # Calculate the position of this tile in the final grid
            # Tiles are positioned from top-left (north-west) corner
            
            # Latitude: Convert from geographic space to pixel space
            # For latitude, larger values are north, so we calculate from top
            lat_offset = int((max_lat - (tile_lat + 1)) * (points_per_degree - 1))
            
            # Longitude: Convert from geographic space to pixel space
            # For longitude in western hemisphere, we need to handle negative values properly
            lon_offset = int((tile_lon - min_lon) * (points_per_degree - 1))
            
            print(f"\nProcessing tile N{tile_lat}W{abs(tile_lon)}:")
            print(f"Position in output: ({lat_offset}, {lon_offset})")
            print(f"Data range: {data.min():.1f}m to {data.max():.1f}m")
            
            # Error checking for offsets
            if lat_offset < 0 or lon_offset < 0:
                print(f"Warning: Adjusting negative offset ({lat_offset}, {lon_offset})")
                # Adjust to avoid array out of bounds issues
                lat_offset = max(0, lat_offset)
                lon_offset = max(0, lon_offset)
            
            # Create weight mask
            weight_mask = np.ones_like(data)
            
            # Apply edge blending
            if blend_width > 0:
                for i in range(blend_width):
                    factor = (i + 1) / (blend_width + 1)
                    # Blend edges that meet other tiles
                    if lat_offset == 0:  # North edge
                        weight_mask[i, :] = factor
                    if lat_offset + points_per_degree >= total_lat_points:  # South edge
                        weight_mask[-(i + 1), :] = factor
                    if lon_offset == 0:  # West edge
                        weight_mask[:, i] = factor
                    if lon_offset + points_per_degree >= total_lon_points:  # East edge
                        weight_mask[:, -(i + 1)] = factor
            
            # Calculate valid regions for both source and target arrays
            src_lat_start = max(0, -lat_offset)
            src_lon_start = max(0, -lon_offset)
            src_lat_end = min(points_per_degree, total_lat_points - lat_offset)
            src_lon_end = min(points_per_degree, total_lon_points - lon_offset)
            
            dst_lat_start = max(0, lat_offset)
            dst_lon_start = max(0, lon_offset)
            dst_lat_end = min(total_lat_points, lat_offset + points_per_degree)
            dst_lon_end = min(total_lon_points, lon_offset + points_per_degree)
            
            # Ensure we have valid ranges
            if src_lat_end <= src_lat_start or src_lon_end <= src_lon_start:
                print("Skipping tile - no valid data range")
                continue
            
            print(f"Source region: [{src_lat_start}:{src_lat_end}, {src_lon_start}:{src_lon_end}]")
            print(f"Target region: [{dst_lat_start}:{dst_lat_end}, {dst_lon_start}:{dst_lon_end}]")
            
            # Extract the valid portion of data and weights
            valid_data = data[src_lat_start:src_lat_end, src_lon_start:src_lon_end]
            valid_weights = weight_mask[src_lat_start:src_lat_end, src_lon_start:src_lon_end]
            
            # Add to the output grid
            merged[dst_lat_start:dst_lat_end, dst_lon_start:dst_lon_end] += (
                valid_data * valid_weights
            )
            weights[dst_lat_start:dst_lat_end, dst_lon_start:dst_lon_end] += valid_weights
            
            # Print progress
            non_zero = np.count_nonzero(merged)
            total = merged.size
            print(f"Progress: {non_zero/total*100:.1f}% of grid filled")

        # Normalize by weights
        valid_mask = weights > 0
        merged[valid_mask] /= weights[valid_mask]
        
        # Print final statistics
        if np.any(valid_mask):
            print(f"\nFinal elevation range: {merged[valid_mask].min():.1f}m to {merged[valid_mask].max():.1f}m")
            print(f"Mean elevation: {merged[valid_mask].mean():.1f}m")
            print(f"Coverage: {np.count_nonzero(valid_mask)/valid_mask.size*100:.1f}% of region")
        else:
            print("\nWarning: No valid elevation data in output grid!")
        
        return merged, pixel_size

    def process_region(self, hgt_paths, bounds, detail_level=1.0):
        """
        Process a region within given bounds and generate 3D terrain model.
        
        Args:
            hgt_paths (list): List of paths to .hgt files or directory containing them
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
            detail_level (float): Level of detail (0.01 to 1.0)
            
        Returns:
            trimesh.Trimesh: Generated 3D mesh
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.001 <= detail_level <= 1.0:
            raise ValueError("Detail level must be between 0.001 and 1.0")

        print("Finding required SRTM tiles...")
        required_tiles = self.find_required_tiles(bounds)
        
        # If hgt_paths is a directory, look for files there
        if isinstance(hgt_paths, str) and os.path.isdir(hgt_paths):
            tile_files = self.find_tile_files(required_tiles, hgt_paths)
        else:
            # Use provided files
            tile_files = {(int(re.search(r'N(\d+)', Path(f).name).group(1)),
                         -int(re.search(r'W(\d+)', Path(f).name).group(1))): f
                        for f in hgt_paths}
        
        print(f"Found {len(tile_files)} of {len(required_tiles)} required tiles")
        
        # Read all tiles
        tile_data = {}
        for coords, file_path in tqdm(tile_files.items(), desc="Reading SRTM files"):
            elevation_data, pixel_size = self.read_hgt_file(file_path)
            tile_data[coords] = (elevation_data, pixel_size)
        
        # Merge elevation data
        elevation_data, pixel_size = self.merge_elevation_data(tile_data, bounds)
        
        # Convert geographic coordinates to Web Mercator
        print("Converting coordinates to Web Mercator...")
        min_x, min_y = self.transformer.transform(bounds[0], bounds[1])
        max_x, max_y = self.transformer.transform(bounds[2], bounds[3])
        
        # Calculate distances in meters
        width_m = max_x - min_x
        height_m = max_y - min_y
        area_km2 = (width_m * height_m) / 1e6
        
        print("Calculating grid dimensions...")
        # Adaptive resolution based on area size and detail level
        print(f"Area size: {area_km2:.1f} km²")
        if area_km2 > 10000:  # For very large areas (>10,000 km²)
            base_resolution = max(90, area_km2 / 50)  # Increase minimum distance between points
        elif area_km2 > 1000:  # For large areas (>1,000 km²)
            base_resolution = max(90, area_km2 / 100)
        else:
            base_resolution = 90
            
        # Apply detail level to base resolution
        actual_resolution = base_resolution / detail_level
        print(f"Base resolution: {base_resolution:.1f}m, Detail-adjusted resolution: {actual_resolution:.1f}m")
        
        # Calculate number of samples based on actual resolution
        target_vertices = int(width_m * height_m / (actual_resolution * actual_resolution))
        target_vertices = min(target_vertices, 1000000)  # Cap at 1 million vertices for memory
        
        # Calculate grid dimensions maintaining aspect ratio
        aspect_ratio = width_m / height_m
        cols = int(np.sqrt(target_vertices * aspect_ratio))
        rows = int(target_vertices / cols)
        
        # Ensure minimum dimensions
        cols = max(cols, 10)
        rows = max(rows, 10)
        
        print(f"Grid dimensions: {rows}x{cols} ({rows*cols:,} vertices)")
        print(f"Approximate memory usage: {rows*cols*32/1024/1024:.1f} MB")
        
        print("\nProcessing elevation data:")
        print("-----------------------")
        print(f"Initial elevation range: {elevation_data.min():.1f}m to {elevation_data.max():.1f}m")
        
        # Create vertex grid first
        x = np.linspace(min_x, max_x, cols)
        y = np.linspace(min_y, max_y, rows)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Resample elevation data to match our grid
        from scipy.ndimage import zoom
        zoom_x = cols / elevation_data.shape[1]
        zoom_y = rows / elevation_data.shape[0]
        
        elevation_data = zoom(elevation_data, (zoom_y, zoom_x), order=1)
        
        # Apply vertical exaggeration based on area size
        # For larger areas, use less exaggeration
        area_scale = np.sqrt(area_km2) / 100
        vertical_exaggeration = 2.0 / max(1.0, area_scale)
        elevation_data = elevation_data * vertical_exaggeration
        
        vertices = np.column_stack((
            x_grid.flatten(),
            y_grid.flatten(),
            elevation_data.flatten()
        ))
        
        # First normalize XY coordinates to 0-1 range while preserving aspect ratio
        xy_scale = 1.0 / max(width_m, height_m)
        vertices[:, 0] = (vertices[:, 0] - min_x) * xy_scale
        vertices[:, 1] = (vertices[:, 1] - min_y) * xy_scale
            
        # Scale elevation to be proportional to horizontal scale
        z_min = vertices[:, 2].min()
        z_range = vertices[:, 2].max() - z_min
        
        if z_range > 0:
            # Calculate terrain scale based on area size
            terrain_scale = 0.15  # Fixed scale factor
            vertices[:, 2] = ((vertices[:, 2] - z_min) / z_range) * terrain_scale
        else:
            print("\nWARNING - No elevation variation detected!")
        
        # Create faces for triangulation with progress bar
        print("Generating terrain mesh...")
        faces = []
        for i in tqdm(range(rows - 1), desc="Creating surface triangles"):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = (i + 1) * cols + j
                v3 = v2 + 1
                # Create two triangles for each grid cell with correct orientation
                faces.extend([
                    [v0, v1, v2],  # Top surface triangle 1
                    [v2, v1, v3]   # Top surface triangle 2
                ])
        
        print("Creating solid base...")
        # Create base vertices by duplicating and setting to minimum Z
        base_vertices = vertices.copy()
        min_z = vertices[:, 2].min()
        base_vertices[:, 2] = min_z - 0.01  # Slightly below to ensure no z-fighting
        
        # Combine top and base vertices
        vertices = np.vstack([vertices, base_vertices])
        vertex_count = rows * cols
        
        # Add side walls - improved to prevent slats
        print("Adding side walls...")
        # Front edge (i = 0)
        for j in tqdm(range(cols - 1), desc="Creating front wall"):
            v0 = j
            v1 = j + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
        
        # Back edge (i = rows-1)
        for j in tqdm(range(cols - 1), desc="Creating back wall"):
            v0 = (rows - 1) * cols + j
            v1 = v0 + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
        
        # Left edge (j = 0)
        for i in tqdm(range(rows - 1), desc="Creating left wall"):
            v0 = i * cols
            v1 = (i + 1) * cols
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
        
        # Right edge (j = cols-1)
        for i in tqdm(range(rows - 1), desc="Creating right wall"):
            v0 = i * cols + (cols - 1)
            v1 = (i + 1) * cols + (cols - 1)
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
        
        # Add base triangles (inverted orientation from top)
        print("Adding base triangles...")
        for i in tqdm(range(rows - 1), desc="Creating base triangles"):
            for j in range(cols - 1):
                v0 = i * cols + j + vertex_count
                v1 = v0 + 1
                v2 = (i + 1) * cols + j + vertex_count
                v3 = v2 + 1
                faces.extend([
                    [v0, v2, v1],  # Base triangle 1 (inverted)
                    [v1, v2, v3]   # Base triangle 2 (inverted)
                ])
        
        # Convert faces to numpy array for the terrain surface
        faces = np.array(faces)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Fix normals to point outward
        mesh.fix_normals()
        
        # Center the mesh at origin
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        
        # Scale to 1 meter length
        scale_factor = 1.0 / max(mesh.extents)
        mesh.apply_scale(scale_factor)
        
        # Ensure the mesh is oriented flat (Z-up)
        rotation = trimesh.transformations.rotation_matrix(
            angle=-np.pi/2,
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation)
        
        return mesh
    
    def export_glb(self, scene, output_path):
        """
        Export scene to .glb format.
        
        Args:
            scene (trimesh.Scene): The scene to export
            output_path (str): Path for the output .glb file
            
        Raises:
            ValueError: If export fails
        """
        try:
            # Create the export directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export the scene
            scene.export(output_path, file_type='glb')
            print(f"Model exported to {output_path}")
        except Exception as e:
            raise ValueError(f"Error exporting to GLB: {str(e)}")

    def generate_terrain(self, bounds, topo_dir="topo", detail_level=0.2, output_prefix="terrain",
                       water_level=-15.0, shore_height=1.0, shore_buffer=1, height_scale=0.05,
                       debug=False):
        """
        Generate terrain model using SRTM data with proper north-up orientation.
        
        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files
            detail_level (float): Detail level (1.0 = highest detail, lower values reduce detail)
            output_prefix (str): Prefix for output files
            water_level (float): Elevation value for water areas (typically negative)
            shore_height (float): Elevation value for shore areas
            shore_buffer (int): Number of cells for shore buffer
            height_scale (float): Scale factor for height relative to horizontal dimensions
            debug (bool): Whether to generate debug visualizations
            
        Returns:
            trimesh.Trimesh: The generated terrain mesh
        """
        total_start_time = time.time()
        print(f"Generating terrain model for bounds: {bounds}")
        
        # Find required tiles
        required_tiles = self.find_required_tiles(bounds)
        tile_files = self.find_tile_files(required_tiles, topo_dir)
        
        if not tile_files:
            raise ValueError(f"No SRTM tiles found in {topo_dir} for the specified bounds")
            
        print(f"Using {len(tile_files)} SRTM tiles:")
        for coords in tile_files.keys():
            print(f"  N{coords[0]}W{-coords[1]}")
        
        # Stitch the tiles with standard SRTM arrangement
        elevation_data = self._stitch_srtm_tiles(tile_files)
        
        # Create debug visualizations if requested (in parallel if debug mode is on)
        if debug:
            debug_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            debug_future = debug_executor.submit(self._generate_debug_visualizations, elevation_data, bounds, output_prefix)
        
        # Create the terrain model
        scene = self._create_terrain_model_from_elevation(
            elevation_data, 
            bounds, 
            detail_level, 
            water_level, 
            shore_height, 
            shore_buffer,
            height_scale
        )
        
        # Wait for debug visualizations to complete if they were requested
        if debug:
            debug_future.result()
            debug_executor.shutdown()
        
        # Export the model
        output_path = f"{output_prefix}_{detail_level:.3f}.glb"
        self.export_glb(scene, output_path)
        
        # Final performance report
        print(f"Total terrain generation pipeline time: {time.time() - total_start_time:.2f} seconds")
        
        return scene
    
    # Function to read a single tile (moved outside for multiprocessing)
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
        
        # Place each tile in its correct position
        for (lat, lon), info in tile_data.items():
            # Calculate row and column in the grid
            # Higher latitudes go at the top (row 0)
            row = max_lat - lat
            # Higher longitudes go to the right
            col = lon - min_lon
            
            print(f"  Placing tile N{lat}W{-lon} at position ({row}, {col})")
            
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
    
    # Function for creating faces for a chunk of rows (moved outside for multiprocessing)
    def _create_faces_for_chunk(self, start_row, end_row, cols, resampled_water_mask_flat):
        water_faces = []
        land_faces = []
        
        for i in range(start_row, end_row):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = (i + 1) * cols + j
                v3 = v2 + 1
                
                # Skip if we're at the last row
                if i >= self.rows - 1:
                    continue
                
                # Check if this quad is entirely water or entirely land
                is_water_quad = (
                    resampled_water_mask_flat[v0] and 
                    resampled_water_mask_flat[v1] and 
                    resampled_water_mask_flat[v2] and 
                    resampled_water_mask_flat[v3]
                )
                
                is_land_quad = (
                    not resampled_water_mask_flat[v0] and 
                    not resampled_water_mask_flat[v1] and 
                    not resampled_water_mask_flat[v2] and 
                    not resampled_water_mask_flat[v3]
                )
                
                # If it's a mixed quad (part water, part land), we need to make a decision
                # For simplicity, we'll assign it to land if at least 3 vertices are land
                if not is_water_quad and not is_land_quad:
                    land_count = (
                        (not resampled_water_mask_flat[v0]) + 
                        (not resampled_water_mask_flat[v1]) + 
                        (not resampled_water_mask_flat[v2]) + 
                        (not resampled_water_mask_flat[v3])
                    )
                    is_land_quad = land_count >= 3
                    is_water_quad = not is_land_quad
                
                if is_water_quad:
                    water_faces.extend([
                        [v0, v1, v2],
                        [v1, v3, v2]
                    ])
                else:
                    land_faces.extend([
                        [v0, v1, v2],
                        [v1, v3, v2]
                    ])
                    
        return water_faces, land_faces
    
    # Function for creating walls for a range (moved outside for multiprocessing)    
    def _create_walls_for_range(self, start_idx, end_idx, is_columns=True, resampled_water_mask_flat=None, cols=None, vertex_count=None):
        wall_faces = []
        
        if is_columns:
            # Process columns (front and back walls)
            for j in range(start_idx, end_idx):
                # Front edge (i=0)
                v0, v1 = j, j+1
                v0_base, v1_base = v0 + vertex_count, v1 + vertex_count
                
                if not resampled_water_mask_flat[v0] and not resampled_water_mask_flat[v1]:
                    wall_faces.extend([
                        [v0, v1, v0_base],
                        [v1, v1_base, v0_base]
                    ])
                
                # Back edge (i=rows-1)
                v0 = (self.rows - 1) * cols + j
                v1 = v0 + 1
                v0_base = v0 + vertex_count
                v1_base = v1 + vertex_count
                
                if not resampled_water_mask_flat[v0] and not resampled_water_mask_flat[v1]:
                    wall_faces.extend([
                        [v0, v1, v0_base],
                        [v1, v1_base, v0_base]
                    ])
        else:
            # Process rows (left and right walls)
            for i in range(start_idx, end_idx):
                # Left edge (j=0)
                v0 = i * cols
                v1 = (i + 1) * cols
                v0_base = v0 + vertex_count
                v1_base = v1 + vertex_count
                
                if not resampled_water_mask_flat[v0] and not resampled_water_mask_flat[v1]:
                    wall_faces.extend([
                        [v0, v1, v0_base],
                        [v1, v1_base, v0_base]
                    ])
                
                # Right edge (j=cols-1)
                v0 = i * cols + (cols - 1)
                v1 = (i + 1) * cols + (cols - 1)
                v0_base = v0 + vertex_count
                v1_base = v1 + vertex_count
                
                if not resampled_water_mask_flat[v0] and not resampled_water_mask_flat[v1]:
                    wall_faces.extend([
                        [v0, v1, v0_base],
                        [v1, v1_base, v0_base]
                    ])
        
        return wall_faces
    
    # Function for creating base faces for a chunk (moved outside for multiprocessing)
    def _create_base_faces_for_chunk(self, start_row, end_row, cols, resampled_water_mask_flat, vertex_count):
        base_faces = []
        
        for i in range(start_row, end_row):
            for j in range(cols - 1):
                v0 = i * cols + j + vertex_count
                v1 = v0 + 1
                v2 = (i + 1) * cols + j + vertex_count
                v3 = v2 + 1
                
                # Skip if we're at the last row
                if i >= self.rows - 1:
                    continue
                
                v0_orig = v0 - vertex_count
                v1_orig = v1 - vertex_count
                v2_orig = v2 - vertex_count
                v3_orig = v3 - vertex_count
                
                if (not resampled_water_mask_flat[v0_orig - vertex_count] and 
                    not resampled_water_mask_flat[v1_orig - vertex_count] and 
                    not resampled_water_mask_flat[v2_orig - vertex_count] and 
                    not resampled_water_mask_flat[v3_orig - vertex_count]):
                    base_faces.extend([
                        [v0, v2, v1],
                        [v1, v2, v3]
                    ])
        
        return base_faces
    
    def _create_terrain_model_from_elevation(self, elevation_data, bounds, detail_level=0.2,
                                           water_level=-15.0, shore_height=1.0, shore_buffer=1,
                                           height_scale=0.05):
        """
        Create a 3D terrain model from elevation data using parallel processing.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) of the region
            detail_level (float): Detail level (1.0 = highest detail, lower values reduce detail)
            water_level (float): Elevation value for water areas (typically negative)
            shore_height (float): Elevation value for shore areas
            shore_buffer (int): Number of cells for shore buffer
            height_scale (float): Scale factor for height relative to horizontal dimensions
            
        Returns:
            trimesh.Scene: The generated terrain scene with water and land as separate meshes
        """
        start_time = time.time()
        print("Starting terrain model creation...")
        
        # Create a pronounced water mask (elevation <= 0 is water)
        water_mask = elevation_data <= 0
        water_coverage = water_mask.sum() / water_mask.size * 100
        
        print(f"Water coverage: {water_coverage:.1f}% of region")
        
        # Make water areas distinctly lower for better visualization
        elevation_data_with_water = elevation_data.copy()
        elevation_data_with_water[water_mask] = water_level
        
        # Create distinct shores
        shore_mask = binary_dilation(water_mask, iterations=shore_buffer) & ~water_mask
        elevation_data_with_water[shore_mask] = shore_height
        
        # Convert geographic coordinates to Web Mercator
        min_lon, min_lat, max_lon, max_lat = bounds
        min_x, min_y = self.transformer.transform(min_lon, min_lat)
        max_x, max_y = self.transformer.transform(max_lon, max_lat)
        
        # Calculate distances in meters
        width_m = max_x - min_x
        height_m = max_y - min_y
        area_km2 = (width_m * height_m) / 1e6
        
        print(f"Area size: {area_km2:.1f} km²")
        
        # Calculate base resolution based on area size
        base_resolution = max(90, area_km2 / 100)
        actual_resolution = base_resolution / detail_level
        print(f"Base resolution: {base_resolution:.1f}m, Detail-adjusted: {actual_resolution:.1f}m")
        
        # Calculate target number of vertices
        target_vertices = int(width_m * height_m / (actual_resolution * actual_resolution))
        target_vertices = min(target_vertices, 50000000)  # Cap at 50 million vertices
        
        # Calculate grid dimensions maintaining aspect ratio
        aspect_ratio = width_m / height_m
        cols = int(np.sqrt(target_vertices * aspect_ratio))
        rows = int(np.sqrt(target_vertices / aspect_ratio))
        
        # Ensure minimum dimensions
        cols = max(cols, 20)
        rows = max(rows, 20)
        
        # Store these values as instance variables so the helper methods can access them
        self.rows = rows
        self.cols = cols
        
        print(f"Grid dimensions: {rows}x{cols} ({rows*cols:,} vertices)")
        
        # Create vertex grid
        x = np.linspace(min_x, max_x, cols)
        y = np.linspace(min_y, max_y, rows)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Resample elevation data to match grid dimensions
        # Handle different aspect ratios between elevation data and target grid
        elev_rows, elev_cols = elevation_data_with_water.shape
        zoom_y = rows / elev_rows
        zoom_x = cols / elev_cols
        
        # Use order=1 for linear interpolation to maintain water boundaries
        print("Resampling elevation data...")
        resampling_start = time.time()
        resampled_data = zoom(elevation_data_with_water, (zoom_y, zoom_x), order=1)
        print(f"Resampling completed in {time.time() - resampling_start:.2f} seconds")
        
        # Re-apply water mask after resampling to maintain clear water boundaries
        resampled_water_mask = resampled_data <= 0
        resampled_data[resampled_water_mask] = water_level
        
        # Create vertices array
        print("Creating vertex array...")
        vertex_start = time.time()
        vertices = np.column_stack((
            x_grid.flatten(),
            y_grid.flatten(),
            resampled_data.flatten()
        ))
        print(f"Vertex array created in {time.time() - vertex_start:.2f} seconds")
        
        # Normalize XY coordinates while preserving aspect ratio
        xy_scale = 1.0 / max(width_m, height_m)
        vertices[:, 0] = (vertices[:, 0] - min_x) * xy_scale
        vertices[:, 1] = (vertices[:, 1] - min_y) * xy_scale

        # Scale the height to some proportion of the width using the height_scale parameter
        vertices[:, 2] = vertices[:, 2] / max(vertices[:, 2]) * height_scale
        
        # Create separate masks for water and land
        vertex_count = rows * cols
        resampled_water_mask_flat = resampled_water_mask.flatten()
        
        # ----- Create separate meshes for water and land using parallel processing -----
        print("Generating terrain faces in parallel...")
        
        # Split the work into chunks based on available CPU cores
        num_cores = multiprocessing.cpu_count()
        chunk_size = max(1, (rows - 1) // num_cores)
        chunks = [(i, min(i + chunk_size, rows - 1)) for i in range(0, rows - 1, chunk_size)]
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
        faces_start = time.time()
        water_faces = []
        land_faces = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Create a partial function with fixed arguments
            create_faces_partial = partial(
                self._create_faces_for_chunk, 
                cols=cols, 
                resampled_water_mask_flat=resampled_water_mask_flat
            )
            
            # Execute in parallel
            future_results = [executor.submit(create_faces_partial, start, end) for start, end in chunks]
            
            # Collect results
            for future in concurrent.futures.as_completed(future_results):
                chunk_water_faces, chunk_land_faces = future.result()
                water_faces.extend(chunk_water_faces)
                land_faces.extend(chunk_land_faces)
        
        print(f"Face generation completed in {time.time() - faces_start:.2f} seconds")
        
        # Create base and side walls for land (this part is less computationally intensive,
        # so we'll keep it serial for now)
        print("Creating base and walls...")
        base_vertices = vertices.copy()
        min_z = vertices[:, 2].min()
        base_vertices[:, 2] = min_z - 0.01
        
        # Combine vertices
        all_vertices = np.vstack([vertices, base_vertices])
        
        # Add side walls for the land mesh
        wall_start = time.time()
        
        # Process side walls in parallel
        
        # Split wall work into chunks
        col_chunks = [(i, min(i + chunk_size, cols - 1)) for i in range(0, cols - 1, chunk_size)]
        row_chunks = [(i, min(i + chunk_size, rows - 1)) for i in range(0, rows - 1, chunk_size)]
        
        # Use ThreadPoolExecutor for walls
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Process column walls
            col_wall_partial = partial(
                self._create_walls_for_range, 
                is_columns=True, 
                resampled_water_mask_flat=resampled_water_mask_flat,
                cols=cols,
                vertex_count=vertex_count
            )
            col_futures = [executor.submit(col_wall_partial, start, end) for start, end in col_chunks]
            
            # Process row walls
            row_wall_partial = partial(
                self._create_walls_for_range, 
                is_columns=False,
                resampled_water_mask_flat=resampled_water_mask_flat,
                cols=cols,
                vertex_count=vertex_count
            )
            row_futures = [executor.submit(row_wall_partial, start, end) for start, end in row_chunks]
            
            # Collect results from all futures
            for future in concurrent.futures.as_completed(col_futures + row_futures):
                land_faces.extend(future.result())
        
        print(f"Wall creation completed in {time.time() - wall_start:.2f} seconds")
        
        # Add base face triangles for land in parallel
        base_start = time.time()
                
        # Use ThreadPoolExecutor for base faces
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Create a partial function with fixed arguments
            create_base_faces_partial = partial(
                self._create_base_faces_for_chunk, 
                cols=cols, 
                resampled_water_mask_flat=resampled_water_mask_flat,
                vertex_count=vertex_count
            )
            
            # Execute in parallel
            base_futures = [executor.submit(create_base_faces_partial, start, end) for start, end in chunks]
            
            # Collect results
            for future in concurrent.futures.as_completed(base_futures):
                land_faces.extend(future.result())
        
        print(f"Base triangle creation completed in {time.time() - base_start:.2f} seconds")
        
        # Convert face lists to numpy arrays
        print("Converting faces to arrays...")
        array_start = time.time()
        water_faces_array = np.array(water_faces) if water_faces else np.empty((0, 3), dtype=np.int64)
        land_faces_array = np.array(land_faces) if land_faces else np.empty((0, 3), dtype=np.int64)
        print(f"Array conversion completed in {time.time() - array_start:.2f} seconds")
        
        # Create separate meshes for water and land
        print("Creating water and land meshes...")
        mesh_start = time.time()
        water_mesh = None
        land_mesh = None
        
        if len(water_faces) > 0:
            # Create water mesh with dark blue material
            water_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=water_faces_array,
                face_colors=[27, 55, 97, 255]  # Dark blue color with alpha
            )
        
        if len(land_faces) > 0:
            # Create land mesh with light gray material
            land_mesh = trimesh.Trimesh(
                vertices=all_vertices, 
                faces=land_faces_array,
                face_colors=[200, 200, 200, 255]  # Light gray color with alpha
            )
        
        print(f"Mesh creation completed in {time.time() - mesh_start:.2f} seconds")
        
        # Combine the meshes into a scene
        scene = trimesh.Scene()
        
        if water_mesh is not None:
            scene.add_geometry(water_mesh, geom_name="water")
        
        if land_mesh is not None:
            scene.add_geometry(land_mesh, geom_name="land")
        
        # Fix normals on all meshes
        normal_start = time.time()
        for mesh in scene.geometry.values():
            mesh.fix_normals()
        print(f"Normal fixing completed in {time.time() - normal_start:.2f} seconds")
        
        # Center the scene - can't use centroid property directly
        transform_start = time.time()
        scene_centroid = scene.centroid
        for mesh in scene.geometry.values():
            mesh.vertices -= scene_centroid
        
        # Scale to a standard size
        scale_factor = 1.0 / max(scene.extents)
        for mesh in scene.geometry.values():
            mesh.apply_scale(scale_factor)
        
        # Ensure proper orientation (Z-up)
        rotation = trimesh.transformations.rotation_matrix(
            angle=-np.pi/2,
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        
        for mesh in scene.geometry.values():
            mesh.apply_transform(rotation)
        
        print(f"Transformation completed in {time.time() - transform_start:.2f} seconds")
        
        # Print mesh information
        print(f"Terrain model created with {len(scene.geometry)} geometries")
        for name, mesh in scene.geometry.items():
            print(f"  {name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        print(f"Total terrain generation time: {time.time() - start_time:.2f} seconds")
        
        # Return the scene object directly
        return scene

    def _generate_debug_visualizations(self, elevation_data, bounds, output_prefix):
        """
        Generate and save debug visualizations for the terrain.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            output_prefix (str): Prefix for output files
        """
        print("Generating debug visualizations...")
        
        # Create land/sea visualization
        self._visualize_land_sea(elevation_data, bounds, output_prefix)
        
        # Create elevation visualization
        self._visualize_elevation(elevation_data, bounds, output_prefix)
    
    def _visualize_land_sea(self, elevation_data, bounds, output_prefix):
        """
        Create and save a visualization of land vs. sea.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            output_prefix (str): Prefix for output files
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create land/sea mask (elevation <= 0 is water)
        water_mask = elevation_data <= 0
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Plot the land/sea mask
        plt.imshow(water_mask, cmap='Blues_r', extent=[min_lon, max_lon, min_lat, max_lat], origin='lower')
        
        # Add color bar and labels
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.set_ticklabels(['Land', 'Water'])
        
        # Add title and labels
        plt.title('Land vs. Sea Visualization')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the visualization
        output_path = f"{output_prefix}_land_sea.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Land/sea visualization saved to {output_path}")
    
    def _visualize_elevation(self, elevation_data, bounds, output_prefix):
        """
        Create and save a visualization of elevation.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            output_prefix (str): Prefix for output files
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Create a terrain colormap
        terrain_cmap = plt.cm.terrain
        
        # Plot the elevation data
        im = plt.imshow(elevation_data, cmap=terrain_cmap, extent=[min_lon, max_lon, min_lat, max_lat], origin='lower')
        
        # Add color bar
        cbar = plt.colorbar(im)
        cbar.set_label('Elevation (m)')
        
        # Add title and labels
        plt.title('Elevation Visualization')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the visualization
        output_path = f"{output_prefix}_elevation.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a 3D visualization of the elevation
        self._visualize_3d_elevation(elevation_data, bounds, output_prefix)
        
        print(f"Elevation visualization saved to {output_path}")
    
    def _visualize_3d_elevation(self, elevation_data, bounds, output_prefix):
        """
        Create and save a 3D visualization of elevation.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            output_prefix (str): Prefix for output files
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create a figure with 3D axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a meshgrid for the plot
        x = np.linspace(min_lon, max_lon, elevation_data.shape[1])
        y = np.linspace(min_lat, max_lat, elevation_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, elevation_data, cmap='terrain', 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Add color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Elevation (m)')
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D Elevation Visualization')
        
        # Adjust the viewing angle for better perspective
        ax.view_init(elev=30, azim=45)
        
        # Save the visualization
        output_path = f"{output_prefix}_elevation_3d.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D elevation visualization saved to {output_path}")
