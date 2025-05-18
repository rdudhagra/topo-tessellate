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
                        
                        # Open the dataset with GDAL
                        ds = gdal.Open(hgt_file_path)
                        if ds is None:
                            raise ValueError("Failed to open the dataset")
                            
                        # Get geotransform (affine transformation coefficients)
                        geotransform = ds.GetGeoTransform()
                        pixel_size = abs(geotransform[1])  # size of pixel in degrees
                        
                        # Read elevation data
                        band = ds.GetRasterBand(1)
                        elevation_data = band.ReadAsArray()
                        
                        # Handle no-data values
                        no_data = band.GetNoDataValue()
                        if no_data is not None:
                            elevation_data = np.where(elevation_data == no_data, 0, elevation_data)
                        
                        ds = None  # Close the dataset
                        return elevation_data, pixel_size
            else:
                # Handle non-zipped .hgt files
                ds = gdal.Open(hgt_path)
                if ds is None:
                    raise ValueError("Failed to open the dataset")
                    
                # Get geotransform (affine transformation coefficients)
                geotransform = ds.GetGeoTransform()
                pixel_size = abs(geotransform[1])  # size of pixel in degrees
                
                # Read elevation data
                band = ds.GetRasterBand(1)
                elevation_data = band.ReadAsArray()
                
                # Handle no-data values
                no_data = band.GetNoDataValue()
                if no_data is not None:
                    elevation_data = np.where(elevation_data == no_data, 0, elevation_data)
                
                ds = None  # Close the dataset
                return elevation_data, pixel_size
            
            # Get geotransform (affine transformation coefficients)
            geotransform = ds.GetGeoTransform()
            pixel_size = geotransform[1]  # size of pixel in degrees
            
            # Read elevation data
            band = ds.GetRasterBand(1)
            elevation_data = band.ReadAsArray()
            
            # Handle no-data values
            no_data = band.GetNoDataValue()
            if no_data is not None:
                elevation_data = np.where(elevation_data == no_data, 0, elevation_data)
            
            ds = None  # Close the dataset
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
    
    def export_glb(self, mesh, output_path):
        """
        Export mesh to .glb format.
        
        Args:
            mesh (trimesh.Trimesh): The mesh to export
            output_path (str): Path for the output .glb file
            
        Raises:
            ValueError: If export fails
        """
        try:
            # Create the export directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export the mesh
            mesh.export(output_path, file_type='glb')
        except Exception as e:
            raise ValueError(f"Error exporting to GLB: {str(e)}")

    def generate_terrain(self, bounds, topo_dir="topo", detail_level=0.2, output_prefix="terrain",
                       water_level=-15.0, shore_height=1.0, shore_buffer=1, height_scale=0.05):
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
            
        Returns:
            trimesh.Trimesh: The generated terrain mesh
        """
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
        
        # Create the terrain model
        mesh = self._create_terrain_model_from_elevation(
            elevation_data, 
            bounds, 
            detail_level, 
            water_level, 
            shore_height, 
            shore_buffer,
            height_scale
        )
        
        # Export the model
        output_path = f"{output_prefix}_{detail_level:.3f}.glb"
        self.export_glb(mesh, output_path)
        
        return mesh
    
    def _stitch_srtm_tiles(self, tile_files):
        """
        Stitch SRTM tiles with proper north-up orientation.
        
        Args:
            tile_files (dict): Dictionary mapping (lat, lon) to file paths
            
        Returns:
            numpy.ndarray: Combined elevation data with proper orientation
        """
        print("Reading and arranging SRTM tiles...")
        tile_data = {}
        
        # Read all tiles
        for (lat, lon), file_path in tile_files.items():
            # Read the raw elevation data
            elevation_data, _ = self.read_hgt_file(file_path)
            
            # Store the data and coordinates
            tile_data[(lat, lon)] = {
                'data': elevation_data,
                'north': lat,
                'south': lat - 1,
                'west': lon,
                'east': lon + 1
            }
        
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
    
    def _create_terrain_model_from_elevation(self, elevation_data, bounds, detail_level=0.2,
                                           water_level=-15.0, shore_height=1.0, shore_buffer=1,
                                           height_scale=0.05):
        """
        Create a 3D terrain model from elevation data.
        
        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) of the region
            detail_level (float): Detail level (1.0 = highest detail, lower values reduce detail)
            water_level (float): Elevation value for water areas (typically negative)
            shore_height (float): Elevation value for shore areas
            shore_buffer (int): Number of cells for shore buffer
            height_scale (float): Scale factor for height relative to horizontal dimensions
            
        Returns:
            trimesh.Trimesh: The generated terrain mesh
        """
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
        resampled_data = zoom(elevation_data_with_water, (zoom_y, zoom_x), order=1)
        
        # Re-apply water mask after resampling to maintain clear water boundaries
        resampled_water = resampled_data <= 0
        resampled_data[resampled_water] = water_level
        
        # Create vertices array
        vertices = np.column_stack((
            x_grid.flatten(),
            y_grid.flatten(),
            resampled_data.flatten()
        ))
        
        # Normalize XY coordinates while preserving aspect ratio
        xy_scale = 1.0 / max(width_m, height_m)
        vertices[:, 0] = (vertices[:, 0] - min_x) * xy_scale
        vertices[:, 1] = (vertices[:, 1] - min_y) * xy_scale

        # Scale the height to some proportion of the width using the height_scale parameter
        vertices[:, 2] = vertices[:, 2] / max(vertices[:, 2]) * height_scale
        
        # Create mesh faces
        faces = []
        for i in tqdm(range(rows - 1), desc="Creating surface triangles"):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = (i + 1) * cols + j
                v3 = v2 + 1
                
                # Correct winding order for proper rendering
                faces.extend([
                    [v0, v1, v2],
                    [v1, v3, v2]
                ])
        
        # Create base and sides
        base_vertices = vertices.copy()
        min_z = vertices[:, 2].min()
        base_vertices[:, 2] = min_z - 0.01
        
        # Combine vertices
        all_vertices = np.vstack([vertices, base_vertices])
        vertex_count = rows * cols
        
        # Add side walls
        for j in tqdm(range(cols - 1), desc="Creating walls"):
            # Front edge (i=0)
            v0, v1 = j, j+1
            v0_base, v1_base = v0 + vertex_count, v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
            
            # Back edge (i=rows-1)
            v0 = (rows - 1) * cols + j
            v1 = v0 + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
        
        for i in range(rows - 1):
            # Left edge (j=0)
            v0 = i * cols
            v1 = (i + 1) * cols
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
            
            # Right edge (j=cols-1)
            v0 = i * cols + (cols - 1)
            v1 = (i + 1) * cols + (cols - 1)
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            faces.extend([
                [v0, v1, v0_base],
                [v1, v1_base, v0_base]
            ])
        
        # Add base face triangles
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j + vertex_count
                v1 = v0 + 1
                v2 = (i + 1) * cols + j + vertex_count
                v3 = v2 + 1
                faces.extend([
                    [v0, v1, v2],
                    [v1, v3, v2]
                ])
        
        # Create mesh
        faces = np.array(faces)
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces)
        
        # Fix normals for proper rendering
        mesh.fix_normals()
        
        # Center the mesh
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        
        # Scale to 1 meter length
        scale_factor = 1.0 / max(mesh.extents)
        mesh.apply_scale(scale_factor)

        # Print dimensions of the mesh
        print(f"Mesh dimensions: {mesh.bounds}")
        
        # Ensure proper orientation (Z-up) for Unity/other 3D software
        # This rotates the model to have Z axis pointing up instead of Y axis
        rotation = trimesh.transformations.rotation_matrix(
            angle=-np.pi/2,
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation)
        
        return mesh
