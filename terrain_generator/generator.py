import os
import numpy as np
import rasterio
from osgeo import gdal
import trimesh
from pyproj import Transformer
from tqdm import tqdm
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, zoom
import concurrent.futures
import multiprocessing
from functools import partial
import time


class TerrainGenerator:
    def __init__(self):
        """Initialize the TerrainGenerator with coordinate transformer."""
        self.transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )

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
            if hgt_path.endswith(".zip"):
                with zipfile.ZipFile(hgt_path, "r") as zip_ref:
                    # Find the .hgt file in the zip
                    hgt_files = [f for f in zip_ref.namelist() if f.endswith(".hgt")]
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
                                elevation_data = np.where(
                                    elevation_data == ds.nodata, 0, elevation_data
                                )

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
                        elevation_data = np.where(
                            elevation_data == ds.nodata, 0, elevation_data
                        )

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
        print(
            f"Region bounds: lon({min_lon:.4f}, {max_lon:.4f}), lat({min_lat:.4f}, {max_lat:.4f})"
        )

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
            print(
                f"Tile N{lat}W{abs(lon)}: elevation {data.min():.0f}m to {data.max():.0f}m"
            )

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
                print(
                    f"Warning: Adjusting negative offset ({lat_offset}, {lon_offset})"
                )
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

            print(
                f"Source region: [{src_lat_start}:{src_lat_end}, {src_lon_start}:{src_lon_end}]"
            )
            print(
                f"Target region: [{dst_lat_start}:{dst_lat_end}, {dst_lon_start}:{dst_lon_end}]"
            )

            # Extract the valid portion of data and weights
            valid_data = data[src_lat_start:src_lat_end, src_lon_start:src_lon_end]
            valid_weights = weight_mask[
                src_lat_start:src_lat_end, src_lon_start:src_lon_end
            ]

            # Add to the output grid
            merged[dst_lat_start:dst_lat_end, dst_lon_start:dst_lon_end] += (
                valid_data * valid_weights
            )
            weights[
                dst_lat_start:dst_lat_end, dst_lon_start:dst_lon_end
            ] += valid_weights

            # Print progress
            non_zero = np.count_nonzero(merged)
            total = merged.size
            print(f"Progress: {non_zero/total*100:.1f}% of grid filled")

        # Normalize by weights
        valid_mask = weights > 0
        merged[valid_mask] /= weights[valid_mask]

        # Print final statistics
        if np.any(valid_mask):
            print(
                f"\nFinal elevation range: {merged[valid_mask].min():.1f}m to {merged[valid_mask].max():.1f}m"
            )
            print(f"Mean elevation: {merged[valid_mask].mean():.1f}m")
            print(
                f"Coverage: {np.count_nonzero(valid_mask)/valid_mask.size*100:.1f}% of region"
            )
        else:
            print("\nWarning: No valid elevation data in output grid!")

        return merged, pixel_size

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
            scene.export(output_path, file_type="glb")
            print(f"Model exported to {output_path}")
        except Exception as e:
            raise ValueError(f"Error exporting to GLB: {str(e)}")

    def export_obj(self, scene, output_path):
        """
        Export scene to .obj format.

        Args:
            scene (trimesh.Scene): The scene to export
            output_path (str): Path for the output .obj file

        Raises:
            ValueError: If export fails
        """
        try:
            # Create the export directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Export the scene
            scene.export(output_path, file_type="obj")
            print(f"Model exported to {output_path}")
        except Exception as e:
            raise ValueError(f"Error exporting to OBJ: {str(e)}")

    def export_model(self, scene, output_path, export_format):
        """
        Export scene to the specified format.

        Args:
            scene (trimesh.Scene): The scene to export
            output_path (str): Path for the output file
            export_format (str): Format to export ('glb' or 'obj')

        Raises:
            ValueError: If export fails or format is not supported
        """
        if export_format.lower() == "glb":
            self.export_glb(scene, output_path)
        elif export_format.lower() == "obj":
            self.export_obj(scene, output_path)
        else:
            raise ValueError(
                f"Unsupported export format: {export_format}. Supported formats are 'glb' and 'obj'."
            )

    def generate_terrain(
        self,
        bounds,
        topo_dir="topo",
        detail_level=0.2,
        output_prefix="terrain",
        water_level=-15.0,
        shore_height=1.0,
        shore_buffer=1,
        height_scale=0.05,
        water_thickness=100,
        debug=False,
        export_format="glb",
        water_alpha=255,
    ):
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
            export_format (str): Format to export ('glb' or 'obj', default: 'glb')
            water_alpha (int): Alpha transparency value for water (0-255, default: 255)

        Returns:
            trimesh.Trimesh: The generated terrain mesh
        """
        total_start_time = time.time()
        print(f"Generating terrain model for bounds: {bounds}")

        # Find required tiles
        required_tiles = self.find_required_tiles(bounds)
        tile_files = self.find_tile_files(required_tiles, topo_dir)

        if not tile_files:
            raise ValueError(
                f"No SRTM tiles found in {topo_dir} for the specified bounds"
            )

        print(f"Using {len(tile_files)} SRTM tiles:")
        for coords in tile_files.keys():
            print(f"  N{coords[0]}W{-coords[1]}")

        # Stitch the tiles with standard SRTM arrangement
        elevation_data = self._stitch_srtm_tiles(tile_files)

        # Create debug visualizations if requested (in parallel if debug mode is on)
        if debug:
            debug_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            debug_future = debug_executor.submit(
                self._generate_debug_visualizations,
                elevation_data,
                bounds,
                output_prefix,
            )

        # Create the terrain model
        scene = self._create_terrain_model_from_elevation(
            elevation_data,
            bounds,
            detail_level,
            water_level,
            shore_height,
            shore_buffer,
            height_scale,
            water_thickness,
            water_alpha,
        )

        # Wait for debug visualizations to complete if they were requested
        if debug:
            debug_future.result()
            debug_executor.shutdown()

        # Export the model
        output_path = f"{output_prefix}_{detail_level:.3f}.{export_format.lower()}"
        self.export_model(scene, output_path, export_format)

        # Final performance report
        print(
            f"Total terrain generation pipeline time: {time.time() - total_start_time:.2f} seconds"
        )

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
            "data": elevation_data,
            "north": coords[0],
            "south": coords[0] - 1,
            "west": coords[1],
            "east": coords[1] + 1,
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
            results = list(
                executor.map(lambda item: self._read_tile(item), tile_files.items())
            )

        # Convert results to a dictionary
        tile_data = {coords: info for coords, info in results}
        print(
            f"Tile reading completed in {time.time() - start_time:.2f} seconds using {num_cores} cores"
        )

        # Find the dimensions of the tile grid
        all_lats = set(lat for lat, _ in tile_data.keys())
        all_lons = set(lon for _, lon in tile_data.keys())

        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)

        # Calculate grid dimensions
        lat_range = max_lat - min_lat + 1
        lon_range = max_lon - min_lon + 1

        # Find our tile dimensions
        sample_data = next(iter(tile_data.values()))["data"]
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
            merged_data[
                start_row : start_row + tile_height, start_col : start_col + tile_width
            ] = info["data"]

        # Flip the entire grid vertically to get north-up orientation
        # This step makes North at the top and South at the bottom
        print("  Applying vertical flip for north-up orientation")
        merged_data = np.flipud(merged_data)

        return merged_data

    def _create_terrain_model_from_elevation(
        self,
        elevation_data,
        bounds,
        detail_level=0.2,
        water_level=-15.0,
        shore_height=1.0,
        shore_buffer=1,
        height_scale=0.05,
        water_thickness=100,
        water_alpha=255,
    ):
        """
        Create a 3D terrain model from elevation data using parallel processing.

        The land mesh covers the entire model, and water is a thin layer on top.

        Args:
            elevation_data (numpy.ndarray): 2D array of elevation values
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) of the region
            detail_level (float): Detail level (1.0 = highest detail, lower values reduce detail)
            water_level (float): Elevation value for water areas (typically negative)
            shore_height (float): Elevation value for shore areas
            shore_buffer (int): Number of cells for shore buffer
            height_scale (float): Scale factor for height relative to horizontal dimensions
            water_thickness (float): Thickness of water layer in model units
            water_alpha (int): Alpha transparency value for water (0-255, default: 255)

        Returns:
            trimesh.Scene: The generated terrain scene with water and land
        """
        start_time = time.time()
        print("Starting terrain model creation...")

        # Create a water mask (elevation <= 0 is water)
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
        print(
            f"Base resolution: {base_resolution:.1f}m, Detail-adjusted: {actual_resolution:.1f}m"
        )

        # Calculate target number of vertices
        target_vertices = int(
            width_m * height_m / (actual_resolution * actual_resolution)
        )
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

        # Use order=1 for linear interpolation
        print("Resampling elevation data...")
        resampling_start = time.time()
        resampled_data = zoom(elevation_data_with_water, (zoom_y, zoom_x), order=1)
        print(f"Resampling completed in {time.time() - resampling_start:.2f} seconds")

        # Create a clean water mask
        resampled_water_mask = resampled_data <= water_level
        resampled_water_mask_flat = resampled_water_mask.flatten()

        # Create vertices array for the land (covers entire terrain)
        print("Creating vertex array...")
        vertex_start = time.time()
        land_vertices = np.column_stack(
            (x_grid.flatten(), y_grid.flatten(), resampled_data.flatten())
        )

        # Create water vertices - identical to land vertices but only for water areas
        # and with a slight offset upward (water thickness)
        water_vertices = np.array(land_vertices)

        # Set the water top surface to be slightly above the water level
        # This creates a thin water layer on top of the land
        for i in range(len(water_vertices)):
            if resampled_water_mask_flat[i]:
                # Only adjust water vertices
                # Water surface is at water_level + water_thickness
                water_vertices[i, 2] = water_level + water_thickness
                land_vertices[i, 2] = water_level

        print(f"Vertex array created in {time.time() - vertex_start:.2f} seconds")

        # Normalize XY coordinates while preserving aspect ratio
        xy_scale = 1.0 / max(width_m, height_m)
        land_vertices[:, 0] = (land_vertices[:, 0] - min_x) * xy_scale
        land_vertices[:, 1] = (land_vertices[:, 1] - min_y) * xy_scale

        water_vertices[:, 0] = (water_vertices[:, 0] - min_x) * xy_scale
        water_vertices[:, 1] = (water_vertices[:, 1] - min_y) * xy_scale

        # Scale the height to some proportion of the width using the height_scale parameter
        land_vertices[:, 2] = (
            land_vertices[:, 2] * height_scale / max(land_vertices[:, 2].max(), 1)
        )
        water_vertices[:, 2] = (
            water_vertices[:, 2] * height_scale / max(water_vertices[:, 2].max(), 1)
        )

        # ----- Create land mesh (entire terrain) and water mesh (only water areas) -----
        print("Generating terrain meshes...")

        # Create land faces (all quad cells become two triangles)
        land_faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = (i + 1) * cols + j
                v3 = v2 + 1

                # Add two triangles for each grid cell
                land_faces.append([v0, v1, v2])
                land_faces.append([v1, v3, v2])

        # Create water faces (only for water areas)
        water_faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = (i + 1) * cols + j
                v3 = v2 + 1

                # Only add faces if all four vertices are water
                if (
                    resampled_water_mask_flat[v0]
                    and resampled_water_mask_flat[v1]
                    and resampled_water_mask_flat[v2]
                    and resampled_water_mask_flat[v3]
                ):

                    water_faces.append([v0, v1, v2])
                    water_faces.append([v1, v3, v2])

        # Create base vertices at the bottom of the model for the land
        land_base_vertices = land_vertices.copy()
        min_z = land_vertices[:, 2].min()
        land_base_vertices[:, 2] = (
            min_z - 0.01
        )  # Slightly below to ensure no z-fighting

        # Combine top and base vertices for the land
        all_land_vertices = np.vstack([land_vertices, land_base_vertices])
        vertex_count = rows * cols

        # Create land base faces
        land_base_faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j + vertex_count  # Base vertex
                v1 = v0 + 1  # Base vertex
                v2 = (i + 1) * cols + j + vertex_count  # Base vertex
                v3 = v2 + 1  # Base vertex

                # Add two triangles with inverted winding order
                land_base_faces.append([v0, v2, v1])
                land_base_faces.append([v1, v2, v3])

        # Add side walls to land
        land_wall_faces = []

        # Front edge (i = 0)
        for j in range(cols - 1):
            v0 = j
            v1 = j + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            land_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Back edge (i = rows-1)
        for j in range(cols - 1):
            v0 = (rows - 1) * cols + j
            v1 = v0 + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            land_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Left edge (j = 0)
        for i in range(rows - 1):
            v0 = i * cols
            v1 = (i + 1) * cols
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            land_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Right edge (j = cols-1)
        for i in range(rows - 1):
            v0 = i * cols + (cols - 1)
            v1 = (i + 1) * cols + (cols - 1)
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            land_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Combine all land faces
        all_land_faces = land_faces + land_wall_faces + land_base_faces

        # Create base vertices for the water
        water_base_vertices = water_vertices.copy()
        water_base_vertices[:, 2] = water_level

        # Combine top and base vertices for the water
        all_water_vertices = np.vstack([water_vertices, water_base_vertices])

        # Create water base faces
        water_base_faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j + vertex_count  # Base vertex
                v1 = v0 + 1  # Base vertex
                v2 = (i + 1) * cols + j + vertex_count  # Base vertex

        # Add side walls to water
        water_wall_faces = []

        # Front edge (i = 0)
        for j in range(cols - 1):
            v0 = j
            v1 = j + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            water_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Back edge (i = rows-1)
        for j in range(cols - 1):
            v0 = (rows - 1) * cols + j
            v1 = v0 + 1
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            water_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Left edge (j = 0)
        for i in range(rows - 1):
            v0 = i * cols
            v1 = (i + 1) * cols
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            water_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Right edge (j = cols-1)
        for i in range(rows - 1):
            v0 = i * cols + (cols - 1)
            v1 = (i + 1) * cols + (cols - 1)
            v0_base = v0 + vertex_count
            v1_base = v1 + vertex_count
            water_wall_faces.extend([[v0, v1, v0_base], [v1, v1_base, v0_base]])

        # Combine all water faces
        all_water_faces = water_faces + water_wall_faces + water_base_faces

        # Convert to numpy arrays
        land_faces_array = np.array(all_land_faces)
        water_faces_array = np.array(all_water_faces)

        # Create colors for land and water
        land_color = [200, 200, 200, 255]  # Light gray
        water_color = [27, 55, 97, water_alpha]  # Dark blue with configurable alpha

        # Create face colors
        land_face_colors = (
            np.ones((len(all_land_faces), 4), dtype=np.uint8) * land_color
        )
        water_face_colors = (
            np.ones((len(all_water_faces), 4), dtype=np.uint8) * water_color
            if all_water_faces
            else None
        )

        # Create meshes
        print("Creating land and water meshes...")
        mesh_start = time.time()

        # Create the land mesh (covers entire terrain)
        land_mesh = trimesh.Trimesh(
            vertices=all_land_vertices,
            faces=land_faces_array,
            face_colors=land_face_colors,
            process=False,
        )

        # Process the land mesh
        try:
            land_mesh.remove_duplicate_faces()
            land_mesh.remove_infinite_values()
            land_mesh.remove_degenerate_faces()
            land_mesh.fix_normals(multibody=True)
            trimesh.repair.fix_inversion(land_mesh, True)
            trimesh.repair.fix_winding(land_mesh)
            trimesh.repair.broken_faces(land_mesh)
            trimesh.repair.fill_holes(land_mesh)
        except Exception as e:
            print(f"Warning: Land mesh repair encountered an error: {str(e)}")
            print("Attempting simplified repair...")
            try:
                land_mesh.process(validate=False)
            except Exception as e2:
                print(f"Warning: Simplified land mesh repair failed: {str(e2)}")

        # Create the water mesh (only water areas)
        water_mesh = None
        if len(all_water_faces) > 0:
            water_mesh = trimesh.Trimesh(
                vertices=all_water_vertices,
                faces=all_water_faces,
                face_colors=water_face_colors,
                process=False,
            )

            # Process the water mesh
            try:
                water_mesh.remove_duplicate_faces()
                water_mesh.remove_infinite_values()
                water_mesh.remove_degenerate_faces()
                water_mesh.fix_normals(multibody=True)
                trimesh.repair.fix_inversion(water_mesh, True)
                trimesh.repair.fix_winding(water_mesh)
                trimesh.repair.broken_faces(water_mesh)
                trimesh.repair.fill_holes(water_mesh)
            except Exception as e:
                print(f"Warning: Water mesh repair encountered an error: {str(e)}")
                print("Attempting simplified repair...")
                try:
                    water_mesh.process(validate=False)
                except Exception as e2:
                    print(f"Warning: Simplified water mesh repair failed: {str(e2)}")

        print(f"Mesh creation completed in {time.time() - mesh_start:.2f} seconds")

        # Combine the meshes into a scene
        scene = trimesh.Scene()

        # Add the land mesh first (it's the base)
        scene.add_geometry(land_mesh, geom_name="land")

        # Add the water mesh on top if it exists
        if water_mesh is not None:
            scene.add_geometry(water_mesh, geom_name="water")

        # Center the scene
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
            angle=-np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
        )

        for mesh in scene.geometry.values():
            mesh.apply_transform(rotation)

        print(
            f"Transformation completed in {time.time() - transform_start:.2f} seconds"
        )

        # Print mesh information
        print(f"Terrain model created with {len(scene.geometry)} geometries")
        for name, mesh in scene.geometry.items():
            print(f"  {name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            # Report status of boundaries
            boundary_edges = mesh.edges_unique[mesh.edges_unique_length < 2]
            print(f"  {name} mesh has {len(boundary_edges)} boundary edges")

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
        plt.imshow(
            water_mask,
            cmap="Blues_r",
            extent=[min_lon, max_lon, min_lat, max_lat],
            origin="lower",
        )

        # Add color bar and labels
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.set_ticklabels(["Land", "Water"])

        # Add title and labels
        plt.title("Land vs. Sea Visualization")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save the visualization
        output_path = f"{output_prefix}_land_sea.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        im = plt.imshow(
            elevation_data,
            cmap=terrain_cmap,
            extent=[min_lon, max_lon, min_lat, max_lat],
            origin="lower",
        )

        # Add color bar
        cbar = plt.colorbar(im)
        cbar.set_label("Elevation (m)")

        # Add title and labels
        plt.title("Elevation Visualization")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save the visualization
        output_path = f"{output_prefix}_elevation.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        ax = fig.add_subplot(111, projection="3d")

        # Create a meshgrid for the plot
        x = np.linspace(min_lon, max_lon, elevation_data.shape[1])
        y = np.linspace(min_lat, max_lat, elevation_data.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot the surface
        surf = ax.plot_surface(
            X,
            Y,
            elevation_data,
            cmap="terrain",
            linewidth=0,
            antialiased=True,
            alpha=0.8,
        )

        # Add color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Elevation (m)")

        # Set labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Elevation (m)")
        ax.set_title("3D Elevation Visualization")

        # Adjust the viewing angle for better perspective
        ax.view_init(elev=30, azim=45)

        # Save the visualization
        output_path = f"{output_prefix}_elevation_3d.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"3D elevation visualization saved to {output_path}")
