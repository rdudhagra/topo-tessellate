import os
import numpy as np
import rasterio
import glob
import concurrent.futures
import multiprocessing
import time
import numpy as np
from elevation import Elevation


class SRTM(Elevation):
    def _find_required_tiles(self, bounds):
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

    def _find_tile_files(self, required_tiles, topo_dir="topo"):
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

    def get_elevation_data(
        self,
        bounds,
        topo_dir="topo",
    ):
        """
        Generate elevation data using SRTM data with proper north-up orientation.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing SRTM data files


        Returns:
            numpy.ndarray: The generated elevation data
        """
        print(f"Generating elevation data for bounds: {bounds}")

        # Find required tiles
        required_tiles = self._find_required_tiles(bounds)
        tile_files = self._find_tile_files(required_tiles, topo_dir)

        if not tile_files:
            raise ValueError(
                f"No SRTM tiles found in {topo_dir} for the specified bounds"
            )

        print(f"Using {len(tile_files)} SRTM tiles:")
        for coords in tile_files.keys():
            print(f"  N{coords[0]}W{-coords[1]}")

        # Stitch the tiles with standard SRTM arrangement
        elevation_data = self._stitch_srtm_tiles(tile_files)

        # Crop the stitched tiles to the bounds using simple geographic coordinates
        elevation_data = self._crop_elevation_data(elevation_data, bounds, tile_files)

        return elevation_data

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
