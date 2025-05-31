import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from .elevation import Elevation


class GeoTiff(Elevation):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def get_elevation(self, bounds, topo_dir="topo"):
        """
        Extract elevation data from GeoTIFF file for the given bounds.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing the GeoTIFF file (not used, kept for interface compatibility)

        Returns:
            numpy.ndarray: The extracted elevation data
        """
        print(f"Extracting elevation data from GeoTIFF for bounds: {bounds}")
        
        min_lon, min_lat, max_lon, max_lat = bounds

        file_path = os.path.join(topo_dir, self.file_name)
        
        try:
            with rasterio.open(file_path) as src:
                print(f"  GeoTIFF file: {file_path}")
                print(f"  File bounds: {src.bounds}")
                print(f"  File CRS: {src.crs}")
                print(f"  File shape: {src.shape}")
                print(f"  File transform: {src.transform}")
                
                # Check if the requested bounds overlap with the file bounds
                file_bounds = src.bounds
                if (max_lon < file_bounds.left or min_lon > file_bounds.right or
                    max_lat < file_bounds.bottom or min_lat > file_bounds.top):
                    raise ValueError(f"Requested bounds {bounds} do not overlap with GeoTIFF bounds {file_bounds}")
                
                # Clamp the bounds to the file bounds to avoid errors
                clamped_bounds = (
                    max(min_lon, file_bounds.left),
                    max(min_lat, file_bounds.bottom),
                    min(max_lon, file_bounds.right),
                    min(max_lat, file_bounds.top)
                )
                
                if clamped_bounds != bounds:
                    print(f"  Bounds clamped to file extent: {clamped_bounds}")
                
                # Get the window that corresponds to the bounds
                window = from_bounds(*clamped_bounds, src.transform)
                
                print(f"  Reading window: {window}")
                
                # Read the elevation data for the window
                elevation_data = src.read(1, window=window)
                
                # Handle no-data values
                if src.nodata is not None:
                    elevation_data = np.where(
                        elevation_data == src.nodata, 0, elevation_data
                    )

                # Vertically flip the elevation data
                elevation_data = np.flipud(elevation_data)
                
                print(f"  Extracted elevation data shape: {elevation_data.shape}")
                print(f"  Elevation range: {elevation_data.min():.1f}m to {elevation_data.max():.1f}m")
                
                return elevation_data
                
        except Exception as e:
            raise ValueError(f"Error reading GeoTIFF file {self.file_path}: {str(e)}")
