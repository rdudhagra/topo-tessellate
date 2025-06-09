import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from .elevation import Elevation

# Import the new console output system
from .console import output


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
        output.progress_info(f"Extracting elevation data from GeoTIFF for bounds: {bounds}")
        
        min_lon, min_lat, max_lon, max_lat = bounds

        file_path = os.path.join(topo_dir, self.file_name)
        
        with rasterio.open(file_path) as src:
            output.info(f"  GeoTIFF file: {file_path}")
            output.info(f"  File bounds: {src.bounds}")
            output.info(f"  File CRS: {src.crs}")
            output.info(f"  File shape: {src.shape}")
            output.info(f"  File transform: {src.transform}")

            # First convert the CRS to EPSG:4326
            dst_crs = "EPSG:4326"
            if src.crs != dst_crs:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                
                # Prepare the output metadata
                kwargs = src.meta.copy()  # Start with the source metadata
                kwargs.update({           # Update with the new values
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'driver': 'GTiff'  # Specify the output format
                })
                
                # Reproject each band
                with rasterio.open(f"{file_path}.transformed", 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)
                        
                src = rasterio.open(f"{file_path}.transformed")
            
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
                output.warning(f"  Bounds clamped to file extent: {clamped_bounds}")
            
            # Get the window that corresponds to the bounds
            window = from_bounds(*clamped_bounds, src.transform)
            
            output.info(f"  Reading window: {window}")
            
            # Read the elevation data for the window
            elevation_data = src.read(1, window=window)
            
            # Handle no-data values
            if src.nodata is not None:
                elevation_data = np.where(
                    elevation_data == src.nodata, 0, elevation_data
                )

            # Vertically flip the elevation data
            elevation_data = np.flipud(elevation_data)
            
            output.success(f"  Extracted elevation data shape: {elevation_data.shape}")
            output.info(f"  Elevation range: {elevation_data.min():.1f}m to {elevation_data.max():.1f}m")
            
            return elevation_data
