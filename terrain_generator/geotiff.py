import os
import glob
from contextlib import ExitStack
import numpy as np
import rasterio
from rasterio.warp import (
    Resampling,
    transform_bounds,
)
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from .elevation import Elevation

# Import the new console output system
from .console import output


class GeoTiff(Elevation):
    def __init__(self, file_names=None, glob_pattern=None):
        super().__init__()
        # file_names: Optional[List[str]] relative to topo_dir
        # glob_pattern: Optional[str] glob under topo_dir (e.g., "*.tif")
        self.file_names = list(file_names) if file_names else None
        self.glob_pattern = str(glob_pattern) if glob_pattern else None

    @staticmethod
    def _bounds_intersect(a, b):
        # a, b are (min_lon, min_lat, max_lon, max_lat)
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def _discover_candidate_files(self, topo_dir: str) -> list[str]:
        candidates: list[str] = []
        # Include explicit file names (relative to topo_dir)
        if self.file_names:
            for fn in self.file_names:
                candidates.append(os.path.join(topo_dir, fn))
        # Include glob results
        patterns = []
        if self.glob_pattern:
            patterns.append(self.glob_pattern)
        if not self.file_names and not patterns:
            # Default discovery of common raster extensions
            patterns.extend(["*.tif", "*.tiff", "*.TIF", "*.TIFF"])
        for pattern in patterns:
            for p in glob.glob(os.path.join(topo_dir, pattern)):
                candidates.append(p)
        # Deduplicate while preserving order
        seen = set()
        unique: list[str] = []
        for p in candidates:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        # Sort for deterministic order
        unique.sort()
        return unique

    def get_elevation(self, bounds, topo_dir="topo"):
        """
        Extract elevation data by mosaicking one or more GeoTIFFs matching bounds.

        Args:
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat) for the region
            topo_dir (str): Directory containing GeoTIFF files

        Returns:
            numpy.ndarray: The extracted elevation data
        """
        output.progress_info(f"Extracting elevation data from GeoTIFF(s) for bounds: {bounds}")

        # Discover candidate files
        candidate_paths = self._discover_candidate_files(topo_dir)
        if not candidate_paths:
            raise ValueError(f"No GeoTIFF files found in '{topo_dir}'")

        # Filter by intersection with requested bounds (compare in EPSG:4326)
        matched: list[str] = []
        for p in candidate_paths:
            try:
                with rasterio.open(p) as ds:
                    ds_bounds = ds.bounds
                    ds_crs = ds.crs
                    if ds_crs and str(ds_crs).upper() != "EPSG:4326":
                        try:
                            b = transform_bounds(ds_crs, "EPSG:4326", *ds_bounds, densify_pts=21)
                            ds_wgs84 = (b[0], b[1], b[2], b[3])
                        except Exception:
                            # Fallback: assume original bounds
                            ds_wgs84 = (ds_bounds.left, ds_bounds.bottom, ds_bounds.right, ds_bounds.top)
                    else:
                        ds_wgs84 = (ds_bounds.left, ds_bounds.bottom, ds_bounds.right, ds_bounds.top)
                    if self._bounds_intersect(bounds, ds_wgs84):
                        matched.append(p)
            except Exception as exc:
                output.warning(f"  Skipping '{p}': {exc}")

        if not matched:
            raise ValueError(
                f"No GeoTIFF files in '{topo_dir}' intersect requested bounds {bounds}"
            )

        output.info(f"  Using {len(matched)} GeoTIFF file(s):")
        for p in matched:
            output.info(f"    - {os.path.basename(p)}")

        # Build VRTs in EPSG:4326 and merge within requested bounds
        dst_crs = "EPSG:4326"
        with ExitStack() as stack:
            vrts = []
            for p in matched:
                ds = stack.enter_context(rasterio.open(p))
                if not ds.crs or str(ds.crs).upper() != dst_crs:
                    vrt = stack.enter_context(
                        WarpedVRT(ds, crs=dst_crs, resampling=Resampling.nearest)
                    )
                else:
                    # Use dataset directly if already in EPSG:4326
                    vrt = ds
                vrts.append(vrt)

            # Merge on-the-fly, clipped to bounds
            mosaic, out_transform = rio_merge(
                vrts,
                bounds=bounds,
                nodata=0,
            )

        # Use the first band (assumed elevation)
        elevation_data = mosaic[0] if getattr(mosaic, "ndim", 0) == 3 else mosaic
        if elevation_data is None:
            raise ValueError("Failed to assemble elevation mosaic (no data)")

        # Replace any remaining nodata (if present) with 0
        elevation_data = np.where(np.isnan(elevation_data), 0, elevation_data)

        # Vertically flip to maintain prior orientation expectations
        elevation_data = np.flipud(elevation_data)

        output.success(f"  Extracted elevation data shape: {elevation_data.shape}")
        try:
            output.info(
                f"  Elevation range: {np.nanmin(elevation_data):.1f}m to {np.nanmax(elevation_data):.1f}m"
            )
        except Exception:
            pass
        
        return elevation_data
