from abc import abstractmethod
import numpy as np
from geopy.distance import geodesic
from typing import Union

# Import the new console output system
from .console import output

class Elevation:
    def __init__(self):
        pass

    @abstractmethod
    def get_elevation(self, bounds, topo_dir="topo"):
        pass

    def get_model_coordinates(self, elevation_data, bounds, coords: Union[tuple[float, float], list[tuple[float, float]]]) -> tuple[int, int]:
        """
        Convert geographic coordinates to model coordinates

        Args:
            elevation_data (numpy.ndarray): The elevation data to convert to model coordinates
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
            coords (tuple or list of tuples): (lon, lat)

        Returns:
            list of tuples: [(x_index, y_index), ...] or (x_index, y_index)
        """
        if isinstance(coords, tuple):
            coords = [coords]

        indices = []
        for coord in coords:
            x, y = self._get_model_coordinates(elevation_data, bounds, coord)
            indices.append((x, y))

        if len(indices) == 1:
            return indices[0]

        return indices

    def _get_model_coordinates(self, elevation_data, bounds, coord: tuple[float, float]) -> tuple[int, int]:
        """
        Convert a single geographic coordinate to model coordinates

        Args:
            elevation_data (numpy.ndarray): The elevation data to convert to model coordinates
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
            coord (tuple): (lon, lat)

        Returns:
            tuple: (x_index, y_index)
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        lon, lat = coord

        # Calculate distances for SRTM bounds
        ref_point = (min_lat, min_lon)
        x_meters = geodesic(ref_point, (min_lat, lon)).meters
        y_meters = geodesic(ref_point, (lat, min_lon)).meters

        width_meters = geodesic(ref_point, (min_lat, max_lon)).meters
        height_meters = geodesic(ref_point, (max_lat, min_lon)).meters

        x_index = x_meters / width_meters * elevation_data.shape[1]
        y_index = y_meters / height_meters * elevation_data.shape[0]

        return x_index, y_index

    
    def _crop_elevation_data(self, elevation_data, bounds):
        """
        Crop the elevation data to the given bounds

        Args:
            elevation_data (numpy.ndarray): The elevation data to crop
            bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Convert SRTM bounds to geographic coordinates (equivalent to original srtm bounds)
        srtm_min_lon, srtm_min_lat = np.floor(min_lon), np.floor(min_lat)
        srtm_max_lon, srtm_max_lat = np.ceil(max_lon), np.ceil(max_lat)

        # Calculate distances for crop bounds
        ref_point = (srtm_min_lat, srtm_min_lon)
        min_x = geodesic(ref_point, (srtm_min_lat, min_lon)).meters
        min_y = geodesic(ref_point, (min_lat, srtm_min_lon)).meters
        max_x = geodesic(ref_point, (srtm_min_lat, max_lon)).meters
        max_y = geodesic(ref_point, (max_lat, srtm_min_lon)).meters

        # Calculate distances for SRTM bounds
        srtm_min_x = geodesic(
            ref_point, (srtm_min_lat, srtm_min_lon)
        ).meters  # This will be 0
        srtm_min_y = geodesic(
            ref_point, (srtm_min_lat, srtm_min_lon)
        ).meters  # This will be 0
        srtm_max_x = geodesic(ref_point, (srtm_min_lat, srtm_max_lon)).meters
        srtm_max_y = geodesic(ref_point, (srtm_max_lat, srtm_min_lon)).meters

        # Calculate the scale of the elevation data (same as original)
        elevation_data_num_rows, elevation_data_num_cols = elevation_data.shape

        x_scale = elevation_data_num_cols / (srtm_max_x - srtm_min_x)
        y_scale = elevation_data_num_rows / (srtm_max_y - srtm_min_y)

        # Calculate the indices of the elevation data to crop (same logic as original)
        elevation_data_min_index_x = int((min_x - srtm_min_x) * x_scale)
        elevation_data_min_index_y = int((min_y - srtm_min_y) * y_scale)

        elevation_data_max_index_x = elevation_data_num_cols - int(
            (srtm_max_x - max_x) * x_scale
        )
        elevation_data_max_index_y = elevation_data_num_rows - int(
            (srtm_max_y - max_y) * y_scale
        )

        output.info(f"  Cropping elevation data using geopy (original logic):")
        output.info(f"    Original shape: {elevation_data.shape}")
        output.info(
            f"    SRTM bounds: lat {srtm_min_lat}-{srtm_max_lat}, lon {srtm_min_lon}-{srtm_max_lon}"
        )
        output.info(f"    Crop bounds: lat {min_lat}-{max_lat}, lon {min_lon}-{max_lon}")
        output.info(
            f"    SRTM extent: {(srtm_max_x-srtm_min_x)/1000:.2f} km x {(srtm_max_y-srtm_min_y)/1000:.2f} km"
        )
        output.info(
            f"    Scale: {x_scale:.6f} pixels/meter (x), {y_scale:.6f} pixels/meter (y)"
        )
        output.info(
            f"    Crop indices: x {elevation_data_min_index_x}-{elevation_data_max_index_x}, y {elevation_data_min_index_y}-{elevation_data_max_index_y}"
        )

        return elevation_data[
            elevation_data_min_index_y:elevation_data_max_index_y,
            elevation_data_min_index_x:elevation_data_max_index_x,
        ]
