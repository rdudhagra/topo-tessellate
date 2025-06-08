from terrain_generator.buildingsextractor import Building
from terrain_generator.console import output
from geopy.distance import geodesic
from shapely.geometry import Polygon, Point


class BuildingsGeoBins:
    """
    A data structure for quick lookup of buildings by their location.
    The bins are defined by a grid of latitude and longitude bins in a hash table.
    Depending on the size of the bulildings, they can be in multiple bins.

    There are also convenience functions for finding buildings within a certain radius of a given location.
    """

    def __init__(
        self,
        buildings: list[Building],
        bin_size_meters: float = 100,
        debug: bool = True,
    ):
        self.buildings = buildings
        self.bin_size_meters = bin_size_meters
        self.bins: dict[tuple[int, int], list[Building]] = {}
        self.debug = debug
        self.centroid_lat, self.centroid_lon = self.get_buildings_centroid()
        self.build_bins()

    def get_buildings_centroid(self):
        """Get the centroid of all the buildings to use a reference for geodesic calculations."""
        centroid_lat = sum(
            building.polygon.centroid.y for building in self.buildings
        ) / len(self.buildings)
        centroid_lon = sum(
            building.polygon.centroid.x for building in self.buildings
        ) / len(self.buildings)
        return centroid_lat, centroid_lon

    def add_building(self, building: Building):
        """Add a building to the bins."""
        minlon, minlat, maxlon, maxlat = building.polygon.bounds

        minx = geodesic((minlat, self.centroid_lon), (minlat, minlon)).meters
        miny = geodesic((self.centroid_lat, minlon), (minlat, minlon)).meters
        maxx = geodesic((maxlat, self.centroid_lon), (maxlat, maxlon)).meters
        maxy = geodesic((self.centroid_lat, maxlon), (maxlat, maxlon)).meters

        for x_bin in range(
            int(minx / self.bin_size_meters), int(maxx / self.bin_size_meters) + 1
        ):
            for y_bin in range(
                int(miny / self.bin_size_meters), int(maxy / self.bin_size_meters) + 1
            ):
                if (x_bin, y_bin) not in self.bins:
                    self.bins[(x_bin, y_bin)] = []
                self.bins[(x_bin, y_bin)].append(building)

    def remove_building(self, building: Building):
        """Remove a building from the bins."""
        minlon, minlat, maxlon, maxlat = building.polygon.bounds

        minx = geodesic((minlat, self.centroid_lon), (minlat, minlon)).meters
        miny = geodesic((self.centroid_lat, minlon), (minlat, minlon)).meters
        maxx = geodesic((maxlat, self.centroid_lon), (maxlat, maxlon)).meters
        maxy = geodesic((self.centroid_lat, maxlon), (maxlat, maxlon)).meters

        for x_bin in range(
            int(minx / self.bin_size_meters), int(maxx / self.bin_size_meters) + 1
        ):
            for y_bin in range(
                int(miny / self.bin_size_meters), int(maxy / self.bin_size_meters) + 1
            ):
                if (x_bin, y_bin) in self.bins:
                    self.bins[(x_bin, y_bin)].remove(building)
                    if len(self.bins[(x_bin, y_bin)]) == 0:
                        del self.bins[(x_bin, y_bin)]

    def build_bins(self):
        if self.debug:
            output.info(f"Building bins with size {self.bin_size_meters} meters...")

        for i, building in enumerate(self.buildings):
            if self.debug:
                if i % 1000 == 0:
                    output.info(f"Building {i} of {len(self.buildings)}...")

            self.add_building(building)

    def get_buildings_in_radius(
        self, lat: float, lon: float, radius_meters: float
    ) -> list[Building]:
        """
        Get all buildings within a certain radius of a given location.

        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            radius_meters: Radius in meters

        Returns:
        """
        x = geodesic((lat, self.centroid_lon), (lat, lon)).meters
        y = geodesic((self.centroid_lat, lon), (lat, lon)).meters
        buffer = Point(x, y).buffer(radius_meters)
        buildings = []

        minx = x - radius_meters
        maxx = x + radius_meters
        miny = y - radius_meters
        maxy = y + radius_meters

        for x_bin in range(
            int(minx / self.bin_size_meters), int(maxx / self.bin_size_meters) + 1
        ):
            for y_bin in range(
                int(miny / self.bin_size_meters), int(maxy / self.bin_size_meters) + 1
            ):
                if (x_bin, y_bin) in self.bins:
                    for building in self.bins[(x_bin, y_bin)]:
                        if building.polygon.within(buffer):
                            buildings.append(building)

        return buildings


class BuildingsProcessor:
    def exclude_buildings_outside_bbox(
        self, buildings: list[Building], bbox: tuple[float, float, float, float]
    ) -> list[Building]:
        return [building for building in buildings if building.is_inside_bbox(bbox)]
