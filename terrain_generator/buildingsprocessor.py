from terrain_generator.buildingbase import Building
from terrain_generator.console import output
from geopy.point import Point as GeopyPoint
from geopy.distance import geodesic, distance
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely import unary_union
import random
from collections import defaultdict


class BuildingCoordinatesWrapper:
    """A wrapper for a building's coordinates."""

    def __init__(self, building: Building, ref_lat: float, ref_lon: float):
        self.building = building
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon

        # Pre-calculate bounds for performance
        minlon, minlat, _, _ = building.polygon.bounds
        assert (
            ref_lat <= minlat and ref_lon <= minlon
        ), "Reference lat/lon must be less than the building's min lat/lon"

        geodesic_poly = []
        for coord in building.polygon.exterior.coords:
            lon, lat = coord
            x = geodesic((lat, ref_lon), (lat, lon)).meters
            y = geodesic((ref_lat, lon), (lat, lon)).meters
            geodesic_poly.append((x, y))

        self.geodesic_polygon = Polygon(geodesic_poly)
        # Cache centroid for repeated access
        self._centroid = None

    @property
    def centroid(self):
        """Cached centroid property to avoid repeated calculations."""
        if self._centroid is None:
            self._centroid = self.geodesic_polygon.centroid
        return self._centroid

    def __hash__(self):
        return hash((self.building.osm_id, self.geodesic_polygon.wkt))

    def __eq__(self, other):
        return (
            self.building.osm_id == other.building.osm_id
            and self.geodesic_polygon == other.geodesic_polygon
        )


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
        self.bins: dict[tuple[int, int], list[BuildingCoordinatesWrapper]] = (
            defaultdict(list)
        )
        self.debug = debug
        self.min_lat, self.min_lon = self.get_buildings_min()
        self.build_bins()

    def get_buildings_min(self):
        """Get the minimum latitude and longitude of all the buildings to use a reference for geodesic calculations."""
        # Optimized to compute min values in a single pass
        min_lat = float("inf")
        min_lon = float("inf")

        for building in self.buildings:
            bounds = building.polygon.bounds
            min_lon = min(min_lon, bounds[0])
            min_lat = min(min_lat, bounds[1])

        return min_lat, min_lon

    def add_building(self, building: Building | BuildingCoordinatesWrapper):
        """Add a building to the bins."""
        if isinstance(building, BuildingCoordinatesWrapper):
            building_wrapper = building
        else:
            building_wrapper = BuildingCoordinatesWrapper(
                building, self.min_lat, self.min_lon
            )

        minx, miny, maxx, maxy = building_wrapper.geodesic_polygon.bounds

        for x_bin in range(
            int(minx / self.bin_size_meters), int(maxx / self.bin_size_meters) + 1
        ):
            for y_bin in range(
                int(miny / self.bin_size_meters), int(maxy / self.bin_size_meters) + 1
            ):
                self.bins[(x_bin, y_bin)].append(building_wrapper)

    def remove_building(self, building: Building | BuildingCoordinatesWrapper):
        """Remove a building from the bins."""
        if isinstance(building, BuildingCoordinatesWrapper):
            building_wrapper = building
        else:
            building_wrapper = BuildingCoordinatesWrapper(
                building, self.min_lat, self.min_lon
            )

        minx, miny, maxx, maxy = building_wrapper.geodesic_polygon.bounds

        for x_bin in range(
            int(minx / self.bin_size_meters), int(maxx / self.bin_size_meters) + 1
        ):
            for y_bin in range(
                int(miny / self.bin_size_meters), int(maxy / self.bin_size_meters) + 1
            ):
                bin_key = (x_bin, y_bin)
                if bin_key in self.bins and building_wrapper in self.bins[bin_key]:
                    self.bins[bin_key].remove(building_wrapper)
                    if len(self.bins[bin_key]) == 0:
                        del self.bins[bin_key]

    def build_bins(self):
        if self.debug:
            output.info(f"Building bins with size {self.bin_size_meters} meters...")

        for i, building in enumerate(self.buildings):
            if self.debug and i % 1000 == 0:
                output.info(f"Building {i} of {len(self.buildings)}...")

            self.add_building(building)

    def get_building_wrappers_within_radius(
        self, x: float, y: float, radius_meters: float
    ) -> list[BuildingCoordinatesWrapper]:
        """
        Get all buildings within a certain radius of a given location.

        Args:
            x: X coordinate of the location
            y: Y coordinate of the location
            radius_meters: Radius in meters

        Returns:
        """
        buffer = Point(x, y).buffer(radius_meters)
        building_wrappers_set = set()  # Use set to avoid duplicates

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
                    for building_wrapper in self.bins[(x_bin, y_bin)]:
                        if building_wrapper.geodesic_polygon.within(buffer):
                            building_wrappers_set.add(building_wrapper)

        return list(building_wrappers_set)

    def is_empty(self) -> bool:
        """Check if the bins are empty."""
        return len(self.bins) == 0

    def __bool__(self) -> bool:
        """Check if the bins are not empty."""
        return not self.is_empty()


class BuildingsProcessor:
    def __init__(self, buildings: list[Building]):
        self.buildings = buildings

    def exclude_buildings_outside_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> list[Building]:
        self.buildings = [
            building for building in self.buildings if building.is_inside_bbox(bbox)
        ]
        return self.buildings

    def cluster_and_merge_buildings(
        self, max_building_distance_meters: float = 35
    ) -> list[list[Building]]:
        """Cluster buildings into groups based on their distance from each other.
        Then, combine the polygons of the buildings in each group into a single polygon.
        """
        output.info(f"Clustering and merging {len(self.buildings)} buildings...")

        # Create a geobins object
        geo_bins = BuildingsGeoBins(self.buildings)

        # List of new buildings to return
        new_buildings = []
        processed_buildings = 0
        last_update_printout_number = 0

        # Cache keys list to avoid repeated conversion
        bin_keys = list(geo_bins.bins.keys())

        while geo_bins:
            if processed_buildings > last_update_printout_number:
                output.info(f"Processed {processed_buildings}/{len(self.buildings)} buildings...")
                last_update_printout_number += 1000

            # Pick a random building from the bins - use cached keys
            if not bin_keys:
                bin_keys = list(geo_bins.bins.keys())

            bin_key = random.choice(bin_keys)
            if bin_key not in geo_bins.bins:  # Key might have been removed
                bin_keys.remove(bin_key)
                continue

            building_wrapper = random.choice(geo_bins.bins[bin_key])

            # Skip the building if significant (don't cluster it)
            if building_wrapper.building.significant:
                new_buildings.append(building_wrapper.building)
                geo_bins.remove_building(building_wrapper)
                continue

            building_wrappers_in_cluster = {building_wrapper}

            # Create a queue of buildings to query
            building_wrappers_query_queue = [building_wrapper]
            geo_bins.remove_building(building_wrapper)

            while building_wrappers_query_queue:
                building_wrapper = building_wrappers_query_queue.pop(0)

                building_wrappers_in_radius = (
                    geo_bins.get_building_wrappers_within_radius(
                        building_wrapper.centroid.x,  # Use cached centroid
                        building_wrapper.centroid.y,
                        max_building_distance_meters,
                    )
                )

                new_wrappers = []
                for wrapper in building_wrappers_in_radius:
                    if wrapper not in building_wrappers_in_cluster:
                        building_wrappers_in_cluster.add(wrapper)
                        new_wrappers.append(wrapper)
                        geo_bins.remove_building(wrapper)

                building_wrappers_query_queue.extend(new_wrappers)

            # Convert set back to list for merge_buildings
            building_wrappers_list = list(building_wrappers_in_cluster)

            # Combine the polygons of the buildings in the cluster into a single polygon
            new_buildings.extend(
                self.merge_buildings(
                    building_wrappers_list, geo_bins, max_building_distance_meters
                )
            )

            processed_buildings += len(building_wrappers_list)

        return new_buildings

    def merge_buildings(
        self,
        building_wrappers: list[BuildingCoordinatesWrapper],
        geo_bins: BuildingsGeoBins,
        max_building_distance_meters: float,
    ) -> list[Building]:
        """Merge buildings into a single polygon. If the result is a MultiPolygon, return a list of buildings."""
        # Get all the polygons from the buildings
        geodesic_polys: list[Polygon] = [
            building_wrapper.geodesic_polygon for building_wrapper in building_wrappers
        ]

        grown = [
            p.buffer(+max_building_distance_meters) for p in geodesic_polys
        ]  # dilate each polygon, buffer by d meters
        merged = unary_union(grown)  # dissolve overlaps fast - use direct import
        result_poly = merged.buffer(
            -max_building_distance_meters
        )  # erode back to near-original size

        # If the result is a MultiPolygon, iterate over each of the sub-polygons
        # Else, use the result
        if isinstance(result_poly, MultiPolygon):
            results = result_poly.geoms
        else:
            results = [result_poly]

        # Convert the polygon back to lat/lon coordinates
        new_buildings: list[Building] = []

        for result in results:
            lat_lon_poly = []
            for coord in result.exterior.coords:
                x, y = coord
                # Create new point for each coordinate transformation
                point = GeopyPoint(geo_bins.min_lat, geo_bins.min_lon)
                point = distance(meters=x).destination(point=point, bearing=90)
                point = distance(meters=y).destination(point=point, bearing=0)
                lat_lon_poly.append((point.longitude, point.latitude))

            new_poly = Polygon(lat_lon_poly)

            # Cache max height calculation
            max_height = max(
                building_wrapper.building.height
                for building_wrapper in building_wrappers
            )

            new_building = Building(
                polygon=new_poly,
                height=max_height,
                building_type=building_wrappers[0].building.building_type,
                osm_id=building_wrappers[0].building.osm_id,
                area=new_poly.area,
            )
            new_building_wrapper = BuildingCoordinatesWrapper(
                new_building,
                new_building.polygon.bounds[1],
                new_building.polygon.bounds[0],
            )
            new_building.area = new_building_wrapper.geodesic_polygon.area

            new_buildings.append(new_building)
        return new_buildings
