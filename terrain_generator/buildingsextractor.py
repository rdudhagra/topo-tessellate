#!/usr/bin/env python3
"""
Buildings Data Extractor for 3D Modeling

This module extracts building and structure data from OpenStreetMap for a given
latitude/longitude bounding box, focusing on buildings with polygon coordinates.
Uses a default height of 5 meters for buildings without explicit height data.
"""

import requests
import json
import time
import pickle
import gzip
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from shapely.geometry import Polygon

# Import the new console output system
from .console import output


@dataclass
class Building:
    """Represents a building extracted from OpenStreetMap with polygon and height data."""

    osm_id: int
    building_type: str
    polygon: Polygon
    area: float
    height: float

    def is_inside_bbox(self, bbox: tuple[float, float, float, float]) -> bool:
        """Check if the building is inside the given bbox."""
        min_lon, min_lat, max_lon, max_lat = bbox

        # Validate bbox: min values should be less than max values
        if min_lon >= max_lon or min_lat >= max_lat:
            return False

        return Polygon(
            [
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
            ]
        ).contains(self.polygon)
    
    def __hash__(self):
        return hash((self.osm_id, self.polygon.wkt))


class BuildingsExtractor:
    """Extracts building data from OpenStreetMap using the Overpass API, with default height support."""

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    CACHE_DIR = "building_cache"
    DEFAULT_HEIGHT = 5.0  # Default height in meters for buildings without height data

    # Building types we want to include (major structures)
    MAJOR_BUILDING_TYPES = {
        "yes",
        "house",
        "residential",
        "apartments",
        "commercial",
        "office",
        "retail",
        "industrial",
        "warehouse",
        "school",
        "university",
        "hospital",
        "hotel",
        "church",
        "mosque",
        "synagogue",
        "temple",
        "civic",
        "government",
        "public",
        "sports_hall",
        "stadium",
        "theatre",
        "cinema",
        "museum",
        "library",
        "fire_station",
        "police",
        "prison",
        "courthouse",
        "embassy",
        "supermarket",
        "mall",
        "shop",
        "restaurant",
        "cafe",
        "bar",
        "pub",
        "bank",
        "clinic",
        "pharmacy",
        "factory",
        "garage",
        "hangar",
        "barn",
        "greenhouse",
        "cathedral",
        "chapel",
        "monastery",
        "tower",
        "bunker",
        "castle",
        "fort",
        "palace",
        "manor",
    }

    # Tags to exclude (small structures, utilities, etc.)
    EXCLUDED_TAGS = {
        "building": [
            "roof",
            "entrance",
            "steps",
            "porch",
            "balcony",
            "canopy",
            "carport",
            "shed",
            "hut",
            "cabin",
            "kiosk",
            "shelter",
            "toilets",
            "utility",
            "service",
            "transformer_tower",
            "water_tower",
            "silo",
            "tank",
            "bunker",
            "ruins",
            "bridge",
            "tunnel",
            "dam",
            "pier",
        ],
        "amenity": [
            "toilets",
            "waste_basket",
            "recycling",
            "bench",
            "drinking_water",
            "fountain",
            "post_box",
            "telephone",
            "atm",
            "vending_machine",
            "charging_station",
            "fuel",
            "parking",
            "bicycle_parking",
            "motorcycle_parking",
            "taxi",
        ],
        "leisure": ["playground", "picnic_table", "bbq", "firepit"],
        "tourism": ["information", "viewpoint", "picnic_site"],
        "highway": ["*"],
        "railway": ["*"],
        "waterway": ["*"],
        "natural": ["*"],
        "landuse": ["*"],
        "power": ["*"],
    }

    def __init__(
        self, timeout: int = 30, use_cache: bool = True, cache_max_age_days: int = 30
    ):
        """Initialize the buildings extractor.

        Args:
            timeout: Request timeout in seconds
            use_cache: Whether to use local caching
            cache_max_age_days: Maximum age of cached data in days
        """
        self.timeout = timeout
        self.use_cache = use_cache
        self.cache_max_age_days = cache_max_age_days
        self.buildings: List[Building] = []
        self.stats = defaultdict(int)

        # Create cache directory if it doesn't exist
        if self.use_cache:
            Path(self.CACHE_DIR).mkdir(exist_ok=True)

    def _get_cache_filename(self, bounds: Tuple[float, float, float, float]) -> str:
        """Generate a cache filename based on the bounds.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box

        Returns:
            Cache filename
        """
        bounds_str = f"{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        bounds_hash = hashlib.md5(bounds_str.encode()).hexdigest()[:8]
        return os.path.join(
            self.CACHE_DIR, f"buildings_cache_{bounds_hash}_{bounds_str}.pkl.gz"
        )

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if the cache file is still valid.

        Args:
            cache_file: Path to cache file

        Returns:
            True if cache is valid and not expired
        """
        if not os.path.exists(cache_file):
            return False

        # Check if cache is not too old
        cache_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        max_age_hours = self.cache_max_age_days * 24

        return cache_age_hours <= max_age_hours

    def _save_to_cache(
        self,
        bounds: Tuple[float, float, float, float],
        buildings: List[Building],
        stats: Dict,
    ) -> None:
        """Save buildings data to cache.

        Args:
            bounds: Bounding box used for the query
            buildings: List of extracted buildings
            stats: Statistics dictionary
        """
        if not self.use_cache:
            return

        cache_file = self._get_cache_filename(bounds)

        cache_data = {
            "bounds": bounds,
            "buildings": buildings,
            "stats": stats,
            "timestamp": time.time(),
            "version": "5.0",
            "filter_type": "default_height",
            "default_height": self.DEFAULT_HEIGHT,
        }

        try:
            with gzip.open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            output.cache_info(
                f"Saved {len(buildings)} buildings (with default heights)", is_hit=False
            )
        except Exception as e:
            output.warning(f"Could not save cache: {e}")

    def _load_from_cache(
        self, bounds: Tuple[float, float, float, float]
    ) -> Optional[Tuple[List[Building], Dict]]:
        """Load buildings data from cache.

        Args:
            bounds: Bounding box for the query

        Returns:
            Tuple of (buildings, stats) if cache hit, None otherwise
        """
        if not self.use_cache:
            return None

        cache_file = self._get_cache_filename(bounds)

        if not self._is_cache_valid(cache_file):
            return None

        try:
            with gzip.open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify the bounds match and it's the right filter type
            if (
                cache_data.get("bounds") != bounds
                or cache_data.get("filter_type") != "default_height"
            ):
                output.warning("Cache mismatch, ignoring cache")
                return None

            buildings = cache_data["buildings"]
            stats = cache_data["stats"]
            cache_age_hours = (time.time() - cache_data["timestamp"]) / 3600

            output.cache_info(
                f"Loaded {len(buildings)} buildings (age: {cache_age_hours:.1f} hours)"
            )
            return buildings, stats

        except Exception as e:
            output.warning(f"Could not load cache: {e}")
            return None

    def clear_cache(
        self, bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """Clear cached data.

        Args:
            bounds: If provided, clear only cache for these bounds. Otherwise clear all cache.
        """
        if bounds is not None:
            # Clear specific cache file
            cache_file = self._get_cache_filename(bounds)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                output.success(f"Cleared cache for bounds {bounds}")
        else:
            # Clear all cache files
            cache_dir = Path(self.CACHE_DIR)
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("buildings_cache_*.pkl.gz"))
                for cache_file in cache_files:
                    cache_file.unlink()
                output.success(f"Cleared {len(cache_files)} cache files")

    def build_overpass_query(self, bounds: Tuple[float, float, float, float]) -> str:
        """Build Overpass API query for buildings in the given bounds.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box

        Returns:
            Overpass QL query string for all buildings (height not required)
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Query for all buildings (no height requirement)
        query = f"""[out:json][timeout:{self.timeout}];
(
  way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["building"]["type"="multipolygon"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out geom;"""
        return query

    def is_excluded_feature(self, tags: Dict[str, str]) -> bool:
        """Check if a feature should be excluded based on its tags.

        Args:
            tags: OpenStreetMap tags dictionary

        Returns:
            True if the feature should be excluded
        """
        for tag_key, excluded_values in self.EXCLUDED_TAGS.items():
            if tag_key in tags:
                tag_value = tags[tag_key]
                if "*" in excluded_values or tag_value in excluded_values:
                    return True
        return False

    def calculate_area(self, polygon: Polygon) -> float:
        """Calculate the area of a polygon using the shoelace formula.

        Args:
            polygon: Polygon of the building

        Returns:
            Area in square meters (approximate)
        """
        if len(polygon.exterior.coords) < 3:
            return 0.0

        area = polygon.area

        # Convert to approximate square meters (very rough conversion)
        # 1 degree ≈ 111,320 meters at equator
        area_m2 = area * (111320**2)

        return area_m2
    
    def extract_height(self, tags: Dict[str, str]) -> float:
        """Extract building height from tags, or return default height.

        Args:
            tags: OpenStreetMap tags dictionary

        Returns:
            Height in meters (explicit from tags or default 5.0m)
        """
        # Try different height tags
        for height_tag in ["height", "building:height"]:
            if height_tag in tags:
                height_str = tags[height_tag]
                try:
                    # Handle different units
                    if height_str.endswith("m"):
                        height = float(height_str[:-1])
                        if height > 0:
                            return height
                    elif height_str.endswith("ft"):
                        height = (
                            float(height_str[:-2]) * 0.3048
                        )  # Convert feet to meters
                        if height > 0:
                            return height
                    else:
                        height = float(height_str)
                        if height > 0:
                            return height
                except ValueError:
                    continue

        # Try to estimate from building levels
        if "building:levels" in tags:
            try:
                levels = int(tags["building:levels"])
                if levels > 0:
                    return levels * 3.0  # Assume 3 meters per level
            except ValueError:
                pass

        # Return default height if no explicit height found
        return self.DEFAULT_HEIGHT

    def extract_buildings(
        self, bounds: Tuple[float, float, float, float], force_refresh: bool = False
    ) -> List[Building]:
        """Extract building data from OpenStreetMap for the given bounds.
        Returns buildings with polygon and height (explicit or default 5m).

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            List of Building objects with polygon and height data
        """
        output.subheader(f"Extracting buildings for bounds: {bounds}")

        # Try to load from cache first
        if not force_refresh:
            cached_result = self._load_from_cache(bounds)
            if cached_result is not None:
                buildings, stats = cached_result
                self.buildings = buildings
                self.stats = defaultdict(int, stats)
                return buildings

        # Build and execute query
        query = self.build_overpass_query(bounds)

        try:
            with output.progress_context("Fetching building data from OpenStreetMap"):
                response = requests.post(
                    self.OVERPASS_URL,
                    data=query,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()

        except requests.RequestException as e:
            output.error(f"Error fetching data from Overpass API: {e}")
            return []

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            output.error(f"Error parsing JSON response: {e}")
            return []

        buildings = []
        processed = 0
        excluded_no_geometry = 0
        excluded_tags = 0
        default_height_count = 0
        explicit_height_count = 0

        for element in data.get("elements", []):
            processed += 1

            # Skip if no geometry
            if "geometry" not in element:
                excluded_no_geometry += 1
                continue

            # Extract tags
            tags = element.get("tags", {})

            # Check if this should be excluded
            if self.is_excluded_feature(tags):
                excluded_tags += 1
                continue

            # Extract coordinates from geometry
            coordinates = []
            for coord in element["geometry"]:
                coordinates.append((coord["lon"], coord["lat"]))

            # Skip if insufficient coordinates for a polygon
            if len(coordinates) < 3:
                excluded_no_geometry += 1
                continue
            
            polygon = Polygon(coordinates)

            # Calculate area
            area = self.calculate_area(polygon)

            # Extract height (explicit or default)
            height = self.extract_height(tags)

            # Track if height was explicit or default
            has_explicit_height = (
                "height" in tags
                or "building:height" in tags
                or "building:levels" in tags
            )

            if has_explicit_height:
                explicit_height_count += 1
            else:
                default_height_count += 1

            # Determine building type
            building_type = tags.get("building", "yes")
            if building_type == "yes":
                # Try to get more specific type from other tags
                for tag in ["amenity", "leisure", "shop", "tourism"]:
                    if tag in tags:
                        building_type = f"{tag}:{tags[tag]}"
                        break

            # Create building object
            building = Building(
                osm_id=element.get("id", "unknown"),
                building_type=building_type,
                polygon=polygon,
                area=area,
                height=height,
            )

            buildings.append(building)

            # Update stats
            self.stats[building_type] += 1
            self.stats["total_buildings"] += 1

        self.stats["processed_elements"] = processed
        self.stats["excluded_no_geometry"] = excluded_no_geometry
        self.stats["excluded_tags"] = excluded_tags
        self.stats["explicit_height_count"] = explicit_height_count
        self.stats["default_height_count"] = default_height_count
        self.stats["total_excluded"] = excluded_no_geometry + excluded_tags

        self.buildings = buildings

        output.success(
            f"Extracted {len(buildings)} buildings from {processed} elements"
        )
        output.info(
            f"Height data: {explicit_height_count} explicit, {default_height_count} default ({self.DEFAULT_HEIGHT}m)"
        )
        output.info(
            f"Excluded: {excluded_no_geometry} (no geometry), {excluded_tags} (unwanted tags)"
        )

        # Save to cache
        self._save_to_cache(bounds, buildings, dict(self.stats))

        return buildings

    def get_stats(self) -> Dict:
        """Get statistics about the extracted buildings.

        Returns:
            Dictionary with extraction statistics
        """
        if not self.buildings:
            return {"error": "No buildings extracted yet"}

        # Calculate additional stats (all buildings have area and height at this point)
        areas = [b.area for b in self.buildings]
        heights = [b.height for b in self.buildings]

        stats = dict(self.stats)

        # Area statistics
        stats["total_building_area_m2"] = sum(areas)
        stats["average_building_area_m2"] = sum(areas) / len(areas)
        stats["largest_building_area_m2"] = max(areas)
        stats["smallest_building_area_m2"] = min(areas)

        # Height statistics
        stats["average_building_height_m"] = sum(heights) / len(heights)
        stats["tallest_building_height_m"] = max(heights)
        stats["shortest_building_height_m"] = min(heights)

        # Building type distribution
        building_types = {}
        for building in self.buildings:
            bt = building.building_type
            building_types[bt] = building_types.get(bt, 0) + 1

        stats["building_type_distribution"] = building_types

        # Building readiness
        stats["buildings_with_area"] = len(self.buildings)  # All have area
        stats["buildings_with_height"] = len(
            self.buildings
        )  # All have height (explicit or default)
        stats["buildings_with_explicit_height"] = stats.get("explicit_height_count", 0)
        stats["buildings_with_default_height"] = stats.get("default_height_count", 0)
        stats["default_height_meters"] = self.DEFAULT_HEIGHT

        return stats

    def print_stats(self):
        """Print extraction statistics."""
        stats = self.get_stats()

        if "error" in stats:
            output.error(stats["error"])
            return

        output.info(f"\n=== Buildings Extraction Statistics ===")
        output.info(f"Total buildings extracted: {stats['total_buildings']}")
        output.info(f"Buildings with area: {stats['buildings_with_area']}")
        output.info(f"Buildings with height: {stats['buildings_with_height']}")

        output.info(f"\nHeight Information:")
        output.info(f"  Explicit height: {stats['buildings_with_explicit_height']}")
        output.info(
            f"  Default height: {stats['buildings_with_default_height']} (using {stats['default_height_meters']}m)"
        )

        if stats["buildings_with_height"] > 0:
            output.info(f"\nHeight Statistics:")
            output.info(f"  Average height: {stats['average_building_height_m']:.1f}m")
            output.info(
                f"  Tallest building: {stats['tallest_building_height_m']:.1f}m"
            )
            output.info(
                f"  Shortest building: {stats['shortest_building_height_m']:.1f}m"
            )

        if stats["buildings_with_area"] > 0:
            output.info(f"\nArea Statistics:")
            output.info(
                f"  Total area: {stats['total_building_area_m2']:.0f} m² ({stats['total_building_area_m2']/1_000_000:.2f} km²)"
            )
            output.info(f"  Average area: {stats['average_building_area_m2']:.0f} m²")
            output.info(
                f"  Largest building: {stats['largest_building_area_m2']:.0f} m²"
            )
            output.info(
                f"  Smallest building: {stats['smallest_building_area_m2']:.0f} m²"
            )

        # Processing statistics
        if "processed_elements" in stats:
            output.info(f"\nProcessing Statistics:")
            output.info(f"  Elements processed: {stats['processed_elements']}")
            output.info(f"  Elements excluded: {stats.get('total_excluded', 0)}")
            output.info(f"    - No geometry: {stats.get('excluded_no_geometry', 0)}")
            output.info(f"    - Unwanted tags: {stats.get('excluded_tags', 0)}")

        # Building types
        if (
            "building_type_distribution" in stats
            and stats["building_type_distribution"]
        ):
            output.info(f"\nBuilding Type Distribution:")
            building_types = stats["building_type_distribution"]
            # Sort by count, show top 10
            sorted_types = sorted(
                building_types.items(), key=lambda x: x[1], reverse=True
            )
            for building_type, count in sorted_types[:10]:
                output.info(f"  {building_type}: {count}")
            if len(sorted_types) > 10:
                output.info(f"  ... and {len(sorted_types) - 10} more types")


if __name__ == "__main__":
    # Example usage with default height support
    output.header("=== Buildings Extractor with Default Height Support ===")

    # Downtown SF bounds (smaller area for testing)
    bounds = (-122.42, 37.77, -122.38, 37.80)

    extractor = BuildingsExtractor(timeout=30, use_cache=True)
    buildings = extractor.extract_buildings(bounds)

    extractor.print_stats()

    output.info(f"\nExample: First building details:")
    if buildings:
        building = buildings[0]
        output.info(f"  OSM ID: {building.osm_id}")
        output.info(f"  Type: {building.building_type}")
        output.info(f"  Area: {building.area:.0f} m²")
        output.info(f"  Height: {building.height:.1f} m")
        output.info(f"  Coordinates: {len(building.polygon.exterior.coords)} points")

        # Check if height was explicit or default
        has_explicit = (
            "height" in building.tags
            or "building:height" in building.tags
            or "building:levels" in building.tags
        )
        height_source = (
            "explicit" if has_explicit else f"default ({extractor.DEFAULT_HEIGHT}m)"
        )
        output.info(f"  Height source: {height_source}")

    # Demonstrate cache usage
    output.info(f"\n=== Demonstrating Cache ===")
    output.info("Running the same query again (should load from cache)...")
    extractor2 = BuildingsExtractor()
    buildings2 = extractor2.extract_buildings(bounds)
    output.info(f"Loaded {len(buildings2)} buildings")

    # Show cache management
    output.info(f"\nCache management:")
    output.info(f"- Cache directory: {extractor.CACHE_DIR}")
    output.info(f"- Default height: {extractor.DEFAULT_HEIGHT} meters")
    output.info(f"- To clear cache: extractor.clear_cache()")
    output.info(
        f"- To force refresh: extractor.extract_buildings(bounds, force_refresh=True)"
    )
