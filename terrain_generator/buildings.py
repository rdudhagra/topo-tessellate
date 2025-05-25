#!/usr/bin/env python3
"""
Buildings Data Extractor for 3D Modeling

This module extracts building and structure data from OpenStreetMap for a given
latitude/longitude bounding box, focusing only on buildings with both polygon
coordinates and height information for 3D modeling purposes.
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


@dataclass
class Building:
    """Represents a building extracted from OpenStreetMap with guaranteed polygon and height data."""
    osm_id: str
    building_type: str
    coordinates: List[Tuple[float, float]]  # Always has polygon data
    tags: Dict[str, str]
    area: float  # Always calculated from polygon
    height: float  # Always has height data


class BuildingsExtractor:
    """Extracts building data from OpenStreetMap using the Overpass API, focusing on 3D-ready buildings."""
    
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    CACHE_DIR = "building_cache"
    
    # Building types we want to include (major structures)
    MAJOR_BUILDING_TYPES = {
        'yes', 'house', 'residential', 'apartments', 'commercial', 'office', 
        'retail', 'industrial', 'warehouse', 'school', 'university', 'hospital',
        'hotel', 'church', 'mosque', 'synagogue', 'temple', 'civic', 'government',
        'public', 'sports_hall', 'stadium', 'theatre', 'cinema', 'museum',
        'library', 'fire_station', 'police', 'prison', 'courthouse', 'embassy',
        'supermarket', 'mall', 'shop', 'restaurant', 'cafe', 'bar', 'pub',
        'bank', 'clinic', 'pharmacy', 'factory', 'garage', 'hangar', 'barn',
        'greenhouse', 'cathedral', 'chapel', 'monastery', 'tower', 'bunker',
        'castle', 'fort', 'palace', 'manor'
    }
    
    # Tags to exclude (small structures, utilities, etc.)
    EXCLUDED_TAGS = {
        'amenity': ['lamp', 'bench', 'waste_basket', 'bicycle_parking', 
                    'parking_meter', 'post_box', 'telephone', 'vending_machine'],
        'barrier': ['*'],  # All barriers
        'highway': ['*'],  # All highway features
        'natural': ['tree', 'shrub'],
        'power': ['*'],  # Power infrastructure
        'man_made': ['utility_pole', 'street_lamp', 'surveillance', 'antenna',
                      'flagpole', 'chimney', 'mast', 'pole', 'pipeline'],
        'leisure': ['playground']
    }
    
    def __init__(self, timeout: int = 30, use_cache: bool = True, cache_max_age_days: int = 30):
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
        # Create a hash of the bounds for the filename
        bounds_str = f"{bounds[0]:.6f},{bounds[1]:.6f},{bounds[2]:.6f},{bounds[3]:.6f}"
        bounds_hash = hashlib.md5(bounds_str.encode()).hexdigest()[:16]
        return os.path.join(self.CACHE_DIR, f"buildings_3d_{bounds_hash}.pkl.gz")
    
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
        cache_age_seconds = time.time() - os.path.getmtime(cache_file)
        cache_age_days = cache_age_seconds / (24 * 3600)
        
        return cache_age_days <= self.cache_max_age_days
    
    def _save_to_cache(self, bounds: Tuple[float, float, float, float], 
                       buildings: List[Building], stats: Dict) -> None:
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
            'bounds': bounds,
            'buildings': buildings,
            'stats': stats,
            'timestamp': time.time(),
            'version': '2.0',  # Updated version for 3D-focused data
            'filter_type': '3d_ready'  # Indicates this cache contains only 3D-ready buildings
        }
        
        try:
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached {len(buildings)} 3D-ready buildings to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _load_from_cache(self, bounds: Tuple[float, float, float, float]) -> Optional[Tuple[List[Building], Dict]]:
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
            with gzip.open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify the bounds match and it's the right filter type
            if (cache_data.get('bounds') != bounds or 
                cache_data.get('filter_type') != '3d_ready'):
                print(f"Warning: Cache mismatch, ignoring cache")
                return None
            
            buildings = cache_data['buildings']
            stats = cache_data['stats']
            cache_age_hours = (time.time() - cache_data['timestamp']) / 3600
            
            print(f"Loaded {len(buildings)} 3D-ready buildings from cache (age: {cache_age_hours:.1f} hours)")
            return buildings, stats
            
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return None
    
    def clear_cache(self, bounds: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Clear cached data.
        
        Args:
            bounds: If provided, clear only cache for these bounds. Otherwise clear all cache.
        """
        if bounds is not None:
            # Clear specific cache file
            cache_file = self._get_cache_filename(bounds)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Cleared cache for bounds {bounds}")
        else:
            # Clear all cache files
            cache_dir = Path(self.CACHE_DIR)
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("buildings_3d_*.pkl.gz"))
                for cache_file in cache_files:
                    cache_file.unlink()
                print(f"Cleared {len(cache_files)} cache files")
    
    def build_overpass_query(self, bounds: Tuple[float, float, float, float]) -> str:
        """Build Overpass API query for buildings with height data in the given bounds.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box
            
        Returns:
            Overpass QL query string focused on buildings with height information
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Query specifically for buildings with height data
        query = f"""[out:json][timeout:{self.timeout}];
(
  // Buildings with explicit height tag
  way["building"]["height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["building"]["building:height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["building"]["building:levels"]({min_lat},{min_lon},{max_lat},{max_lon});
  // Major structures with height data
  way["amenity"~"^(hospital|school|university|library|fire_station|police|courthouse|embassy|cinema|theatre|museum|town_hall|community_centre)$"]["height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["amenity"~"^(hospital|school|university|library|fire_station|police|courthouse|embassy|cinema|theatre|museum|town_hall|community_centre)$"]["building:height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["amenity"~"^(hospital|school|university|library|fire_station|police|courthouse|embassy|cinema|theatre|museum|town_hall|community_centre)$"]["building:levels"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["leisure"~"^(sports_centre|stadium|swimming_pool|fitness_centre)$"]["height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["leisure"~"^(sports_centre|stadium|swimming_pool|fitness_centre)$"]["building:height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["leisure"~"^(sports_centre|stadium|swimming_pool|fitness_centre)$"]["building:levels"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["shop"~"^(supermarket|mall|department_store)$"]["height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["shop"~"^(supermarket|mall|department_store)$"]["building:height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["shop"~"^(supermarket|mall|department_store)$"]["building:levels"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["tourism"~"^(hotel|hostel|motel|guest_house|attraction|museum)$"]["height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["tourism"~"^(hotel|hostel|motel|guest_house|attraction|museum)$"]["building:height"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["tourism"~"^(hotel|hostel|motel|guest_house|attraction|museum)$"]["building:levels"]({min_lat},{min_lon},{max_lat},{max_lon});
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
                if '*' in excluded_values or tag_value in excluded_values:
                    return True
        return False
    
    def calculate_area(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate the area of a polygon using the shoelace formula.
        
        Args:
            coordinates: List of (lon, lat) coordinate pairs
            
        Returns:
            Area in square meters (approximate)
        """
        if len(coordinates) < 3:
            return 0.0
        
        # Convert to Cartesian coordinates (rough approximation)
        # For more accurate calculations, we'd need to project to a local coordinate system
        area = 0.0
        n = len(coordinates)
        
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        
        area = abs(area) / 2.0
        
        # Convert to approximate square meters (very rough conversion)
        # 1 degree ≈ 111,320 meters at equator
        area_m2 = area * (111320 ** 2)
        
        return area_m2
    
    def extract_height(self, tags: Dict[str, str]) -> Optional[float]:
        """Extract building height from tags.
        
        Args:
            tags: OpenStreetMap tags dictionary
            
        Returns:
            Height in meters if available, None otherwise
        """
        # Try different height tags
        for height_tag in ['height', 'building:height']:
            if height_tag in tags:
                height_str = tags[height_tag]
                try:
                    # Handle different units
                    if height_str.endswith('m'):
                        return float(height_str[:-1])
                    elif height_str.endswith('ft'):
                        return float(height_str[:-2]) * 0.3048  # Convert feet to meters
                    else:
                        return float(height_str)
                except ValueError:
                    continue
        
        # Try to estimate from building levels
        if 'building:levels' in tags:
            try:
                levels = int(tags['building:levels'])
                return levels * 3.0  # Assume 3 meters per level
            except ValueError:
                pass
        
        return None
    
    def extract_buildings(self, bounds: Tuple[float, float, float, float], 
                         force_refresh: bool = False) -> List[Building]:
        """Extract building data from OpenStreetMap for the given bounds.
        Only returns buildings with both polygon coordinates AND height data.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) bounding box
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            List of Building objects (guaranteed to have both polygon and height data)
        """
        print(f"Extracting 3D-ready buildings for bounds: {bounds}")
        
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
            response = requests.post(
                self.OVERPASS_URL,
                data=query,
                timeout=self.timeout,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()
            
        except requests.RequestException as e:
            print(f"Error fetching data from Overpass API: {e}")
            return []
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return []
        
        buildings = []
        processed = 0
        excluded_no_geometry = 0
        excluded_no_height = 0
        excluded_tags = 0
        
        for element in data.get('elements', []):
            processed += 1
            
            # Extract tags
            tags = element.get('tags', {})
            
            # Skip excluded features
            if self.is_excluded_feature(tags):
                excluded_tags += 1
                continue
            
            # Extract coordinates - REQUIRED for 3D modeling
            coordinates = []
            if element['type'] == 'way' and 'geometry' in element:
                coordinates = [(node['lon'], node['lat']) for node in element['geometry']]
            elif element['type'] == 'relation':
                # For relations, we'd need to resolve member ways - simplified for now
                continue
            
            if len(coordinates) < 3:  # Need at least 3 points for a polygon
                excluded_no_geometry += 1
                continue
            
            # Extract height - REQUIRED for 3D modeling
            height = self.extract_height(tags)
            if height is None or height <= 0:
                excluded_no_height += 1
                continue
            
            # Determine building type
            building_type = tags.get('building', 'unknown')
            if building_type == 'yes':
                # Try to get more specific type from other tags
                for tag in ['amenity', 'leisure', 'shop', 'tourism']:
                    if tag in tags:
                        building_type = f"{tag}:{tags[tag]}"
                        break
            
            # Calculate area (guaranteed to have coordinates at this point)
            area = self.calculate_area(coordinates)
            if area <= 0:  # Skip buildings with invalid area
                excluded_no_geometry += 1
                continue
            
            # Create building object - guaranteed to have both polygon and height data
            building = Building(
                osm_id=str(element['id']),
                building_type=building_type,
                coordinates=coordinates,
                tags=tags,
                area=area,
                height=height
            )
            
            buildings.append(building)
            
            # Update stats
            self.stats[building_type] += 1
            self.stats['total_buildings'] += 1
        
        self.stats['processed_elements'] = processed
        self.stats['excluded_no_geometry'] = excluded_no_geometry
        self.stats['excluded_no_height'] = excluded_no_height
        self.stats['excluded_tags'] = excluded_tags
        self.stats['total_excluded'] = excluded_no_geometry + excluded_no_height + excluded_tags
        
        self.buildings = buildings
        
        print(f"Extracted {len(buildings)} 3D-ready buildings from {processed} elements")
        print(f"Excluded: {excluded_no_height} (no height), {excluded_no_geometry} (no geometry), {excluded_tags} (unwanted tags)")
        
        # Save to cache
        self._save_to_cache(bounds, buildings, dict(self.stats))
        
        return buildings
    
    def get_stats(self) -> Dict:
        """Get statistics about the extracted buildings.
        
        Returns:
            Dictionary with extraction statistics
        """
        if not self.buildings:
            return {'error': 'No buildings extracted yet'}
        
        # Calculate additional stats (all buildings have area and height at this point)
        areas = [b.area for b in self.buildings]
        heights = [b.height for b in self.buildings]
        
        stats = dict(self.stats)
        
        # Area statistics
        stats['total_building_area_m2'] = sum(areas)
        stats['average_building_area_m2'] = sum(areas) / len(areas)
        stats['largest_building_area_m2'] = max(areas)
        stats['smallest_building_area_m2'] = min(areas)
        
        # Height statistics
        stats['average_building_height_m'] = sum(heights) / len(heights)
        stats['tallest_building_height_m'] = max(heights)
        stats['shortest_building_height_m'] = min(heights)
        
        # Building type distribution
        building_types = {}
        for building in self.buildings:
            bt = building.building_type
            building_types[bt] = building_types.get(bt, 0) + 1
        
        stats['building_type_distribution'] = building_types
        
        # 3D modeling readiness
        stats['buildings_with_area'] = len(self.buildings)  # All have area
        stats['buildings_with_height'] = len(self.buildings)  # All have height
        stats['3d_modeling_ready'] = len(self.buildings)  # All are 3D ready
        
        return stats
    
    def print_stats(self):
        """Print detailed statistics about the extracted buildings."""
        stats = self.get_stats()
        
        if 'error' in stats:
            print(stats['error'])
            return
        
        print("\n=== 3D-Ready Building Extraction Statistics ===")
        print(f"Total 3D-ready buildings extracted: {stats['total_buildings']}")
        print(f"Elements processed: {stats['processed_elements']}")
        print(f"Elements excluded: {stats['total_excluded']}")
        print(f"  - No height data: {stats.get('excluded_no_height', 0)}")
        print(f"  - No geometry: {stats.get('excluded_no_geometry', 0)}")
        print(f"  - Unwanted tags: {stats.get('excluded_tags', 0)}")
        print(f"3D modeling ready: {stats['3d_modeling_ready']} (100%)")
        
        print(f"\n=== Area Statistics ===")
        print(f"Total building area: {stats['total_building_area_m2']:,.0f} m²")
        print(f"Average building area: {stats['average_building_area_m2']:,.0f} m²")
        print(f"Largest building: {stats['largest_building_area_m2']:,.0f} m²")
        print(f"Smallest building: {stats['smallest_building_area_m2']:,.0f} m²")
        
        print(f"\n=== Height Statistics ===")
        print(f"Average building height: {stats['average_building_height_m']:.1f} m")
        print(f"Tallest building: {stats['tallest_building_height_m']:.1f} m")
        print(f"Shortest building: {stats['shortest_building_height_m']:.1f} m")
        
        print(f"\n=== Building Types ===")
        building_types = stats['building_type_distribution']
        for building_type, count in sorted(building_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{building_type}: {count}")


if __name__ == "__main__":
    # Example usage with 3D-focused extraction
    print("=== 3D-Ready Buildings Extractor Example ===")
    
    # Downtown SF bounds (smaller area for testing)
    bounds = (-122.42, 37.77, -122.38, 37.80)
    
    extractor = BuildingsExtractor(timeout=30, use_cache=True)
    buildings = extractor.extract_buildings(bounds)
    
    extractor.print_stats()
    
    print(f"\nExample: First building details:")
    if buildings:
        building = buildings[0]
        print(f"  OSM ID: {building.osm_id}")
        print(f"  Type: {building.building_type}")
        print(f"  Area: {building.area:.0f} m²")
        print(f"  Height: {building.height:.1f} m")
        print(f"  Coordinates: {len(building.coordinates)} points")
        print(f"  3D Ready: ✓ (guaranteed polygon + height)")
    
    # Demonstrate cache usage
    print(f"\n=== Demonstrating Cache ===")
    print("Running the same query again (should load from cache)...")
    extractor2 = BuildingsExtractor()
    buildings2 = extractor2.extract_buildings(bounds)
    print(f"Loaded {len(buildings2)} 3D-ready buildings")
    
    # Show cache management
    print(f"\nCache management:")
    print(f"- Cache directory: {extractor.CACHE_DIR}")
    print(f"- Cache files are named with '3d_' prefix for 3D-ready buildings")
    print(f"- To clear cache: extractor.clear_cache()")
    print(f"- To force refresh: extractor.extract_buildings(bounds, force_refresh=True)") 
