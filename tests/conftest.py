"""
Pytest configuration and shared fixtures for topo-tessellate tests.
"""

import pytest
import random
from typing import List
from shapely.geometry import Polygon

from terrain_generator.buildingbase import Building


@pytest.fixture
def sample_building():
    """Create a single test building."""
    polygon = Polygon([
        (-122.420, 37.775), (-122.419, 37.775), 
        (-122.419, 37.776), (-122.420, 37.776), 
        (-122.420, 37.775)
    ])
    return Building(
        osm_id=999,
        building_type="test",
        polygon=polygon,
        area=polygon.area,
        height=10.0
    )


@pytest.fixture
def sample_buildings():
    """Create a set of sample buildings for testing."""
    def create_building(osm_id: int, building_type: str, 
                       polygon_coords: List[tuple], height: float = 10.0) -> Building:
        polygon = Polygon(polygon_coords)
        return Building(
            osm_id=osm_id,
            building_type=building_type,
            polygon=polygon,
            area=polygon.area,
            height=height
        )
    
    buildings = []
    
    # Building 1: Small house in downtown SF
    buildings.append(create_building(
        1, "house", 
        [(-122.420, 37.775), (-122.419, 37.775), (-122.419, 37.776), (-122.420, 37.776), (-122.420, 37.775)],
        8.0
    ))
    
    # Building 2: Large commercial building in downtown SF
    buildings.append(create_building(
        2, "commercial",
        [(-122.405, 37.780), (-122.400, 37.780), (-122.400, 37.785), (-122.405, 37.785), (-122.405, 37.780)],
        50.0
    ))
    
    # Building 3: Office building on the edge of bbox
    buildings.append(create_building(
        3, "office",
        [(-122.450, 37.770), (-122.449, 37.770), (-122.449, 37.771), (-122.450, 37.771), (-122.450, 37.770)],
        30.0
    ))
    
    # Building 4: School outside the main bbox
    buildings.append(create_building(
        4, "school",
        [(-122.500, 37.800), (-122.499, 37.800), (-122.499, 37.801), (-122.500, 37.801), (-122.500, 37.800)],
        15.0
    ))
    
    # Building 5: Hospital partially overlapping bbox edge
    buildings.append(create_building(
        5, "hospital",
        [(-122.399, 37.789), (-122.398, 37.789), (-122.398, 37.791), (-122.399, 37.791), (-122.399, 37.789)],
        25.0
    ))
    
    # Building 6: Large industrial building
    buildings.append(create_building(
        6, "industrial",
        [(-122.410, 37.770), (-122.405, 37.770), (-122.405, 37.775), (-122.410, 37.775), (-122.410, 37.770)],
        20.0
    ))
    
    return buildings


@pytest.fixture
def random_buildings():
    """Factory fixture to create random buildings for performance testing."""
    def _create_random_buildings(count: int, bounds: tuple = (-122.5, 37.7, -122.3, 37.9)) -> List[Building]:
        min_lon, min_lat, max_lon, max_lat = bounds
        buildings = []
        building_types = ["house", "commercial", "office", "industrial", "residential", "retail"]
        
        for i in range(count):
            center_lon = random.uniform(min_lon, max_lon)
            center_lat = random.uniform(min_lat, max_lat)
            size_lon = random.uniform(0.0001, 0.001)
            size_lat = random.uniform(0.0001, 0.001)
            
            coords = [
                (center_lon - size_lon/2, center_lat - size_lat/2),
                (center_lon + size_lon/2, center_lat - size_lat/2),
                (center_lon + size_lon/2, center_lat + size_lat/2),
                (center_lon - size_lon/2, center_lat + size_lat/2),
                (center_lon - size_lon/2, center_lat - size_lat/2)
            ]
            
            polygon = Polygon(coords)
            buildings.append(Building(
                osm_id=i + 1000,
                building_type=random.choice(building_types),
                polygon=polygon,
                area=polygon.area,
                height=random.uniform(5.0, 100.0)
            ))
        
        return buildings
    
    return _create_random_buildings

