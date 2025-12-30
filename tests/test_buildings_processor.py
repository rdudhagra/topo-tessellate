"""
Pytest tests for BuildingsProcessor and BuildingsGeoBins classes.
"""

import pytest
import time
from shapely.geometry import Polygon

from terrain_generator.buildingbase import Building
from terrain_generator.buildingsprocessor import BuildingsProcessor, BuildingsGeoBins


class TestBuildingIsInsideBbox:
    """Tests for Building.is_inside_bbox method."""

    def test_building_inside_bbox(self, sample_building):
        """Building fully inside bbox should return True."""
        bbox = (-122.425, 37.770, -122.415, 37.780)
        assert sample_building.is_inside_bbox(bbox) is True

    def test_building_outside_bbox(self, sample_building):
        """Building outside bbox should return False."""
        bbox = (-122.430, 37.780, -122.425, 37.785)
        assert sample_building.is_inside_bbox(bbox) is False

    def test_bbox_inside_building(self, sample_building):
        """Bbox inside building should return False (building not contained)."""
        bbox = (-122.4195, 37.7745, -122.4185, 37.7755)
        assert sample_building.is_inside_bbox(bbox) is False

    def test_bbox_completely_contains_building(self, sample_building):
        """Bbox that completely contains building should return True."""
        bbox = (-122.425, 37.770, -122.418, 37.777)
        assert sample_building.is_inside_bbox(bbox) is True

    def test_bbox_partially_overlaps_building(self, sample_building):
        """Partial overlap should return False."""
        bbox = (-122.4205, 37.7745, -122.4190, 37.7755)
        assert sample_building.is_inside_bbox(bbox) is False


class TestBuildingsProcessorBboxFiltering:
    """Tests for BuildingsProcessor bbox filtering."""

    def test_basic_bbox_filtering(self, sample_buildings):
        """Test that bbox filtering returns correct buildings."""
        # Bbox should include buildings 1, 2, and 6 (downtown SF area)
        bbox = (-122.425, 37.770, -122.395, 37.790)
        
        processor = BuildingsProcessor(sample_buildings)
        filtered = processor.exclude_buildings_outside_bbox(bbox)
        
        expected_ids = {1, 2, 6}
        actual_ids = {b.osm_id for b in filtered}
        
        assert expected_ids == actual_ids

    def test_empty_bbox_returns_no_buildings(self, sample_buildings):
        """Impossible bbox (min > max) should return no buildings."""
        empty_bbox = (-122.400, 37.780, -122.410, 37.770)  # min > max
        
        processor = BuildingsProcessor(list(sample_buildings))
        filtered = processor.exclude_buildings_outside_bbox(empty_bbox)
        
        assert len(filtered) == 0

    def test_large_bbox_returns_all_buildings(self, sample_buildings):
        """Very large bbox should include all buildings."""
        large_bbox = (-123.0, 37.0, -121.0, 38.0)
        
        processor = BuildingsProcessor(list(sample_buildings))
        filtered = processor.exclude_buildings_outside_bbox(large_bbox)
        
        assert len(filtered) == len(sample_buildings)

    def test_small_bbox_returns_few_buildings(self, sample_buildings):
        """Very small bbox should include few or no buildings."""
        small_bbox = (-122.4001, 37.7801, -122.4000, 37.7802)
        
        processor = BuildingsProcessor(list(sample_buildings))
        filtered = processor.exclude_buildings_outside_bbox(small_bbox)
        
        assert len(filtered) <= 1

    def test_empty_building_list(self):
        """Empty building list should return empty result."""
        large_bbox = (-123.0, 37.0, -121.0, 38.0)
        
        processor = BuildingsProcessor([])
        filtered = processor.exclude_buildings_outside_bbox(large_bbox)
        
        assert len(filtered) == 0


class TestBuildingsGeoBinsBasic:
    """Tests for BuildingsGeoBins basic functionality."""

    @pytest.mark.parametrize("bin_size", [50, 100, 200])
    def test_geo_bins_creation(self, sample_buildings, bin_size):
        """Test that geo bins can be created with different sizes."""
        geo_bins = BuildingsGeoBins(sample_buildings, bin_size_meters=bin_size, debug=False)
        
        # All buildings should be accessible (may be in multiple bins)
        total_in_bins = sum(len(bl) for bl in geo_bins.bins.values())
        
        assert total_in_bins >= len(sample_buildings)
        assert len(geo_bins.bins) > 0

    def test_geo_bins_has_correct_reference_point(self, sample_buildings):
        """GeoBins should have valid min lat/lon reference."""
        geo_bins = BuildingsGeoBins(sample_buildings, bin_size_meters=100, debug=False)
        
        assert geo_bins.min_lat is not None
        assert geo_bins.min_lon is not None
        assert geo_bins.min_lat > 0  # SF area
        assert geo_bins.min_lon < 0  # West of prime meridian


class TestBuildingsGeoBinsRadiusSearch:
    """Tests for BuildingsGeoBins radius search functionality."""

    def test_radius_search_returns_no_duplicates(self, sample_buildings):
        """Radius search should not return duplicate buildings."""
        geo_bins = BuildingsGeoBins(sample_buildings, bin_size_meters=100, debug=False)
        
        found_wrappers = geo_bins.get_building_wrappers_within_radius(500, 500, 500)
        found_ids = [w.building.osm_id for w in found_wrappers]
        
        assert len(found_ids) == len(set(found_ids))

    def test_radius_search_empty_area(self, sample_buildings):
        """Radius search in empty area should return empty list."""
        geo_bins = BuildingsGeoBins(sample_buildings, bin_size_meters=100, debug=False)
        
        # Search far from any buildings
        found = geo_bins.get_building_wrappers_within_radius(50000, 50000, 100)
        
        assert len(found) == 0

    def test_radius_search_is_fast(self, sample_buildings):
        """Radius search should complete quickly."""
        geo_bins = BuildingsGeoBins(sample_buildings, bin_size_meters=100, debug=False)
        
        start = time.time()
        geo_bins.get_building_wrappers_within_radius(1000, 1000, 1000)
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should complete in under 100ms


class TestBuildingsGeoBinsAddRemove:
    """Tests for BuildingsGeoBins add/remove functionality."""

    def test_add_building(self, sample_buildings):
        """Adding a building should increase bin entries."""
        initial_buildings = sample_buildings[:3]
        geo_bins = BuildingsGeoBins(initial_buildings, bin_size_meters=100, debug=False)
        
        initial_count = sum(len(bl) for bl in geo_bins.bins.values())
        
        # Add new building
        new_polygon = Polygon([
            (-122.415, 37.777), (-122.414, 37.777), 
            (-122.414, 37.778), (-122.415, 37.778), 
            (-122.415, 37.777)
        ])
        new_building = Building(
            osm_id=9999,
            building_type="test_add",
            polygon=new_polygon,
            area=new_polygon.area,
            height=10.0
        )
        
        geo_bins.add_building(new_building)
        after_add_count = sum(len(bl) for bl in geo_bins.bins.values())
        
        assert after_add_count > initial_count

    def test_remove_building_restores_state(self, sample_buildings):
        """Removing a building should restore previous state."""
        initial_buildings = sample_buildings[:3]
        geo_bins = BuildingsGeoBins(initial_buildings, bin_size_meters=100, debug=False)
        
        initial_count = sum(len(bl) for bl in geo_bins.bins.values())
        
        # Add then remove
        new_polygon = Polygon([
            (-122.415, 37.777), (-122.414, 37.777), 
            (-122.414, 37.778), (-122.415, 37.778), 
            (-122.415, 37.777)
        ])
        new_building = Building(
            osm_id=9999,
            building_type="test_add",
            polygon=new_polygon,
            area=new_polygon.area,
            height=10.0
        )
        
        geo_bins.add_building(new_building)
        geo_bins.remove_building(new_building)
        
        after_remove_count = sum(len(bl) for bl in geo_bins.bins.values())
        
        assert after_remove_count == initial_count


class TestPerformance:
    """Performance tests for BuildingsProcessor and GeoBins."""

    @pytest.mark.parametrize("count", [100, 500, 1000])
    def test_bbox_filtering_performance(self, random_buildings, count):
        """Bbox filtering should complete within reasonable time."""
        buildings = random_buildings(count)
        bbox = (-122.45, 37.75, -122.35, 37.85)
        
        start = time.time()
        processor = BuildingsProcessor(list(buildings))
        processor.exclude_buildings_outside_bbox(bbox)
        elapsed = time.time() - start
        
        # Should complete in under 1 second
        assert elapsed < 1.0

    @pytest.mark.parametrize("count", [100, 500, 1000])
    def test_geo_bins_creation_performance(self, random_buildings, count):
        """GeoBins creation should complete within reasonable time."""
        buildings = random_buildings(count)
        
        start = time.time()
        BuildingsGeoBins(buildings, bin_size_meters=100, debug=False)
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0

    def test_radius_search_performance(self, random_buildings):
        """Radius search should be fast even with many buildings."""
        buildings = random_buildings(1000)
        geo_bins = BuildingsGeoBins(buildings, bin_size_meters=100, debug=False)
        
        start = time.time()
        geo_bins.get_building_wrappers_within_radius(5000, 5000, 500)
        elapsed = time.time() - start
        
        # Should complete in under 100ms
        assert elapsed < 0.1

