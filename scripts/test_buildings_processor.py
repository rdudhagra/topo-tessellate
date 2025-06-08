#!/usr/bin/env python3
"""
Test script for BuildingsProcessor module

This script tests the functionality of BuildingsProcessor and BuildingsGeoBins classes
with various test cases including edge cases and performance tests.

Usage:
    python test_buildings_processor.py
"""

import sys
import os
import time
import random
import math
from typing import List
from shapely.geometry import Polygon

# Add the parent directory to the path to import terrain_generator modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from terrain_generator.buildingsextractor import Building
from terrain_generator.buildingsprocessor import BuildingsProcessor, BuildingsGeoBins
from terrain_generator.console import output


class BuildingsProcessorTest:
    """Test suite for BuildingsProcessor and BuildingsGeoBins classes."""
    
    def __init__(self):
        self.processor = BuildingsProcessor()
        self.test_buildings = []
        self.test_results = []
    
    def create_test_building(self, osm_id: int, building_type: str, 
                           polygon_coords: List[tuple], height: float = 10.0) -> Building:
        """Create a test building with given parameters."""
        polygon = Polygon(polygon_coords)
        return Building(
            osm_id=osm_id,
            building_type=building_type,
            polygon=polygon,
            area=polygon.area,
            height=height
        )
    
    def create_sample_buildings(self) -> List[Building]:
        """Create a set of sample buildings for testing."""
        buildings = []
        
        # Building 1: Small house in downtown SF
        buildings.append(self.create_test_building(
            1, "house", 
            [(-122.420, 37.775), (-122.419, 37.775), (-122.419, 37.776), (-122.420, 37.776), (-122.420, 37.775)],
            8.0
        ))
        
        # Building 2: Large commercial building in downtown SF
        buildings.append(self.create_test_building(
            2, "commercial",
            [(-122.405, 37.780), (-122.400, 37.780), (-122.400, 37.785), (-122.405, 37.785), (-122.405, 37.780)],
            50.0
        ))
        
        # Building 3: Office building on the edge of bbox
        buildings.append(self.create_test_building(
            3, "office",
            [(-122.450, 37.770), (-122.449, 37.770), (-122.449, 37.771), (-122.450, 37.771), (-122.450, 37.770)],
            30.0
        ))
        
        # Building 4: School outside the main bbox
        buildings.append(self.create_test_building(
            4, "school",
            [(-122.500, 37.800), (-122.499, 37.800), (-122.499, 37.801), (-122.500, 37.801), (-122.500, 37.800)],
            15.0
        ))
        
        # Building 5: Hospital partially overlapping bbox edge
        buildings.append(self.create_test_building(
            5, "hospital",
            [(-122.399, 37.789), (-122.398, 37.789), (-122.398, 37.791), (-122.399, 37.791), (-122.399, 37.789)],
            25.0
        ))
        
        # Building 6: Large industrial building
        buildings.append(self.create_test_building(
            6, "industrial",
            [(-122.410, 37.770), (-122.405, 37.770), (-122.405, 37.775), (-122.410, 37.775), (-122.410, 37.770)],
            20.0
        ))
        
        return buildings
    
    def create_random_buildings(self, count: int, bounds: tuple = (-122.5, 37.7, -122.3, 37.9)) -> List[Building]:
        """Create random buildings within given bounds for performance testing."""
        min_lon, min_lat, max_lon, max_lat = bounds
        buildings = []
        
        building_types = ["house", "commercial", "office", "industrial", "residential", "retail"]
        
        for i in range(count):
            # Random center point
            center_lon = random.uniform(min_lon, max_lon)
            center_lat = random.uniform(min_lat, max_lat)
            
            # Random building size (in degrees, roughly)
            size_lon = random.uniform(0.0001, 0.001)  # ~10-100m
            size_lat = random.uniform(0.0001, 0.001)
            
            # Create a rectangular building
            coords = [
                (center_lon - size_lon/2, center_lat - size_lat/2),
                (center_lon + size_lon/2, center_lat - size_lat/2),
                (center_lon + size_lon/2, center_lat + size_lat/2),
                (center_lon - size_lon/2, center_lat + size_lat/2),
                (center_lon - size_lon/2, center_lat - size_lat/2)
            ]
            
            buildings.append(self.create_test_building(
                i + 1000,
                random.choice(building_types),
                coords,
                random.uniform(5.0, 100.0)
            ))
        
        return buildings
    
    def test_bbox_filtering_basic(self) -> bool:
        """Test basic bounding box filtering functionality."""
        output.subheader("Testing: Basic bbox filtering")
        
        buildings = self.create_sample_buildings()
        
        # Define a bbox that should include buildings 1, 2, and 6 (downtown SF area)
        bbox = (-122.425, 37.770, -122.395, 37.790)
        
        filtered = self.processor.exclude_buildings_outside_bbox(buildings, bbox)
        
        expected_ids = {1, 2, 6}  # Buildings that should be included
        actual_ids = {b.osm_id for b in filtered}
        
        success = expected_ids == actual_ids
        
        output.info(f"Original buildings: {len(buildings)}")
        output.info(f"Filtered buildings: {len(filtered)}")
        output.info(f"Expected IDs: {expected_ids}")
        output.info(f"Actual IDs: {actual_ids}")
        
        if success:
            output.success("✓ Basic bbox filtering test passed")
        else:
            output.error("✗ Basic bbox filtering test failed")
        
        return success
    
    def test_bbox_filtering_edge_cases(self) -> bool:
        """Test edge cases for bounding box filtering."""
        output.subheader("Testing: Bbox filtering edge cases")
        
        buildings = self.create_sample_buildings()
        all_passed = True
        
        # Test 1: Empty bbox (impossible bbox)
        try:
            empty_bbox = (-122.400, 37.780, -122.410, 37.770)  # min > max
            filtered = self.processor.exclude_buildings_outside_bbox(buildings, empty_bbox)
            output.info(f"Empty bbox test: {len(filtered)} buildings (expected: 0)")
            if len(filtered) == 0:
                output.success("✓ Empty bbox test passed")
            else:
                output.error("✗ Empty bbox test failed")
                all_passed = False
        except Exception as e:
            output.warning(f"Empty bbox test caused exception: {e}")
        
        # Test 2: Very large bbox (should include all buildings)
        large_bbox = (-123.0, 37.0, -121.0, 38.0)
        filtered = self.processor.exclude_buildings_outside_bbox(buildings, large_bbox)
        output.info(f"Large bbox test: {len(filtered)} buildings (expected: {len(buildings)})")
        if len(filtered) == len(buildings):
            output.success("✓ Large bbox test passed")
        else:
            output.error("✗ Large bbox test failed")
            all_passed = False
        
        # Test 3: Very small bbox (should include very few or no buildings)
        small_bbox = (-122.4001, 37.7801, -122.4000, 37.7802)
        filtered = self.processor.exclude_buildings_outside_bbox(buildings, small_bbox)
        output.info(f"Small bbox test: {len(filtered)} buildings (expected: 0-1)")
        if len(filtered) <= 1:
            output.success("✓ Small bbox test passed")
        else:
            output.error("✗ Small bbox test failed")
            all_passed = False
        
        # Test 4: Empty building list
        filtered = self.processor.exclude_buildings_outside_bbox([], large_bbox)
        output.info(f"Empty building list test: {len(filtered)} buildings (expected: 0)")
        if len(filtered) == 0:
            output.success("✓ Empty building list test passed")
        else:
            output.error("✗ Empty building list test failed")
            all_passed = False
        
        return all_passed
    
    def test_building_is_inside_bbox(self) -> bool:
        """Test the Building.is_inside_bbox method directly."""
        output.subheader("Testing: Building.is_inside_bbox method")
        
        # Create a test building
        building = self.create_test_building(
            999, "test",
            [(-122.420, 37.775), (-122.419, 37.775), (-122.419, 37.776), (-122.420, 37.776), (-122.420, 37.775)]
        )
        
        test_cases = [
            # (bbox, expected_result, description)
            ((-122.425, 37.770, -122.415, 37.780), True, "Building inside bbox"),
            ((-122.430, 37.780, -122.425, 37.785), False, "Building outside bbox"),
            ((-122.4195, 37.7745, -122.4185, 37.7755), False, "Bbox inside building (should be False for contains)"),
            ((-122.425, 37.770, -122.418, 37.777), True, "Bbox completely contains building"),  # Fixed: this bbox actually contains the building
            ((-122.4205, 37.7745, -122.4190, 37.7755), False, "Bbox partially overlaps building"),  # True partial overlap case
        ]
        
        all_passed = True
        for bbox, expected, description in test_cases:
            result = building.is_inside_bbox(bbox)
            output.info(f"{description}: {result} (expected: {expected})")
            
            if result == expected:
                output.success(f"✓ {description} - passed")
            else:
                output.error(f"✗ {description} - failed")
                all_passed = False
        
        return all_passed
    
    def test_geo_bins_basic(self) -> bool:
        """Test basic BuildingsGeoBins functionality."""
        output.subheader("Testing: BuildingsGeoBins basic functionality")
        
        buildings = self.create_sample_buildings()
        
        # Test with different bin sizes
        bin_sizes = [50, 100, 200]  # meters
        all_passed = True
        
        for bin_size in bin_sizes:
            output.info(f"Testing with bin size: {bin_size}m")
            
            try:
                geo_bins = BuildingsGeoBins(buildings, bin_size_meters=bin_size, debug=False)
                
                # Test that all buildings are accessible
                total_buildings_in_bins = sum(len(building_list) for building_list in geo_bins.bins.values())
                
                output.info(f"  Total buildings in bins: {total_buildings_in_bins}")
                output.info(f"  Number of bins: {len(geo_bins.bins)}")
                output.info(f"  Centroid: ({geo_bins.centroid_lat:.6f}, {geo_bins.centroid_lon:.6f})")
                
                # Note: total_buildings_in_bins can be > len(buildings) because buildings can span multiple bins
                if total_buildings_in_bins >= len(buildings):
                    output.success(f"✓ Bin size {bin_size}m test passed")
                else:
                    output.error(f"✗ Bin size {bin_size}m test failed - not enough buildings in bins")
                    all_passed = False
                    
            except Exception as e:
                output.error(f"✗ Bin size {bin_size}m test failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def test_geo_bins_radius_search(self) -> bool:
        """Test radius search functionality in BuildingsGeoBins."""
        output.subheader("Testing: BuildingsGeoBins radius search")
        
        buildings = self.create_sample_buildings()
        geo_bins = BuildingsGeoBins(buildings, bin_size_meters=100, debug=False)
        
        all_passed = True
        
        # Test cases: (lat, lon, radius_meters, description)
        test_cases = [
            (37.775, -122.420, 100, "Small radius around building 1"),
            (37.780, -122.402, 500, "Medium radius around building 2"),
            (37.775, -122.407, 1000, "Large radius in central area"),
            (37.900, -122.300, 100, "Small radius in empty area"),
        ]
        
        for lat, lon, radius, description in test_cases:
            try:
                start_time = time.time()
                found_buildings = geo_bins.get_buildings_in_radius(lat, lon, radius)
                search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                output.info(f"{description}:")
                output.info(f"  Location: ({lat:.6f}, {lon:.6f})")
                output.info(f"  Radius: {radius}m")
                output.info(f"  Found buildings: {len(found_buildings)}")
                output.info(f"  Search time: {search_time:.2f}ms")
                
                # Basic validation: found buildings should be unique
                found_ids = [b.osm_id for b in found_buildings]
                if len(found_ids) == len(set(found_ids)):
                    output.success(f"✓ {description} - no duplicates")
                else:
                    output.error(f"✗ {description} - found duplicate buildings")
                    all_passed = False
                    
            except Exception as e:
                output.error(f"✗ {description} failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def test_geo_bins_add_remove(self) -> bool:
        """Test adding and removing buildings from GeoBins."""
        output.subheader("Testing: BuildingsGeoBins add/remove functionality")
        
        initial_buildings = self.create_sample_buildings()[:3]  # Use first 3 buildings
        geo_bins = BuildingsGeoBins(initial_buildings, bin_size_meters=100, debug=False)
        
        initial_bin_count = len(geo_bins.bins)
        initial_total_buildings = sum(len(building_list) for building_list in geo_bins.bins.values())
        
        output.info(f"Initial state: {initial_bin_count} bins, {initial_total_buildings} building entries")
        
        # Test adding a new building
        new_building = self.create_test_building(
            9999, "test_add",
            [(-122.415, 37.777), (-122.414, 37.777), (-122.414, 37.778), (-122.415, 37.778), (-122.415, 37.777)]
        )
        
        geo_bins.add_building(new_building)
        after_add_total = sum(len(building_list) for building_list in geo_bins.bins.values())
        
        output.info(f"After adding: {len(geo_bins.bins)} bins, {after_add_total} building entries")
        
        # Test removing the building
        geo_bins.remove_building(new_building)
        after_remove_total = sum(len(building_list) for building_list in geo_bins.bins.values())
        
        output.info(f"After removing: {len(geo_bins.bins)} bins, {after_remove_total} building entries")
        
        # Verify we're back to the initial state
        success = after_remove_total == initial_total_buildings
        
        if success:
            output.success("✓ Add/remove functionality test passed")
        else:
            output.error("✗ Add/remove functionality test failed")
        
        return success
    
    def test_performance(self) -> bool:
        """Test performance with larger datasets."""
        output.subheader("Testing: Performance with larger datasets")
        
        # Test with increasing numbers of buildings
        building_counts = [100, 500, 1000, 2000]
        all_passed = True
        
        for count in building_counts:
            output.info(f"Testing with {count} buildings...")
            
            try:
                # Create random buildings
                start_time = time.time()
                buildings = self.create_random_buildings(count)
                creation_time = time.time() - start_time
                
                # Test bbox filtering performance
                bbox = (-122.45, 37.75, -122.35, 37.85)
                start_time = time.time()
                filtered = self.processor.exclude_buildings_outside_bbox(buildings, bbox)
                bbox_time = time.time() - start_time
                
                # Test geo bins creation performance
                start_time = time.time()
                geo_bins = BuildingsGeoBins(buildings, bin_size_meters=100, debug=False)
                bins_creation_time = time.time() - start_time
                
                # Test radius search performance
                start_time = time.time()
                found = geo_bins.get_buildings_in_radius(37.78, -122.42, 500)
                radius_search_time = time.time() - start_time
                
                output.info(f"  Building creation: {creation_time:.3f}s")
                output.info(f"  Bbox filtering: {bbox_time:.3f}s ({len(filtered)} buildings)")
                output.info(f"  GeoBins creation: {bins_creation_time:.3f}s ({len(geo_bins.bins)} bins)")
                output.info(f"  Radius search: {radius_search_time:.3f}s ({len(found)} buildings)")
                
                # Performance thresholds (adjust as needed)
                if bbox_time < 1.0 and bins_creation_time < 5.0 and radius_search_time < 0.1:
                    output.success(f"✓ Performance test with {count} buildings passed")
                else:
                    output.warning(f"⚠ Performance test with {count} buildings slower than expected")
                    
            except Exception as e:
                output.error(f"✗ Performance test with {count} buildings failed: {e}")
                all_passed = False
        
        return all_passed
    
    def run_all_tests(self) -> None:
        """Run all tests and report results."""
        output.header("BuildingsProcessor Test Suite")
        
        tests = [
            ("Building.is_inside_bbox method", self.test_building_is_inside_bbox),
            ("Basic bbox filtering", self.test_bbox_filtering_basic),
            ("Bbox filtering edge cases", self.test_bbox_filtering_edge_cases),
            ("GeoBins basic functionality", self.test_geo_bins_basic),
            ("GeoBins radius search", self.test_geo_bins_radius_search),
            ("GeoBins add/remove", self.test_geo_bins_add_remove),
            ("Performance tests", self.test_performance),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            output.info(f"\n{'-' * 60}")
            try:
                if test_func():
                    passed += 1
                    self.test_results.append((test_name, "PASSED"))
                else:
                    self.test_results.append((test_name, "FAILED"))
            except Exception as e:
                output.error(f"Test '{test_name}' crashed with exception: {e}")
                self.test_results.append((test_name, f"CRASHED: {e}"))
        
        # Print summary
        output.info(f"\n{'=' * 60}")
        output.header("Test Results Summary")
        
        for test_name, result in self.test_results:
            if result == "PASSED":
                output.success(f"✓ {test_name}: {result}")
            elif result == "FAILED":
                output.error(f"✗ {test_name}: {result}")
            else:
                output.error(f"💥 {test_name}: {result}")
        
        output.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            output.success("🎉 All tests passed!")
        else:
            output.warning(f"⚠ {total - passed} test(s) failed")


def main():
    """Main function to run the test suite."""
    try:
        test_suite = BuildingsProcessorTest()
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        output.warning("\nTests interrupted by user")
    except Exception as e:
        output.error(f"Test suite failed with exception: {e}")


if __name__ == "__main__":
    main() 