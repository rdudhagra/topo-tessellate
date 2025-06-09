#!/usr/bin/env python3
"""
Test Terrain Generator

This script generates a 3D terrain model of the test area using SRTM elevation data
and extracts 3D-ready building data from OpenStreetMap.
"""

import meshlib.mrmeshpy as mr
from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.buildingsextractor import BuildingsExtractor
from terrain_generator.geotiff import GeoTiff
from terrain_generator.srtm import SRTM
from terrain_generator.console import output
from terrain_generator.buildingsgenerator import BuildingsGenerator


def generate_terrain(prefix, bounds):
    """Generate a detailed Bay Area terrain model."""
    output.header("Test Terrain Model Generation")

    # Create model generator
    elevation = GeoTiff("USGS_1M_10_x55y419_CA_SanFrancisco_B23.tif")

    # elevation = SRTM()
    generator = ModelGenerator(elevation)

    elevation_multiplier = 1
    base_height = 250

    # Generate detailed terrain model
    result = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir="topo",
        base_height=base_height,
        water_threshold=1,
        elevation_multiplier=elevation_multiplier,
        downsample_factor=1,
        force_refresh=True,
    )

    # Extract buildings
    buildings = BuildingsExtractor(timeout=120).extract_buildings(bounds, max_building_distance_meters=0, force_refresh=True)
    buildings_generator = BuildingsGenerator(elevation)
    buildings_mesh = buildings_generator.generate_buildings(
        base_height,
        result["elevation_data"],
        elevation_multiplier,
        bounds,
        buildings,
        min_building_height=10,
    )

    # Save meshes
    output.progress_info(f"Saving meshes...")
    generator.save_mesh(result["land_mesh"], f"{prefix}_land.obj")
    generator.save_mesh(result["base_mesh"], f"{prefix}_base.obj")
    generator.save_mesh(buildings_mesh, f"{prefix}_buildings.obj")

    output.success("Bay Area terrain model generation complete!")


def generate_test():
    """Generate both terrain and extract buildings for the test area."""
    output.header(
        "Test 3D Model Generator",
        "Generating terrain and extracting buildings for the test area",
    )

    # Test bounds (covers FiDi area in San Francisco)
    bounds = (-122.42846, 37.77723, -122.37645, 37.81792)

    output.info(f"Processing region: {bounds}")
    output.info("Longitude: {:.3f}° to {:.3f}°".format(bounds[0], bounds[2]))
    output.info("Latitude: {:.3f}° to {:.3f}°".format(bounds[1], bounds[3]))

    output.print_section_divider()

    # Generate terrain
    generate_terrain("test", bounds)

    output.print_section_divider()

    # Extract buildings and create visualization
    # buildings = extract_buildings(bounds)

    output.success("Test 3D model generation complete!")


if __name__ == "__main__":
    generate_test()
