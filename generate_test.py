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


def generate_terrain(prefix, bounds, *,
                     downsample_factor: int = 1,
                     decimate: bool = False,
                     decimate_max_error: float | None = None,
                     decimate_target_face_count: int | None = None,
                     generate_buildings: bool = False):
    """Generate a terrain model with configurable parameters."""
    output.header("Test Terrain Model Generation")

    # Create model generator
    elevation = GeoTiff("USGS_1M_10_x55y419_CA_SanFrancisco_B23.tif")

    # elevation = SRTM()
    generator = ModelGenerator(elevation)

    elevation_multiplier = 1
    building_height_multiplier = 1
    base_height = 150

    # Generate detailed terrain model
    result = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir="topo",
        base_height=base_height,
        water_threshold=1,
        elevation_multiplier=elevation_multiplier,
        downsample_factor=downsample_factor,
        force_refresh=False,
        decimate=decimate,
        decimate_max_error=decimate_max_error,
        decimate_target_face_count=decimate_target_face_count,
    )

    # Optionally extract buildings once
    buildings_mesh = None
    if generate_buildings:
        buildings = BuildingsExtractor(timeout=120).extract_buildings(
            bounds, max_building_distance_meters=0, force_refresh=False
        )
        buildings_generator = BuildingsGenerator(elevation)
        buildings_mesh = buildings_generator.generate_buildings(
            base_height,
            result["elevation_data"],
            elevation_multiplier,
            building_height_multiplier,
            bounds,
            buildings,
            min_building_height=10,
        )

    # Save meshes
    output.progress_info(f"Saving meshes...")
    generator.save_mesh(result["land_mesh"], f"{prefix}_land.obj")
    generator.save_mesh(result["base_mesh"], f"{prefix}_base.obj")
    if buildings_mesh is not None:
        generator.save_mesh(buildings_mesh, f"{prefix}_buildings.obj")

    output.success("Terrain model generation complete!")


def generate_comparison(bounds):
    """Generate multiple variants to compare decimation and light_base settings."""
    variants = [
        {"name": "lightbase_ds1", "params": {"downsample_factor": 1, "decimate": False}, "buildings": False},
        {"name": "lightbase_dec_err1m_ds2", "params": {"downsample_factor": 2, "decimate": True, "decimate_max_error": 1.0}, "buildings": False},
        {"name": "lightbase_dec_err2m_ds2", "params": {"downsample_factor": 2, "decimate": True, "decimate_max_error": 2.0}, "buildings": False},
        {"name": "lightbase_dec_250k_ds2", "params": {"downsample_factor": 2, "decimate": True, "decimate_target_face_count": 250_000}, "buildings": False},
        {"name": "lightbase_dec_150k_ds4", "params": {"downsample_factor": 4, "decimate": True, "decimate_target_face_count": 150_000}, "buildings": False},
    ]

    for v in variants:
        prefix = f"test_{v['name']}"
        output.print_section_divider()
        output.info(f"Generating variant: {prefix}")
        generate_terrain(prefix, bounds, generate_buildings=v.get("buildings", False), **v["params"]) 


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

    # Generate comparison variants
    generate_comparison(bounds)

    output.print_section_divider()

    # Extract buildings and create visualization
    # buildings = extract_buildings(bounds)

    output.success("Test 3D model generation complete!")


if __name__ == "__main__":
    generate_test()
