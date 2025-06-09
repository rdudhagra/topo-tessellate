#!/usr/bin/env python3
"""
Washington DC Terrain Generator

This script generates a 3D terrain model of Washington DC using SRTM elevation data
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
    """Generate a detailed Washington DC terrain model."""
    output.header("Washington DC Terrain Model Generation")

    # Create model generator
    elevation = GeoTiff("USGS_one_meter_x32y431_MD_VA_Sandy_NCR_2014.tif")

    # elevation = SRTM()
    generator = ModelGenerator(elevation)

    elevation_multiplier = 3.5
    building_height_multiplier = 1
    base_height = 250

    # Generate detailed terrain model
    result = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir="topo",
        base_height=base_height,
        water_threshold=0,
        elevation_multiplier=elevation_multiplier,
        downsample_factor=1,
        force_refresh=False,
    )

    # Extract buildings
    buildings = BuildingsExtractor(timeout=120).extract_buildings(bounds, max_building_distance_meters=30, force_refresh=False)
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
    generator.save_mesh(buildings_mesh, f"{prefix}_buildings.obj")

    output.success("Washington DC terrain model generation complete!")


def generate_dc():
    """Generate both terrain and extract buildings for Washington DC."""
    output.header(
        "Washington DC 3D Model Generator",
        "Generating terrain and extracting buildings for Washington DC",
    )

    # Washington DC bounds
    bounds = (-77.0728342158, 38.8349886875, -76.9620071625, 38.9178722916)

    output.info(f"Processing region: {bounds}")
    output.info("Longitude: {:.3f}° to {:.3f}°".format(bounds[0], bounds[2]))
    output.info("Latitude: {:.3f}° to {:.3f}°".format(bounds[1], bounds[3]))

    output.print_section_divider()

    # Generate terrain
    generate_terrain("dc", bounds)

    output.print_section_divider()

    # Extract buildings and create visualization
    # buildings = extract_buildings(bounds)

    output.success("Washington DC 3D model generation complete!")


if __name__ == "__main__":
    generate_dc()
