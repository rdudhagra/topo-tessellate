#!/usr/bin/env python3
"""
Bay Area Terrain Generator

This script generates a 3D terrain model of the Bay Area using SRTM elevation data
and extracts 3D-ready building data from OpenStreetMap.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.buildingsextractor import BuildingsExtractor
from terrain_generator.geotiff import GeoTiff
from terrain_generator.srtm import SRTM
from terrain_generator.console import output
from terrain_generator.buildingsgenerator import BuildingsGenerator


def generate_terrain(prefix, bounds):
    """Generate a detailed Bay Area terrain model."""
    output.header("Bay Area Terrain Model Generation")

    # Create model generator
    elevation = GeoTiff("output_USGS10m.tif")

    # elevation = SRTM()
    generator = ModelGenerator(elevation)

    try:
        elevation_multiplier = 3.5

        # Generate detailed terrain model
        result = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=2500,
            water_threshold=1,
            elevation_multiplier=elevation_multiplier,
            downsample_factor=3,
        )

        # Save the terrain and water meshes
        generator.save_mesh(result["land_mesh"], f"{prefix}_land.obj")
        generator.save_mesh(result["base_mesh"], f"{prefix}_base.obj")

        # Extract buildings
        buildings = BuildingsExtractor(timeout=120).extract_buildings(bounds)
        buildings_generator = BuildingsGenerator(elevation)
        buildings_mesh = buildings_generator.generate_buildings(
            result["land_mesh"],
            result["elevation_data"],
            elevation_multiplier,
            bounds,
            buildings,
        )

        generator.save_mesh(buildings_mesh, f"{prefix}_buildings.obj")

        output.success("Bay Area terrain model generation complete!")

    except Exception as e:
        output.error(f"Could not generate Bay Area terrain model: {e}")
        return None


def generate_bay_area():
    """Generate both terrain and extract buildings for the Bay Area."""
    output.header(
        "Bay Area 3D Model Generator",
        "Generating terrain and extracting buildings for the San Francisco Bay Area",
    )

    # Bay Area bounds (covers San Francisco to San Jose area)
    bounds = (-122.67, 37.22, -121.75, 38.18)

    output.info(f"Processing region: {bounds}")
    output.info("Longitude: {:.3f}° to {:.3f}°".format(bounds[0], bounds[2]))
    output.info("Latitude: {:.3f}° to {:.3f}°".format(bounds[1], bounds[3]))

    output.print_section_divider()

    # Generate terrain
    generate_terrain("bay_area", bounds)

    output.print_section_divider()

    # Extract buildings and create visualization
    # buildings = extract_buildings(bounds)

    output.success("Bay Area 3D model generation complete!")


if __name__ == "__main__":
    generate_bay_area()
