#!/usr/bin/env python3

from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.geotiff import GeoTiff
from terrain_generator.srtm import SRTM
from terrain_generator.console import output


def generate_terrain(prefix, bounds):
    """Generate a detailed New York terrain model."""
    output.header("New York Terrain Model Generation")

    # Create model generator
    # generator = ModelGenerator(GeoTiff("output_USGS10m.tif"))
    generator = ModelGenerator(SRTM())

    try:
        # Generate detailed terrain model
        result = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=500,
            water_threshold=1,
            elevation_multiplier=3,
            downsample_factor=1,
        )

        # Save the terrain and water meshes
        generator.save_mesh(result['land_mesh'], f"{prefix}_land.obj")
        generator.save_mesh(result['base_mesh'], f"{prefix}_base.obj")

        output.success("New York terrain model generation complete!")

    except Exception as e:
        output.error(f"Could not generate New York terrain model: {e}")
        return None


def generate_newyork():
    """Generate both terrain and extract buildings for the New York area."""
    output.header("New York 3D Model Generator", 
                 "Generating terrain and extracting buildings for the New York area")

    # Pittsburgh bounds
    bounds = (-74.1687, 40.5295, -73.7979, 40.9166)

    output.info(f"Processing region: {bounds}")
    output.info("Longitude: {:.3f}° to {:.3f}°".format(bounds[0], bounds[2]))
    output.info("Latitude: {:.3f}° to {:.3f}°".format(bounds[1], bounds[3]))

    output.print_section_divider()

    # Generate terrain
    generate_terrain("newyork", bounds)

    output.print_section_divider()

    output.success("New York 3D model generation complete!")


if __name__ == "__main__":
    generate_newyork()
