#!/usr/bin/env python3

from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.geotiff import GeoTiff
from terrain_generator.srtm import SRTM
from terrain_generator.console import output


def generate_terrain(prefix, bounds):
    """Generate a detailed Pittsburgh terrain model."""
    output.header("Pittsburgh Terrain Model Generation")

    # Create model generator
    # generator = ModelGenerator(GeoTiff("output_USGS10m.tif"))
    generator = ModelGenerator(SRTM())

    try:
        # Generate detailed terrain model
        result = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=500,
            water_threshold=61,
            elevation_multiplier=2,
            downsample_factor=1,
        )

        # Save the terrain and water meshes
        generator.save_mesh(result['land_mesh'], f"{prefix}_land.obj")
        generator.save_mesh(result['base_mesh'], f"{prefix}_base.obj")

        output.success("Pittsburgh terrain model generation complete!")

    except Exception as e:
        output.error(f"Could not generate Pittsburgh terrain model: {e}")
        return None


def generate_pittsburgh():
    """Generate both terrain and extract buildings for the Pittsburgh area."""
    output.header("Pittsburgh 3D Model Generator", 
                 "Generating terrain and extracting buildings for the Pittsburgh area")

    # Pittsburgh bounds
    bounds = (-80.0557, 40.3881, -79.8541, 40.5135)

    output.info(f"Processing region: {bounds}")
    output.info("Longitude: {:.3f}° to {:.3f}°".format(bounds[0], bounds[2]))
    output.info("Latitude: {:.3f}° to {:.3f}°".format(bounds[1], bounds[3]))

    output.print_section_divider()

    # Generate terrain
    generate_terrain("pittsburgh", bounds)

    output.print_section_divider()

    output.success("Pittsburgh 3D model generation complete!")


if __name__ == "__main__":
    generate_pittsburgh()
