#!/usr/bin/env python3

import argparse
from terrain_generator.generator import TerrainGenerator
import os


def main():
    """
    Generate terrain models using TerrainGenerator.
    """
    print("CityModelGenerator - World to Model")

    # Create the terrain generator
    generator = TerrainGenerator()

    # Generate San Francisco Bay Area model with proper north-up orientation
    # Bounds: San Francisco Bay Area (-123.0, 36.9, -121.7, 38.1)
    # This includes San Francisco, Oakland, Berkeley, San Jose, etc.
    print("\nGenerating San Francisco Bay Area terrain model...")

    # Generate at medium detail level (0.2)
    generator.generate_bay_area_terrain(
        bounds=(-122.673340, 37.225955, -121.753235, 38.184228),
        topo_dir="topo",
        detail_level=0.2,
        output_prefix="bay_area",
    )

    print("\nTerrain generation complete!")


if __name__ == "__main__":
    main()
