#!/usr/bin/env python3

import argparse
from terrain_generator.generator import TerrainGenerator
import os


def main():
    """
    Generate terrain models using TerrainGenerator.
    """
    print("CityModelGenerator - World to Model")

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate 3D terrain models from SRTM elevation data")
    
    # Required arguments
    parser.add_argument("--min-lon", type=float, required=True, 
                        help="Minimum longitude of the region")
    parser.add_argument("--min-lat", type=float, required=True, 
                        help="Minimum latitude of the region")
    parser.add_argument("--max-lon", type=float, required=True, 
                        help="Maximum longitude of the region")
    parser.add_argument("--max-lat", type=float, required=True, 
                        help="Maximum latitude of the region")
    
    # Optional arguments
    parser.add_argument("--topo-dir", type=str, default="topo", 
                        help="Directory containing SRTM data (default: 'topo')")
    parser.add_argument("--detail-level", type=float, default=0.2, 
                        help="Level of detail, 0.01-1.0 (default: 0.2)")
    parser.add_argument("--output-prefix", type=str, default="terrain", 
                        help="Prefix for output files (default: 'terrain')")
    parser.add_argument("--water-level", type=float, default=-15.0, 
                        help="Elevation value for water areas (default: -15.0)")
    parser.add_argument("--shore-height", type=float, default=1.0, 
                        help="Elevation value for shore areas (default: 1.0)")
    parser.add_argument("--shore-buffer", type=int, default=1, 
                        help="Number of cells for shore buffer (default: 1)")
    parser.add_argument("--height-scale", type=float, default=0.05, 
                        help="Scale factor for height (default: 0.05)")
    parser.add_argument("--debug", action="store_true",
                        help="Generate debug visualizations as JPG files")
    
    args = parser.parse_args()

    # Create the terrain generator
    generator = TerrainGenerator()

    # Construct bounds tuple from arguments
    bounds = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    
    # Generate terrain model with the specified parameters
    print(f"\nGenerating terrain model for region: {bounds}...")
    
    # Reference bounds for the San Francisco Bay Area:
    # San Francisco Bay Area bounds: (-122.673340, 37.225955, -121.753235, 38.184228)
    # This includes San Francisco, Oakland, Berkeley, San Jose, etc.
    
    generator.generate_terrain(
        bounds=bounds,
        topo_dir=args.topo_dir,
        detail_level=args.detail_level,
        output_prefix=args.output_prefix,
        water_level=args.water_level,
        shore_height=args.shore_height,
        shore_buffer=args.shore_buffer,
        height_scale=args.height_scale,
        debug=args.debug
    )

    print("\nTerrain generation complete!")


if __name__ == "__main__":
    main()
