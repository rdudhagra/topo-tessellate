#!/usr/bin/env python3

import argparse
from terrain_generator.generator import TerrainGenerator
from terrain_generator.meshlib_terrain_generator import MeshLibTerrainGenerator, MESHLIB_AVAILABLE
import os


def main():
    """
    Generate terrain models using either the original or MeshLib-optimized TerrainGenerator.
    """
    print("TerrainGenerator - World to Model (with MeshLib support)")

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
    parser.add_argument("--water-level", type=float, default=0.0, 
                        help="Elevation value for water areas (default: 0.0)")
    parser.add_argument("--height-scale", type=float, default=0.05, 
                        help="Scale factor for height (default: 0.05)")
    parser.add_argument("--export-format", type=str, choices=["glb", "obj"], default="glb",
                        help="Format to export the model (default: 'glb')")
    
    # New argument for choosing generator
    parser.add_argument("--use-meshlib", action="store_true",
                        help="Use MeshLib-optimized terrain generator (requires meshlib)")

    args = parser.parse_args()

    # Choose generator based on user preference and availability
    if args.use_meshlib:
        if not MESHLIB_AVAILABLE:
            print("ERROR: MeshLib is not available. Please install with: pip install meshlib")
            print("Falling back to original generator...")
            generator = TerrainGenerator()
            output_prefix = args.output_prefix
        else:
            print("Using MeshLib-optimized terrain generator...")
            generator = MeshLibTerrainGenerator()
            output_prefix = f"{args.output_prefix}_meshlib"
    else:
        print("Using original terrain generator...")
        generator = TerrainGenerator()
        output_prefix = args.output_prefix

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
        output_prefix=output_prefix,
        water_level=args.water_level,
        height_scale=args.height_scale,
        export_format=args.export_format
    )

    print("\nTerrain generation complete!")


if __name__ == "__main__":
    main() 