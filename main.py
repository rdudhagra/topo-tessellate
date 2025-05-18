#!/usr/bin/env python3

import argparse
from terrain_generator.generator import TerrainGenerator
import os

def main():
    parser = argparse.ArgumentParser(description='Generate 3D terrain model from SRTM data')
    parser.add_argument('hgt_file', help='Path to input .hgt file')
    parser.add_argument('output_file', help='Path to output .glb file')
    parser.add_argument('--bounds', type=float, nargs=4,
                      metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                      help='Bounding box coordinates (min_lon min_lat max_lon max_lat)')
    parser.add_argument('--detail', type=float, default=1.0,
                      help='Detail level (0.1 to 1.0, default: 1.0)')
    
    args = parser.parse_args()
    
    generator = TerrainGenerator()
    mesh = generator.process_region(
        args.hgt_file,
        args.bounds,
        args.detail
    )
    generator.export_glb(mesh, args.output_file)
    print(f"Terrain model exported to {args.output_file}")

if __name__ == '__main__':
    main()
