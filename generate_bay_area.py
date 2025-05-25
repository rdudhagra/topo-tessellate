#!/usr/bin/env python3
"""
Bay Area Terrain Generator

This script generates a 3D terrain model of the Bay Area using SRTM elevation data
and extracts 3D-ready building data from OpenStreetMap.
"""

from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.buildings import BuildingsExtractor


def extract_buildings(bounds):
    """Extract 3D-ready buildings for the given bounds."""
    print("\n=== Bay Area Buildings Extraction ===")
    
    # Create buildings extractor
    extractor = BuildingsExtractor(timeout=120, use_cache=True)  # Longer timeout for large area
    
    try:
        # Extract 3D-ready buildings
        buildings = extractor.extract_buildings(bounds)
        
        # Print statistics
        extractor.print_stats()
        
        return buildings
        
    except Exception as e:
        print(f"Could not extract buildings: {e}")
        print("This requires internet connection to access OpenStreetMap data")
        return []


def generate_terrain():
    """Generate a detailed Bay Area terrain model."""
    print("=== Bay Area Terrain Model Generation ===")

    # Create model generator
    generator = ModelGenerator()

    # Bay Area bounds (covers San Francisco to San Jose area)
    bounds = (-122.67, 37.22, -121.75, 38.18)

    try:
        # Generate detailed terrain model
        mesh = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=0.1,
            elevation_multiplier=5,
            downsample_factor=5,
            output_prefix="bay_area_terrain",
        )

        print("Bay Area terrain model created successfully!")
        return mesh

    except Exception as e:
        print(f"Could not generate Bay Area terrain model: {e}")
        print("This requires SRTM data files in the 'topo/' directory")
        print("You can download SRTM data from: https://dwtkns.com/srtm30m/")
        return None


def generate_bay_area():
    """Generate both terrain and extract buildings for the Bay Area."""
    # Bay Area bounds (covers San Francisco to San Jose area)
    bounds = (-122.67, 37.22, -121.75, 38.18)
    
    # Generate terrain
    terrain_mesh = generate_terrain()
    
    # Extract buildings
    buildings = extract_buildings(bounds)
    
    return terrain_mesh, buildings


if __name__ == "__main__":
    generate_bay_area()
