#!/usr/bin/env python3
"""
Example usage of the ModelGenerator class

This script demonstrates how to use the ModelGenerator to create 3D terrain models
from both dummy data and real SRTM elevation data.
"""

from terrain_generator.modelgenerator import ModelGenerator


def example_dummy_terrain():
    """Generate a dummy terrain model for testing."""
    print("=== Example: Dummy Terrain Model ===")
    
    # Create model generator
    generator = ModelGenerator()
    
    # Generate dummy terrain model
    mesh = generator.generate_dummy_terrain_model(
        width=50,              # Smaller for faster processing
        height=50,
        max_elevation=10.0,    # 10 unit height
        base_height=5.0,       # 5 unit base
        scale_factor=0.2,      # Scale factor for coordinates
        output_prefix="demo_dummy_terrain"
    )
    
    print("Dummy terrain model created successfully!\n")
    return mesh


def example_real_terrain():
    """Generate a real terrain model from SRTM data."""
    print("=== Example: Real Terrain Model ===")
    
    # Create model generator
    generator = ModelGenerator()
    
    # Example bounds for a small area in San Francisco
    # Format: (min_lon, min_lat, max_lon, max_lat)
    bounds = (-122.52, 37.75, -122.48, 37.78)
    
    try:
        # Generate real terrain model
        mesh = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",           # Directory with SRTM data
            base_height=50.0,          # 50 meter base height
            scale_factor=0.001,        # Scale down for manageable model size
            output_prefix="demo_real_terrain"
        )
        
        print("Real terrain model created successfully!\n")
        return mesh
        
    except Exception as e:
        print(f"Could not generate real terrain model: {e}")
        print("This requires SRTM data files in the 'topo/' directory")
        print("You can download SRTM data from: https://dwtkns.com/srtm30m/\n")
        return None


def main():
    """Main function demonstrating ModelGenerator usage."""
    print("ModelGenerator Usage Examples")
    print("=" * 40)
    
    # Example 1: Dummy terrain (always works)
    dummy_mesh = example_dummy_terrain()
    
    # Example 2: Real terrain (requires SRTM data)
    real_mesh = example_real_terrain()
    
    print("Examples completed!")
    print("\nGenerated files:")
    print("- demo_dummy_terrain.ply/.stl/.obj")
    if real_mesh:
        print("- demo_real_terrain.ply/.stl/.obj")


if __name__ == "__main__":
    main() 