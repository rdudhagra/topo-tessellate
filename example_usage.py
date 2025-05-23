#!/usr/bin/env python3
"""
Example usage of the ModelGenerator class

This script demonstrates how to use the ModelGenerator to create 3D terrain models
from real SRTM elevation data with various parameter options.
"""

from terrain_generator.modelgenerator import ModelGenerator


def example_small_terrain():
    """Generate a small, fast terrain model for testing."""
    print("=== Example: Small Terrain Model (Fast) ===")
    
    # Create model generator
    generator = ModelGenerator()
    
    # Example bounds for a very small area
    # Format: (min_lon, min_lat, max_lon, max_lat)
    bounds = (-122.52, 37.75, -122.48, 37.78)  # Small San Francisco area
    
    try:
        # Generate small terrain model with high downsampling for speed
        mesh = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=5.0,            # 5 unit base height
            scale_factor=0.01,          # Scale down coordinates
            elevation_scale_ratio=0.02, # Default elevation scaling (2% of max dimension)
            downsample_factor=20,       # High downsampling for speed
            output_prefix="small_terrain"
        )
        
        print("Small terrain model created successfully!\n")
        return mesh
        
    except Exception as e:
        print(f"Could not generate terrain model: {e}")
        print("This requires SRTM data files in the 'topo/' directory")
        print("You can download SRTM data from: https://dwtkns.com/srtm30m/\n")
        return None


def example_detailed_terrain():
    """Generate a more detailed terrain model."""
    print("=== Example: Detailed Terrain Model ===")
    
    # Create model generator
    generator = ModelGenerator()
    
    # Example bounds for a small area
    bounds = (-122.52, 37.75, -122.50, 37.77)  # Even smaller area for detail
    
    try:
        # Generate detailed terrain model with less downsampling
        mesh = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=10.0,           # 10 unit base height
            scale_factor=0.02,          # Larger scale factor
            elevation_scale_ratio=0.03, # Slightly more pronounced elevation (3%)
            downsample_factor=5,        # Less downsampling for more detail
            output_prefix="detailed_terrain"
        )
        
        print("Detailed terrain model created successfully!\n")
        return mesh
        
    except Exception as e:
        print(f"Could not generate detailed terrain model: {e}")
        print("This requires SRTM data files in the 'topo/' directory\n")
        return None


def example_flat_terrain():
    """Generate a terrain model with subtle elevation changes."""
    print("=== Example: Flat Terrain Model (Subtle Elevation) ===")
    
    # Create model generator
    generator = ModelGenerator()
    
    # Example bounds for a flatter area
    bounds = (-122.45, 37.75, -122.43, 37.77)
    
    try:
        # Generate terrain with subtle elevation changes
        mesh = generator.generate_terrain_model(
            bounds=bounds,
            topo_dir="topo",
            base_height=2.0,            # Lower base height
            scale_factor=0.01,          # Standard scale
            elevation_scale_ratio=0.01, # Very subtle elevation (1%)
            downsample_factor=15,       # Moderate downsampling
            output_prefix="flat_terrain"
        )
        
        print("Flat terrain model created successfully!\n")
        return mesh
        
    except Exception as e:
        print(f"Could not generate flat terrain model: {e}")
        print("This requires SRTM data files in the 'topo/' directory\n")
        return None


def main():
    """Main function demonstrating ModelGenerator usage with different parameters."""
    print("ModelGenerator Usage Examples")
    print("=" * 50)
    print("This demonstrates different parameter combinations for terrain generation:")
    print("- elevation_scale_ratio: Controls how pronounced the elevation is")
    print("- downsample_factor: Controls detail level (higher = faster, less detail)")
    print("- scale_factor: Controls overall model size")
    print("- base_height: Controls the height of the flat base\n")
    
    # Example 1: Fast generation with high downsampling
    small_mesh = example_small_terrain()
    
    # Example 2: More detailed generation
    detailed_mesh = example_detailed_terrain()
    
    # Example 3: Subtle elevation changes
    flat_mesh = example_flat_terrain()
    
    print("Examples completed!")
    print("\nGenerated files:")
    if small_mesh:
        print("- small_terrain.obj")
    if detailed_mesh:
        print("- detailed_terrain.obj") 
    if flat_mesh:
        print("- flat_terrain.obj")
    
    print("\nTips:")
    print("- Use higher downsample_factor (e.g., 20) for faster processing")
    print("- Use lower downsample_factor (e.g., 5) for more detail")
    print("- Adjust elevation_scale_ratio to make terrain more/less pronounced")
    print("- Only .obj files are generated for better compatibility")


if __name__ == "__main__":
    main() 