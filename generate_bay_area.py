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
from terrain_generator.buildings import BuildingsExtractor


def extract_buildings(bounds):
    """Extract 3D-ready buildings for the given bounds and create a visualization."""
    print("\n=== Bay Area Buildings Extraction ===")
    
    # Create buildings extractor
    extractor = BuildingsExtractor(timeout=120, use_cache=True)  # Longer timeout for large area
    
    try:
        # Extract 3D-ready buildings
        buildings = extractor.extract_buildings(bounds)
        
        if not buildings:
            print("No buildings found!")
            return []
        
        print(f"Creating visualization of {len(buildings)} 3D-ready buildings...")
        
        # Create the visualization
        create_buildings_map(buildings, bounds)
        
        return buildings
        
    except Exception as e:
        print(f"Could not extract buildings: {e}")
        print("This requires internet connection to access OpenStreetMap data")
        return []


def create_buildings_map(buildings, bounds):
    """Create a map visualization of all buildings colored by height."""
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Get height data for color mapping
    heights = [building.height for building in buildings]
    min_height = min(heights)
    max_height = max(heights)
    
    print(f"Building heights range from {min_height:.1f}m to {max_height:.1f}m")
    
    # Create colormap
    colormap = plt.cm.viridis
    
    # Plot each building
    for building in buildings:
        # Normalize height for coloring (0-1 range)
        height_norm = (building.height - min_height) / (max_height - min_height) if max_height > min_height else 0
        color = colormap(height_norm)
        
        # Create polygon from coordinates
        if len(building.coordinates) >= 3:
            # Convert coordinates to numpy array
            coords = np.array(building.coordinates)
            
            # Create polygon patch
            polygon = patches.Polygon(coords, closed=True, facecolor=color, 
                                    edgecolor='black', linewidth=0.1, alpha=0.8)
            ax.add_patch(polygon)
    
    # Set map bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Bay Area Buildings - {len(buildings)} 3D-Ready Buildings\n'
                 f'Colored by Height ({min_height:.1f}m - {max_height:.1f}m)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_height, vmax=max_height))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Building Height (meters)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Buildings: {len(buildings)}
Avg Height: {np.mean(heights):.1f}m
Max Height: {max_height:.1f}m
Total Area: {sum(b.area for b in buildings)/1_000_000:.1f} km²"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    # Save the image
    output_file = "bay_area_buildings_map.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Buildings map saved as: {output_file}")
    
    # Show basic statistics
    print(f"\n=== Building Statistics ===")
    print(f"Total buildings: {len(buildings)}")
    print(f"Height range: {min_height:.1f}m - {max_height:.1f}m")
    print(f"Average height: {np.mean(heights):.1f}m")
    print(f"Total building area: {sum(b.area for b in buildings)/1_000_000:.2f} km²")
    
    # Height distribution
    tall_buildings = len([b for b in buildings if b.height > 100])
    mid_buildings = len([b for b in buildings if 30 <= b.height <= 100])
    low_buildings = len([b for b in buildings if b.height < 30])
    
    print(f"Height distribution:")
    print(f"  Low-rise (<30m): {low_buildings} ({low_buildings/len(buildings)*100:.1f}%)")
    print(f"  Mid-rise (30-100m): {mid_buildings} ({mid_buildings/len(buildings)*100:.1f}%)")
    print(f"  High-rise (>100m): {tall_buildings} ({tall_buildings/len(buildings)*100:.1f}%)")


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
        print(f"You can download SRTM data from: https://dwtkns.com/srtm30m/")
        return None


def generate_bay_area():
    """Generate both terrain and extract buildings for the Bay Area."""
    # Bay Area bounds (covers San Francisco to San Jose area)
    bounds = (-122.67, 37.22, -121.75, 38.18)
    
    # Generate terrain
    terrain_mesh = generate_terrain()
    
    # Extract buildings and create visualization
    # buildings = extract_buildings(bounds)
    
    return terrain_mesh


if __name__ == "__main__":
    generate_bay_area()
