#!/usr/bin/env python3
"""
Test script for terrain_generator/elevation.py

This script tests the ElevationMap class with San Francisco Bay Area coordinates.
Usage: python test_elevation.py --min-lon -122.67 --min-lat 37.22 --max-lon -121.75 --max-lat 38.18
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import time

# Add the base directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from terrain_generator.elevation import ElevationMap
except ImportError as e:
    print(f"Error importing ElevationMap: {e}")
    print("Make sure the terrain_generator directory exists and contains elevation.py")
    sys.exit(1)


def test_elevation_map(bounds, topo_dir="topo", save_plots=True):
    """
    Test the ElevationMap with given bounds.

    Args:
        bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        topo_dir (str): Directory containing SRTM data
        save_plots (bool): Whether to save visualization plots
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    print("=" * 60)
    print("ELEVATION MAP TEST")
    print("=" * 60)
    print(f"Testing bounds: {bounds}")
    print(
        f"Region: Longitude {min_lon}° to {max_lon}°, Latitude {min_lat}° to {max_lat}°"
    )
    print(f"SRTM data directory: {topo_dir}")
    print()

    # Check if topo directory exists
    if not os.path.exists(topo_dir):
        print(f"Warning: SRTM data directory '{topo_dir}' not found!")
        print("Please ensure you have SRTM data files in the 'topo' directory.")
        return False

    # List available SRTM files
    srtm_files = [
        f for f in os.listdir(topo_dir) if f.endswith(".hgt.zip") or f.endswith(".hgt")
    ]
    print(f"Found {len(srtm_files)} SRTM files in {topo_dir}:")
    for f in sorted(srtm_files)[:10]:  # Show first 10
        print(f"  {f}")
    if len(srtm_files) > 10:
        print(f"  ... and {len(srtm_files) - 10} more")
    print()

    try:
        # Initialize the ElevationMap
        print("Initializing ElevationMap...")
        elevation_map = ElevationMap()

        # Test finding required tiles
        print("Finding required SRTM tiles...")
        required_tiles = elevation_map._find_required_tiles(bounds)
        print(f"Required tiles: {required_tiles}")

        # Test finding tile files
        tile_files = elevation_map._find_tile_files(required_tiles, topo_dir)
        print(f"Found {len(tile_files)} of {len(required_tiles)} required tiles:")
        for coords, filepath in tile_files.items():
            print(f"  {coords}: {os.path.basename(filepath)}")

        if not tile_files:
            print("No SRTM tiles found for the specified bounds!")
            print("Please ensure you have the appropriate SRTM data files.")
            return False

        print()

        # Get elevation data
        print("Getting elevation data...")
        start_time = time.time()
        elevation_data = elevation_map.get_elevation_data(bounds, topo_dir)
        end_time = time.time()

        print(
            f"Elevation data generation completed in {end_time - start_time:.2f} seconds"
        )
        print(f"Elevation data shape: {elevation_data.shape}")
        print(f"Elevation data type: {elevation_data.dtype}")
        print()

        # Analyze the elevation data
        print("ELEVATION DATA ANALYSIS:")
        print("-" * 30)
        print(f"Min elevation: {np.min(elevation_data):.1f} meters")
        print(f"Max elevation: {np.max(elevation_data):.1f} meters")
        print(f"Mean elevation: {np.mean(elevation_data):.1f} meters")
        print(f"Std deviation: {np.std(elevation_data):.1f} meters")
        print(f"Data points: {elevation_data.size:,}")
        print()

        # Check for any issues
        zero_count = np.sum(elevation_data == 0)
        if zero_count > 0:
            print(
                f"Warning: {zero_count} zero elevation points found (possible no-data values)"
            )

        negative_count = np.sum(elevation_data < 0)
        if negative_count > 0:
            print(f"Info: {negative_count} below sea level points found")

        print()

        # Create visualizations if requested
        if save_plots:
            print("Creating visualizations...")
            create_elevation_plots(elevation_data, bounds, save_plots)

        print("Test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during elevation data processing: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_elevation_plots(elevation_data, bounds, save_plots=True):
    """
    Create visualization plots of the elevation data.

    Args:
        elevation_data (numpy.ndarray): The elevation data
        bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        save_plots (bool): Whether to save the plots to files
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Elevation Data Analysis\nRegion: {min_lon}° to {max_lon}°E, {min_lat}° to {max_lat}°N",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Basic elevation map
    im1 = axes[0, 0].imshow(elevation_data, cmap="terrain", aspect="auto")
    axes[0, 0].set_title("Elevation Map")
    axes[0, 0].set_xlabel("Longitude (pixels)")
    axes[0, 0].set_ylabel("Latitude (pixels)")
    plt.colorbar(im1, ax=axes[0, 0], label="Elevation (m)")

    # 2. Hillshade visualization
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elevation_data, vert_exag=2.0, dx=1.0, dy=1.0)
    axes[0, 1].imshow(hillshade, cmap="gray", aspect="auto")
    axes[0, 1].set_title("Hillshade Relief")
    axes[0, 1].set_xlabel("Longitude (pixels)")
    axes[0, 1].set_ylabel("Latitude (pixels)")

    # 3. Elevation histogram
    axes[1, 0].hist(elevation_data.flatten(), bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Elevation Distribution")
    axes[1, 0].set_xlabel("Elevation (m)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 3D-like contour plot
    contour = axes[1, 1].contour(
        elevation_data, levels=20, colors="brown", linewidths=0.5
    )
    contour_filled = axes[1, 1].contourf(
        elevation_data, levels=20, cmap="terrain", alpha=0.8
    )
    axes[1, 1].set_title("Topographic Contours")
    axes[1, 1].set_xlabel("Longitude (pixels)")
    axes[1, 1].set_ylabel("Latitude (pixels)")
    plt.colorbar(contour_filled, ax=axes[1, 1], label="Elevation (m)")

    plt.tight_layout()

    if save_plots:
        output_file = "elevation_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved as: {output_file}")

    # Show the plot (comment out if running headless)
    try:
        plt.show()
    except:
        print("Display not available, plot saved only.")


def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test the ElevationMap class")
    parser.add_argument(
        "--min-lon", type=float, required=True, help="Minimum longitude"
    )
    parser.add_argument("--min-lat", type=float, required=True, help="Minimum latitude")
    parser.add_argument(
        "--max-lon", type=float, required=True, help="Maximum longitude"
    )
    parser.add_argument("--max-lat", type=float, required=True, help="Maximum latitude")
    parser.add_argument(
        "--topo-dir",
        type=str,
        default="topo",
        help="Directory containing SRTM data (default: topo)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip creating visualization plots"
    )

    args = parser.parse_args()

    # Validate bounds
    if args.min_lon >= args.max_lon:
        print("Error: min-lon must be less than max-lon")
        sys.exit(1)

    if args.min_lat >= args.max_lat:
        print("Error: min-lat must be less than max-lat")
        sys.exit(1)

    bounds = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)

    success = test_elevation_map(bounds, args.topo_dir, not args.no_plots)

    if success:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
