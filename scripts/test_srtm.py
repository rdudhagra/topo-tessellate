#!/usr/bin/env python3
"""
SRTM Testing Script

This script tests the SRTM elevation data loading and processing functionality.
"""

import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from terrain_generator.srtm import SRTM
    from terrain_generator.console import output
except ImportError as e:
    output.error(f"Error importing SRTM: {e}")
    output.error("Make sure the terrain_generator directory exists and contains srtm.py")
    sys.exit(1)


def test_srtm(bounds, topo_dir="topo", save_plots=True):
    """
    Test the SRTM class with given bounds and visualize the results.

    Args:
        bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        topo_dir (str): Directory containing SRTM data files
        save_plots (bool): Whether to save visualization plots
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    output.header("ELEVATION MAP TEST", f"Testing bounds: {bounds}")
    
    # Display region information
    region_info = {
        "Longitude Range": f"{min_lon}° to {max_lon}°",
        "Latitude Range": f"{min_lat}° to {max_lat}°",
        "SRTM Data Directory": topo_dir
    }
    output.stats_table("Test Configuration", region_info)

    # Check if topo directory exists
    if not os.path.exists(topo_dir):
        output.error(f"SRTM data directory '{topo_dir}' not found!")
        output.info("Please ensure you have SRTM data files in the 'topo' directory.")
        return False

    # List available SRTM files
    srtm_files = [
        f for f in os.listdir(topo_dir) if f.endswith(".hgt.zip") or f.endswith(".hgt")
    ]
    
    output.info(f"Found {len(srtm_files)} SRTM files in {topo_dir}")
    if len(srtm_files) > 0:
        output.info("Sample files:")
        for f in sorted(srtm_files)[:5]:  # Show first 5
            output.info(f"  • {f}")
        if len(srtm_files) > 5:
            output.info(f"  ... and {len(srtm_files) - 5} more")

    try:
        # Initialize the SRTM
        output.subheader("Initializing SRTM")
        srtm = SRTM()

        # Test finding required tiles
        output.progress_info("Finding required SRTM tiles...")
        required_tiles = srtm._find_required_tiles(bounds)
        output.info(f"Required tiles: {required_tiles}")

        # Test finding tile files
        tile_files = srtm._find_tile_files(required_tiles, topo_dir)
        
        if tile_files:
            tile_info = {
                "Required Tiles": len(required_tiles),
                "Found Tiles": len(tile_files),
                "Coverage": f"{len(tile_files)/len(required_tiles)*100:.1f}%"
            }
            output.stats_table("Tile Availability", tile_info)
            
            output.info("Found tile files:")
            for coords, filepath in tile_files.items():
                output.info(f"  • {coords}: {os.path.basename(filepath)}")
        else:
            output.error("No SRTM tiles found for the specified bounds!")
            output.info("Please ensure you have the appropriate SRTM data files.")
            return False

        # Get elevation data
        output.subheader("Processing elevation data")
        start_time = time.time()
        
        with output.progress_context("Loading elevation data"):
            elevation_data = srtm.get_elevation(bounds, topo_dir)
        
        end_time = time.time()

        # Analyze the elevation data
        output.success(f"Elevation data generation completed in {end_time - start_time:.2f} seconds")
        
        elevation_stats = {
            "Data Shape": f"{elevation_data.shape[1]} × {elevation_data.shape[0]}",
            "Data Type": str(elevation_data.dtype),
            "Total Points": f"{elevation_data.size:,}"
        }
        output.stats_table("Data Information", elevation_stats)

        # Statistical analysis
        analysis_stats = {
            "Min Elevation": f"{np.min(elevation_data):.1f} m",
            "Max Elevation": f"{np.max(elevation_data):.1f} m",
            "Mean Elevation": f"{np.mean(elevation_data):.1f} m",
            "Std Deviation": f"{np.std(elevation_data):.1f} m"
        }
        output.stats_table("Elevation Analysis", analysis_stats)

        # Check for any issues
        zero_count = np.sum(elevation_data == 0)
        negative_count = np.sum(elevation_data < 0)
        
        if zero_count > 0:
            output.warning(f"{zero_count:,} zero elevation points found (possible no-data values)")

        if negative_count > 0:
            output.info(f"{negative_count:,} below sea level points found")

        # Create visualizations if requested
        if save_plots:
            output.subheader("Creating visualizations")
            create_elevation_plots(elevation_data, bounds, save_plots)

        output.success("Test completed successfully!")
        return True

    except Exception as e:
        output.error(f"Error during elevation data processing: {e}")
        import traceback
        output.error("Full traceback:")
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
        output.file_saved(output_file, "visualization")

    # Show the plot (comment out if running headless)
    try:
        plt.show()
    except:
        output.info("Display not available, plot saved only.")


def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test the SRTM class")
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
        output.error("min-lon must be less than max-lon")
        sys.exit(1)

    if args.min_lat >= args.max_lat:
        output.error("min-lat must be less than max-lat")
        sys.exit(1)

    bounds = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    success = test_srtm(bounds, args.topo_dir, not args.no_plots)

    if success:
        output.success("🎉 All tests passed successfully!")
    else:
        output.error("❌ Tests failed!")


if __name__ == "__main__":
    main()
