#!/usr/bin/env python3
"""
USGS National Map DEM Downloader

This script automatically downloads Digital Elevation Model (DEM) GeoTIFF data
from the USGS National Map (https://apps.nationalmap.gov/downloader/).

It reads a config file with bounds, queries the TNM Access API for available data,
and downloads either 1-meter DEMs (if they cover the entire area) or 1/3 arc-second
DEMs as a fallback.

Usage:
    python scripts/download_dem.py --config configs/sf_fidi.yaml [--output-dir topo] [--dry-run]
"""

import os
import sys
import re
import argparse
import json
from typing import Optional
from dataclasses import dataclass, field
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

import requests
import yaml

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from terrain_generator.console import output
except ImportError:
    # Fallback to simple print if console module not available
    class SimpleOutput:
        def header(self, title, subtitle=None):
            print(f"\n{'='*60}\n{title}\n{subtitle or ''}\n{'='*60}")
        def subheader(self, title):
            print(f"\n▶ {title}")
        def success(self, msg):
            print(f"✓ {msg}")
        def warning(self, msg):
            print(f"⚠ {msg}")
        def error(self, msg):
            print(f"✗ {msg}")
        def info(self, msg):
            print(f"ℹ {msg}")
        def progress_info(self, msg):
            print(f"⟳ {msg}")
        def stats_table(self, title, data):
            print(f"\n{title}:")
            for k, v in data.items():
                print(f"  {k}: {v}")
        def file_saved(self, filename, file_type="file"):
            print(f"✓ {file_type.title()} saved: {filename}")
    output = SimpleOutput()


# TNM Access API endpoint
TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

# Dataset identifiers for elevation data
# Note: These are the official USGS dataset names as used in the TNM API
DATASETS = {
    "1m": "Digital Elevation Model (DEM) 1 meter",
    "1/3_arcsec": "National Elevation Dataset (NED) 1/3 arc-second",
    # Alternative names that may be used
    "1m_alt": "1 meter DEM",
    "1/3_arcsec_alt": "1/3 arc-second DEM",
}


@dataclass
class DEMProduct:
    """Represents a DEM product from the TNM API."""
    title: str
    source_id: str
    download_url: str
    format: str
    extent: Optional[dict] = None
    file_size: Optional[int] = None
    publication_date: Optional[str] = None
    
    @classmethod
    def from_api_item(cls, item: dict) -> "DEMProduct":
        """Create a DEMProduct from a TNM API response item."""
        return cls(
            title=item.get("title", "Unknown"),
            source_id=item.get("sourceId", ""),
            download_url=item.get("downloadURL", ""),
            format=item.get("format", ""),
            extent=item.get("boundingBox"),
            file_size=item.get("sizeInBytes"),
            publication_date=item.get("publicationDate", item.get("dateCreated", "")),
        )
    
    def get_tile_key(self) -> str:
        """
        Extract a tile identifier from the filename to group duplicate tiles.
        
        For 1m DEMs, we include the survey name so different surveys are kept separate:
          USGS_1M_10_x55y419_CA_SanFrancisco_B23.tif -> 1m_10_x55y419_CA_SanFrancisco
          USGS_1M_10_x55y419_CA_CaliforniaGaps_B23.tif -> 1m_10_x55y419_CA_CaliforniaGaps
        
        For 1/3 arcsec, these are true duplicates with different dates:
          USGS_13_n38w123_20210301.tif -> 13_n38w123
        """
        filename = os.path.basename(self.download_url)
        
        # Pattern for 1m DEMs: extract zone, tile coordinates, and survey name
        # Examples:
        #   USGS_1M_10_x55y419_CA_SanFrancisco_B23.tif
        #   USGS_1m_x55y419_CA_NoCAL_Wildfires_B5b_2018.tif
        match_1m_with_zone = re.search(r'USGS_1[Mm]_(\d+)_x(\d+)y(\d+)_([A-Za-z_]+)_B', filename)
        if match_1m_with_zone:
            zone, x, y, survey = match_1m_with_zone.groups()
            return f"1m_{zone}_x{x}y{y}_{survey}"
        
        # Pattern for 1m DEMs without zone number (older format)
        match_1m_no_zone = re.search(r'USGS_1[Mm]_x(\d+)y(\d+)_([A-Za-z_]+)_B', filename)
        if match_1m_no_zone:
            x, y, survey = match_1m_no_zone.groups()
            return f"1m_x{x}y{y}_{survey}"
        
        # Pattern for 1/3 arc-second: USGS_13_nXXwYYY_YYYYMMDD.tif
        # These are true duplicates with different publication dates
        match_13 = re.search(r'USGS_13_(n\d+w\d+)', filename)
        if match_13:
            return f"13_{match_13.group(1)}"
        
        # Fallback: use full filename (no deduplication)
        return filename


def filter_latest_products(products: list[DEMProduct]) -> list[DEMProduct]:
    """
    Filter products to keep only the latest version of each tile.
    
    Groups products by tile key and keeps the one with the most recent
    publication date.
    
    Args:
        products: List of DEMProduct objects
        
    Returns:
        Filtered list with only the latest version of each tile
    """
    if not products:
        return products
    
    # Group by tile key
    tile_groups: dict[str, list[DEMProduct]] = {}
    for product in products:
        key = product.get_tile_key()
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append(product)
    
    # Keep only the latest from each group
    latest_products = []
    for key, group in tile_groups.items():
        if len(group) == 1:
            latest_products.append(group[0])
        else:
            # Sort by publication date (descending) and take the first
            # Handle None dates by sorting them last
            sorted_group = sorted(
                group,
                key=lambda p: p.publication_date or "",
                reverse=True
            )
            latest_products.append(sorted_group[0])
            
    return latest_products


def load_config(config_path: str) -> dict:
    """Load and parse a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_bounds_from_config(config: dict) -> tuple:
    """
    Extract bounds from config file.
    Supports both single-job and multi-job formats.
    
    Returns: (min_lon, min_lat, max_lon, max_lat)
    """
    # Check for direct bounds (single-job format)
    if "bounds" in config:
        bounds = config["bounds"]
        return tuple(bounds)
    
    # Check for jobs array (multi-job format)
    if "jobs" in config and config["jobs"]:
        # Get the union of all job bounds
        all_bounds = []
        for job in config["jobs"]:
            if "bounds" in job:
                all_bounds.append(job["bounds"])
        
        if all_bounds:
            # Calculate the bounding box that encompasses all jobs
            min_lon = min(b[0] for b in all_bounds)
            min_lat = min(b[1] for b in all_bounds)
            max_lon = max(b[2] for b in all_bounds)
            max_lat = max(b[3] for b in all_bounds)
            return (min_lon, min_lat, max_lon, max_lat)
    
    raise ValueError("No bounds found in config file")


def query_tnm_api(bounds: tuple, dataset_name: str, max_results: int = 500) -> list[DEMProduct]:
    """
    Query the TNM Access API for DEM products within the given bounds.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        dataset_name: Name of the dataset to query
        max_results: Maximum number of results to return
        
    Returns:
        List of DEMProduct objects
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    params = {
        "datasets": dataset_name,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "prodFormats": "GeoTIFF",
        "outputFormat": "JSON",
        "max": max_results,
    }
    
    try:
        response = requests.get(TNM_API_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        products = [DEMProduct.from_api_item(item) for item in items]
        
        # Filter to only GeoTIFF files
        products = [p for p in products if p.download_url.endswith(('.tif', '.tiff', '.TIF', '.TIFF'))]
        
        return products
        
    except requests.RequestException as e:
        output.warning(f"API request failed for {dataset_name}: {e}")
        return []


def check_full_coverage(bounds: tuple, products: list[DEMProduct], coverage_threshold: float = 1.0) -> bool:
    """
    Check if the products cover the entire bounding box.
    
    Uses Shapely to compute the union of all product extents and compare
    against the requested bounding box.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        products: List of DEMProduct objects with extent information
        coverage_threshold: Minimum fraction of area that must be covered (default 100%)
        
    Returns:
        True if products cover at least coverage_threshold of the bounds
    """
    if not products:
        return False
    
    min_lon, min_lat, max_lon, max_lat = bounds
    request_box = shapely_box(min_lon, min_lat, max_lon, max_lat)
    request_area = request_box.area
    
    # Build list of product extents
    product_boxes = []
    for product in products:
        extent = product.extent
        if extent:
            # TNM API returns bounding box as {minX, minY, maxX, maxY}
            try:
                p_box = shapely_box(
                    extent.get("minX", extent.get("xmin", 0)),
                    extent.get("minY", extent.get("ymin", 0)),
                    extent.get("maxX", extent.get("xmax", 0)),
                    extent.get("maxY", extent.get("ymax", 0)),
                )
                product_boxes.append(p_box)
            except Exception:
                continue
    
    if not product_boxes:
        # No extent info available, fall back to checking if we have any products
        return len(products) > 0
    
    # Compute union of all product extents
    try:
        products_union = unary_union(product_boxes)
        
        # Intersect with request box to get covered area
        covered = request_box.intersection(products_union)
        covered_area = covered.area
        
        coverage_ratio = covered_area / request_area if request_area > 0 else 0
        
        output.info(f"  Coverage: {coverage_ratio*100:.1f}% of requested area")
        
        return coverage_ratio >= coverage_threshold
        
    except Exception as e:
        output.warning(f"  Could not compute coverage: {e}")
        return len(products) > 0


def download_file(url: str, output_path: str, timeout: int = 300) -> bool:
    """
    Download a file from URL to output path with progress indication.
    
    Args:
        url: Download URL
        output_path: Local file path to save to
        timeout: Request timeout in seconds
        
    Returns:
        True if download succeeded
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        # Print progress on same line
                        print(f"\r  Downloading: {pct:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end="", flush=True)
        
        if total_size > 0:
            print()  # New line after progress
            
        return True
        
    except requests.RequestException as e:
        output.error(f"  Download failed: {e}")
        return False


def download_products(products: list[DEMProduct], output_dir: str, dry_run: bool = False) -> list[str]:
    """
    Download a list of DEM products to the output directory.
    
    Args:
        products: List of DEMProduct objects to download
        output_dir: Directory to save files to
        dry_run: If True, only print what would be downloaded
        
    Returns:
        List of successfully downloaded file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_files = []
    
    for i, product in enumerate(products, 1):
        filename = os.path.basename(product.download_url)
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists with same name
        if os.path.exists(output_path):
            output.info(f"  [{i}/{len(products)}] Skipping (exists): {filename}")
            downloaded_files.append(output_path)
            continue
        
        if dry_run:
            size_str = f" ({product.file_size/(1024*1024):.1f} MB)" if product.file_size else ""
            output.info(f"  [{i}/{len(products)}] Would download: {filename}{size_str}")
        else:
            output.progress_info(f"  [{i}/{len(products)}] Downloading: {filename}")
            
            if download_file(product.download_url, output_path):
                output.success(f"  Downloaded: {filename}")
                downloaded_files.append(output_path)
            else:
                output.error(f"  Failed: {filename}")
    
    return downloaded_files


def main():
    """Main entry point for the DEM downloader."""
    parser = argparse.ArgumentParser(
        description="Download DEM data from USGS National Map based on config bounds"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="topo",
        help="Output directory for downloaded files (default: topo)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--force-1m",
        action="store_true",
        help="Only download 1m data, skip if not available for full area"
    )
    parser.add_argument(
        "--force-1-3-arcsec",
        action="store_true",
        help="Force download of 1/3 arc-second data instead of checking 1m first"
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        default=True,
        help="Download only the latest version of each tile (default: True)"
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Download all versions of each tile, not just the latest"
    )
    
    args = parser.parse_args()
    
    # Load config
    output.header("USGS National Map DEM Downloader", f"Config: {args.config}")
    
    if not os.path.exists(args.config):
        output.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        config = load_config(args.config)
        bounds = get_bounds_from_config(config)
    except Exception as e:
        output.error(f"Failed to parse config: {e}")
        sys.exit(1)
    
    min_lon, min_lat, max_lon, max_lat = bounds
    
    config_info = {
        "Min Longitude": f"{min_lon:.6f}°",
        "Max Longitude": f"{max_lon:.6f}°",
        "Min Latitude": f"{min_lat:.6f}°",
        "Max Latitude": f"{max_lat:.6f}°",
        "Output Directory": args.output_dir,
        "Dry Run": "Yes" if args.dry_run else "No",
    }
    output.stats_table("Configuration", config_info)
    
    # Query for available data
    output.subheader("Querying USGS National Map API")
    
    selected_products = []
    selected_resolution = None
    
    if not args.force_1_3_arcsec:
        # First, try 1-meter DEM
        output.progress_info("Searching for 1-meter DEM data...")
        
        products_1m = query_tnm_api(bounds, DATASETS["1m"])
        
        if not products_1m:
            # Try alternative dataset name
            products_1m = query_tnm_api(bounds, DATASETS["1m_alt"])
        
        if products_1m:
            output.info(f"  Found {len(products_1m)} 1-meter DEM products")
            
            # Check if 1m covers the full area
            if check_full_coverage(bounds, products_1m, 1.0):
                output.success("  1-meter data covers the full requested area!")
                selected_products = products_1m
                selected_resolution = "1m"
            else:
                output.warning("  1-meter data does NOT cover the full area")
                if args.force_1m:
                    output.error("  --force-1m specified but full coverage not available")
                    sys.exit(1)
        else:
            output.warning("  No 1-meter DEM data found for this area")
            if args.force_1m:
                output.error("  --force-1m specified but no 1m data available")
                sys.exit(1)
    
    # If 1m not selected, try 1/3 arc-second
    if not selected_products:
        output.progress_info("Searching for 1/3 arc-second DEM data...")
        
        products_1_3 = query_tnm_api(bounds, DATASETS["1/3_arcsec"])
        
        if not products_1_3:
            # Try alternative dataset name
            products_1_3 = query_tnm_api(bounds, DATASETS["1/3_arcsec_alt"])
        
        if products_1_3:
            output.info(f"  Found {len(products_1_3)} 1/3 arc-second DEM products")
            
            if check_full_coverage(bounds, products_1_3, 1.0):
                output.success("  1/3 arc-second data covers the full requested area!")
                selected_products = products_1_3
                selected_resolution = "1/3 arc-second"
            else:
                output.warning("  1/3 arc-second data may not fully cover the area, proceeding anyway")
                selected_products = products_1_3
                selected_resolution = "1/3 arc-second (partial)"
        else:
            output.warning("  No 1/3 arc-second DEM data found for this area")
    
    if not selected_products:
        output.error("No DEM data available for the specified area!")
        output.info("You may need to download data manually from: https://apps.nationalmap.gov/downloader/")
        sys.exit(1)
    
    # Filter to latest versions only (unless --all-versions is specified)
    if not args.all_versions:
        original_count = len(selected_products)
        selected_products = filter_latest_products(selected_products)
        if len(selected_products) < original_count:
            output.info(f"  Filtered to {len(selected_products)} tiles (latest versions only, was {original_count})")
    
    # Download selected products
    output.subheader(f"Downloading {selected_resolution} DEM Data")
    output.info(f"Products to download: {len(selected_products)}")
    
    # Calculate total size if available
    total_size = sum(p.file_size or 0 for p in selected_products)
    if total_size > 0:
        output.info(f"Estimated total download size: {total_size/(1024*1024):.1f} MB")
    
    downloaded = download_products(selected_products, args.output_dir, args.dry_run)
    
    # Summary
    output.subheader("Summary")
    summary = {
        "Resolution": selected_resolution,
        "Products Found": len(selected_products),
        "Files Downloaded": len(downloaded) if not args.dry_run else "N/A (dry run)",
        "Output Directory": os.path.abspath(args.output_dir),
    }
    output.stats_table("Download Summary", summary)
    
    if downloaded and not args.dry_run:
        output.success(f"Successfully downloaded {len(downloaded)} file(s)")
        output.info("Downloaded files:")
        for f in downloaded:
            output.info(f"  • {os.path.basename(f)}")
    elif args.dry_run:
        output.info("Dry run complete - no files were downloaded")
    else:
        output.warning("No files were downloaded")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

