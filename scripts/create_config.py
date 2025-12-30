#!/usr/bin/env python3
"""
Interactive Configuration Creator for Topo-Tessellate

This script walks the user through creating a YAML configuration file
by prompting for each option with descriptions and defaults.

Usage:
    python scripts/create_config.py configs/my_config.yaml
"""

import os
import sys
import argparse
import yaml


def prompt(description: str, default=None, required: bool = False, validator=None):
    """
    Prompt the user for input with a description and optional default.
    
    Args:
        description: Description of the option
        default: Default value (None means no default)
        required: If True and no default, user must provide a value
        validator: Optional function to validate input, returns (valid, error_msg)
    
    Returns:
        The user's input or the default value
    """
    if default is not None:
        prompt_text = f"\n{description}\n  [default: {default}]: "
    elif required:
        prompt_text = f"\n{description}\n  (required): "
    else:
        prompt_text = f"\n{description}\n  : "
    
    while True:
        user_input = input(prompt_text).strip()
        
        if user_input == "":
            if default is not None:
                return default
            elif required:
                print("  Error: This field is required.")
                continue
            else:
                return None
        
        if validator:
            valid, error_msg = validator(user_input)
            if not valid:
                print(f"  Error: {error_msg}")
                continue
        
        return user_input


def validate_float(value: str):
    """Validate that a string can be converted to a float."""
    try:
        float(value)
        return True, None
    except ValueError:
        return False, "Must be a valid number"


def validate_positive_float(value: str):
    """Validate that a string is a positive float."""
    try:
        f = float(value)
        if f <= 0:
            return False, "Must be a positive number"
        return True, None
    except ValueError:
        return False, "Must be a valid number"


def validate_non_negative_float(value: str):
    """Validate that a string is a non-negative float."""
    try:
        f = float(value)
        if f < 0:
            return False, "Must be a non-negative number"
        return True, None
    except ValueError:
        return False, "Must be a valid number"


def validate_positive_int(value: str):
    """Validate that a string is a positive integer."""
    try:
        i = int(value)
        if i <= 0:
            return False, "Must be a positive integer"
        return True, None
    except ValueError:
        return False, "Must be a valid integer"


def validate_longitude(value: str):
    """Validate longitude is in valid range."""
    try:
        lon = float(value)
        if lon < -180 or lon > 180:
            return False, "Longitude must be between -180 and 180"
        return True, None
    except ValueError:
        return False, "Must be a valid number"


def validate_latitude(value: str):
    """Validate latitude is in valid range."""
    try:
        lat = float(value)
        if lat < -90 or lat > 90:
            return False, "Latitude must be between -90 and 90"
        return True, None
    except ValueError:
        return False, "Must be a valid number"


def validate_base_height(value: str):
    """Validate base height is >= 20mm."""
    try:
        h = float(value)
        if h < 20:
            return False, "Base height must be at least 20mm for structural integrity"
        return True, None
    except ValueError:
        return False, "Must be a valid number"


def validate_yes_no(value: str):
    """Validate yes/no input."""
    if value.lower() in ('y', 'yes', 'n', 'no'):
        return True, None
    return False, "Enter 'y' or 'n'"


def parse_yes_no(value: str) -> bool:
    """Parse yes/no string to boolean."""
    return value.lower() in ('y', 'yes')


def main():
    parser = argparse.ArgumentParser(
        description="Interactively create a Topo-Tessellate configuration file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output YAML configuration file"
    )
    
    args = parser.parse_args()
    output_path = args.output_file
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("  Topo-Tessellate Configuration Creator")
    print("=" * 60)
    print("\nThis wizard will help you create a configuration file.")
    print("Press Enter to accept the default value shown in brackets.")
    print("For coordinates, use decimal degrees (e.g., -122.4194 for San Francisco).")
    
    config = {"version": 1}
    
    # ==================== Basic Info ====================
    print("\n" + "=" * 60)
    print("  BASIC INFORMATION")
    print("=" * 60)
    
    name = prompt(
        "Project name (used for display and job selection)",
        required=True
    )
    config["name"] = name
    
    output_prefix = prompt(
        "Output prefix (used for filenames, e.g., 'sf' produces sf_land.stl)",
        default=name.lower().replace(" ", "_")
    )
    config["output_prefix"] = output_prefix
    
    # ==================== Bounds ====================
    print("\n" + "=" * 60)
    print("  GEOGRAPHIC BOUNDS")
    print("=" * 60)
    print("\nDefine the bounding box for your terrain model.")
    print("Use map apps or GIS tools to find coordinates.")
    print("Tip: Keep areas modest on first runs (a few km²).")
    
    min_lon = float(prompt(
        "Minimum longitude (west edge, negative for western hemisphere)",
        required=True,
        validator=validate_longitude
    ))
    
    min_lat = float(prompt(
        "Minimum latitude (south edge)",
        required=True,
        validator=validate_latitude
    ))
    
    max_lon = float(prompt(
        "Maximum longitude (east edge)",
        required=True,
        validator=validate_longitude
    ))
    
    max_lat = float(prompt(
        "Maximum latitude (north edge)",
        required=True,
        validator=validate_latitude
    ))
    
    # Validate bounds order
    if min_lon >= max_lon:
        print("\nWarning: min_lon should be less than max_lon. Swapping values.")
        min_lon, max_lon = max_lon, min_lon
    
    if min_lat >= max_lat:
        print("\nWarning: min_lat should be less than max_lat. Swapping values.")
        min_lat, max_lat = max_lat, min_lat
    
    config["bounds"] = [min_lon, min_lat, max_lon, max_lat]
    
    # ==================== Global Settings ====================
    print("\n" + "=" * 60)
    print("  GLOBAL SETTINGS")
    print("=" * 60)
    
    scale_max = float(prompt(
        "Maximum model length in mm (longest side of output)",
        default="225.0",
        validator=validate_positive_float
    ))
    config["global"] = {"scale_max_length_mm": scale_max}
    
    # ==================== Base Settings ====================
    print("\n" + "=" * 60)
    print("  BASE SETTINGS")
    print("=" * 60)
    
    base_height = float(prompt(
        "Base height in mm (minimum 20mm for structural integrity)",
        default="20.0",
        validator=validate_base_height
    ))
    config["base"] = {"height": base_height}
    
    # ==================== Elevation Source ====================
    # Hardcoded to geotiff with topo dir and glob
    print("\n" + "=" * 60)
    print("  ELEVATION SOURCE")
    print("=" * 60)
    print("\nElevation data will be read from GeoTIFF files.")
    print("Use the download-dem.sh script to download USGS elevation data.")
    
    config["elevation_source"] = {
        "type": "geotiff",
        "topo_dir": "topo",
        "glob": "*"
    }
    print("\n  Configured: GeoTIFF files from 'topo/' directory")
    
    # ==================== Terrain Settings ====================
    print("\n" + "=" * 60)
    print("  TERRAIN SETTINGS")
    print("=" * 60)
    
    elevation_mult = float(prompt(
        "Elevation multiplier (vertical exaggeration, 1.0 = real scale)",
        default="1",
        validator=validate_positive_float
    ))
    
    downsample = int(prompt(
        "Downsample factor (1 = full resolution, higher = faster but lower detail)",
        default="10",
        validator=validate_positive_int
    ))
    
    # Hardcoded terrain settings per user request
    config["terrain"] = {
        "elevation_multiplier": elevation_mult,
        "downsample_factor": downsample,
        "water_threshold": 1,
        "adaptive_tolerance_z": 10.0,
        "adaptive_max_gap_fraction": 0.001,
        "adaptive_max_sampled_rows": 10000,
        "adaptive_max_sampled_cols": 10000,
        "split_at_water_level": True
    }
    
    # ==================== Buildings ====================
    print("\n" + "=" * 60)
    print("  BUILDING SETTINGS")
    print("=" * 60)
    
    buildings_enabled = parse_yes_no(prompt(
        "Enable building extraction from OpenStreetMap? (y/n)",
        default="y",
        validator=validate_yes_no
    ))
    
    # Hardcoded building settings per user request
    config["buildings"] = {
        "enabled": buildings_enabled,
        "extract": {
            "max_building_distance_meters": 0
        },
        "generate": {
            "building_height_multiplier": 1,
            "min_building_height": 25
        }
    }
    
    # ==================== Tiling ====================
    print("\n" + "=" * 60)
    print("  TILING SETTINGS")
    print("=" * 60)
    
    tiling_enabled = parse_yes_no(prompt(
        "Enable tiling? (split into multiple pieces for large prints) (y/n)",
        default="n",
        validator=validate_yes_no
    ))
    
    config["tiling"] = {"enabled": tiling_enabled}
    
    if tiling_enabled:
        rows = int(prompt(
            "Number of tile rows",
            default="2",
            validator=validate_positive_int
        ))
        cols = int(prompt(
            "Number of tile columns",
            default="2",
            validator=validate_positive_int
        ))
        config["tiling"]["rows"] = rows
        config["tiling"]["cols"] = cols
    
    # ==================== Output ====================
    print("\n" + "=" * 60)
    print("  OUTPUT SETTINGS")
    print("=" * 60)
    
    output_dir = prompt(
        "Output directory for generated STL files",
        default=f"outputs/{output_prefix}"
    )
    config["output"] = {"directory": output_dir}
    
    # ==================== Write Config ====================
    print("\n" + "=" * 60)
    print("  SAVING CONFIGURATION")
    print("=" * 60)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Configuration saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Download DEM data: ./download-dem.sh --config {output_path} --topo-dir ./topo")
    print(f"  2. Generate terrain:  ./run.sh --config {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

