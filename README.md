# Terrain Model Generator

A Python tool for generating 3D terrain models from SRTM elevation data. The tool reads .hgt files, processes the elevation data within specified coordinate bounds, and exports the terrain as a .glb 3D model file.

## Features

- Read SRTM elevation data (.hgt files)
- Process terrain within specified coordinate bounds
- Generate 3D terrain models with customizable detail levels
- Export to .glb format with:
  - Models centered at (0,0)
  - Flat orientation
  - 1-meter length scaling

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python main.py input.hgt output.glb --bounds MIN_LON MIN_LAT MAX_LON MAX_LAT [--detail DETAIL_LEVEL]
```

Arguments:
- `input.hgt`: Path to input SRTM .hgt file
- `output.glb`: Path for output .glb file
- `--bounds`: Bounding box coordinates (min_lon min_lat max_lon max_lat)
- `--detail`: Detail level (0.1 to 1.0, default: 1.0)

Example:
```bash
python main.py N37W123.hgt terrain.glb --bounds -122.5 37.7 -122.4 37.8 --detail 0.8
```

## Dependencies

- GDAL/rasterio for elevation data processing
- NumPy for numerical operations
- Trimesh for 3D mesh generation
- PyProj for coordinate transformations
