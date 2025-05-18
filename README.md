# Terrain Model Generator

A Python tool for generating 3D terrain models from SRTM elevation data. The tool reads .hgt files, processes the elevation data within specified coordinate bounds, and exports the terrain as a 3D model file (GLB or OBJ).

## Features

- Read and merge multiple SRTM elevation data files (.hgt)
- Process terrain within specified coordinate bounds
- Generate 3D terrain models with customizable detail levels
- Separate water and land objects in the scene
- Control water transparency, shore height, and other parameters
- Export to GLB or OBJ format with proper scaling and orientation
- Debug visualization options for land/sea masks and elevation maps
- Parallel processing for improved performance
- Optimized mesh quality at water-land boundaries

## Installation

1. Create a virtual environment:
```bash
conda create -f environment.yml
conda activate terrain-gen
```

## Usage

Basic usage:
```bash
python main.py --min-lon MIN_LON --min-lat MIN_LAT --max-lon MAX_LON --max-lat MAX_LAT [options]
```

Required Arguments:
- `--min-lon`: Minimum longitude of the region
- `--min-lat`: Minimum latitude of the region
- `--max-lon`: Maximum longitude of the region
- `--max-lat`: Maximum latitude of the region

Optional Arguments:
- `--topo-dir`: Directory containing SRTM data (default: 'topo')
- `--detail-level`: Level of detail, 0.01-1.0 (default: 0.2)
- `--output-prefix`: Prefix for output files (default: 'terrain')
- `--water-level`: Elevation value for water areas (default: -15.0)
- `--shore-height`: Elevation value for shore areas (default: 1.0)
- `--shore-buffer`: Number of cells for shore buffer (default: 1)
- `--height-scale`: Scale factor for height (default: 0.05)
- `--debug`: Generate debug visualizations as JPG files
- `--export-format`: Format to export the model, 'glb' or 'obj' (default: 'glb')
- `--water-alpha`: Water transparency level, 0-255 (default: 255)

Example for San Francisco Bay Area:
```bash
python main.py --min-lon -122.5 --max-lon -122.4 --min-lat 37.7 --max-lat 37.8 --detail-level 0.3 --water-level -15.0 --output-prefix sf_bay
```

## Verifying Mesh Quality

Use the included verification script to analyze mesh quality, particularly at water-land boundaries:

```bash
python verify_mesh.py your_model.glb
```

The script will provide detailed analysis and generate visualizations of the boundary areas.

## Dependencies

- rasterio/GDAL for elevation data processing
- NumPy for numerical operations
- Trimesh for 3D mesh generation
- PyProj for coordinate transformations
- SciPy for image processing and interpolation
- Matplotlib for debug visualizations
- tqdm for progress bars
- concurrent.futures for parallel processing
