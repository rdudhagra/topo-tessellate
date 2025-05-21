# World to Model - Terrain Generator

A Python application to generate 3D CAD models of terrain from SRTM elevation data.

## Overview

This tool creates high-quality 3D models of terrain for various purposes including:

- CAD design and engineering projects
- Digital fabrication (3D printing)
- Visualization and presentations
- GIS and mapping applications

The application uses SRTM (Shuttle Radar Topography Mission) data to generate accurate 3D terrain models with proper elevation, including water bodies.

## New CadQuery Support

The latest version uses CadQuery, a powerful parametric CAD scripting library, to create high-quality 3D models. This offers advantages over the previous mesh-based approach:

- Higher quality CAD models with proper solid geometry
- Support for STEP, STL, and 3MF formats
- Better integration with CAD software
- Cleaner, more parametric terrain representations
- Improved water body representation

## Installation

1. Clone this repository

```bash
git clone https://github.com/yourusername/world-to-model.git
cd world-to-model
```

2. Create a conda environment with required dependencies

```bash
conda env create -f environment.yml
conda activate world-to-model
```

3. Download SRTM data for your region of interest. Place the SRTM data files (.hgt.zip) in a directory named `topo` in the project root.

## Usage

Run the script with your desired region's coordinates:

```bash
python main.py --min-lon -122.673340 --min-lat 37.225955 --max-lon -121.753235 --max-lat 38.184228 --output-prefix bay_area
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--min-lon` | Minimum longitude of the region | Required |
| `--min-lat` | Minimum latitude of the region | Required |
| `--max-lon` | Maximum longitude of the region | Required |
| `--max-lat` | Maximum latitude of the region | Required |
| `--topo-dir` | Directory containing SRTM data | "topo" |
| `--detail-level` | Level of detail (0.01-1.0) | 0.2 |
| `--output-prefix` | Prefix for output files | "terrain" |
| `--water-level` | Elevation value for water areas | 0.0 |
| `--height-scale` | Scale factor for height | 0.05 |
| `--export-format` | Format to export (step, stl, 3mf) | "step" |

## Example

Generate a 3D model of the San Francisco Bay Area:

```bash
python main.py --min-lon -122.673340 --min-lat 37.225955 --max-lon -121.753235 --max-lat 38.184228 --water-level 0 --detail-level 0.2 --output-prefix sf_bay_area --export-format step
```

This will create a STEP file named `sf_bay_area.step` with the terrain model of the San Francisco Bay Area.

## Tips for Better Results

- Use an appropriate `detail-level` (lower values create simpler models but process faster)
- Set the `water-level` parameter to match the water elevation in your region
- Choose the right export format:
  - STEP for CAD software
  - STL for 3D printing
  - 3MF for modern 3D printing workflows

## License

MIT

## Acknowledgments

- SRTM data provided by NASA/USGS
- CadQuery for CAD modeling capabilities
- PyProj for coordinate transformations
