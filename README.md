# World to Model - Terrain Generator

A Python application to generate 3D CAD models of terrain from SRTM elevation data and extract 3D-ready building data from OpenStreetMap.

## Overview

This tool creates high-quality 3D models of terrain and extracts 3D-ready building data for various purposes including:

- CAD design and engineering projects
- Digital fabrication (3D printing)
- Visualization and presentations
- GIS and mapping applications
- Urban planning and analysis
- 3D city modeling and visualization

The application uses SRTM (Shuttle Radar Topography Mission) data to generate accurate 3D terrain models with proper elevation, including water bodies. Additionally, it extracts building and structure data from OpenStreetMap that is specifically optimized for 3D modeling applications.

## Features

### Terrain Generation
- High-quality CAD models from SRTM elevation data
- Support for STEP, STL, and 3MF formats
- Water body representation
- Parametric terrain modeling with CadQuery

### 3D-Ready Building Data Extraction (`buildings.py`)
- **3D-Focused**: Only extracts buildings with BOTH polygon coordinates AND height data
- **Quality Filtering**: Focuses on major buildings/structures (filters out small items like lamps, trees, etc.)
- **Fast Caching**: Local caching with compression (pickle + gzip) 
- **Comprehensive Analytics**: Building statistics, areas, heights, and types
- **Guaranteed 3D Data**: Every building has polygon geometry and height information
- **Cache Management**: Configurable expiration and cache control

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

### Terrain Generation

Run the script with your desired region's coordinates:

```bash
python main.py --min-lon -122.673340 --min-lat 37.225955 --max-lon -121.753235 --max-lat 38.184228 --output-prefix bay_area
```

### 3D-Ready Building Data Extraction

Use the `buildings.py` module to extract 3D-ready building data:

```python
from buildings import BuildingsExtractor

# Create extractor with caching enabled
extractor = BuildingsExtractor(use_cache=True, cache_max_age_days=30)

# Extract buildings for a bounding box (min_lon, min_lat, max_lon, max_lat)
bounds = (-122.42, 37.77, -122.38, 37.80)  # Downtown San Francisco
buildings = extractor.extract_buildings(bounds)

# Print statistics
extractor.print_stats()

# Access individual building data (all buildings guaranteed to have polygon + height)
for building in buildings[:5]:
    print(f"Building {building.osm_id}: {building.building_type}")
    print(f"  Area: {building.area:.0f} m²")  # Always available
    print(f"  Height: {building.height:.1f} m")  # Always available
    print(f"  Coordinates: {len(building.coordinates)} points")  # Always available
```

#### Cache Management

The buildings extractor includes efficient caching:

```python
# Cache is automatically used by default
extractor = BuildingsExtractor(use_cache=True)

# Force refresh (ignore cache)
buildings = extractor.extract_buildings(bounds, force_refresh=True)

# Clear cache for specific bounds
extractor.clear_cache(bounds)

# Clear all cache
extractor.clear_cache()
```

**3D-Ready Cache Performance:**
- **Quality**: Only buildings with both polygon geometry AND height data
- **Efficiency**: 5,556 3D-ready buildings vs 7,771 total buildings (71% have height data)
- **File Size**: ~930KB for 5,556 3D-ready buildings (smaller, higher quality dataset)
- **Loading Speed**: Lightning-fast cache loading for repeated queries
- **Storage**: Compressed pickle files with gzip, named with `3d_` prefix

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

## Examples

### Terrain Generation
Generate a 3D model of the San Francisco Bay Area:

```bash
python main.py --min-lon -122.673340 --min-lat 37.225955 --max-lon -121.753235 --max-lat 38.184228 --water-level 0 --detail-level 0.2 --output-prefix sf_bay_area --export-format step
```

### 3D-Ready Building Extraction
Extract 3D-ready buildings for downtown Oakland:

```python
from buildings import BuildingsExtractor

# Downtown Oakland bounds
bounds = (-122.28, 37.79, -122.25, 37.82)

extractor = BuildingsExtractor()
buildings = extractor.extract_buildings(bounds)

# All buildings are guaranteed to have both geometry and height
print(f"Extracted {len(buildings)} 3D-ready buildings")

# Filter by building type
hotels = [b for b in buildings if 'hotel' in b.building_type.lower()]
offices = [b for b in buildings if 'office' in b.building_type.lower()]

print(f"Found {len(hotels)} hotels and {len(offices)} office buildings")

# Find tallest buildings (all have height data)
tallest = sorted(buildings, key=lambda b: b.height, reverse=True)[:10]
for i, building in enumerate(tallest):
    print(f"{i+1}. {building.height:.1f}m - {building.building_type}")
```

## 3D Modeling Benefits

The 3D-focused building extraction provides several advantages:

- **Guaranteed Data Quality**: Every building has both polygon geometry and height information
- **3D Modeling Ready**: No need to filter or validate data before 3D modeling
- **Smaller Datasets**: More efficient storage and processing (71% of buildings have height data)
- **Better Performance**: Focused queries reduce API load and processing time
- **Reliable Heights**: Uses explicit height tags or estimates from building levels

## Tips for Better Results

- Use an appropriate `detail-level` (lower values create simpler models but process faster)
- Set the `water-level` parameter to match the water elevation in your region
- Choose the right export format:
  - STEP for CAD software
  - STL for 3D printing
  - 3MF for modern 3D printing workflows
- For building extraction, use smaller bounding boxes for faster processing
- All extracted buildings are 3D-ready with guaranteed polygon and height data
- Cache is automatically managed, but you can clear it if you need fresh data

## License

MIT

## Acknowledgments

- SRTM data provided by NASA/USGS
- Building data from OpenStreetMap contributors
- CadQuery for CAD modeling capabilities
- PyProj for coordinate transformations
