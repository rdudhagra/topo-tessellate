# Topo-Tessellate - 3D Terrain Generator from REAL elevation+buildings data

Generate high-quality 3D terrain models from SRTM/GeoTIFF elevation data with optional building extraction from OpenStreetMap.

## Features

- **Terrain Generation**: Create 3D meshes from SRTM or GeoTIFF elevation data
- **Building Extraction**: Pull 3D-ready building data from OpenStreetMap
- **Adaptive Meshing**: Intelligent mesh simplification for optimal file sizes
- **Tiling Support**: Split large regions into printable tiles with interlocking joints
- **Multiple Formats**: Export to STL for 3D printing
- **Caching**: Local caching for faster repeat runs
- **Beautiful Console Output**: Rich terminal output with progress bars and formatting

## Quick Start

### Using Docker (Recommended)

The easiest way to run the generator is using the provided Docker container:

```bash
# Pull and run with the helper script
./run.sh --config configs/sf_fidi.yaml

# Or run directly with Docker
docker run --rm \
    -v "$PWD/configs:/app/configs" \
    -v "$PWD/topo:/app/topo" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/rdudhagra/topo-tessellate:latest \
    --config configs/sf_fidi.yaml
```

### Native Installation (Conda)

The project requires conda due to complex dependencies (meshlib, GDAL, rasterio):

```bash
# Clone the repository
git clone https://github.com/rdudhagra/topo-tessellate.git
cd topo-tessellate

# Create conda environment
conda env create -f environment.yml
conda activate topo-tessellate

# Run the generator
python generate.py --config configs/sf_fidi.yaml
```

## Configuration

The generator is driven by YAML configuration files. See `configs/template.yaml` for a fully documented example.

### Basic Usage

```bash
# Run with a configuration file
python generate.py --config configs/sf_fidi.yaml

# Run a specific job from a multi-job config
python generate.py --config configs/bay_area.yaml --job sf_downtown

# Override output directory
python generate.py --config configs/sf_fidi.yaml --outdir ./my_outputs
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML configuration file (required) |
| `--job` | Run only the named job (matches `name` or `output_prefix`) |
| `--outdir` | Override output directory for all jobs |

### Configuration File Structure

```yaml
version: 1
name: my_terrain
output_prefix: my_terrain

bounds: [-122.42, 37.77, -122.38, 37.80]  # [min_lon, min_lat, max_lon, max_lat]

elevation_source:
  type: srtm  # or "geotiff"
  topo_dir: topo

terrain:
  elevation_multiplier: 1.0
  water_threshold: 0
  adaptive_tolerance_z: 1.0

buildings:
  enabled: true
  timeout: 120

output:
  directory: outputs/my_terrain
```

See `configs/template.yaml` for all available options.

## Elevation Data

### SRTM Data

Place SRTM tiles (`.hgt.zip` files) in the `topo/` directory. Files follow the naming convention `N37W122.SRTMGL3.hgt.zip`.

Download SRTM data from:
- [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
- [OpenTopography](https://opentopography.org/)

### GeoTIFF Data

For higher resolution, use GeoTIFF files. Configure in your YAML:

```yaml
elevation_source:
  type: geotiff
  topo_dir: topo
  glob: "USGS_*.tif"  # or specify individual files
```

## Output Files

The generator creates STL files in the output directory:

| File | Description |
|------|-------------|
| `{prefix}_land.stl` | Terrain mesh (above water level) |
| `{prefix}_base.stl` | Base/platform mesh with joints |
| `{prefix}_buildings.stl` | Building footprints (if enabled) |

## Building Extraction

Extract 3D-ready building data from OpenStreetMap:

```python
from terrain_generator.buildingsextractor import BuildingsExtractor

# Create extractor with caching
extractor = BuildingsExtractor(use_cache=True, cache_max_age_days=30)

# Extract buildings for a bounding box
bounds = (-122.42, 37.77, -122.38, 37.80)  # Downtown San Francisco
buildings = extractor.extract_buildings(bounds)

# Print statistics
extractor.print_stats()

# Access building data
for building in buildings[:5]:
    print(f"Building {building.osm_id}: {building.building_type}")
    print(f"  Area: {building.area:.0f} m²")
    print(f"  Height: {building.height:.1f} m")
```

## Docker Container

### Building Locally

```bash
docker build -t topo-tessellate .
```

### Running the Container

```bash
# Using the helper script (recommended)
./run.sh --config configs/sf_fidi.yaml

# Manual Docker run
docker run --rm \
    -v "$PWD/configs:/app/configs:ro" \
    -v "$PWD/topo:/app/topo:ro" \
    -v "$PWD/outputs:/app/outputs" \
    topo-tessellate --config configs/sf_fidi.yaml
```

### Volume Mounts

| Mount | Purpose |
|-------|---------|
| `/app/configs` | Configuration files |
| `/app/topo` | Elevation data (SRTM/GeoTIFF) |
| `/app/outputs` | Generated STL files |

## Development

### Running Tests

Tests run inside the Docker container using pytest:

```bash
# Run all tests (builds image if needed)
./test.sh

# Run with verbose output
./test.sh -v

# Run specific test file
./test.sh tests/test_buildings_processor.py

# Run tests matching a pattern
./test.sh -k 'bbox'

# Force rebuild of test image
./test.sh --rebuild
```

Or run tests natively with conda:

```bash
conda activate topo-tessellate
pytest -v
```

### Project Structure

```
topo-tessellate/
├── generate.py              # Main entry point
├── run.sh                   # Docker runner script
├── test.sh                  # Test runner script
├── Dockerfile               # Container definition
├── terrain_generator/       # Core library
│   ├── buildingsextractor.py
│   ├── buildingsgenerator.py
│   ├── buildingsprocessor.py
│   ├── basegenerator.py
│   ├── modelgenerator.py
│   ├── elevation.py
│   ├── srtm.py
│   ├── geotiff.py
│   └── console.py
├── tests/                   # Pytest test suite
│   ├── conftest.py          # Shared fixtures
│   └── test_buildings_processor.py
├── configs/                 # Example configurations
│   ├── template.yaml
│   ├── bay_area.yaml
│   ├── sf_fidi.yaml
│   └── sf_full.yaml
├── scripts/                 # Utility scripts
├── topo/                    # Elevation data (not in repo)
└── outputs/                 # Generated files (not in repo)
```

## License

MIT

## Acknowledgments

- SRTM data provided by NASA/USGS
- Building data from OpenStreetMap contributors
- [MeshLib](https://github.com/MeshInspector/MeshLib) for mesh processing
- [Rasterio](https://rasterio.readthedocs.io/) for geospatial raster I/O
