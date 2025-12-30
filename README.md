# Topo-Tessellate

Generate 3D-printable terrain models from real-world elevation and building data.

- **Terrain**: Create meshes from SRTM or high-resolution GeoTIFF elevation data
- **Buildings**: Extract 3D building footprints from OpenStreetMap
- **Tiling**: Split large regions into interlocking tiles for printing
- **Adaptive Meshing**: Intelligent simplification for optimal file sizes

## Quick Start

```bash
# 1. Download the helper scripts
curl -fsSL https://github.com/rdudhagra/topo-tessellate/releases/latest/download/download-dem.sh -o download-dem.sh
curl -fsSL https://github.com/rdudhagra/topo-tessellate/releases/latest/download/create-config.sh -o create-config.sh
chmod +x download-dem.sh create-config.sh

# 2. Create a config file (interactive wizard)
./create-config.sh configs/my_terrain.yaml

# 3. Download elevation data for your config
./download-dem.sh --config configs/my_terrain.yaml --topo-dir ./topo

# 4. Generate the terrain model
docker run --rm \
    -v "$PWD/configs:/app/configs:ro" \
    -v "$PWD/topo:/app/topo:ro" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/rdudhagra/topo-tessellate:latest \
    --config configs/my_terrain.yaml
```

## License

MIT

## Acknowledgments

- SRTM data provided by NASA/USGS
- Building data from OpenStreetMap contributors
- [MeshLib](https://github.com/MeshInspector/MeshLib) for mesh processing
- [Rasterio](https://rasterio.readthedocs.io/) for geospatial raster I/O
