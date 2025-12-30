# Topo-Tessellate

Generate 3D-printable terrain models from real-world elevation and building data.

- **Terrain**: Create meshes from SRTM or high-resolution GeoTIFF elevation data
- **Buildings**: Extract 3D building footprints from OpenStreetMap
- **Tiling**: Split large regions into interlocking tiles for printing
- **Adaptive Meshing**: Intelligent simplification for optimal file sizes

## Quick Start

```bash
# 1. Create a config file (interactive wizard)
bash <(curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/create-config.sh) configs/my_terrain.yaml

# 2. Download elevation data for your config
bash <(curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/download-dem.sh) --config configs/my_terrain.yaml --topo-dir ./topo

# 3. Generate the terrain model
docker run --rm \
    -v "$PWD/configs:/app/configs:ro" \
    -v "$PWD/topo:/app/topo:ro" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/rdudhagra/topo-tessellate:latest \
    --config configs/my_terrain.yaml
```

Or download the scripts locally:
```bash
curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/create-config.sh -o create-config.sh
curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/download-dem.sh -o download-dem.sh
chmod +x create-config.sh download-dem.sh
```

## License

MIT

## Acknowledgments

- SRTM data provided by NASA/USGS
- Building data from OpenStreetMap contributors
- [MeshLib](https://github.com/MeshInspector/MeshLib) for mesh processing
- [Rasterio](https://rasterio.readthedocs.io/) for geospatial raster I/O
