# Topo-Tessellate

Generate 3D-printable terrain models from real-world elevation and building data.

- **Terrain**: Create meshes from SRTM or high-resolution GeoTIFF elevation data
- **Buildings**: Extract 3D building footprints from OpenStreetMap
- **Tiling**: Split large regions into interlocking tiles for printing
- **Adaptive Meshing**: Intelligent simplification/building merging for very large regions

![](images/IMG_1982.jpeg)
<table>
  <tr>
    <td><img src="images/IMG_2691.jpeg" alt="Terrain 1" width="400"></td>
    <td><img src="images/IMG_2679.jpeg" alt="Terrain 2" width="400"></td>
  </tr>
  <tr>
    <td><img src="images/IMG_2692.jpeg" alt="Terrain 3" width="400"></td>
    <td><img src="images/IMG_1717.jpeg" alt="Terrain 4" width="400"></td>
  </tr>
</table>

## [Get Started](docs/get_started.md)

## License

GPL-3.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Elevation data provided by [USGS National Map](https://apps.nationalmap.gov/downloader/)
- Building data from [OpenStreetMap](https://www.openstreetmap.org/) contributors
- [MeshLib](https://github.com/MeshInspector/MeshLib) for mesh processing
- [Rasterio](https://rasterio.readthedocs.io/) for geospatial raster I/O
