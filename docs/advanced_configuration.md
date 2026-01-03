# Advanced Configuration

This guide explains how to customize your 3D terrain models using the YAML configuration file. Each setting controls some aspect of how your model is generated—from the area covered to how tall the buildings appear. You don't need to understand every parameter; start with the important ones and experiment from there.

> **Tip:** For a fully annotated example with all available options, see [`configs/template.yaml`](../configs/template.yaml).

---

## Important Parameters

These are the settings you'll most likely want to adjust. They have the biggest impact on your final model.

### `bounds` — The Area You Want to Model

This is the most important setting. It defines the geographic rectangle that will become your 3D model.

```yaml
bounds: [-122.42846, 37.77723, -122.37645, 37.81792]
```

The four numbers represent:
1. **West edge** (minimum longitude) — negative numbers for locations west of the Prime Meridian
2. **South edge** (minimum latitude) — the bottom of your rectangle
3. **East edge** (maximum longitude)
4. **North edge** (maximum latitude)

**How to find coordinates:** [bboxfinder.com](http://bboxfinder.com/) is a great tool to generate these coordinates.

**When to adjust:**
- Start with a small area (a few city blocks) to test your settings before scaling up
- Larger areas take longer to process and require more memory
- A few square kilometers is usually a good starting size

---

### `terrain.elevation_multiplier` — How Exaggerated the Hills Look

This controls how tall the hills and valleys appear relative to reality.

```yaml
terrain:
  elevation_multiplier: 3.5
```

| Value | Effect |
|-------|--------|
| `1.0` | Real-world proportions (hills may look flat on small models) |
| `2.0–3.0` | Moderate exaggeration (good starting point for hilly areas) |
| `4.0+` | Dramatic, stylized terrain |

**When to adjust:**
- **Increase** if your area has subtle terrain that looks flat on the model
- **Decrease** if mountains look unnaturally steep or pointy
- For flat areas like coastal cities, a higher multiplier (3.5–5.0) helps show what little elevation change exists

---

### `terrain.downsample_factor` — Resolution vs. Speed Tradeoff

This reduces the resolution of your terrain data. Higher values mean faster processing and smaller files, but less detail.

```yaml
terrain:
  downsample_factor: 5
```

| Value | Effect |
|-------|--------|
| `1` | Full resolution (slowest, most detailed) |
| `5` | 5× reduction in each direction (a good balance) |
| `10+` | Fast processing, but may lose fine terrain details |

**When to adjust:**
- **Start high** (e.g., `10`) for test runs to see results quickly
- **Decrease** to `1`–`5` for your final model when you want maximum detail
- Large areas may require higher values to avoid running out of memory
- Denser elevation data (USGS 1M) may require higher values to avoid running out of memory

---

### `buildings.enabled` — Include Buildings or Not

Toggles whether buildings are extracted from OpenStreetMap and added to your model.

```yaml
buildings:
  enabled: true
```

**When to set to `false`:**
- You only want the terrain/landscape
- You want faster processing (building extraction adds time)

---

### `buildings.generate.building_height_multiplier` — How Tall Buildings Appear

Similar to the terrain elevation multiplier, this scales how tall buildings appear.

```yaml
buildings:
  generate:
    building_height_multiplier: 3.5
```

**When to adjust:**
- This should typically match your `terrain.elevation_multiplier` so buildings and terrain are scaled consistently. For larger areas, increase this value to make buildings more prominent.
- **Increase** if buildings look too short compared to the terrain
- **Decrease** if skyscrapers look out of proportion

---

### `buildings.extract.max_building_distance_meters` — Building Merging

Controls whether nearby buildings are merged together into combined shapes. Buildings that are within this distance of each other get fused into a single polygon.

```yaml
buildings:
  extract:
    max_building_distance_meters: 30
```

**How it works:** Each building footprint is expanded outward by this distance, then all overlapping shapes are combined, and finally the result is shrunk back. This effectively bridges gaps smaller than the specified distance. Significant buildings (large or tall) are kept separate and not merged.

| Value | Effect |
|-------|--------|
| `0` | Disabled—every building remains a separate shape |
| `5-10` | Merges tightly-packed buildings (row houses, attached units) |
| `10–50` | Merges buildings within a city block that have narrow gaps between them |

**When to adjust:**
- **Set to `0`** for detailed models where you want every building as its own distinct shape
- **Set to typical road width** (e.g., `15–25` meters) if you want buildings on the same block to merge, but buildings across streets to stay separate
- **Set to building spacing** within blocks (e.g., `5–10` meters) to only merge truly adjacent structures
- Merging reduces the total number of shapes, which makes files smaller and printing easier for large areas

**Visual impact:** At `0`, you'll see distinct building footprints. With merging enabled, adjacent buildings blend into continuous masses—useful for large-scale regional models where individual small buildings would be too tiny to print anyway.

---

### `global.scale_max_length_mm` — Physical Size of Your Model

Controls how large the final 3D model will be (in millimeters).

```yaml
global:
  scale_max_length_mm: 200.0
```

The longest side of your model (either width or height) will be scaled to this length. The other dimension is scaled proportionally.

**When to adjust:**
- Set to match your 3D printer's build plate size
- A 200mm model is a good default for most printers
- For larger prints or multi-tile models, you may want smaller individual tiles

---

## Detailed Parameter Reference

### Basic Settings

| Parameter | Description |
|-----------|-------------|
| `version` | Always set to `1`. Required for the config to work. |
| `name` | A friendly name for your model (e.g., `sf_downtown`). Used in logging. |
| `output_prefix` | Prefix for output filenames. If set to `sf_fidi`, your files will be named `sf_fidi_land.stl`, `sf_fidi_buildings.stl`, etc. |

---

### Elevation Source Settings

These control where the terrain elevation data comes from.

```yaml
elevation_source:
  type: geotiff
  topo_dir: topo
  glob: "USGS_1M_10_*.tif"
```

| Parameter | Description |
|-----------|-------------|
| `type` | The format of your elevation files. Use `geotiff` for USGS DEM files (`.tif`) or `srtm` for SRTM tiles (`.hgt`). |
| `topo_dir` | The folder containing your elevation data files. |
| `glob` | A pattern to match specific files (e.g., `"*.tif"` matches all TIF files). |
| `file` | Alternatively, specify a single file by name. |
| `files` | Or provide a list of specific files. |

**Tips:**
- GeoTIFF files from USGS typically provide higher resolution than SRTM
- You can download SRTM tiles from [dwtkns.com/srtm30m/](https://dwtkns.com/srtm30m/)
- You can download USGS `.tif` files from [apps.nationalmap.gov/downloader/](https://apps.nationalmap.gov/downloader/), or by using the convenience script `download-dem.sh`
- The system automatically finds and stitches together whichever files overlap your bounds
- Make sure you have elevation data that covers your chosen `bounds`

---

### Terrain Settings

These fine-tune how the terrain mesh is generated.

```yaml
terrain:
  base_height: 250
  elevation_multiplier: 3.5
  downsample_factor: 1
  water_threshold: 1
  split_at_water_level: true
```

| Parameter | Description | When to Adjust |
|-----------|-------------|----------------|
| `base_height` | Height (in meters) of the solid base under your terrain. | Increase if your model has deep valleys that cut through the base. |
| `elevation_multiplier` | Vertical exaggeration factor. See above. | Adjust to make hills more or less dramatic. |
| `downsample_factor` | Resolution reduction. See above. | Increase for faster test runs; decrease for final quality. |
| `water_threshold` | Elevation (in meters) below which is considered water. | The land mesh will be split at this elevation and translated so the water plane is at z=0. |
| `adaptive_tolerance_z` | Controls mesh simplification for flat areas. Higher = fewer triangles. | Increase if your files are too large; decrease for more terrain detail. |

---

### Building Settings

These control how buildings are extracted and rendered.

```yaml
buildings:
  enabled: true
  timeout: 120
  extract:
    max_building_distance_meters: 30
  generate:
    building_height_multiplier: 1.0
    min_building_height: 10.0
```

| Parameter | Description | When to Adjust |
|-----------|-------------|----------------|
| `enabled` | Master switch for buildings. | Set `false` to skip building extraction entirely. |
| `timeout` | Seconds to wait for OpenStreetMap data. | Increase if you get timeout errors on large areas. |
| `extract.max_building_distance_meters` | Merges buildings within this distance into clusters. | Set to `0` to keep buildings separate. Increase (e.g., `30`) if you want simplified building groups. |
| `generate.building_height_multiplier` | Scales building heights. | Match to your terrain's `elevation_multiplier` for consistency. |
| `generate.min_building_height` | Minimum height for any building (in meters). | Increase if small buildings are too thin to print. |

**Tips:**
- Building extraction uses OpenStreetMap data, so coverage varies by location
- Rural areas may have incomplete building data
- Dense urban areas may take longer to process

---

### Tiling Settings

For large areas, you can split your model into a grid of tiles that can be printed separately and assembled.

```yaml
tiling:
  rows: 3
  cols: 3
```

| Parameter | Description |
|-----------|-------------|
| `rows` | Number of tiles vertically (north-south). |
| `cols` | Number of tiles horizontally (east-west). |

**When to use tiling:**
- Your model is too large for a single print
- You want to print different sections in different colors
- A 3×3 grid gives you 9 separate pieces to assemble

**Note:** Each tile will include registration features (cleats and joints) for alignment during assembly. See the [Assembly Guide](assembly.md) for instructions.

---

### Base Settings

Controls the solid base under your terrain.

```yaml
base:
  height: 20.0
```

| Parameter | Description |
|-----------|-------------|
| `height` | Thickness of the base in millimeters. |

**When to adjust:**
- Use at least `20.0` mm for tiled models (provides enough material for assembly hardware)
- If you want the base to be thinner, be sure to disable the cleat cutout

---

### Output Settings

Controls where and how files are saved.

```yaml
output:
  directory: ./outputs/sf_fidi
```

| Parameter | Description |
|-----------|-------------|
| `directory` | Folder where output files are saved. Created automatically. |

---

## Troubleshooting

### "My terrain looks completely flat"
- Increase `terrain.elevation_multiplier` (try 3.0–5.0)
- Check that your elevation data files actually cover your `bounds`

### "Processing takes forever / runs out of memory"
- Increase `terrain.downsample_factor` (try 5 or 10)
- Reduce the size of your `bounds`
- Try disabling buildings temporarily

### "Buildings look too short/tall compared to terrain"
- Make sure `buildings.generate.building_height_multiplier` matches `terrain.elevation_multiplier`

### "My STL file is too large to open/print"
- Increase `terrain.downsample_factor`
- Increase `terrain.adaptive_tolerance_z` for more aggressive mesh simplification

---

## Full Template Reference

For a complete list of every available parameter with technical comments, see:

📄 **[`configs/template.yaml`](../configs/template.yaml)**

This file includes all options with their default values and detailed explanations.
