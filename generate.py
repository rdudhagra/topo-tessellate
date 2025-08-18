#!/usr/bin/env python3

"""
Unified terrain + buildings generator driven by a YAML configuration.

Usage:
  python generate.py --config configs/test_srtm.yaml
  python generate.py --config configs/dc.yaml --job dc

Config schema (high level):
  - version: 1
  - jobs: [ { name, output_prefix, bounds, elevation_source, terrain, buildings, output } ]
    or a single job object with the same fields at the root.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import yaml  # type: ignore

import meshlib.mrmeshnumpy as mn  # type: ignore
import meshlib.mrmeshpy as mr  # type: ignore

from terrain_generator.console import output
from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.srtm import SRTM
from terrain_generator.geotiff import GeoTiff
from terrain_generator.buildingsextractor import BuildingsExtractor
from terrain_generator.buildingsgenerator import BuildingsGenerator
from terrain_generator.basegenerator import BaseGenerator


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _as_jobs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Support either top-level jobs list or a single job object at root
    if isinstance(config.get("jobs"), list):
        return config["jobs"]
    else:
        # treat the whole doc as a single job definition
        return [config]


def _validate_bounds(bounds: Any) -> Tuple[float, float, float, float]:
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
        raise ValueError("bounds must be a list/tuple of 4 numbers: [min_lon, min_lat, max_lon, max_lat]")
    try:
        return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    except Exception as exc:
        raise ValueError(f"Invalid bounds values: {bounds}") from exc


def _build_elevation_source(cfg: Dict[str, Any]) -> Tuple[Any, str]:
    es = cfg.get("elevation_source", {}) or {}
    src_type = str(es.get("type", "srtm")).lower()
    topo_dir = str(es.get("topo_dir", "topo"))

    if src_type == "geotiff":
        # Support single file, multiple files, or glob auto-discovery
        file_name = es.get("file")
        file_names = es.get("files") or []
        if isinstance(file_names, str):
            file_names = [file_names]
        if file_name:
            file_names = list(file_names) + [str(file_name)]
        glob_pattern = es.get("glob")
        elevation = GeoTiff(file_names or None, glob_pattern)
    elif src_type == "srtm":
        elevation = SRTM()
    else:
        raise ValueError(f"Unknown elevation_source.type: {src_type}")

    return elevation, topo_dir


def _terrain_defaults() -> Dict[str, Any]:
    return {
        "elevation_multiplier": 1.0,
        "downsample_factor": 1,
        "water_threshold": None,
        "force_refresh": False,
        # Adaptive mesh controls (Z in meters)
        "adaptive_tolerance_z": 1.0,            # controls simplification aggressiveness
        "adaptive_max_gap_fraction": 1/256,      # max gap vs full resolution to avoid huge faces
        "adaptive_max_sampled_rows": 400,        # cap rows sampled for RDP pre-selection
        "adaptive_max_sampled_cols": 400,        # cap cols sampled for RDP pre-selection
        "split_at_water_level": True,
        # ModelGenerator ctor
        "use_cache": True,
        "cache_max_age_days": 30,
    }


def _buildings_defaults() -> Dict[str, Any]:
    return {
        "enabled": False,
        # extractor options
        "timeout": 120,
        "use_cache": True,
        "cache_max_age_days": 30,
        "extract": {
            "force_refresh": False,
            "max_building_distance_meters": 35,
        },
        # generator options
        "generate": {
            "building_height_multiplier": 1.0,
            "min_building_height": 10.0,
        },
    }


def _global_defaults() -> Dict[str, Any]:
    return {
        "scale_max_length_mm": 200.0,
    }


def _base_defaults() -> Dict[str, Any]:
    return {
        "height": 20.0,
    }


def _tiling_defaults() -> Dict[str, Any]:
    return {
        "enabled": False,
        "rows": 1,
        "cols": 1,
    }


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _generate_base_for_tile(
    generator: ModelGenerator,
    bounds: Tuple[float, float, float, float],
    overall_bounds: Tuple[float, float, float, float],
    tile_row: int,
    tile_col: int,
    total_rows: int,
    total_cols: int,
    scale_max_length_mm: float,
    base_cfg: Dict[str, Any],
) -> Optional[Any]:
    """Generate a base mesh for a specific tile with appropriate cutouts.
    
    Args:
        generator: ModelGenerator instance
        bounds: Tile bounds (min_lon, min_lat, max_lon, max_lat)
        overall_bounds: Overall bounds for consistent scaling
        tile_row: Current tile row (0-based)
        tile_col: Current tile column (0-based)
        total_rows: Total number of rows
        total_cols: Total number of columns
        scale_max_length_mm: Maximum length for scaling each tile
        
    Returns:
        Base mesh or None if generation fails
    """
    try:
        # Calculate dimensions based on overall bounds for consistent scaling
        overall_width_m, overall_height_m = generator.elevation.calculate_bounds_dimensions_meters(overall_bounds)
        
        # Each tile should have dimensions = overall_dimensions / tile_count
        tile_width_m = overall_width_m / total_cols
        tile_height_m = overall_height_m / total_rows
        
        # Scale to target size in mm (each tile should fit within scale_max_length_mm)
        target_len_m = float(scale_max_length_mm) / 1000.0
        denom = max(tile_width_m, tile_height_m) or 1.0
        scale_factor = float(target_len_m) / float(denom)
        
        # Convert to mm for base generation (consistent across all tiles)
        length_mm = tile_width_m * 1000.0 * scale_factor
        width_mm = tile_height_m * 1000.0 * scale_factor
        height_mm = float(base_cfg.get("height", 20.0))
        
        # Determine which sides need cutouts (only inner faces)
        cutout_left = tile_col > 0  # Not leftmost column
        cutout_right = tile_col < total_cols - 1  # Not rightmost column
        cutout_front = tile_row > 0  # Not top row
        cutout_back = tile_row < total_rows - 1  # Not bottom row
        
        # Generate base with appropriate cutouts
        base_gen = BaseGenerator()
        base_mesh = base_gen.generate_base_with_cutouts(
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            cutout_left=cutout_left,
            cutout_right=cutout_right,
            cutout_front=cutout_front,
            cutout_back=cutout_back,
        )
        
        return base_mesh
        
    except Exception as exc:
        output.warning(f"Failed to generate base for tile ({tile_row}, {tile_col}): {exc}")
        return None


def _save_outputs(
    generator: ModelGenerator,
    result: Dict[str, Any],
    buildings_mesh: Optional[Any],
    base_mesh: Optional[Any],
    out_dir: str,
    prefix: str,
    bounds: Tuple[float, float, float, float],
    overall_bounds: Tuple[float, float, float, float],
    scale_max_length_mm: float,
    tile_row: int = 0,
    tile_col: int = 0,
    total_rows: int = 1,
    total_cols: int = 1,
    output_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_dir(out_dir)
    land = result.get("land_mesh")
            
    # Apply vertical offset to both meshes
    base_bbox = base_mesh.computeBoundingBox()
    base_height = base_bbox.max.z - base_bbox.min.z
    offset_transform = mr.AffineXf3f(mr.Matrix3f(), mr.Vector3f(0, 0, base_height))
    
    if land is not None:
        land.transform(offset_transform)
    
    if buildings_mesh is not None:
        buildings_mesh.transform(offset_transform)
    
    # Save each mesh as separate STL files
    saved_files = []
    
    # Save base mesh
    if base_mesh is not None:
        try:
            base_path = os.path.join(out_dir, f"{prefix}_base.stl")
            mr.saveMesh(base_mesh, base_path)
            output.file_saved(base_path, "base mesh")
            saved_files.append("base")
        except Exception as exc:
            output.warning(f"Failed to save base mesh: {exc}")
    
    # Save land mesh
    if land is not None:
        try:
            land_path = os.path.join(out_dir, f"{prefix}_land.stl")
            mr.saveMesh(land, land_path)
            output.file_saved(land_path, "land mesh")
            saved_files.append("land")
        except Exception as exc:
            output.warning(f"Failed to save land mesh: {exc}")
    
    # Save buildings mesh
    if buildings_mesh is not None:
        try:
            buildings_path = os.path.join(out_dir, f"{prefix}_buildings.stl")
            mr.saveMesh(buildings_mesh, buildings_path)
            output.file_saved(buildings_path, "buildings mesh")
            saved_files.append("buildings")
        except Exception as exc:
            output.warning(f"Failed to save buildings mesh: {exc}")
    
    if saved_files:
        output.success(f"Saved {len(saved_files)} STL files: {', '.join(saved_files)}")


def run_job(job_cfg: Dict[str, Any], global_output_dir: Optional[str] = None, only_prefix: Optional[str] = None) -> None:
    # Basic required fields
    name = str(job_cfg.get("name") or job_cfg.get("output_prefix") or "job")
    prefix = str(job_cfg.get("output_prefix") or name)
    if only_prefix and (prefix != only_prefix and name != only_prefix):
        return

    bounds = _validate_bounds(job_cfg.get("bounds"))

    output.header(f"Generating: {name}", f"Bounds: {bounds}")

    # Build elevation
    elevation, topo_dir = _build_elevation_source(job_cfg)

    # Terrain configuration
    terrain_cfg = _merge(_terrain_defaults(), job_cfg.get("terrain", {}))

    # ModelGenerator with caching options
    generator = ModelGenerator(
        elevation,
        use_cache=bool(terrain_cfg.get("use_cache", True)),
        cache_max_age_days=int(terrain_cfg.get("cache_max_age_days", 30)),
    )

    # Global configuration
    global_cfg = _merge(_global_defaults(), job_cfg.get("global", {}))
    
    # Base configuration
    base_cfg = _merge(_base_defaults(), job_cfg.get("base", {}))
    
    # Tiling (separate section)
    tiling_cfg = _merge(_tiling_defaults(), job_cfg.get("tiling", {}))
    tile_enabled = bool(tiling_cfg.get("enabled", False))
    tile_rows = max(1, int(tiling_cfg.get("rows", 1)))
    tile_cols = max(1, int(tiling_cfg.get("cols", 1)))

    def _gen(bounds_tile: Tuple[float, float, float, float], max_length_mm: float) -> Dict[str, Any]:
        return generator.generate_terrain_model(
            bounds=bounds_tile,
            topo_dir=topo_dir,
            water_threshold=terrain_cfg["water_threshold"],
            elevation_multiplier=float(terrain_cfg["elevation_multiplier"]),
            downsample_factor=int(terrain_cfg["downsample_factor"]),
            force_refresh=bool(terrain_cfg["force_refresh"]),
            adaptive_tolerance_z=float(terrain_cfg.get("adaptive_tolerance_z", 1.0)),
            adaptive_max_gap_fraction=float(terrain_cfg.get("adaptive_max_gap_fraction", 1/256)),
            adaptive_max_sampled_rows=int(terrain_cfg.get("adaptive_max_sampled_rows", 400)),
            adaptive_max_sampled_cols=int(terrain_cfg.get("adaptive_max_sampled_cols", 400)),
            split_at_water_level=bool(terrain_cfg["split_at_water_level"]),
            max_length_mm=max_length_mm,
        )

    # Helper: per-tile buildings mesh generation
    def _gen_buildings(bounds_tile: Tuple[float, float, float, float], elev_data) -> Optional[Any]:
        buildings_cfg = _merge(_buildings_defaults(), job_cfg.get("buildings", {}))
        if not bool(buildings_cfg.get("enabled", False)):
            return None
        extractor = BuildingsExtractor(
            timeout=int(buildings_cfg.get("timeout", 120)),
            use_cache=bool(buildings_cfg.get("use_cache", True)),
            cache_max_age_days=int(buildings_cfg.get("cache_max_age_days", 30)),
        )
        buildings = extractor.extract_buildings(
            bounds_tile,
            force_refresh=bool(buildings_cfg["extract"].get("force_refresh", False)),
            max_building_distance_meters=float(
                buildings_cfg["extract"].get("max_building_distance_meters", 35)
            ),
        )
        extractor.print_stats()
        bgen = BuildingsGenerator(elevation)
        bmesh = bgen.generate_buildings(
            elev_data,
            float(terrain_cfg["elevation_multiplier"]),
            float(buildings_cfg["generate"].get("building_height_multiplier", 1.0)),
            bounds_tile,
            buildings,
            min_building_height=float(buildings_cfg["generate"].get("min_building_height", 10.0)),
            max_length_mm=float(global_cfg.get("scale_max_length_mm", 200.0)),
        )
        return bmesh

    # Result entries: (prefix, terrain_result, tile_bounds, buildings_mesh, base_mesh, tile_row, tile_col)
    results: List[Tuple[str, Dict[str, Any], Tuple[float, float, float, float], Optional[Any], Optional[Any], int, int]] = []
    if tile_enabled and (tile_rows > 1 or tile_cols > 1):
        min_lon, min_lat, max_lon, max_lat = bounds
        dlon = (max_lon - min_lon) / float(tile_cols)
        dlat = (max_lat - min_lat) / float(tile_rows)
        # Process tiles sequentially to avoid display conflicts
        for r in range(tile_rows):
            for c in range(tile_cols):
                t_bounds = (
                    min_lon + c * dlon,
                    min_lat + r * dlat,
                    min_lon + (c + 1) * dlon,
                    min_lat + (r + 1) * dlat,
                )
                t_prefix = f"{prefix}_r{r+1}c{c+1}"
                try:
                    tres = _gen(t_bounds, float(global_cfg.get("scale_max_length_mm", 200.0)))
                except Exception as exc:
                    output.warning(f"Skipping tile {t_prefix}: {exc}")
                    continue
                bmesh = _gen_buildings(t_bounds, tres.get("elevation_data")) if (tres and "elevation_data" in tres) else None
                # Generate base mesh for this tile
                base_mesh = _generate_base_for_tile(
                    generator, t_bounds, bounds, r, c, tile_rows, tile_cols, 
                    float(global_cfg.get("scale_max_length_mm", 200.0)),
                    base_cfg
                )
                results.append((t_prefix, tres, t_bounds, bmesh, base_mesh, r, c))
    else:
        t_res = _gen(bounds, float(global_cfg.get("scale_max_length_mm", 200.0)))
        bmesh = _gen_buildings(bounds, t_res["elevation_data"]) if t_res else None
        # Generate base for single tile (no cutouts needed for single tile)
        base_mesh = _generate_base_for_tile(
            generator, bounds, bounds, 0, 0, 1, 1, 
            float(global_cfg.get("scale_max_length_mm", 200.0)),
            base_cfg
        )
        results.append((prefix, t_res, bounds, bmesh, base_mesh, 0, 0))

    # Output
    output_cfg = job_cfg.get("output", {}) or {}
    out_dir = str(output_cfg.get("directory") or global_output_dir or ".")
    # Save outputs for each (possibly tiled) result with scaling
    for t_prefix, res, t_bounds, bmesh, base_mesh, tr, tc in results:
        _save_outputs(
            generator,
            res,
            bmesh,
            base_mesh,
            out_dir,
            t_prefix,
            t_bounds,
            bounds,  # Pass overall bounds for consistent scaling
            float(tiling_cfg.get("scale_max_length_mm", 200.0)),
            tr,
            tc,
            tile_rows,
            tile_cols,
            output_cfg,
        )

    output.success(f"Completed: {name}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate terrain and buildings from YAML configuration")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--job", help="Run only the named job (matches output_prefix)")
    parser.add_argument("--outdir", help="Override output directory for all jobs")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)

    jobs = _as_jobs(cfg)
    if not jobs:
        output.error("No jobs found in configuration")
        return 2

    for job in jobs:
        try:
            run_job(job, global_output_dir=args.outdir, only_prefix=args.job)
        except Exception as exc:
            output.error(f"Job failed: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


