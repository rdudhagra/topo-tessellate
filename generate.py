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

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print("Missing dependency: PyYAML. Please install with: pip install PyYAML", file=sys.stderr)
    raise

from terrain_generator.console import output
from terrain_generator.modelgenerator import ModelGenerator
from terrain_generator.srtm import SRTM
from terrain_generator.geotiff import GeoTiff
from terrain_generator.buildingsextractor import BuildingsExtractor
from terrain_generator.buildingsgenerator import BuildingsGenerator


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
        file_name = es.get("file")
        if not file_name:
            raise ValueError("elevation_source.file is required for type=geotiff")
        elevation = GeoTiff(str(file_name))
    elif src_type == "srtm":
        elevation = SRTM()
    else:
        raise ValueError(f"Unknown elevation_source.type: {src_type}")

    return elevation, topo_dir


def _terrain_defaults() -> Dict[str, Any]:
    return {
        "base_height": 500.0,
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
        "merge_land_and_base": False,
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


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _save_outputs(
    generator: ModelGenerator,
    result: Dict[str, Any],
    buildings_mesh: Optional[Any],
    out_dir: str,
    prefix: str,
    output_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_dir(out_dir)
    land = result.get("land_mesh")
    base = result.get("base_mesh")
    merged = result.get("merged_mesh")
    if land is not None:
        generator.save_mesh(land, os.path.join(out_dir, f"{prefix}_land.obj"))
    if base is not None:
        generator.save_mesh(base, os.path.join(out_dir, f"{prefix}_base.obj"))
    if buildings_mesh is not None:
        generator.save_mesh(buildings_mesh, os.path.join(out_dir, f"{prefix}_buildings.obj"))
    if merged is not None and output_cfg and bool(output_cfg.get("save_merged", False)):
        merged_filename = str(output_cfg.get("merged_filename") or f"{prefix}_merged.obj")
        generator.save_mesh(merged, os.path.join(out_dir, merged_filename))


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

    # Generate terrain
    result = generator.generate_terrain_model(
        bounds=bounds,
        topo_dir=topo_dir,
        base_height=float(terrain_cfg["base_height"]),
        water_threshold=terrain_cfg["water_threshold"],
        elevation_multiplier=float(terrain_cfg["elevation_multiplier"]),
        downsample_factor=int(terrain_cfg["downsample_factor"]),
        force_refresh=bool(terrain_cfg["force_refresh"]),
        adaptive_tolerance_z=float(terrain_cfg.get("adaptive_tolerance_z", 1.0)),
        adaptive_max_gap_fraction=float(terrain_cfg.get("adaptive_max_gap_fraction", 1/256)),
        adaptive_max_sampled_rows=int(terrain_cfg.get("adaptive_max_sampled_rows", 400)),
        adaptive_max_sampled_cols=int(terrain_cfg.get("adaptive_max_sampled_cols", 400)),
        split_at_water_level=bool(terrain_cfg["split_at_water_level"]),
        merge_land_and_base=bool(terrain_cfg["merge_land_and_base"]),
    )

    # Buildings (optional)
    buildings_cfg = _merge(_buildings_defaults(), job_cfg.get("buildings", {}))
    buildings_mesh = None
    if bool(buildings_cfg.get("enabled", False)):
        extractor = BuildingsExtractor(
            timeout=int(buildings_cfg.get("timeout", 120)),
            use_cache=bool(buildings_cfg.get("use_cache", True)),
            cache_max_age_days=int(buildings_cfg.get("cache_max_age_days", 30)),
        )
        buildings = extractor.extract_buildings(
            bounds,
            force_refresh=bool(buildings_cfg["extract"].get("force_refresh", False)),
            max_building_distance_meters=float(
                buildings_cfg["extract"].get("max_building_distance_meters", 35)
            ),
        )
        extractor.print_stats()

        bgen = BuildingsGenerator(elevation)
        buildings_mesh = bgen.generate_buildings(
            float(terrain_cfg["base_height"]),
            result["elevation_data"],
            float(terrain_cfg["elevation_multiplier"]),
            float(buildings_cfg["generate"].get("building_height_multiplier", 1.0)),
            bounds,
            buildings,
            min_building_height=float(buildings_cfg["generate"].get("min_building_height", 10.0)),
        )

    # Output
    output_cfg = job_cfg.get("output", {}) or {}
    out_dir = str(output_cfg.get("directory") or global_output_dir or ".")
    _save_outputs(generator, result, buildings_mesh, out_dir, prefix, output_cfg)

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


