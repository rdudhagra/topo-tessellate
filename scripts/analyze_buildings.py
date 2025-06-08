#!/usr/bin/env python3
"""
Building Analysis Script

This script analyzes cached building data to generate comprehensive analytics and visualizations.
It can analyze buildings within a specific bounding box or analyze all buildings in a cache file.

Usage:
    python analyze_buildings.py cache_file.pkl.gz [--bbox min_lon min_lat max_lon max_lat] [--output-dir ./analysis]
"""

import argparse
import gzip
import pickle
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import json

# Add the parent directory to the path to import terrain_generator modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from terrain_generator.buildingsextractor import Building
from terrain_generator.console import output
from terrain_generator.buildingsprocessor import BuildingsProcessor


class BuildingAnalyzer:
    """Analyzes cached building data and generates analytics and visualizations."""
    
    def __init__(self, cache_file: str, output_dir: str = "./analysis"):
        """Initialize the analyzer with a cache file.
        
        Args:
            cache_file: Path to the building cache file (.pkl.gz)
            output_dir: Directory to save analysis outputs
        """
        self.cache_file = cache_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.buildings: List[Building] = []
        self.cache_metadata: Dict[str, Any] = {}
        self.filtered_buildings: List[Building] = []
        
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load buildings data from the cache file."""
        output.progress_info(f"Loading building data from {self.cache_file}")
        
        try:
            with gzip.open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.buildings = cache_data.get('buildings', [])
            self.cache_metadata = {
                'bounds': cache_data.get('bounds'),
                'timestamp': cache_data.get('timestamp'),
                'version': cache_data.get('version'),
                'stats': cache_data.get('stats', {}),
                'default_height': cache_data.get('default_height', 5.0)
            }
            
            output.success(f"Loaded {len(self.buildings)} buildings from cache")
            
            # Show cache info
            if self.cache_metadata['bounds']:
                bounds = self.cache_metadata['bounds']
                output.info(f"Cache bounds: {bounds[0]:.4f}, {bounds[1]:.4f} to {bounds[2]:.4f}, {bounds[3]:.4f}")
            
            if self.cache_metadata['timestamp']:
                import time
                cache_age_hours = (time.time() - self.cache_metadata['timestamp']) / 3600
                output.info(f"Cache age: {cache_age_hours:.1f} hours")
                
        except Exception as e:
            output.error(f"Failed to load cache file: {e}")
            sys.exit(1)
    
    def filter_by_bbox(self, bbox: Tuple[float, float, float, float]) -> None:
        """Filter buildings by bounding box.
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) bounding box
        """        
        output.progress_info(f"Filtering buildings by bbox: {bbox}")
        
        self.processor = BuildingsProcessor(self.buildings)
        self.filtered_buildings = self.processor.exclude_buildings_outside_bbox(bbox)

        output.success(f"Filtered to {len(self.filtered_buildings)} buildings within bbox")
    
    def get_analysis_buildings(self) -> List[Building]:
        """Get the buildings to analyze (filtered or all)."""
        return self.filtered_buildings if self.filtered_buildings else self.buildings
    
    def generate_basic_statistics(self) -> Dict[str, Any]:
        """Generate basic statistics about the buildings."""
        buildings = self.get_analysis_buildings()
        
        if not buildings:
            return {"error": "No buildings to analyze"}
        
        # Basic counts
        total_buildings = len(buildings)
        
        # Area statistics
        areas = [b.area for b in buildings]
        total_area = sum(areas)
        avg_area = total_area / total_buildings
        
        # Height statistics  
        heights = [b.height for b in buildings]
        avg_height = sum(heights) / total_buildings
        
        # Building types
        building_types = Counter(b.building_type for b in buildings)
        
        # Coordinate bounds
        all_coords = []
        for building in buildings:
            all_coords.extend(building.polygon.exterior.coords)
        
        if all_coords:
            lons, lats = zip(*all_coords)
            coord_bounds = (min(lons), min(lats), max(lons), max(lats))
        else:
            coord_bounds = None
        
        return {
            "total_buildings": total_buildings,
            "total_area_m2": total_area,
            "total_area_km2": total_area / 1_000_000,
            "average_area_m2": avg_area,
            "min_area_m2": min(areas),
            "max_area_m2": max(areas),
            "average_height_m": avg_height,
            "min_height_m": min(heights),
            "max_height_m": max(heights),
            "building_types": dict(building_types.most_common()),
            "coordinate_bounds": coord_bounds,
            "unique_building_types": len(building_types)
        }
    
    def generate_detailed_analytics(self) -> Dict[str, Any]:
        """Generate detailed analytics about the buildings."""
        buildings = self.get_analysis_buildings()
        
        if not buildings:
            return {"error": "No buildings to analyze"}
        
        basic_stats = self.generate_basic_statistics()
        
        # Height distribution
        heights = [b.height for b in buildings]
        height_percentiles = {
            "p10": np.percentile(heights, 10),
            "p25": np.percentile(heights, 25),
            "p50": np.percentile(heights, 50),
            "p75": np.percentile(heights, 75),
            "p90": np.percentile(heights, 90),
            "p95": np.percentile(heights, 95),
            "p99": np.percentile(heights, 99)
        }
        
        # Area distribution
        areas = [b.area for b in buildings]
        area_percentiles = {
            "p10": np.percentile(areas, 10),
            "p25": np.percentile(areas, 25),
            "p50": np.percentile(areas, 50),
            "p75": np.percentile(areas, 75),
            "p90": np.percentile(areas, 90),
            "p95": np.percentile(areas, 95),
            "p99": np.percentile(areas, 99)
        }
        
        # Building type analysis
        type_stats = defaultdict(lambda: {"count": 0, "total_area": 0, "total_height": 0})
        for building in buildings:
            bt = building.building_type
            type_stats[bt]["count"] += 1
            type_stats[bt]["total_area"] += building.area
            type_stats[bt]["total_height"] += building.height
        
        # Calculate averages for each type
        for bt in type_stats:
            count = type_stats[bt]["count"]
            type_stats[bt]["avg_area"] = type_stats[bt]["total_area"] / count
            type_stats[bt]["avg_height"] = type_stats[bt]["total_height"] / count
        
        # Density analysis (buildings per km²)
        if basic_stats["coordinate_bounds"]:
            bounds = basic_stats["coordinate_bounds"]
            # Approximate area calculation (not perfectly accurate for large areas)
            lon_diff = bounds[2] - bounds[0]
            lat_diff = bounds[3] - bounds[1]
            # Rough conversion: 1 degree ≈ 111 km at equator
            area_km2 = abs(lon_diff * lat_diff) * (111 ** 2)
            building_density = len(buildings) / area_km2 if area_km2 > 0 else 0
        else:
            building_density = 0
        
        return {
            **basic_stats,
            "height_percentiles": height_percentiles,
            "area_percentiles": area_percentiles,
            "building_type_details": dict(type_stats),
            "building_density_per_km2": building_density,
            "height_std": np.std(heights),
            "area_std": np.std(areas)
        }
    
    def create_height_distribution_plot(self) -> str:
        """Create a histogram of building heights."""
        buildings = self.get_analysis_buildings()
        heights = [b.height for b in buildings]
        
        plt.figure(figsize=(12, 8))
        
        # Create histogram
        plt.subplot(2, 2, 1)
        plt.hist(heights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Height (m)')
        plt.ylabel('Number of Buildings')
        plt.title('Building Height Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(heights, vert=True)
        plt.ylabel('Height (m)')
        plt.title('Building Height Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_heights = np.sort(heights)
        cumulative = np.arange(1, len(sorted_heights) + 1) / len(sorted_heights)
        plt.plot(sorted_heights, cumulative, color='red', linewidth=2)
        plt.xlabel('Height (m)')
        plt.ylabel('Cumulative Probability')
        plt.title('Building Height CDF')
        plt.grid(True, alpha=0.3)
        
        # Height by building type (top 10 types)
        plt.subplot(2, 2, 4)
        type_heights = defaultdict(list)
        for building in buildings:
            type_heights[building.building_type].append(building.height)
        
        # Get top 10 most common types
        top_types = sorted(type_heights.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        
        if top_types:
            types, height_lists = zip(*top_types)
            plt.boxplot(height_lists, labels=[t[:15] for t in types])  # Truncate long labels
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Height (m)')
            plt.title('Height by Building Type (Top 10)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.output_dir / "height_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_area_distribution_plot(self) -> str:
        """Create a histogram of building areas."""
        buildings = self.get_analysis_buildings()
        areas = [b.area for b in buildings]
        
        plt.figure(figsize=(12, 8))
        
        # Log scale histogram (areas can vary greatly)
        plt.subplot(2, 2, 1)
        plt.hist(areas, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Area (m²)')
        plt.ylabel('Number of Buildings')
        plt.title('Building Area Distribution')
        plt.grid(True, alpha=0.3)
        
        # Log scale histogram
        plt.subplot(2, 2, 2)
        plt.hist(areas, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Area (m²)')
        plt.ylabel('Number of Buildings')
        plt.title('Building Area Distribution (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(areas, vert=True)
        plt.ylabel('Area (m²)')
        plt.title('Building Area Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Area vs Height scatter plot
        plt.subplot(2, 2, 4)
        heights = [b.height for b in buildings]
        plt.scatter(areas, heights, alpha=0.5, s=1)
        plt.xlabel('Area (m²)')
        plt.ylabel('Height (m)')
        plt.title('Building Area vs Height')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.output_dir / "area_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_building_type_plot(self) -> str:
        """Create visualizations for building types."""
        buildings = self.get_analysis_buildings()
        
        type_counts = Counter(b.building_type for b in buildings)
        top_types = type_counts.most_common(20)
        
        if not top_types:
            return None
        
        plt.figure(figsize=(15, 10))
        
        # Bar chart of top building types
        plt.subplot(2, 2, 1)
        types, counts = zip(*top_types[:15])
        plt.bar(range(len(types)), counts, color='steelblue')
        plt.xlabel('Building Type')
        plt.ylabel('Count')
        plt.title('Top 15 Building Types by Count')
        plt.xticks(range(len(types)), [t[:15] for t in types], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Pie chart of top building types
        plt.subplot(2, 2, 2)
        top_10_types = type_counts.most_common(10)
        other_count = sum(type_counts.values()) - sum(count for _, count in top_10_types)
        
        pie_labels = [t for t, _ in top_10_types]
        pie_values = [c for _, c in top_10_types]
        
        if other_count > 0:
            pie_labels.append('Others')
            pie_values.append(other_count)
        
        plt.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Building Type Distribution')
        
        # Building type by average area
        plt.subplot(2, 2, 3)
        type_areas = defaultdict(list)
        for building in buildings:
            type_areas[building.building_type].append(building.area)
        
        avg_areas = [(bt, np.mean(areas)) for bt, areas in type_areas.items() if len(areas) >= 5]  # At least 5 buildings
        avg_areas.sort(key=lambda x: x[1], reverse=True)
        
        if avg_areas:
            top_area_types = avg_areas[:15]
            types, areas = zip(*top_area_types)
            plt.bar(range(len(types)), areas, color='orange')
            plt.xlabel('Building Type')
            plt.ylabel('Average Area (m²)')
            plt.title('Top 15 Building Types by Average Area')
            plt.xticks(range(len(types)), [t[:15] for t in types], rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        # Building type by average height
        plt.subplot(2, 2, 4)
        type_heights = defaultdict(list)
        for building in buildings:
            type_heights[building.building_type].append(building.height)
        
        avg_heights = [(bt, np.mean(heights)) for bt, heights in type_heights.items() if len(heights) >= 5]
        avg_heights.sort(key=lambda x: x[1], reverse=True)
        
        if avg_heights:
            top_height_types = avg_heights[:15]
            types, heights = zip(*top_height_types)
            plt.bar(range(len(types)), heights, color='red')
            plt.xlabel('Building Type')
            plt.ylabel('Average Height (m)')
            plt.title('Top 15 Building Types by Average Height')
            plt.xticks(range(len(types)), [t[:15] for t in types], rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.output_dir / "building_types_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_spatial_plot(self) -> str:
        """Create a spatial visualization of buildings showing actual polygons."""
        buildings = self.get_analysis_buildings()
        
        if not buildings:
            return None
        
        # Extract building polygons and data
        building_polygons = []
        building_data = []
        all_coords = []
        
        for building in buildings:
            try:
                coords = np.array(building.polygon.exterior.coords)
                
                if len(coords) == 0:
                    output.error(f"Building {building.osm_id} has no coordinates")
                    continue
                
                # Store polygon coordinates and building data
                building_polygons.append(coords)
                building_data.append({
                    'coords': coords,
                    'height': building.height,
                    'area': building.area,
                    'type': building.building_type
                })
                
                # Collect all coordinates for bounds calculation
                all_coords.extend(coords)
                
            except Exception as e:
                output.error(f"Error processing building {building.osm_id}: {e}")
                continue
        
        if not building_polygons:
            return None
        
        # Calculate bounds for consistent axes
        all_lons, all_lats = zip(*all_coords)
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        
        plt.figure(figsize=(16, 12))
        
        # Building polygons colored by height
        ax1 = plt.subplot(2, 2, 1)
        heights = [bd['height'] for bd in building_data]
        height_min, height_max = min(heights), max(heights)
        
        for i, bd in enumerate(building_data):
            if len(bd['coords']) >= 3:  # Valid polygon needs at least 3 points
                # Normalize height for colormap
                norm_height = (bd['height'] - height_min) / (height_max - height_min) if height_max > height_min else 0
                color = plt.cm.viridis(norm_height)
                
                polygon = mpatches.Polygon(bd['coords'], closed=True, 
                                         facecolor=color, edgecolor='black', 
                                         linewidth=0.1, alpha=0.7)
                ax1.add_patch(polygon)
        
        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Building Polygons by Height')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for height
        if height_max > height_min:
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=height_min, vmax=height_max))
            sm.set_array([])
            plt.colorbar(sm, ax=ax1, label='Height (m)')
        
        # Building polygons colored by area
        ax2 = plt.subplot(2, 2, 2)
        areas = [bd['area'] for bd in building_data]
        area_min, area_max = min(areas), max(areas)
        
        for i, bd in enumerate(building_data):
            if len(bd['coords']) >= 3:
                # Normalize area for colormap (use log scale for better visualization)
                log_area = np.log(bd['area']) if bd['area'] > 0 else 0
                log_area_min = np.log(area_min) if area_min > 0 else 0
                log_area_max = np.log(area_max) if area_max > 0 else 0
                
                if log_area_max > log_area_min:
                    norm_area = (log_area - log_area_min) / (log_area_max - log_area_min)
                else:
                    norm_area = 0
                    
                color = plt.cm.plasma(norm_area)
                
                polygon = mpatches.Polygon(bd['coords'], closed=True,
                                         facecolor=color, edgecolor='black',
                                         linewidth=0.1, alpha=0.7)
                ax2.add_patch(polygon)
        
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Building Polygons by Area')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for area
        if area_max > area_min:
            sm_area = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=area_min, vmax=area_max))
            sm_area.set_array([])
            plt.colorbar(sm_area, ax=ax2, label='Area (m²)')
        
        # Building density heatmap (using centroids)
        ax3 = plt.subplot(2, 2, 3)
        centroids = [np.mean(bd['coords'], axis=0) for bd in building_data if len(bd['coords']) > 0]
        if centroids:
            centroid_lons, centroid_lats = zip(*centroids)
            h = ax3.hist2d(centroid_lons, centroid_lats, bins=50, cmap='Blues')
            plt.colorbar(h[3], ax=ax3, label='Building Count')
        ax3.set_xlim(lon_min, lon_max)
        ax3.set_ylim(lat_min, lat_max)
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title('Building Density Heatmap')
        
        # Building polygons by type (top 5 types)
        ax4 = plt.subplot(2, 2, 4)
        type_counts = Counter(bd['type'] for bd in building_data)
        top_5_types = [t for t, _ in type_counts.most_common(5)]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, building_type in enumerate(top_5_types):
            type_buildings = [bd for bd in building_data if bd['type'] == building_type]
            
            for bd in type_buildings:
                if len(bd['coords']) >= 3:
                    polygon = mpatches.Polygon(bd['coords'], closed=True,
                                             facecolor=colors[i], edgecolor='black',
                                             linewidth=0.1, alpha=0.6)
                    ax4.add_patch(polygon)
        
        ax4.set_xlim(lon_min, lon_max)
        ax4.set_ylim(lat_min, lat_max)
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Building Polygons by Type (Top 5)')
        ax4.grid(True, alpha=0.3)
        
        # Create legend
        legend_patches = [mpatches.Patch(color=colors[i], label=building_type[:20]) 
                         for i, building_type in enumerate(top_5_types)]
        ax4.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        
        filename = self.output_dir / "spatial_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def save_detailed_report(self, analytics: Dict[str, Any]) -> str:
        """Save a detailed text report of the analysis."""
        filename = self.output_dir / "building_analysis_report.txt"
        
        with open(filename, 'w') as f:
            f.write("Building Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Cache info
            f.write(f"Cache File: {self.cache_file}\n")
            if self.cache_metadata['bounds']:
                bounds = self.cache_metadata['bounds']
                f.write(f"Original Cache Bounds: {bounds[0]:.6f}, {bounds[1]:.6f} to {bounds[2]:.6f}, {bounds[3]:.6f}\n")
            
            buildings = self.get_analysis_buildings()
            if self.filtered_buildings:
                f.write(f"Analysis Mode: Filtered by bounding box\n")
                f.write(f"Buildings Analyzed: {len(buildings)} (out of {len(self.buildings)} total)\n")
            else:
                f.write(f"Analysis Mode: All buildings in cache\n")
                f.write(f"Buildings Analyzed: {len(buildings)}\n")
            
            f.write("\n" + "=" * 30 + "\n")
            f.write("BASIC STATISTICS\n")
            f.write("=" * 30 + "\n")
            
            f.write(f"Total Buildings: {analytics['total_buildings']:,}\n")
            f.write(f"Total Area: {analytics['total_area_m2']:,.0f} m² ({analytics['total_area_km2']:.3f} km²)\n")
            f.write(f"Average Area: {analytics['average_area_m2']:,.0f} m²\n")
            f.write(f"Area Range: {analytics['min_area_m2']:,.0f} - {analytics['max_area_m2']:,.0f} m²\n")
            f.write(f"Average Height: {analytics['average_height_m']:.1f} m\n")
            f.write(f"Height Range: {analytics['min_height_m']:.1f} - {analytics['max_height_m']:.1f} m\n")
            f.write(f"Unique Building Types: {analytics['unique_building_types']}\n")
            
            if 'building_density_per_km2' in analytics:
                f.write(f"Building Density: {analytics['building_density_per_km2']:.1f} buildings/km²\n")
            
            f.write("\n" + "=" * 30 + "\n")
            f.write("PERCENTILE STATISTICS\n")
            f.write("=" * 30 + "\n")
            
            if 'height_percentiles' in analytics:
                f.write("Height Percentiles:\n")
                for p, v in analytics['height_percentiles'].items():
                    f.write(f"  {p}: {v:.1f} m\n")
                
                f.write("\nArea Percentiles:\n")
                for p, v in analytics['area_percentiles'].items():
                    f.write(f"  {p}: {v:,.0f} m²\n")
            
            f.write("\n" + "=" * 30 + "\n")
            f.write("TOP 20 BUILDING TYPES\n")
            f.write("=" * 30 + "\n")
            
            for i, (bt, count) in enumerate(list(analytics['building_types'].items())[:20], 1):
                f.write(f"{i:2d}. {bt:<25} {count:>8,} buildings\n")
            
            if 'building_type_details' in analytics:
                f.write("\n" + "=" * 40 + "\n")
                f.write("DETAILED BUILDING TYPE STATISTICS\n")
                f.write("=" * 40 + "\n")
                
                type_details = analytics['building_type_details']
                for bt in sorted(type_details.keys(), key=lambda x: type_details[x]['count'], reverse=True)[:20]:
                    details = type_details[bt]
                    f.write(f"\n{bt}:\n")
                    f.write(f"  Count: {details['count']:,}\n")
                    f.write(f"  Total Area: {details['total_area']:,.0f} m²\n")
                    f.write(f"  Average Area: {details['avg_area']:,.0f} m²\n")
                    f.write(f"  Average Height: {details['avg_height']:.1f} m\n")
        
        return str(filename)
    
    def save_json_data(self, analytics: Dict[str, Any]) -> str:
        """Save analytics data as JSON for further processing."""
        filename = self.output_dir / "building_analysis_data.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the analytics dict
        json_analytics = {}
        for key, value in analytics.items():
            if isinstance(value, dict):
                json_analytics[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                json_analytics[key] = convert_numpy(value)
        
        with open(filename, 'w') as f:
            json.dump(json_analytics, f, indent=2, default=convert_numpy)
        
        return str(filename)
    
    def run_complete_analysis(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Run complete analysis and generate all outputs."""
        output.header("Building Analysis")
        
        if bbox:
            self.filter_by_bbox(bbox)
            output.info(f"Analyzing buildings within bounding box: {bbox}")
        else:
            output.info("Analyzing all buildings in cache")

        # Cluster and merge buildings
        output.progress_info("Clustering and merging buildings...")
        self.filtered_buildings = self.processor.cluster_and_merge_buildings()
        
        buildings = self.get_analysis_buildings()
        if not buildings:
            output.error("No buildings to analyze")
            return
        
        # Generate analytics
        output.progress_info("Generating detailed analytics...")
        analytics = self.generate_detailed_analytics()
        
        # Create visualizations
        output.progress_info("Creating height distribution plots...")
        height_plot = self.create_height_distribution_plot()
        
        output.progress_info("Creating area distribution plots...")
        area_plot = self.create_area_distribution_plot()
        
        output.progress_info("Creating building type plots...")
        type_plot = self.create_building_type_plot()
        
        output.progress_info("Creating spatial plots...")
        spatial_plot = self.create_spatial_plot()
        
        # Save reports
        output.progress_info("Saving detailed report...")
        report_file = self.save_detailed_report(analytics)
        
        output.progress_info("Saving JSON data...")
        json_file = self.save_json_data(analytics)
        
        # Summary
        output.success("Analysis complete!")
        output.info(f"\nGenerated files in {self.output_dir}:")
        
        files_created = [
            ("Detailed Report", report_file),
            ("JSON Data", json_file),
            ("Height Analysis", height_plot),
            ("Area Analysis", area_plot),
        ]
        
        if type_plot:
            files_created.append(("Building Types", type_plot))
        if spatial_plot:
            files_created.append(("Spatial Analysis", spatial_plot))
        
        for desc, filepath in files_created:
            output.file_saved(filepath, desc.lower())
        
        # Print key statistics
        output.subheader("Key Statistics")
        output.info(f"Total Buildings: {analytics['total_buildings']:,}")
        output.info(f"Total Area: {analytics['total_area_km2']:.3f} km²")
        output.info(f"Average Height: {analytics['average_height_m']:.1f} m")
        output.info(f"Building Types: {analytics['unique_building_types']}")
        
        if 'building_density_per_km2' in analytics:
            output.info(f"Density: {analytics['building_density_per_km2']:.1f} buildings/km²")


def main():
    """Main function to run the building analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze cached building data and generate analytics and visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all buildings in cache
  python analyze_buildings.py building_cache/buildings_cache_abc123_-122.6700_37.2200_-121.7500_38.1800.pkl.gz
  
  # Analyze buildings within a specific bounding box
  python analyze_buildings.py cache_file.pkl.gz --bbox -122.45 37.75 -122.40 37.80
  
  # Specify custom output directory
  python analyze_buildings.py cache_file.pkl.gz --output-dir ./my_analysis
        """
    )
    
    parser.add_argument(
        'cache_file',
        help='Path to the building cache file (.pkl.gz)'
    )
    
    parser.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        help='Bounding box to filter buildings (min_lon min_lat max_lon max_lat)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./analysis',
        help='Directory to save analysis outputs (default: ./analysis)'
    )
    
    args = parser.parse_args()
    
    # Validate cache file
    if not os.path.exists(args.cache_file):
        output.error(f"Cache file not found: {args.cache_file}")
        sys.exit(1)
    
    # Validate bounding box
    bbox = None
    if args.bbox:
        min_lon, min_lat, max_lon, max_lat = args.bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            output.error("Invalid bounding box: min values must be less than max values")
            sys.exit(1)
        bbox = tuple(args.bbox)
    
    # Run analysis
    try:
        analyzer = BuildingAnalyzer(args.cache_file, args.output_dir)
        analyzer.run_complete_analysis(bbox)
    except KeyboardInterrupt:
        output.warning("\nAnalysis interrupted by user")
        sys.exit(1)



if __name__ == "__main__":
    main()
