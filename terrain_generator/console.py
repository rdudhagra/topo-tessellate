#!/usr/bin/env python3
"""
Console Output Utilities

Provides consistent, colorful console output formatting across the application.
Uses the rich library for beautiful terminal output with colors, progress bars, and formatting.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich.text import Text
from rich import box
from typing import Dict, List, Optional, Any
import time
from contextlib import contextmanager

# Create global console instance
console = Console()

# Color scheme
COLORS = {
    "primary": "cyan",
    "secondary": "blue", 
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "white",
    "accent": "magenta",
    "muted": "dim white"
}

class ConsoleOutput:
    """Centralized console output manager with consistent styling."""
    
    def __init__(self):
        self.console = console
        
    def header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print a styled header section."""
        content = f"[bold {COLORS['primary']}]{title}[/bold {COLORS['primary']}]"
        if subtitle:
            content += f"\n[{COLORS['muted']}]{subtitle}[/{COLORS['muted']}]"
            
        panel = Panel(
            content,
            border_style=COLORS['primary'],
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def subheader(self, title: str) -> None:
        """Print a styled subheader."""
        self.console.print(f"\n[bold {COLORS['secondary']}]▶ {title}[/bold {COLORS['secondary']}]")
        
    def success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[{COLORS['success']}]✓[/{COLORS['success']}] {message}")
        
    def warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[{COLORS['warning']}]⚠[/{COLORS['warning']}] {message}")
        
    def error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[{COLORS['error']}]✗[/{COLORS['error']}] {message}")
        
    def info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[{COLORS['info']}]ℹ[/{COLORS['info']}] {message}")
        
    def progress_info(self, message: str) -> None:
        """Print a progress-related info message."""
        self.console.print(f"[{COLORS['accent']}]⟳[/{COLORS['accent']}] {message}")
        
    def stats_table(self, title: str, data: Dict[str, Any]) -> None:
        """Display statistics in a formatted table."""
        table = Table(title=f"[bold {COLORS['primary']}]{title}[/bold {COLORS['primary']}]", 
                     show_header=True, header_style=f"bold {COLORS['secondary']}")
        table.add_column("Metric", style=COLORS['info'])
        table.add_column("Value", style=f"bold {COLORS['success']}")
        
        for key, value in data.items():
            if isinstance(value, float):
                if value >= 1000:
                    formatted_value = f"{value:,.1f}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            table.add_row(key, formatted_value)
            
        self.console.print(table)
        
    def building_stats(self, buildings: List[Any], stats: Dict[str, Any]) -> None:
        """Display building extraction statistics."""
        if not buildings:
            self.warning("No buildings found!")
            return
            
        heights = [building.height for building in buildings]
        min_height = min(heights)
        max_height = max(heights)
        avg_height = sum(heights) / len(heights)
        
        # Main statistics table
        main_stats = {
            "Total Buildings": len(buildings),
            "Height Range": f"{min_height:.1f}m - {max_height:.1f}m",
            "Average Height": f"{avg_height:.1f}m",
            "Total Area": f"{sum(b.area for b in buildings)/1_000_000:.2f} km²"
        }
        self.stats_table("Building Summary", main_stats)
        
        # Height distribution
        tall_buildings = len([b for b in buildings if b.height > 100])
        mid_buildings = len([b for b in buildings if 30 <= b.height <= 100])
        low_buildings = len([b for b in buildings if b.height < 30])
        
        height_dist = {
            "Low-rise (<30m)": f"{low_buildings} ({low_buildings/len(buildings)*100:.1f}%)",
            "Mid-rise (30-100m)": f"{mid_buildings} ({mid_buildings/len(buildings)*100:.1f}%)",
            "High-rise (>100m)": f"{tall_buildings} ({tall_buildings/len(buildings)*100:.1f}%)"
        }
        self.stats_table("Height Distribution", height_dist)
        
        # Processing statistics if available
        if stats:
            process_stats = {
                "Elements Processed": stats.get('processed_elements', 'N/A'),
                "Elements Excluded": stats.get('total_excluded', 'N/A'),
                "Buildings with Height": stats.get('buildings_with_height', 'N/A')
            }
            self.stats_table("Processing Statistics", process_stats)
    
    def terrain_info(self, bounds, elevation_data_shape, water_info: Optional[Dict] = None) -> None:
        """Display terrain generation information."""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        terrain_stats = {
            "Longitude Range": f"{min_lon:.3f}° to {max_lon:.3f}°",
            "Latitude Range": f"{min_lat:.3f}° to {max_lat:.3f}°",
            "Grid Size": f"{elevation_data_shape[1]} × {elevation_data_shape[0]}",
            "Total Data Points": f"{elevation_data_shape[0] * elevation_data_shape[1]:,}"
        }
        
        if water_info:
            terrain_stats.update({
                "Water Pixels": f"{water_info.get('pixel_count', 0):,}",
                "Water Coverage": f"{water_info.get('percentage', 0):.1f}%"
            })
            
        self.stats_table("Terrain Information", terrain_stats)
        
    def file_saved(self, filename: str, file_type: str = "file") -> None:
        """Indicate a file has been saved."""
        self.success(f"{file_type.title()} saved: [bold]{filename}[/bold]")
        
    def cache_info(self, message: str, is_hit: bool = True) -> None:
        """Display cache-related information."""
        if is_hit:
            self.success(f"Cache: {message}")
        else:
            self.info(f"Cache: {message}")
            
    @contextmanager
    def progress_context(self, description: str):
        """Context manager for progress operations."""
        with Progress() as progress:
            task = progress.add_task(f"[{COLORS['accent']}]{description}[/{COLORS['accent']}]", total=None)
            yield progress, task
            progress.update(task, completed=100)
            
    def print_section_divider(self) -> None:
        """Print a visual section divider."""
        self.console.print(f"\n[{COLORS['muted']}]{'─' * 60}[/{COLORS['muted']}]\n")

# Global console output instance
output = ConsoleOutput() 