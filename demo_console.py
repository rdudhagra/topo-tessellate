#!/usr/bin/env python3
"""
Console Output Demo

This script demonstrates the new colorful console output system with all its features.
Run this to see how the improved output looks!
"""

import time
from terrain_generator.console import output


def demo_basic_messages():
    """Demonstrate basic message types."""
    output.header("Basic Message Types", "Showcasing different types of console messages")
    
    output.info("This is an informational message")
    output.success("This indicates a successful operation")
    output.warning("This is a warning message")
    output.error("This shows an error occurred")
    output.progress_info("This shows progress or ongoing operations")
    
    output.print_section_divider()


def demo_structured_output():
    """Demonstrate structured output like tables and stats."""
    output.header("Structured Output")
    
    # Demo stats table
    sample_stats = {
        "Total Files": 1249,
        "Processing Time": "3.45 seconds", 
        "Success Rate": "98.7%",
        "Memory Usage": "245 MB",
        "Cache Hit Rate": "87.3%"
    }
    output.stats_table("System Performance", sample_stats)
    
    # Demo terrain info
    elevation_shape = (1024, 768)
    bounds = (-122.5, 37.7, -122.3, 37.9)
    water_info = {'pixel_count': 45678, 'percentage': 12.3}
    
    output.terrain_info(bounds, elevation_shape, water_info)
    
    output.print_section_divider()


def demo_building_stats():
    """Demonstrate building statistics display.""" 
    output.header("Building Statistics Demo")
    
    # Create mock building data
    class MockBuilding:
        def __init__(self, height, area):
            self.height = height
            self.area = area
    
    # Generate sample buildings with varying heights
    buildings = [
        MockBuilding(5.0, 120.5),   # Low-rise
        MockBuilding(25.0, 450.2),  # Low-rise
        MockBuilding(45.0, 890.1),  # Mid-rise
        MockBuilding(85.0, 1250.7), # Mid-rise  
        MockBuilding(120.0, 2100.3), # High-rise
        MockBuilding(180.0, 3500.8), # High-rise
        MockBuilding(8.0, 95.3),    # Low-rise
        MockBuilding(15.0, 234.6),  # Low-rise
        MockBuilding(65.0, 1150.4), # Mid-rise
        MockBuilding(200.0, 4200.1) # High-rise
    ]
    
    mock_stats = {
        'processed_elements': 1500,
        'total_excluded': 200,
        'buildings_with_height': len(buildings)
    }
    
    output.building_stats(buildings, mock_stats)
    
    output.print_section_divider()


def demo_progress_operations():
    """Demonstrate progress tracking."""
    output.header("Progress Operations")
    
    output.subheader("File Operations")
    
    # Simulate some operations with progress
    operations = [
        "Loading configuration files",
        "Downloading elevation data", 
        "Processing terrain mesh",
        "Extracting building data",
        "Generating output files"
    ]
    
    for operation in operations:
        with output.progress_context(operation):
            time.sleep(0.8)  # Simulate work
        output.success(f"Completed: {operation}")
    
    output.print_section_divider()


def demo_cache_and_files():
    """Demonstrate cache and file operations."""
    output.header("Cache & File Operations")
    
    output.cache_info("Loading 1,249 buildings from cache (age: 2.3 hours)", is_hit=True)
    output.cache_info("Saved 856 buildings to cache", is_hit=False)
    
    output.file_saved("bay_area_terrain.obj", "terrain mesh")
    output.file_saved("buildings_map.png", "visualization")
    output.file_saved("elevation_analysis.png", "chart")
    
    output.print_section_divider()


def demo_real_world_scenario():
    """Demonstrate a realistic terrain generation workflow."""
    output.header("Realistic Workflow Demo", "Bay Area terrain and building extraction")
    
    # Simulate bounds input
    bounds = (-122.67, 37.22, -121.75, 38.18)
    output.info(f"Processing region: {bounds}")
    output.info("Area: San Francisco Bay Area")
    
    output.print_section_divider()
    
    # Simulate terrain processing
    output.subheader("Terrain Processing")
    
    with output.progress_context("Loading elevation data"):
        time.sleep(1.0)
    
    terrain_stats = {
        "Grid Size": "2048 × 1536",
        "Real-world Size": "92.5 × 69.2 km",
        "Elevation Range": "-5.2 to 875.3 m",
        "Data Points": "3,145,728"
    }
    output.stats_table("Terrain Information", terrain_stats)
    
    with output.progress_context("Detecting water areas"):
        time.sleep(0.7)
    
    output.success("Found 47,832 water pixels (15.2% coverage)")
    
    with output.progress_context("Creating terrain mesh"):
        time.sleep(1.2)
    
    output.success("Generated mesh with 3,145,728 vertices and 6,291,456 faces")
    
    output.print_section_divider()
    
    # Simulate building extraction
    output.subheader("Building Extraction")
    
    with output.progress_context("Fetching building data from OpenStreetMap"):
        time.sleep(1.5)
    
    output.success("Extracted 12,847 buildings from 18,932 elements")
    
    # Mock building stats
    class MockBuilding:
        def __init__(self, height, area):
            self.height = height
            self.area = area
    
    # Generate realistic building distribution
    import random
    buildings = []
    for _ in range(12847):
        # Weight toward lower buildings (realistic distribution)
        if random.random() < 0.7:  # 70% low-rise
            height = random.uniform(3.0, 25.0)
            area = random.uniform(80, 400)
        elif random.random() < 0.25:  # 25% mid-rise  
            height = random.uniform(25.0, 100.0)
            area = random.uniform(400, 1500)
        else:  # 5% high-rise
            height = random.uniform(100.0, 300.0) 
            area = random.uniform(1500, 5000)
        buildings.append(MockBuilding(height, area))
    
    mock_stats = {
        'processed_elements': 18932,
        'total_excluded': 6085,
        'buildings_with_height': len(buildings)
    }
    
    output.building_stats(buildings, mock_stats)
    
    # Final results
    output.subheader("Output Files")
    output.file_saved("bay_area_terrain.obj", "terrain mesh")
    output.file_saved("bay_area_water.obj", "water mesh") 
    output.file_saved("bay_area_buildings_map.png", "buildings visualization")
    
    output.success("Bay Area 3D model generation complete!")


def main():
    """Run the complete console output demonstration."""
    output.header("🎨 Console Output System Demo", 
                 "Showcasing the new colorful, structured terminal output")
    
    output.info("This demo shows all the features of the new console output system")
    output.info("Press Ctrl+C at any time to exit")
    
    print()  # Add some spacing
    
    try:
        demo_basic_messages()
        demo_structured_output() 
        demo_building_stats()
        demo_progress_operations()
        demo_cache_and_files()
        demo_real_world_scenario()
        
        output.header("🎉 Demo Complete!", "The console output system is ready to use!")
        output.success("All print statements in the codebase have been upgraded!")
        output.info("Run 'python generate_bay_area.py' to see it in action")
        
    except KeyboardInterrupt:
        output.warning("Demo interrupted by user")
        output.info("Thanks for checking out the console output system!")


if __name__ == "__main__":
    main() 