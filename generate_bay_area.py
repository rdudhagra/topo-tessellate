from terrain_generator.generator import TerrainGenerator
import os

def main():
    # Create generator
    generator = TerrainGenerator()
    
    # Bay Area bounds (approximately)
    bounds = (-123.0, 37.2, -121.7, 38.1)  # Covers SF, Oakland, San Jose, and surrounding areas
    
    # Use all SRTM files in the topo directory
    topo_dir = "topo"
    
    print("Generating Bay Area terrain model...")
    print(f"Bounds: {bounds}")
    
    # Detail level explanation:
    # 1.0 = highest detail (90m between points)
    # 0.5 = medium detail (180m between points)
    # 0.1 = low detail (900m between points)
    # 0.01 = very low detail (9km between points)
    detail_level = 1.0  # ~450m between points, good balance for Bay Area scale
    
    print(f"Using detail level {detail_level} (~{90/detail_level:.0f}m between points)")
    
    # Process the region
    mesh = generator.process_region(topo_dir, bounds, detail_level)
    
    # Export the model
    output_path = f"bay_area_detail_{detail_level:.3f}.glb"
    print(f"Exporting to {output_path}...")
    generator.export_glb(mesh, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
