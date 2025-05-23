import cadquery as cq
from cqmore import Workplane
from cqmore.polyhedron import Polyhedron
import numpy as np
import time
import cProfile
import pstats
from pstats import SortKey
import trimesh
WIDTH = 501
HEIGHT = 501
THICKNESS = 1

if __name__ == "__main__":
    # Enable profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Overall timing
    start_time_total = time.time()
    
    # Create randomly generated heightmap
    start_time = time.time()
    heightmap = np.random.rand(WIDTH, HEIGHT) * 1 + THICKNESS
    heightmap_time = time.time() - start_time
    print(f"Heightmap generation time: {heightmap_time:.4f} seconds")

    # Create a 3D model from the heightmap
    model = Workplane("XY")
    
    # Create lists to store all points and faces
    all_points = []
    all_faces = []
    
    # Calculate points and faces for all cells
    start_time = time.time()
    for x in range(WIDTH - 1):
        for y in range(HEIGHT - 1):
            # Current point index offset
            point_index = len(all_points)
            
            my_elevation = heightmap[x, y]

            # Get neighboring elevations
            southeast_elevation = heightmap[x + 1, y + 1]
            east_elevation = heightmap[x + 1, y]
            south_elevation = heightmap[x, y + 1]

            # Calculate positions for this cell
            pos_x = x - WIDTH / 2
            pos_y = y - HEIGHT / 2
            
            # Calculate corner coordinates
            northwest_corner_top = (
                pos_x,
                pos_y,
                my_elevation,
            )
            northeast_corner_top = (
                pos_x + 1,
                pos_y,
                east_elevation,
            )
            southeast_corner_top = (
                pos_x + 1,
                pos_y + 1,
                southeast_elevation,
            )
            southwest_corner_top = (
                pos_x,
                pos_y + 1,
                south_elevation,
            )

            # Add points for this cell
            all_points.extend(
                [
                    northeast_corner_top,
                    northwest_corner_top,
                    southwest_corner_top,
                    southeast_corner_top,
                ]
            )

            # Add faces for this cell with proper indices
            all_faces.extend(
                [
                    (point_index + 2, point_index + 1, point_index + 0),
                    (point_index + 3, point_index + 2, point_index + 0),
                ]
            )
    terrain_faces_time = time.time() - start_time
    print(f"Terrain points and faces generation time: {terrain_faces_time:.4f} seconds")
    print(f"Points count: {len(all_points)}, Faces count: {len(all_faces)}")

    # # Create each side wall
    # start_time = time.time()

    # # Left side
    # point_index = len(all_points)
    # for y in range(HEIGHT):
    #     y_pos = y - HEIGHT / 2
    #     all_points.extend([(-WIDTH / 2, y_pos, heightmap[0, y])])
    # all_points.extend([(-WIDTH / 2, HEIGHT / 2 - 1, 0), (-WIDTH / 2, -HEIGHT / 2, 0)])
    # all_faces.append(tuple(range(point_index, len(all_points))))

    # # Right side
    # point_index = len(all_points)
    # for y in range(HEIGHT):
    #     y_pos = y - HEIGHT / 2
    #     all_points.extend([(WIDTH / 2 - 1, y_pos, heightmap[WIDTH - 1, y])])
    # all_points.extend(
    #     [(WIDTH / 2 - 1, HEIGHT / 2 - 1, 0), (WIDTH / 2 - 1, -HEIGHT / 2, 0)]
    # )
    # all_faces.append(tuple(range(point_index, len(all_points))))

    # # Far side
    # point_index = len(all_points)
    # for x in range(WIDTH):
    #     x_pos = x - WIDTH / 2
    #     all_points.extend([(x_pos, HEIGHT / 2 - 1, heightmap[x, HEIGHT - 1])])
    # all_points.extend(
    #     [(WIDTH / 2 - 1, HEIGHT / 2 - 1, 0), (-WIDTH / 2, HEIGHT / 2 - 1, 0)]
    # )
    # all_faces.append(tuple(range(point_index, len(all_points))))

    # # Near side
    # point_index = len(all_points)
    # for x in range(WIDTH):
    #     x_pos = x - WIDTH / 2
    #     all_points.extend([(x_pos, -HEIGHT / 2, heightmap[x, 0])])
    # all_points.extend([(WIDTH / 2 - 1, -HEIGHT / 2, 0), (-WIDTH / 2, -HEIGHT / 2, 0)])
    # all_faces.append(tuple(range(point_index, len(all_points))))

    # # Create bottom
    # point_index = len(all_points)
    # all_points.extend(
    #     [
    #         (-WIDTH / 2, -HEIGHT / 2, 0),
    #         (WIDTH / 2 - 1, -HEIGHT / 2, 0),
    #         (WIDTH / 2 - 1, HEIGHT / 2 - 1, 0),
    #         (-WIDTH / 2, HEIGHT / 2 - 1, 0),
    #     ]
    # )
    # all_faces.append(tuple(range(point_index, len(all_points))))
    
    # walls_time = time.time() - start_time
    # print(f"Walls and bottom generation time: {walls_time:.4f} seconds")
    
    # Create a single polyhedron with all points and faces
    start_time = time.time()
    print(f"Final points count: {len(all_points)}, Final faces count: {len(all_faces)}")
    entire_terrain = Polyhedron(
        points=tuple(all_points),
        faces=tuple(all_faces),
    )
    
    # Create the final model
    # model = Workplane("XY").polyhedron(*entire_terrain)
    model = trimesh.Trimesh(vertices=np.array(all_points), faces=np.array(all_faces))
    polyhedron_time = time.time() - start_time
    print(f"Polyhedron creation time: {polyhedron_time:.4f} seconds")

    # Export the model
    start_time = time.time()
    # cq.exporters.export(model, "hello_world.stl")
    model.export("hello_world.stl")
    export_time = time.time() - start_time
    print(f"Export time: {export_time:.4f} seconds")
    
    # Total time
    total_time = time.time() - start_time_total
    print(f"\nTotal execution time: {total_time:.4f} seconds")
    
    # Disable profiling and print stats
    profiler.disable()
    
    # Sort the profiling stats by cumulative time
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Print top 20 functions by cumulative time
    
    # Also print by internal time (time spent in the function itself)
    print("\nStats by internal time:")
    stats.sort_stats(SortKey.TIME).print_stats(20)
