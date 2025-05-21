import cadquery as cq
from cqmore import Workplane
from cqmore.polyhedron import Polyhedron
import numpy as np

WIDTH = 25
HEIGHT = 25
THICKNESS = 1

if __name__ == "__main__":
    # Create randomly generated heightmap
    heightmap = np.random.rand(WIDTH, HEIGHT) * 3 + THICKNESS

    # Create a 3D model from the heightmap
    model = Workplane("XY")

    for x in range(1, WIDTH - 1):
        for y in range(1, HEIGHT - 1):
            my_elevation = heightmap[x, y]

            # Create a prism with the height of the elevation, sloping the face such that
            # the height of the corner is the average of the height of the four neighboring
            # elevation squares.
            northeast_elevation = heightmap[x + 1, y + 1]
            northwest_elevation = heightmap[x - 1, y + 1]
            southeast_elevation = heightmap[x + 1, y - 1]
            southwest_elevation = heightmap[x - 1, y - 1]
            east_elevation = heightmap[x + 1, y]
            west_elevation = heightmap[x - 1, y]
            north_elevation = heightmap[x, y + 1]
            south_elevation = heightmap[x, y - 1]

            northeast_corner_top = (
                0.5,
                0.5,
                (northeast_elevation + east_elevation + north_elevation + my_elevation)
                / 4,
            )
            northwest_corner_top = (
                -0.5,
                0.5,
                (northwest_elevation + west_elevation + north_elevation + my_elevation)
                / 4,
            )
            southeast_corner_top = (
                0.5,
                -0.5,
                (southeast_elevation + east_elevation + south_elevation + my_elevation)
                / 4,
            )
            southwest_corner_top = (
                -0.5,
                -0.5,
                (southwest_elevation + west_elevation + south_elevation + my_elevation)
                / 4,
            )

            northeast_corner_bottom = (0.5, 0.5, 0)
            northwest_corner_bottom = (-0.5, 0.5, 0)
            southeast_corner_bottom = (0.5, -0.5, 0)
            southwest_corner_bottom = (-0.5, -0.5, 0)

            my_top = (0, 0, my_elevation)

            polyhedron = Polyhedron(
                points=(
                    northeast_corner_top,
                    northwest_corner_top,
                    southwest_corner_top,
                    southeast_corner_top,
                    northeast_corner_bottom,
                    northwest_corner_bottom,
                    southwest_corner_bottom,
                    southeast_corner_bottom,
                    my_top,
                ),
                faces=(
                    (0, 1, 8),
                    (1, 2, 8),
                    (2, 3, 8),
                    (3, 0, 8),
                    (0, 1, 5, 4),
                    (1, 2, 6, 5),
                    (2, 3, 7, 6),
                    (3, 0, 4, 7),
                    (4, 5, 6, 7),
                ),
            )

            prism = Workplane("XY").polyhedron(*polyhedron).consolidateWires()

            # Add the prism at the correct position
            pos_x = x - WIDTH / 2
            pos_y = y - HEIGHT / 2
            model.add(prism.translate((pos_x, pos_y, 0)))

    cq.exporters.export(model, "hello_world.stl")
