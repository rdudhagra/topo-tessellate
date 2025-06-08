from terrain_generator.buildingbase import Building
from terrain_generator.elevation import Elevation
import numpy as np
import meshlib.mrmeshpy as mr
from terrain_generator.console import output


class BuildingsGenerator:
    def __init__(self, elevation: Elevation):
        self.elevation = elevation
        self.pool_context = {}

    def generate_buildings(
        self,
        base_height: float,
        elevation_data: np.ndarray,
        elevation_multiplier: float,
        bounds: tuple[float, float, float, float],
        buildings: list[Building],
        min_building_height: float = 25,
    ):
        """
        Generate buildings in a separate mesh

        Args:
            base_height (float): The height of the base mesh
            elevation_data (np.ndarray): The elevation data
            elevation_multiplier (float): The elevation multiplier
            bounds (tuple[float, float, float, float]): The bounds of the terrain
            buildings (list[Building]): The buildings to generate
            min_building_height (float, optional): The minimum height of a building. Defaults to 25.
        """

        width_meters, height_meters = self.elevation.calculate_bounds_dimensions_meters(
            bounds
        )

        all_chunks_meshes = mr.std_vector_std_shared_ptr_Mesh()

        # Chunk the buildings into 1000 buildings at a time
        buildings_chunks = [
            buildings[i : i + 1000] for i in range(0, len(buildings), 1000)
        ]

        for i, building_chunk in enumerate(buildings_chunks):
            output.progress_info(f"Generating building chunk {i+1}/{len(buildings_chunks)}")

            this_chunk_meshes = mr.std_vector_std_shared_ptr_Mesh()

            for building in building_chunk:
                # First, convert the coordinates to model coordinates
                points = np.array(
                    self.elevation.get_model_coordinates(
                        elevation_data, bounds, building.polygon.exterior.coords
                    )
                )

                # Order the points in a clockwise order
                points = self._order_points_clockwise(points)
                points_indices = [
                    (
                        int(p[0] / width_meters * elevation_data.shape[1]),
                        int(p[1] / height_meters * elevation_data.shape[0]),
                    )
                    for p in points
                ]

                # If points_indices are out of bounds, skip the building
                if any(
                    p[0] < 0
                    or p[0] >= elevation_data.shape[1]
                    or p[1] < 0
                    or p[1] >= elevation_data.shape[0]
                    for p in points_indices
                ):
                    continue

                # Compute the minimum elevation of the building's foundation
                foundation_elevation = np.min(
                    [elevation_data[p[1], p[0]] for p in points_indices]
                )

                # Compute the height of the building
                height = max(building.height * elevation_multiplier, min_building_height)

                # Create a mesh for the building
                contours = mr.std_vector_std_vector_Vector2f()
                contours.resize(1)
                for x, y in points:
                    contours[0].append(mr.Vector2f(x, y))
                # Add the first point again to close the contour
                contours[0].append(contours[0][0])

                building_mesh = mr.triangulateContours(contours)

                # Move the building mesh to the correct position
                building_mesh.transform(
                    mr.AffineXf3f.translation(
                        mr.Vector3f(0, 0, height + foundation_elevation)
                    )
                )

                # Add the base to the building mesh
                mr.addBaseToPlanarMesh(building_mesh, zOffset=-height)

                # Add the building mesh to the chunk meshes
                this_chunk_meshes.append(building_mesh)

            # Merge the chunk meshes
            chunk_mesh = mr.mergeMeshes(this_chunk_meshes)

            # Add the chunk mesh to the buildings mesh group
            all_chunks_meshes.append(chunk_mesh)

        # Merge the chunks
        output.progress_info(f"Merging chunks...")
        all_chunks_mesh = mr.mergeMeshes(all_chunks_meshes)

        # Move the buildings mesh group to the correct position
        all_chunks_mesh.transform(
            mr.AffineXf3f.translation(
                mr.Vector3f(0, 0, base_height)
            )
        )

        return all_chunks_mesh

    def _order_points_clockwise(self, points):
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]
