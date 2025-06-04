from terrain_generator.buildingsextractor import Building
from terrain_generator.elevation import Elevation
import numpy as np
import meshlib.mrmeshpy as mr


class BuildingsGenerator:
    def __init__(self, elevation: Elevation):
        self.elevation = elevation

    def generate_buildings(
        self,
        mesh: mr.Mesh,
        elevation_data: np.ndarray,
        bounds: tuple[float, float, float, float],
        buildings: list[Building],
    ) -> mr.Mesh:
        buildings = list(filter(lambda b: b.osm_id == 431972186, buildings))

        for building in buildings:
            # Create a mesh for the building

            # First, convert the coordinates to model coordinates
            points = self.elevation.get_model_coordinates(
                elevation_data, bounds, building.coordinates
            )

            print(points)
            import matplotlib.pyplot as plt

            plt.scatter([p[0] for p in points], [p[1] for p in points])
            plt.show()
