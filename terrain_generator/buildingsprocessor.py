from terrain_generator.buildingsextractor import Building

class BuildingsProcessor:
    def exclude_buildings_outside_bbox(self, buildings: list[Building], bbox: tuple[float, float, float, float]) -> list[Building]:
        return [building for building in buildings if building.is_inside_bbox(bbox)]