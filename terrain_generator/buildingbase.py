
from dataclasses import dataclass
from shapely import Polygon

@dataclass
class Building:
    """Represents a building extracted from OpenStreetMap with polygon and height data."""

    osm_id: int
    building_type: str
    polygon: Polygon
    area: float
    height: float

    def is_inside_bbox(self, bbox: tuple[float, float, float, float]) -> bool:
        """Check if the building is inside the given bbox."""
        min_lon, min_lat, max_lon, max_lat = bbox

        # Validate bbox: min values should be less than max values
        if min_lon >= max_lon or min_lat >= max_lat:
            return False

        return Polygon(
            [
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
            ]
        ).contains(self.polygon)
    
    @property
    def significant(self) -> bool:
        """Check if the building is significant."""
        return self.area > 17500 or self.height > 100
    
    def __hash__(self):
        return hash((self.osm_id, self.polygon.wkt))
