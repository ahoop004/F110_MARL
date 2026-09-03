import numpy as np
import pytest
from scipy.spatial import cKDTree

from utils.centerline import prepare_centerline_geometry, project_to_centerline
from utils.map_loader import MapLoader


def test_circle_map_width_and_direction_contract() -> None:
    map_data = MapLoader().load(
        {
            "map_dir": "maps",
            "map_bundle": "circle_map",
            "centerline_autoload": True,
            "walls_autoload": True,
        }
    )
    assert map_data.centerline is not None
    assert map_data.walls is not None

    centerline = map_data.centerline[:, :2]
    wall_distances = [
        cKDTree(wall).query(centerline)[0] for wall in map_data.walls.values()
    ]
    assert len(wall_distances) == 2
    for distances in wall_distances:
        assert np.median(distances) == pytest.approx(1.35, abs=0.02)

    geometry = prepare_centerline_geometry(map_data.centerline)
    annotations = map_data.metadata["annotations"]
    for spawn in annotations["spawn_points"]:
        pose = np.asarray(spawn["pose"], dtype=np.float32)
        projection = project_to_centerline(geometry, pose[:2], float(pose[2]))
        assert abs(projection.heading_error) < 0.05

    finish = annotations["finish_line"]
    midpoint = 0.5 * (
        np.asarray(finish["start"], dtype=np.float32)
        + np.asarray(finish["end"], dtype=np.float32)
    )
    heading = float(np.arctan2(finish["direction"][1], finish["direction"][0]))
    projection = project_to_centerline(geometry, midpoint, heading)
    assert abs(projection.heading_error) < 0.05
