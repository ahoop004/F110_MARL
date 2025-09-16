# map_utils.py
"""
Minimal utility: compute x/y bounds from a ROS map yaml + image.
"""

from pathlib import Path
import math
import yaml
from PIL import Image
import numpy as np


def get_map_bounds(yaml_path: str):
    """
    Compute metric x/y bounds from a ROS map YAML.

    Args:
        yaml_path (str): path to the .yaml map file

    Returns:
        dict with keys: x_min, x_max, y_min, y_max
    """
    yaml_file = Path(yaml_path)
    with yaml_file.open("r") as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    ox, oy, oyaw = [float(v) for v in meta["origin"]]
    image_field = meta["image"]

    img_path = Path(image_field)
    if not img_path.is_absolute():
        img_path = (yaml_file.parent / img_path).resolve()

    with Image.open(str(img_path)) as im:
        width_px, height_px = im.size

    # corners in pixel space, scaled to meters
    corners = np.array(
        [
            [0.0, 0.0],
            [width_px * resolution, 0.0],
            [width_px * resolution, height_px * resolution],
            [0.0, height_px * resolution],
        ],
        dtype=np.float64,
    )

    if abs(oyaw) > 1e-9:
        c, s = math.cos(oyaw), math.sin(oyaw)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T

    corners[:, 0] += ox
    corners[:, 1] += oy

    return {
        "x_min": float(np.min(corners[:, 0])),
        "x_max": float(np.max(corners[:, 0])),
        "y_min": float(np.min(corners[:, 1])),
        "y_max": float(np.max(corners[:, 1])),
    }
