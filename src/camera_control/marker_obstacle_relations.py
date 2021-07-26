import math

from src.camera.util import base_marker_offset
from src.utils.obstacle import BoxObstacle
import numpy as np


RED_COLOR = [1, 0, 0, 1]
GREEN_COLOR = [0, 1, 0, 1]
BLUE_COLOR = [0, 0, 1, 1]

block_ids = [6, 7, 10, 11, 12, 13]

def get_rotation_around_z_from_marker(found_marker):
    matrix = found_marker.relative_rotation_matrix
    return math.atan2(matrix[1][0], matrix[0][0])


def get_obstacle_from_marker(found_marker):
    alpha = get_rotation_around_z_from_marker(found_marker)

    marker_id = found_marker.id
    x = float(found_marker.tvec[0] + base_marker_offset[0])
    y = float(found_marker.tvec[1] + base_marker_offset[1])
    z = 0.0
    location = np.array([x, y, z])

    if marker_id == 29:
        return BoxObstacle(np.array([13, 12, 35]), location, alpha=alpha, color=RED_COLOR)
    if marker_id == 17:
        return BoxObstacle(np.array([23, 8, 21]), location, alpha=alpha, color=RED_COLOR)
    if marker_id == 26:
        return BoxObstacle(np.array([19, 5, 15]), location, alpha=alpha, color=RED_COLOR)
    if marker_id == 27:
        return BoxObstacle(np.array([8, 16, 12]), location, alpha=alpha, color=RED_COLOR)
    elif marker_id in block_ids:
        return BoxObstacle(np.array([2.7, 2.7, 2.7]), location, alpha=alpha, color=GREEN_COLOR)
