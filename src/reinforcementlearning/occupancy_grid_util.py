import numpy as np


def is_point_in_obstacle(point_location, obstacle_location, obstacle_half_extends, obstacle_alpha):
    obstacle_to_point = point_location[0:2] - obstacle_location[0:2]

    c, s = np.cos(obstacle_alpha), np.sin(obstacle_alpha)
    rotation_matrix_inv = np.array([[c, s], [-s, c]])

    # Get the vector from obstacle to point expressed in the coordinate frame of the obstacle
    vector = np.dot(rotation_matrix_inv, obstacle_to_point)
    x, y = vector[0], vector[1]

    half_extend_x, half_extend_y = obstacle_half_extends[0], obstacle_half_extends[1]

    return np.abs(x) <= half_extend_x and np.abs(y) <= half_extend_y


# todo
def create_grid_from_obstacles(obstacles, grid_len_x=60, grid_len_y=40, grid_size=2):
    scaled_x = np.ceil(grid_len_x/grid_size)
    scaled_y = np.ceil(grid_len_y/grid_size)

    grid = np.zeros((scaled_x.astype(int), scaled_y.astype(int)))
    grid[10][0] = 1
    return grid

