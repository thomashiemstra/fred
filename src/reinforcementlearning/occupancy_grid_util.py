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


def get_height_tallest_obstacle(obstacles):
    tallest_height = 0
    for obstacle in obstacles:
        obstacle_height = obstacle.dimensions[2]
        if obstacle_height > tallest_height:
            tallest_height = obstacle_height
    return tallest_height


# Return all the obstacles that are overlapping the point
def get_obstacles_for_point(point_location, obstacles):
    res = []
    for obstacle in obstacles:
        if is_point_in_obstacle(point_location, obstacle.base_center_position, obstacle.half_extends, obstacle.alpha):
            res.append(obstacle)
    return np.array(res)


def create_grid_from_obstacles(obstacles, grid_len_x=60, grid_len_y=40, grid_size=1):
    if grid_len_x % grid_size != 0:
        raise ValueError("grid_len_x has to be a multiple of grid_size")
    if grid_len_y % grid_size != 0:
        raise ValueError("grid_len_y has to be a multiple of grid_size")

    scaled_x = np.ceil(grid_len_x/grid_size)
    scaled_y = np.ceil(grid_len_y/grid_size)

    grid = np.zeros((scaled_x.astype(int), scaled_y.astype(int)))

    for grid_x in range(0, grid.shape[0]):
        for grid_y in range(0, grid.shape[1]):
            x_coordinate = grid_x*grid_size + grid_size/2 - grid_len_x/2
            y_coordinate = grid_y*grid_size + grid_size/2
            point_location = np.array([x_coordinate, y_coordinate])
            intersecting_obstacles = get_obstacles_for_point(point_location, obstacles)
            if intersecting_obstacles.size != 0:
                height = get_height_tallest_obstacle(intersecting_obstacles)
                grid[grid_x][grid_y] = 0.5

            print("indices: ({}, {}) coordinates: ({}, {}), intersection_obs: {}".format(grid_x, grid_y, x_coordinate, y_coordinate, intersecting_obstacles))

    return grid


if __name__ == '__main__':
    import pybullet as p

    from src.utils.obstacle import BoxObstacle

    physics_client = p.connect(p.GUI)

    floor = BoxObstacle(physics_client, np.array([60, 40, 1]), np.array([0, 20, -1]))

    obs_1 = BoxObstacle(physics_client, np.array([10, 20, 10]), np.array([10, 20, 0]), alpha=np.pi/4)
    obs_2 = BoxObstacle(physics_client, np.array([10, 20, 10]), np.array([-10, 20, 0]), alpha=-np.pi / 4)

    grid = create_grid_from_obstacles(np.array([obs_1, obs_2]))

    import matplotlib.pyplot as plt

    plt.set_cmap('hot')

    plt.imshow(grid)
    plt.show()

