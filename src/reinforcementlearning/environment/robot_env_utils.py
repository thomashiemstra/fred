import numpy as np
import pybullet as p

# Ids of the control points in the urdf robot
sphere_1_id = 8  # in between frame 3 and the wrist
sphere_2_id = 7  # wrist (frame 4)
sphere_3_id = 6  # tip of the gripper
control_point_ids = np.array([sphere_1_id, sphere_2_id, sphere_3_id])  # todo convert control points to class

control_point_base_radius = 1
repulsive_cutoff_distance = 2


def get_target_points(target_pose, d6):
    """gives the target 3d location for the control points on the robot
    point_3 is the tip of the gripper
    point_2 is the origin of frame 6
    point_1 is only used to calculate repulsive forces

    Args:
      target_pose: A Pose specifying the target pose for the robot
      d6: the length of the gripper

    Returns:
      a 3 vector [x,y,z] for each control point on the robot
    """
    x_3, y_3, z_3 = target_pose.x, target_pose.y, target_pose.z
    point_3 = np.array([x_3, y_3, z_3])

    t = target_pose.get_euler_matrix()
    x_2 = x_3 - d6 * t[0, 2]
    y_2 = y_3 - d6 * t[1, 2]
    z_2 = z_3 - d6 * t[2, 2]
    point_2 = np.array([x_2, y_2, z_2])

    point_1 = None  # we only need two points for the attractive force

    return point_1, point_2, point_3


def get_attractive_force_world(control_points, target_points, attractive_cutoff_distance=2, weights=None):
    """Calculates the vectors pointing from the control points on the robot to the target points
    The vectors will have a norm of max attractive_cutoff_distance

    Args:
        control_points: An array of the current 3d positions of every control point on the robot
        target_points: An array of the 3d positions of the target points
        attractive_cutoff_distance: Distance after which the vector will start to decrease
        weights: a 3 vector of relative importance of the control points.
    Returns:
        workspace_forces: a 3x3 matrix for every control point a vector
                          pointing to the target position of the control point
        total_distance: the sum of distances between the control points and their targets,
                        can be used to determine if the robot has reached the target position
    """
    number_of_control_points = control_points.shape[0]

    if number_of_control_points != target_points.shape[0]:
        raise Exception("control points and target points should have the same dimension!")
    number_of_control_points = control_points.shape[0]

    if weights is None:
        weights = np.ones(number_of_control_points)
    workspace_forces = np.zeros((number_of_control_points, 3))

    total_distance = 0

    for control_point_id in range(number_of_control_points):
        if control_points[control_point_id] is None or target_points[control_point_id] is None:
            continue
        vector = control_points[control_point_id] - target_points[control_point_id]
        distance = np.linalg.norm(vector)
        if distance == 0:
            pass
        elif distance > attractive_cutoff_distance:
            workspace_forces[control_point_id][0] = -attractive_cutoff_distance * weights[control_point_id] * vector[0] / distance
            workspace_forces[control_point_id][1] = -attractive_cutoff_distance * weights[control_point_id] * vector[1] / distance
            workspace_forces[control_point_id][2] = -attractive_cutoff_distance * weights[control_point_id] * vector[2] / distance
        elif distance <= attractive_cutoff_distance:
            workspace_forces[control_point_id][0] = -weights[control_point_id] * vector[0]
            workspace_forces[control_point_id][1] = -weights[control_point_id] * vector[1]
            workspace_forces[control_point_id][2] = -weights[control_point_id] * vector[2]
        total_distance += distance

    return workspace_forces, total_distance


def get_normal_and_distance(robot_body_id, obstacle_id, control_point_id, physics_client_id):
    _, _, _, _, _, _, _, normal_on_b, d, *x = p.getClosestPoints(bodyA=robot_body_id, bodyB=obstacle_id,
                                                                 linkIndexA=control_point_id,
                                                                 distance=repulsive_cutoff_distance,
                                                                 physicsClientId=physics_client_id)[0]
    return normal_on_b, d * 100


def get_repulsive_forces_world(robot_body_id, control_points, obstacle_ids, physics_client_id):
    nr_of_control_points = control_points.shape[0]
    workspace_forces = np.zeros((3, nr_of_control_points))

    for i in range(nr_of_control_points):
        control_point = control_points[i]
        control_point_id = control_point.point_id
        smallest_distance = repulsive_cutoff_distance  # anything further away should not be considered

        closest_obstacle_normal = None
        closest_obstacle_distance = None

        for obstacle_id in obstacle_ids:
            normal, d = get_normal_and_distance(robot_body_id, obstacle_id, control_point_id, physics_client_id)
            distance = d - control_point.radius  # todo handle zero or negative distance
            if distance < smallest_distance:
                closest_obstacle_normal = normal
                closest_obstacle_distance = d
                smallest_distance = distance

        if smallest_distance < repulsive_cutoff_distance:
            distance = closest_obstacle_distance - control_point.radius
            constant_term = control_point.weight * (1 / distance - 1 / repulsive_cutoff_distance) * (1 / (distance * distance))

            workspace_forces[i][0] += constant_term * closest_obstacle_normal[0]
            workspace_forces[i][1] += constant_term * closest_obstacle_normal[1]
            workspace_forces[i][2] += constant_term * closest_obstacle_normal[2]

    return workspace_forces


def get_control_point_pos(robot_body_id, point_id):
    """Get the 3d position of a control point on the robot

    Args:
        robot_body_id: the id of the simulated robot arm
        point_id: the id of the control point on the robot arm
    Returns: A 3d array of the coordinates of the control point
    """
    _, _, _, _, pos, _ = p.getLinkState(robot_body_id, point_id)
    return np.array(pos) * 100  # convert from meters to centimeters


def draw_debug_lines(physics_client_id, control_points, attr_forces, rep_forces, attr_lines, rep_lines, line_size=4):
    number_of_control_points = control_points.shape[0]

    new_attr_lines = [None] * number_of_control_points
    new_rep_lines = [None] * number_of_control_points

    for i in range(number_of_control_points):
        control_point_pos = control_points[i].position
        attr_target = control_point_pos + line_size*attr_forces[i]
        rep_target = control_point_pos + line_size*rep_forces[i]

        if attr_lines is None:
            new_attr_lines[i] = p.addUserDebugLine(control_point_pos / 100, attr_target / 100, lineColorRGB=[0, 1, 0],
                                                   lineWidth=3, physicsClientId=physics_client_id)
        else:
            new_attr_lines[i] = p.addUserDebugLine(control_point_pos / 100, attr_target / 100, lineColorRGB=[0, 1, 0],
                                                   lineWidth=3, replaceItemUniqueId=attr_lines[i],
                                                   physicsClientId=physics_client_id)
        if rep_lines is None:
            new_rep_lines[i] = p.addUserDebugLine(control_point_pos / 100, rep_target / 100, lineColorRGB=[1, 0, 0],
                                                  lineWidth=3, physicsClientId=physics_client_id)
        else:
            new_rep_lines[i] = p.addUserDebugLine(control_point_pos / 100, rep_target / 100, lineColorRGB=[1, 0, 0],
                                                  lineWidth=3, replaceItemUniqueId=rep_lines[i],
                                                  physicsClientId=physics_client_id)
    return new_attr_lines, new_rep_lines