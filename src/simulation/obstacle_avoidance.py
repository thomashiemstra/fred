import inspect
import os

import pybullet as p
from functools import lru_cache

from src.global_constants import simulated_robot_config
from src.kinematics.kinematics import jacobian_transpose_on_f
from src.kinematics.kinematics_utils import Pose
from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController
import numpy as np
from time import sleep

# I don't know why the spheres are reversed in the urdf
sphere_1_id = 8  # in between frame 3 and the wrist
sphere_2_id = 7  # wrist (frame 4)
sphere_3_id = 6  # tip of the gripper
sphere_ids = np.array([sphere_1_id, sphere_2_id, sphere_3_id])  # todo rename to control points or something

total_control_points = 3
attractive_cutoff_distance = 2
repulsive_cutoff_distance = 1
angle_update = 0.005
control_point_1_position = 11.2
control_point_attractive_weights = np.array([1, 1, 1])
control_point_repulsive_weights = np.array([1, 1, 1])

control_point_radii = np.array([5, 5, 3])
control_point_base_radius = 1


def get_target_points(pose, d6):
    x_3, y_3, z_3 = pose.x, pose.y, pose.z
    point_3 = np.array([x_3, y_3, z_3])

    t = pose.get_euler_matrix()
    x_2 = x_3 - d6 * t[0, 2]
    y_2 = y_3 - d6 * t[1, 2]
    z_2 = z_3 - d6 * t[2, 2]
    point_2 = np.array([x_2, y_2, z_2])

    point_1 = None  # we only need to points for the attractive force

    return point_1, point_2, point_3


def get_control_point_pos(robot_body_id, point_id):
    _, _, _, _, pos, _ = p.getLinkState(robot_body_id, point_id)
    return np.array(pos) * 100  # convert from meters to centimeters


def get_attractive_force_world(control_points, target_points, d, weights=None):
    if control_points.size != target_points.size:
        raise Exception("control points and target points should have the same dimension!")
    num_points = control_points.shape[0]

    if weights is None:
        weights = np.ones(num_points)
    workspace_forces = np.zeros((total_control_points, 3))

    total_distance = 0

    for i in range(num_points):
        vector = control_points[i] - target_points[i]  # the vector from control point to target point
        distance = np.linalg.norm(vector)
        if distance == 0:
            pass
        elif distance > d:
            workspace_forces[i][0] = -d * weights[i] * vector[0] / distance
            workspace_forces[i][1] = -d * weights[i] * vector[1] / distance
            workspace_forces[i][2] = -d * weights[i] * vector[2] / distance
        elif distance < d:
            workspace_forces[i][0] = -weights[i] * vector[0]
            workspace_forces[i][1] = -weights[i] * vector[1]
            workspace_forces[i][2] = -weights[i] * vector[2]
        total_distance += distance

    return workspace_forces, total_distance


@lru_cache(maxsize=32)
def get_normal_and_distance(robot_body_id, obstacle_id, control_point_id, physics_client_id):
    _, _, _, _, _, _, _, normal_on_b, d, *x = p.getClosestPoints(bodyA=robot_body_id, bodyB=obstacle_id,
                                                                 linkIndexA=control_point_id,
                                                                 distance=repulsive_cutoff_distance,
                                                                 physicsClientId=physics_client_id)[0]
    return normal_on_b, d*100 + control_point_base_radius


def get_repulsive_forces_world(robot_body_id, control_point_ids, obstacle_ids, physics_client_id):
    workspace_forces = np.zeros((total_control_points, 3))

    get_normal_and_distance.cache_clear()

    for i in range(control_point_ids.size):
        control_point_id = control_point_ids[i]
        smallest_distance = repulsive_cutoff_distance  # anything further away should not be considered
        closest_obstacle_id = -1

        for obstacle_id in obstacle_ids:
            normal, d = get_normal_and_distance(robot_body_id, obstacle_id, control_point_id, physics_client_id)
            distance = d - control_point_radii[i]
            if distance < smallest_distance:
                smallest_distance = distance
                closest_obstacle_id = obstacle_id

        if smallest_distance < repulsive_cutoff_distance:
            normal_on_b, d = get_normal_and_distance(robot_body_id, closest_obstacle_id, control_point_id, physics_client_id)
            distance = d - control_point_radii[i]
            constant_term = control_point_repulsive_weights[i]*(1/distance - 1/repulsive_cutoff_distance)*(1/(distance*distance))

            workspace_forces[i][0] += constant_term * normal_on_b[0]
            workspace_forces[i][1] += constant_term * normal_on_b[1]
            workspace_forces[i][2] += constant_term * normal_on_b[2]

    return workspace_forces


def draw_debug_lines(physics_client_id, robot_id, control_point_ids, attr_forces, rep_forces, attr_lines, rep_lines):
    number_of_control_points = control_point_ids.size

    new_attr_lines = [None] * number_of_control_points
    new_rep_lines = [None] * number_of_control_points

    for i in range(number_of_control_points):
        control_point_pos = get_control_point_pos(robot_id, control_point_ids[i])
        attr_target = control_point_pos + 4*attr_forces[i]
        rep_target = control_point_pos + 4*rep_forces[i]

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


def create_visual_sphere(body_id, location, physics_client):
    target_sphere_1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01, physicsClientId=physics_client)
    target_sphere_1_id = p.createMultiBody(0, target_sphere_1, -1, location/100, [0, 0, 0, 1], physicsClientId=physics_client)
    p.changeVisualShape(target_sphere_1_id, -1, rgbaColor=[1, 1, 0, 1], physicsClientId=physics_client)

    p.setCollisionFilterPair(body_id, target_sphere_1_id, 2, -1, 0)
    p.setCollisionFilterPair(body_id, target_sphere_1_id, 3, -1, 0)
    p.setCollisionFilterPair(body_id, target_sphere_1_id, 4, -1, 0)
    p.setCollisionFilterPair(body_id, target_sphere_1_id, 5, -1, 0)
    p.setCollisionFilterPair(body_id, target_sphere_1_id, 6, -1, 0)
    p.setCollisionFilterPair(body_id, target_sphere_1_id, 7, -1, 0)
    p.setCollisionFilterPair(body_id, target_sphere_1_id, 8, -1, 0)


physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -10)
# planeId = p.loadURDF("urdf/plane.urdf")
p.setRealTimeSimulation(1)

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
start_pos = [0, 0, 0]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
body_id = p.loadURDF(current_dir + "/urdf/fred_with_spheres.urdf", start_pos, start_orientation,
                     physicsClientId=physics_client)

simulated_robot = SimulatedRobotController(simulated_robot_config, physics_client, body_id)

collision_box_id_1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2], physicsClientId=physics_client)
floor = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.1], physicsClientId=physics_client)


box1 = p.createMultiBody(0, collision_box_id_1, -1, [0, 0.35, 0.1], [0, 0, 0, 1], physicsClientId=physics_client)
floor_id = p.createMultiBody(0, floor, -1, [0, 0.0, -0.1], [0, 0, 0, 1], physicsClientId=physics_client)

p.changeVisualShape(box1, -1, rgbaColor=[0, 1, 0, 1], physicsClientId=physics_client)
p.changeVisualShape(floor_id, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=physics_client)

obstacles = np.array([box1, floor_id])

arc_1 = Pose(-25, 20, 10)
arc_2 = Pose(25, 20, 7.1)


simulated_robot.reset_to_pose(arc_1)

_, target_point_2, target_point_3 = get_target_points(arc_2, simulated_robot.robot_config.d6)

create_visual_sphere(body_id, target_point_2, physics_client)
create_visual_sphere(body_id, target_point_3, physics_client)


current_angles = simulated_robot.pose_to_angles(arc_1)

done = False
attr_lines = None
rep_lines = None
while not done:
    zero = np.zeros(3)  # the first control point is not used to determine the attractive force right now,
    # mainly because getting the position of the target  point for this one is annoying
    c2_pos = get_control_point_pos(body_id, sphere_2_id)
    c3_pos = get_control_point_pos(body_id, sphere_3_id)

    attractive_forces, total_distance = get_attractive_force_world(np.array([zero, c2_pos, c3_pos]),
                                                        np.array([zero, target_point_2, target_point_3]),
                                                        attractive_cutoff_distance,
                                                        weights=control_point_attractive_weights)

    repulsive_forces = get_repulsive_forces_world(body_id, sphere_ids, obstacles, physics_client)

    attr_lines, rep_lines = draw_debug_lines(physics_client, body_id,
                                             sphere_ids, attractive_forces, repulsive_forces, attr_lines, rep_lines)

    joint_forces = jacobian_transpose_on_f(attractive_forces + repulsive_forces, current_angles,
                                           simulated_robot.robot_config, control_point_1_position)

    absolute_force = np.linalg.norm(joint_forces)

    current_angles += angle_update * (joint_forces/absolute_force)

    simulated_robot.move_servos(current_angles)
    sleep(0.01)
