import inspect
import os

import numpy as np
import pybullet as p

from src import global_constants
from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController
from numpy import pi


def start_simulated_robot(use_gui=False, robot_config=global_constants.simulated_robot_config):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    if use_gui:
        physics_client = p.connect(p.GUI, options="--width=1920 --height=1080")
    else:
        physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF(current_dir + "/urdf/plane.urdf", physicsClientId=physics_client)

    if use_gui:
        p.setRealTimeSimulation(1, physicsClientId=physics_client)
        p.resetDebugVisualizerCamera(1, -40, -40, cameraTargetPosition=[-0.2, -0.1, 0.5])

    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    body_id = p.loadURDF(current_dir + "/urdf/fred_with_spheres_new.urdf", start_pos, start_orientation,
                         physicsClientId=physics_client)

    robot = SimulatedRobotController(robot_config, physics_client, body_id)
    robot.reset_to_angels([0, pi/2, pi/2, 0, 0, 0, 0])

    return SimulatedRobotController(robot_config, physics_client, body_id)


class ControlPoint:

    def __init__(self, point_id, radius, body_id, physics_client, weight=1):
        self.point_id = point_id
        self.body_id = body_id
        self.radius = radius
        self._physics_client = physics_client
        self._position = None
        self.weight = weight

    @property
    def position(self):
        self._update_position()
        return self._position

    def _update_position(self):
        """
        Function used to sync the position of the control point with the robot,
        needs to be called before getting the position of the control point
        """
        _, _, _, _, pos, _ = p.getLinkState(self.body_id, self.point_id, physicsClientId=self._physics_client)
        self._position = np.array(pos) * 100  # convert from meters to centimeters


def generate_control_points(body_id, physics_client):
    c1 = ControlPoint(8, 6, body_id, physics_client)   # in between frame 3 and the wrist
    c2 = ControlPoint(7, 6, body_id, physics_client, weight=2)  # wrist (frame 4)
    c3 = ControlPoint(6, 4, body_id, physics_client)  # tip of the gripper
    return c1, c2, c3