import inspect
import os

import pybullet as p

from src import global_constants
from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController


def start_simulated_robot(use_gui=False):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    if use_gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -10, physicsClientId=physics_client)
    p.loadURDF(current_dir + "/urdf/plane.urdf", physicsClientId=physics_client)

    if use_gui:
        p.setRealTimeSimulation(1, physicsClientId=physics_client)

    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    body_id = p.loadURDF(current_dir + "/urdf/fred_with_spheres.urdf", start_pos, start_orientation,
                         physicsClientId=physics_client)

    return SimulatedRobotController(global_constants.simulated_robot_config, physics_client, body_id)

