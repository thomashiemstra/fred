import inspect
import os

import pybullet as p

from src import global_constants
from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController


def start_simulation():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.loadURDF(currentdir + "/urdf/plane.urdf")
    p.setRealTimeSimulation(1)

    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    body_id = p.loadURDF(currentdir + "/urdf/fred_with_spheres.urdf", start_pos, start_orientation, physicsClientId=physics_client)

    return SimulatedRobotController(global_constants.simulated_robot_config, physics_client, body_id)

