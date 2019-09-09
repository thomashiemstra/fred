import inspect
import os

import pybullet as p

from src.global_constants import simulated_robot_config
from src.kinematics.kinematics_utils import Pose
from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController
from src.xbox_control.xbox_robot_controller import create_move

physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("urdf/plane.urdf")
p.setRealTimeSimulation(1)


collision_box_id_1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2], physicsClientId=physics_client)
box1 = p.createMultiBody(0, collision_box_id_1, -1, [-0.4, 0.1, 0.1], [0, 0, 0, 1], physicsClientId=physics_client)

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
start_pos = [0, 0, 0]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])

body_id = p.loadURDF(current_dir + "/urdf/fred_with_spheres.urdf", start_pos, start_orientation,
                         physicsClientId=physics_client)

simulated_robot = SimulatedRobotController(simulated_robot_config, physics_client, body_id)
robot_body_id = simulated_robot.body_id

arc_1 = Pose(-25, 30, 25)
arc_2 = Pose(0, 30, 25)
arc_3 = Pose(25, 30, 25)

poses = [arc_1, arc_2, arc_3]

move = create_move(simulated_robot, poses, 15, None, None)
simulated_robot.reset_to_pose(arc_1)
_, _, _, _, pos, orientation_quaterions = p.getLinkState(robot_body_id, 7)

print(pos)

_, body_a, body_b, link_index_a, link_index_b, pos_on_a, pos_on_b, normal_on_b, distance, *x = p.getClosestPoints(bodyA=simulated_robot.body_id, bodyB=box1, linkIndexA=6, distance=10, physicsClientId=physics_client)[0]

print(normal_on_b, distance)

print(distance)
# # move.go_to_start_of_move(simulated_robot, 2)
while True:
    move.move(simulated_robot)
    move.move_reversed(simulated_robot)
