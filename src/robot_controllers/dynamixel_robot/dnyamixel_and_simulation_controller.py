from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
from src.simulation.simulation_utils import start_simulated_robot


class DynamixelAndSimulationController(AbstractRobotController):

    def __init__(self, port, robot_config):
        self.dynamixel_controller = DynamixelRobotController(port, robot_config)
        self.simulation_controller = start_simulated_robot()
        self.use_dynamixel_robot = False

    def move_to_pose(self, pose):
        self.simulation_controller.move_to_pose(pose)
        if self.use_dynamixel_robot:
            self.dynamixel_controller.move_to_pose(pose)

    def move_servos(self, angles):
        self.simulation_controller.move_servos(angles)
        if self.use_dynamixel_robot:
            self.dynamixel_controller.move_servos(angles)

    def get_current_angles(self):
        if self.use_dynamixel_robot:
            return self.dynamixel_controller.get_current_angles()
        else:
            return self.simulation_controller.get_current_angles()

    def pose_to_angles(self, pose):
        if self.use_dynamixel_robot:
            return self.dynamixel_controller.pose_to_angles(pose)
        else:
            return self.simulation_controller.pose_to_angles()

    def set_gripper(self, new_gripper_state):
        if self.use_dynamixel_robot:
            self.dynamixel_controller.set_gripper(new_gripper_state)

    def use_dynamixel_robot(self, use_dynamixel_robot):
        """
        enable or disable the use of the phisical robot
        :param use_dynamixel_robot: boolean
        """
        self.use_dynamixel_robot = use_dynamixel_robot
