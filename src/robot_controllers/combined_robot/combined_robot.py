from src.kinematics.kinematics import inverse_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController


class CombinedRobot(AbstractRobotController):

    def __init__(self, dynamixel_robot, simulated_robot):
        self.dynamixel_robot = dynamixel_robot
        self.simulated_robot = simulated_robot

        self.robot_config = dynamixel_robot.robot_config
        self.physics_client = simulated_robot.physics_client
        self.body_id = simulated_robot.body_id
        self.control_points = simulated_robot.control_points

    def enable_servos(self):
        self.dynamixel_robot.enable_servos()
        self.simulated_robot.enable_servos()

    def disable_servos(self):
        self.dynamixel_robot.disable_servos()
        self.simulated_robot.disable_servos()

    def move_to_pose(self, pose):
        self.simulated_robot.move_to_pose(pose)
        return self.dynamixel_robot.move_to_pose(pose)

    def move_to_pose_and_give_new_angles(self, pose):
        self.simulated_robot.move_to_pose_and_give_new_angles(pose)
        return self.dynamixel_robot.move_to_pose_and_give_new_angles(pose)

    def move_servos(self, angles):
        self.simulated_robot.move_servos(angles)
        return self.dynamixel_robot.move_servos(angles)

    def get_current_angles(self):
        return self.dynamixel_robot.get_current_angles()

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.dynamixel_robot.robot_config)

    def set_gripper(self, new_gripper_state):
        self.dynamixel_robot.set_gripper(new_gripper_state)

    def reset_to_pose(self, pose):
        self.simulated_robot.reset_to_pose(pose)
        self.dynamixel_robot.reset_to_pose(pose)

    def set_servo_1_and_2_low_current(self):
        self.dynamixel_robot.set_servo_1_and_2_low_current()

    def set_servo_1_and_2_full_current(self):
        self.dynamixel_robot.set_servo_1_and_2_full_current()