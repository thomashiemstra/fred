from __future__ import division
from flask import jsonify

import time

from flask import Blueprint

from kinematics.kinematics_utils import RobotConfig
from movement_utils import point_to_point
from servo_handling.servo_controller import ServoController
from xbox_controller.xbox_utill import PosePoller

xbox_api = Blueprint('xbox_api', __name__)


@xbox_api.route('/start', methods=['POST'])
def start():
    pose_poller = PosePoller()
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)
    dynamixel_servo_controller = ServoController("COM5", dynamixel_robot_config)
    dynamixel_servo_controller.enable_servos()

    start_pose = dynamixel_servo_controller.get_current_pose()

    xbox_pose = start_pose.copy()
    xbox_pose.z += 5
    xbox_pose.reset_orientation()

    point_to_point(start_pose, xbox_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)

    try:
        while True:
            pose_poller.update_positions_of_pose(xbox_pose)

            dynamixel_servo_controller.move_to_pose(xbox_pose)

            time.sleep(pose_poller.dt)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        # TODO gracefully exit (move back, disable servos etc...)
        pose_poller.stop()


@xbox_api.route('/stop', methods=['POST'])
def stop():
    print("whoop")

    resp = jsonify(success=True)
    return resp
