from __future__ import division

import threading

from src import global_constants
from src.kinematics.kinematics_utils import Pose
import numpy as np
from numpy import pi
from src.global_constants import WorkSpaceLimits


# class used to update a pose using the inputs from the xbox360 controller
# i.e. move the pose (and thus robot) with the xbox360 controller
from src.utils.decorators import synchronized_with_lock, timer
from src.utils.movement_utils import get_angles_center
from src.utils.os_utils import is_linux


class XboxPoseUpdater:

    def __init__(self, controller, maximum_speed=5.0, ramp_up_time=0.1,
                 workspace_limits=WorkSpaceLimits):
        self.workspace_limits = workspace_limits
        self.v_x, self.v_y, self.v_z = 0, 0, 0
        self.v_alpha, self.v_gamma = 0, 0
        self.steps_per_second = global_constants.steps_per_second
        self.dt = 1.0 / self.steps_per_second
        self._maximum_speed = maximum_speed  # cm/sec
        self.ramp_up_time = ramp_up_time  # time to speed up/slow down
        self._dv = self._maximum_speed / (self.ramp_up_time * self.steps_per_second)  # v/step
        self.controller = controller
        self.lock = threading.RLock()
        self.is_linux = is_linux()

    @property
    @synchronized_with_lock("lock")
    def maximum_speed(self):
        return self._maximum_speed

    @maximum_speed.setter
    @synchronized_with_lock("lock")
    def maximum_speed(self, value):
        self._maximum_speed = value
        self._dv = self._maximum_speed / (self.ramp_up_time * self.steps_per_second)

    def input_to_delta_velocity(self, controller_input, velocity, maximum_velocity):
        new_velocity = 0
        direction = np.sign(controller_input)
        if controller_input != 0:
            new_velocity = velocity + direction * self._dv
            return np.clip(new_velocity, -np.abs(maximum_velocity), np.abs(maximum_velocity))
        else:
            if velocity > 0:
                new_velocity = velocity - self._dv if velocity - self._dv > 0 else 0
            elif velocity < 0:
                new_velocity = velocity + self._dv if velocity + self._dv < 0 else 0
        return new_velocity

    def get_xyz_from_poller(self):
        x_in, y_in = self.controller.get_left_thumb()
        z_in = self.controller.get_lr_trigger()
        return x_in, y_in, z_in

    def __update_orientation_velocities(self, find_center_mode):
        right_thumb_x, right_thumb_y = self.controller.get_right_thumb()
        if not self.is_linux:
            right_thumb_x *= -1
        right_thumb_y *= -1

        v_alpha_max = self.maximum_speed * (right_thumb_x / 1000)
        v_gamma_max = self.maximum_speed * (right_thumb_y / 1000)

        # todo get rid of these global variables and just return them
        self.v_alpha = self.input_to_delta_velocity(right_thumb_x, self.v_alpha, v_alpha_max)
        if find_center_mode:
            self.v_gamma = 0
        else:
            self.v_gamma = self.input_to_delta_velocity(right_thumb_y, self.v_gamma, v_gamma_max)

    def __update_position_velocities(self, find_center_mode):
        x, y, z = self.get_xyz_from_poller()

        v_x_max = self.maximum_speed * (x / 100)
        v_y_max = self.maximum_speed * (y / 100)
        v_z_max = self.maximum_speed * (z / 100)

        # todo get rid of these global variables and just return them
        self.v_x = self.input_to_delta_velocity(x, self.v_x, v_x_max)
        self.v_y = self.input_to_delta_velocity(y, self.v_y, v_y_max)
        if find_center_mode:
            self.v_z = 0
        else:
            self.v_z = self.input_to_delta_velocity(z, self.v_z, v_z_max)

    def get_updated_pose_from_controller(self, old_pose, find_center_mode, center):
        self.__update_position_velocities(find_center_mode)
        self.__update_orientation_velocities(find_center_mode)

        dx = self.dt * self.v_x
        dy = self.dt * self.v_y
        dz = self.dt * self.v_z

        if self.workspace_limits is None:
            x = old_pose.x + dx
            y = old_pose.y + dy
            z = old_pose.z + dz
        else:
            new_x = old_pose.x + dx
            new_y = old_pose.y + dy
            new_z = old_pose.z + dz
            new_r = np.sqrt(new_x*new_x + new_y*new_y + new_z*new_z)

            if new_r > self.workspace_limits.radius_max or new_r < self.workspace_limits.radius_min:
                x, y, z = old_pose.x, old_pose.y, old_pose.z
            else:
                x, y = new_x, new_y
                z = new_z if new_z > self.workspace_limits.z_min else old_pose.z

        alpha, gamma = self.get_orientation(old_pose, center)
        alpha = np.clip(alpha, -pi / 2, pi / 2)
        gamma = np.clip(gamma, -pi / 2, pi / 2)

        return Pose(x, y, z, flip=old_pose.flip, alpha=alpha, gamma=gamma, beta=0.0)

    def get_orientation(self, old_pose, center):
        if center is not None:
            alpha, _, gamma = get_angles_center(old_pose.x, old_pose.y, old_pose.z, center)
        else:
            alpha = old_pose.alpha + self.dt * self.v_alpha
            gamma = old_pose.gamma + self.dt * self.v_gamma

        return alpha, gamma

    def stop(self):
        self.controller.stop()


