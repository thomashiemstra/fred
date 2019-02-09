from __future__ import division
from xbox_controller.xbox_poller import XboxPoller
import numpy as np


class PosePoller:

    def __init__(self, maximum_speed=20.0, ramp_up_time=0.1, position_limit=30.0):
        self.v_x, self.v_y, self.v_z = 0, 0, 0
        self.steps_per_second = 15
        self.dt = 1.0 / self.steps_per_second
        self.maximum_speed = maximum_speed  # cm/sec
        self.ramp_up_time = ramp_up_time  # time to speed up/slow down
        self.dv = self.maximum_speed / (self.ramp_up_time * self.steps_per_second)  # v/step
        self.poller = XboxPoller()
        self.position_limit = position_limit

    def input_to_delta_velocity(self, controller_input, velocity, maximum_velocity):
        new_velocity = 0
        if controller_input > 0:
            new_velocity = velocity + self.dv if velocity < maximum_velocity else maximum_velocity
        elif controller_input < 0:
            new_velocity = velocity - self.dv if velocity > maximum_velocity else maximum_velocity
        else:
            if velocity > 0:
                new_velocity = velocity - self.dv if velocity - self.dv > 0 else 0
            elif velocity < 0:
                new_velocity = velocity + self.dv if velocity + self.dv < 0 else 0
        return new_velocity

    def get_xyz_from_poller(self):
        x_in, y_in = self.poller.get_left_thumb()
        z_in = -self.poller.get_lr_trigger()
        return x_in, y_in, z_in

    def __update_velocities(self):
        x, y, z = self.get_xyz_from_poller()

        v_x_max = self.maximum_speed * (x / 100)
        v_y_max = self.maximum_speed * (y / 100)
        v_z_max = self.maximum_speed * (z / 100)

        self.v_x = self.input_to_delta_velocity(x, self.v_x, v_x_max)
        self.v_y = self.input_to_delta_velocity(y, self.v_y, v_y_max)
        self.v_z = self.input_to_delta_velocity(z, self.v_z, v_z_max)

    def update_positions_of_pose(self, pose):
        self.__update_velocities()

        pose.x += self.dt * self.v_x
        pose.y += self.dt * self.v_y
        pose.z += self.dt * self.v_z

        pose.x = np.clip(pose.x, -self.position_limit, self.position_limit)
        pose.y = np.clip(pose.y, 5, self.position_limit)
        pose.z = np.clip(pose.z, 5, self.position_limit)

    def stop(self):
        self.poller.stop()
