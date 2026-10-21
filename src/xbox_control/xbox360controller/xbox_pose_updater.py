"""
Turns the gamepad inputs into the next pose of the end effector, one step at a time.

The left stick moves the pose in x and y, the triggers in z, the right stick turns it: alpha with x, gamma with y.
Every axis has a velocity that ramps towards the velocity the stick asks for, the stick deflection times
maximum_speed (cm/s) or maximum_angular_speed (rad/s), by at most the acceleration maximum / ramp_up_time. Pushing,
easing off and releasing a stick all take ramp_up_time from rest to full speed. A step covers the time that really
passed since the previous one (dt), so the speeds hold whatever the pace of the control loop is; a stall of the loop
counts as at most MAX_STEP_TIME, so the pose never jumps.

The pose is not checked against the workspace here, the control loop does that with robot.workspace_violation and
holds the pose when a step would leave it. The orientation is clipped to a quarter turn either way and beta is
always 0. With a centre the right stick is ignored and the gripper points at the centre.

The class belongs to the control loop thread. maximum_speed may be set from other threads (the d-pad, the REST
api): it is a plain float that every step reads fresh, nothing is derived from it ahead of time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import pi

from src import global_constants
from src.kinematics.kinematics_utils import Pose
from src.utils.movement import get_angles_center

if TYPE_CHECKING:
    from src.xbox_control.xbox360controller.Xbox360Controller import Xbox360Controller

# the sticks and the triggers deliver -STICK_RANGE..STICK_RANGE (Xbox360Controller(scale=100))
STICK_RANGE = 100.0
# a step never covers more than this, however long the control loop was busy in between
MAX_STEP_TIME = 0.1


class XboxPoseUpdater:

    def __init__(self, controller: Xbox360Controller, maximum_speed: float = 5.0, ramp_up_time: float = 0.1,
                 maximum_angular_speed: float = 0.5) -> None:
        """
        :param maximum_speed: cm/s of the end effector at full stick deflection
        :param ramp_up_time: seconds to get from rest to the maximum speed, and back
        :param maximum_angular_speed: rad/s of alpha and gamma at full stick deflection
        """
        self.controller = controller
        self.maximum_speed = maximum_speed
        self.maximum_angular_speed = maximum_angular_speed
        self.ramp_up_time = ramp_up_time
        self.dt = 1.0 / global_constants.steps_per_second  # the nominal step, used when the caller gives no time
        self.v_x, self.v_y, self.v_z = 0.0, 0.0, 0.0
        self.v_alpha, self.v_gamma = 0.0, 0.0

    def get_updated_pose_from_controller(self, old_pose: Pose, find_center_mode: bool, center: np.ndarray | None,
                                         dt: float | None = None) -> Pose:
        """
        the next pose, from the pose and the gamepad inputs
        :param find_center_mode: z and gamma are held while the centre is being recorded
        :param center: the point the gripper points at, None to turn it with the right stick
        :param dt: seconds since the previous step, by default the nominal step
        """
        dt = self.dt if dt is None else min(dt, MAX_STEP_TIME)
        self._update_velocities(find_center_mode, dt)

        x = old_pose.x + dt * self.v_x
        y = old_pose.y + dt * self.v_y
        z = old_pose.z + dt * self.v_z
        if center is not None:
            alpha, _, gamma = get_angles_center(old_pose.x, old_pose.y, old_pose.z, center)
        else:
            alpha = old_pose.alpha + dt * self.v_alpha
            gamma = old_pose.gamma + dt * self.v_gamma
        alpha = float(np.clip(alpha, -pi / 2, pi / 2))
        gamma = float(np.clip(gamma, -pi / 2, pi / 2))
        return Pose(x, y, z, flip=old_pose.flip, alpha=alpha, gamma=gamma, beta=0.0)

    def _update_velocities(self, find_center_mode: bool, dt: float) -> None:
        x, y = self.controller.get_left_thumb()
        z = self.controller.get_lr_trigger()
        alpha, gamma = self.controller.get_right_thumb()
        if find_center_mode:
            z, gamma = 0.0, 0.0
        self.v_x = self._ramp(self.v_x, x, self.maximum_speed, dt)
        self.v_y = self._ramp(self.v_y, y, self.maximum_speed, dt)
        self.v_z = self._ramp(self.v_z, z, self.maximum_speed, dt)
        self.v_alpha = self._ramp(self.v_alpha, alpha, self.maximum_angular_speed, dt)
        self.v_gamma = self._ramp(self.v_gamma, gamma, self.maximum_angular_speed, dt)

    def _ramp(self, velocity: float, stick: float, maximum: float, dt: float) -> float:
        """the velocity moved towards the one the stick asks for, by at most the acceleration maximum / ramp_up_time"""
        target = maximum * stick / STICK_RANGE
        max_change = maximum / self.ramp_up_time * dt
        return velocity + float(np.clip(target - velocity, -max_change, max_change))

    def stop(self) -> None:
        self.controller.stop()
