from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from src import global_constants
from numpy import  sin, cos, pi


@tf.function
def jacobian_transpose_on_f_tf(workspace_force, angles, robot_config, c1_location):
    """
    compute the jacobian transpose on 3 control points on the robot, this translates world forces on each joint
    into joint forces. The join forces add up in joint space.
    :param workspace_force: 3x3 numpy array of the workspace forces on each of the control points
    :param angles: current angles of the robot
    :param robot_config: robot configuration of link lengths
    :param c1_location: distance of control point 1 located between frame 3 and 4 of the robot,
                        should be less than d4 (the total distance between frame 3 and 4)
    :return:
    """
    x_comp, y_comp, z_comp = 0, 1, 2
    a2, d4, d6 = robot_config.a2, robot_config.d4, robot_config.d6
    c1, c2, c3, c4, c5 = cos(angles[1]), cos(angles[2]), cos(angles[3]), cos(angles[4]), cos(angles[5])
    c23 = cos(angles[2] + angles[3])
    s1, s2, s3, s4, s5 = sin(angles[1]), sin(angles[2]), sin(angles[3]), sin(angles[4]), sin(angles[5])
    s23 = sin(angles[2] + angles[3])

    joint_forces = np.zeros(7)

    # first control point, somewhere between frame 3 and frame 4
    fx, fy, fz = workspace_force[0][x_comp], workspace_force[0][y_comp], workspace_force[0][z_comp]
    joint_forces[1] += (fy * c1 - fx * s1) * (a2 * c2 + c1_location * s23)
    joint_forces[2] += a2 * fz * c2 + (fx * c2 + fy * s1) * (c1_location * c23 - a2 * s2) + c1_location * fz * s23
    joint_forces[3] += c1_location * c23 * (fx * c1 + fy * s1) + c1_location * fz * s23

    # second control point, origin of frame 4
    fx, fy, fz = workspace_force[1][x_comp], workspace_force[1][y_comp], workspace_force[1][z_comp]
    joint_forces[1] += (fy * c1 - fx * s1) * (a2 * c2 + d4 * s23)
    joint_forces[2] += a2 * fz * c2 + (fx * c2 + fy * s1) * (d4 * c23 - a2 * s2) + d4 * fz * s23
    joint_forces[3] += d4 * c23 * (fx * c1 + fy * s1) + d4 * fz * s23

    # third control point, origin of frame 6
    fx, fy, fz = workspace_force[2][x_comp], workspace_force[2][y_comp], workspace_force[2][z_comp]
    joint_forces[1] += (fy * c1 - fx * s1) * (a2 * c2 + (d4 + d6 * c5) * s23) + d6 * (
                c23 * c4 * (fy * c1 - fx * s1) + (fx * c1 + fy * s1) * s4) * s5
    joint_forces[2] += a2 * fz * c2 + c23 * (d4 + d6 * c5) * (fx * c1 + fy * s1) + fz * (
                d4 + d6 * c5) * s23 + d6 * fz * c23 * c4 * s5 - (fx * c1 + fy * s1) * (a2 * s2 + d6 * c4 * s23 * s5)
    joint_forces[3] += (d4 + d6 * c5) * (c23 * (fx * c1 + fy * s1) + fz * s23) + d6 * c4 * (
                fz * c23 - (fx * c1 + fy * s1) * s23) * s5
    joint_forces[4] += -d6 * (-fx * c4 * s1 + (fy * c23 * s1 + fz * s23) * s4 + c1 * (fy * c4 + fx * c23 * s4)) * s5
    joint_forces[5] += d6 * c5 * (c4 * (c23 * (fx * c1 + fy * s1) + fz * s23) + (-fy * c1 + fx * s1) * s4) + d6 * (
                fz * c23 - (fx * c1 + fy * s1) * s23) * s5

    return joint_forces


class CustomPolicy(tf_policy.Base):
    """Returns random samples of the given action_spec.

  Note: the values in the info_spec (except for the log_probability) are random
    values that have nothing to do with the emitted actions.
  """

    def __init__(self, time_step_spec, action_spec, *args, **kwargs):
        observation_and_action_constraint_splitter = (
            kwargs.get('observation_and_action_constraint_splitter', None))
        self._accepts_per_arm_features = (
            kwargs.pop('accepts_per_arm_features', False))

        if observation_and_action_constraint_splitter is not None:
            if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
                raise NotImplementedError(
                    'RandomTFPolicy only supports action constraints for '
                    'BoundedTensorSpec action specs.')

            scalar_shape = action_spec.shape.rank == 0
            single_dim_shape = (
                    action_spec.shape.rank == 1 and action_spec.shape.dims == [1])

            if not scalar_shape and not single_dim_shape:
                raise NotImplementedError(
                    'RandomTFPolicy only supports action constraints for action specs '
                    'shaped as () or (1,) or their equivalent list forms.')

        super(CustomPolicy, self).__init__(time_step_spec, action_spec, *args,
                                           **kwargs)

    def _variables(self):
        return []

    def get_de_normalized_current_angles(self, normalized_angles):
        return [
            (pi / 2) * (normalized_angles[0] + 1),
            (pi / 2) * (normalized_angles[1] + 1),
            (pi / 2) * (normalized_angles[2] + (1 / 3)),
            pi * normalized_angles[3],
            (3 * pi / 4) * normalized_angles[4]
        ]

    @tf.function
    def act(self, raw_observation):
        observation = raw_observation[0]

        c1_attr = np.zeros(3)
        c2_attr = observation[0:3]
        c3_attr = observation[3:6]

        c1_rep = observation[6:9]
        c2_rep = observation[9:12]
        c3_rep = observation[12:15]

        attractive_forces = tf.stack([c1_attr, c2_attr, c3_attr])
        repulsive_forces = tf.stack([c1_rep, c2_rep, c3_rep])

        forces = attractive_forces + repulsive_forces

        current_angles = self.get_de_normalized_current_angles(observation[15:20])

        joint_forces = jacobian_transpose_on_f_tf(forces, np.append([0], current_angles),
                                               global_constants.simulated_robot_config, 11.5)

        absolute_force = np.linalg.norm(joint_forces)

        action = (joint_forces / absolute_force)

        return tf.constant([action[1:6]], shape=(1, 5), dtype=tf.float32)



    def _action(self, time_step, policy_state, seed):
        # action_ = tf.constant([1, 0, 0, 0, 0], shape=(1, 5), dtype=tf.float32)
        action_ = self.act(time_step.observation)

        # if time_step is not None:
        #     with tf.control_dependencies(tf.nest.flatten(time_step)):
        #         action_ = tf.nest.map_structure(tf.identity, action_)

        step = policy_step.PolicyStep(action_, policy_state, ())
        return step

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError(
            'CustomPolicy does not support distributions yet.')
