from time import sleep

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import pybullet as p
import numpy as np

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.robot_env_utils import get_control_point_pos, sphere_2_id, sphere_3_id, \
    get_attractive_force_world, get_target_points
from src.simulation.simulation_utils import start_simulated_robot
from tf_agents.environments import utils


class RobotEnv(py_environment.PyEnvironment):

    def __init__(self, use_gui=False):
        super().__init__()
        self._use_gui = use_gui
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.float32, minimum=-1, maximum=1, name='observation')
        self._update_step_size = 0.05
        self._simulation_steps_per_step = 5
        self._wait_time_per_step = self._simulation_steps_per_step / 240  # Pybullet simulations run at 240HZ
        self._episode_ended = False
        self._robot_controller = start_simulated_robot(use_gui)
        self._gripper_length = self._robot_controller.robot_config.d6
        self._physics_client = self._robot_controller.physics_client
        self.start_pose = Pose(-25, 20, 10)
        self._previous_distance_to_target = 0
        self._robot_body_id = self._robot_controller.body_id
        self._target_pose = Pose(25, 20, 20)
        self._attractive_cutoff_distance = 1
        self._steps_taken = 0
        self._state = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self._robot_body_id, start_pos, start_orientation,
                                          physicsClientId=self._physics_client)
        p.resetBaseVelocity(self._robot_body_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._physics_client)

        self._robot_controller.reset_to_pose(self.start_pose)
        self._advance_simulation()
        self._state = self._robot_controller.get_current_angles()
        observation, self._previous_distance_to_target = self._get_observations()
        self._episode_ended = False
        self._steps_taken = 0
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        if not action.shape == (6,):
            raise ValueError("Action should be of shape (6,)")

        if self._state is None:
            raise ValueError("Please reset the environment before taking steps!")

        converted_action = np.append([0], action)  # angles are not 0 indexed
        self._state = self._state + self._update_step_size * converted_action
        self._robot_controller.move_servos(self._state)
        self._advance_simulation()

        observation, total_distance = self._get_observations()
        delta_distance = self._previous_distance_to_target - total_distance
        self._previous_distance_to_target = total_distance

        self._steps_taken += 1

        if self._steps_taken > 100:
            return ts.termination(np.array(observation, dtype=np.float32), reward=0)
        elif total_distance < 10:  # target reached
            return ts.termination(np.array(observation, dtype=np.float32), reward=100)
        else:
            return ts.transition(np.array(observation, dtype=np.float32), reward=delta_distance, discount=1.0)

    def get_control_point_positions(self):
        c2_pos = get_control_point_pos(self._robot_body_id, sphere_2_id)
        c3_pos = get_control_point_pos(self._robot_body_id, sphere_3_id)
        return c2_pos, c3_pos

    def _get_observations(self):
        c2_pos, c3_pos = self.get_control_point_positions()
        _, target_point_2, target_point_3 = get_target_points(self._target_pose, self._gripper_length)

        attractive_forces, total_distance = get_attractive_force_world(np.array([c2_pos, c3_pos]),
                                                                       np.array([target_point_2, target_point_3]),
                                                                       self._attractive_cutoff_distance)

        normalized_attr_vec_1 = attractive_forces[0] / np.linalg.norm(attractive_forces[0])
        normalized_attr_vec_2 = attractive_forces[1] / np.linalg.norm(attractive_forces[1])

        return np.append(normalized_attr_vec_1, normalized_attr_vec_2), total_distance

    def _advance_simulation(self):
        if self._use_gui:
            sleep(self._wait_time_per_step)
        else:
            for _ in range(self._simulation_steps_per_step):
                p.stepSimulation(self._physics_client)


if __name__ == '__main__':
    env = RobotEnv(use_gui=True)
    state = env.reset()
    print(state)
    for _ in range(50):
        simple_action = np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
        res = env.step(simple_action)
        print(res.reward)

    print('hoi')
    env.reset()
    for _ in range(50):
        simple_action = np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
        res = env.step(simple_action)
        print(res.reward)
    print('hoi')




