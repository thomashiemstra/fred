from time import sleep

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import pybullet as p
import numpy as np
from numpy import pi

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.robot_env_utils import get_control_point_pos, sphere_2_id, sphere_3_id, \
    get_attractive_force_world, get_target_points, draw_debug_lines, get_repulsive_forces_world
from src.simulation.simulation_utils import start_simulated_robot
from tf_agents.environments import utils

from src.utils.obstacle import BoxObstacle, SphereObstacle


class RobotEnv(py_environment.PyEnvironment):

    def __init__(self, use_gui=False, raw_obs=False):
        super().__init__()
        self._use_gui = use_gui
        self._raw_obs = raw_obs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='observation')
        self._update_step_size = 0.03
        self._simulation_steps_per_step = 5
        self._wait_time_per_step = self._simulation_steps_per_step / 240  # Pybullet simulations run at 240HZ
        self._episode_ended = False
        self._robot_controller = start_simulated_robot(use_gui)
        self._physics_client = self._robot_controller.physics_client
        self.start_pose = Pose(-25, 20, 10)
        self._previous_distance_to_target = 0
        self._robot_body_id = self._robot_controller.body_id
        self._target_pose = None
        self._steps_taken = 0
        self._current_angles = None
        self._floor = BoxObstacle(self._physics_client, [1000, 1000, 1], [0, 0, -1], color=[1, 1, 1, 1])
        self._obstacles = None
        self._target_spheres = None
        self._attr_lines = None
        self._rep_lines = None

    @property
    def current_angles(self):
        return self._current_angles

    @property
    def robot_controller(self):
        return self._robot_controller

    def _generate_obstacles_and_target_pose(self):
        obstacle = BoxObstacle(self._physics_client, [20, 20, 40], [0, 40, 0], color=[1, 0, 0, 1])
        target_pose = Pose(25, 20, 8)
        return np.array([obstacle]), target_pose

    def _create_visual_target_spheres(self, target_pose):
        if not self._use_gui:
            return

        if self._target_spheres is not None:
            for target_sphere in self._target_spheres:
                p.removeBody(target_sphere.obstacle_id)

        _, target_point_2, target_point_3 = get_target_points(target_pose, self._robot_controller.robot_config.d6)

        target_sphere_2 = SphereObstacle(self._physics_client, 1, target_point_2.tolist(), color=[1, 1, 0, 1])
        target_sphere_3 = SphereObstacle(self._physics_client, 1, target_point_3.tolist(), color=[1, 1, 0, 1])

        for sphere_id in [target_sphere_2.obstacle_id, target_sphere_3.obstacle_id]:
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 2, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 3, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 4, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 5, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 6, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 7, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 8, -1, 0)

        self._target_spheres = [target_sphere_2, target_sphere_3]

    def _remove_obstacles(self):
        if self._obstacles is None:
            return
        for obstacle in self._obstacles:
            p.removeBody(obstacle.obstacle_id)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        start_pos = [0, 0, 0]
        self._remove_obstacles()
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self._robot_body_id, start_pos, start_orientation,
                                          physicsClientId=self._physics_client)
        p.resetBaseVelocity(self._robot_body_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._physics_client)

        self._robot_controller.reset_to_pose(self.start_pose)
        self._advance_simulation()

        self._obstacles, self._target_pose = self._generate_obstacles_and_target_pose()
        self._create_visual_target_spheres(self._target_pose)

        self._current_angles = self._robot_controller.get_current_angles()
        observation, self._previous_distance_to_target = self._get_observations()
        self._episode_ended = False
        self._steps_taken = 0
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        if not action.shape == (6,):
            raise ValueError("Action should be of shape (6,)")

        if self._current_angles is None:
            raise ValueError("Please reset the environment before taking steps!")

        converted_action = np.append([0], action)  # angles are not 0 indexed
        self._current_angles = self._current_angles + self._update_step_size * converted_action
        self._clip_state()

        self._robot_controller.move_servos(self._current_angles)
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

    def _clip_state(self):
        np.clip(self._current_angles[1], 0, pi)
        np.clip(self._current_angles[2], 0, pi)
        np.clip(self._current_angles[3], -pi / 3, 2 * pi / 3)
        np.clip(self._current_angles[4], 0, pi)
        np.clip(self._current_angles[5], -3 * pi / 4, 3 * pi / 4)
        np.clip(self._current_angles[6], 0, pi)

    def _get_observations(self):
        c1, c2, c3 = self._robot_controller.control_points

        _, target_point_2, target_point_3 = get_target_points(self._target_pose, self._robot_controller.robot_config.d6)

        # Control point 1 is not used for the attractive forces
        attractive_forces, total_distance = get_attractive_force_world(np.array([c1.position, c2.position, c3.position]),
                                                                       np.array([None, target_point_2, target_point_3]))

        obstacle_ids = [obstacle.obstacle_id for obstacle in self._obstacles] + [self._floor.obstacle_id]

        repulsive_forces = get_repulsive_forces_world(self._robot_body_id, np.array([c1, c2, c3]),
                                                      obstacle_ids, self._physics_client)

        if self._use_gui:
            self._attr_lines, self._rep_lines = draw_debug_lines(self._physics_client, np.array([c1, c2, c3]),
                                                                 attractive_forces, repulsive_forces,
                                                                 self._attr_lines, self._rep_lines)

        total_observation = []

        # attractive forces[0] is for a control point which is not considered for the attractive forces
        total_observation += self._get_normalized_vector_as_list(attractive_forces[1])
        total_observation += self._get_normalized_vector_as_list(attractive_forces[2])

        total_observation += self._get_normalized_vector_as_list(repulsive_forces[0])
        total_observation += self._get_normalized_vector_as_list(repulsive_forces[1])
        total_observation += self._get_normalized_vector_as_list(repulsive_forces[2])

        return np.array(total_observation), total_distance

    def _get_normalized_vector_as_list(self, vec):
        if self._raw_obs:
            return vec.tolist()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        normalized_vec = vec / np.linalg.norm(vec)
        return normalized_vec.tolist()

    def _get_control_point_positions(self):
        c2_pos = get_control_point_pos(self._robot_body_id, sphere_2_id)
        c3_pos = get_control_point_pos(self._robot_body_id, sphere_3_id)
        return c2_pos, c3_pos

    def _advance_simulation(self):
        if self._use_gui:
            sleep(self._wait_time_per_step)
        else:
            for _ in range(self._simulation_steps_per_step):
                p.stepSimulation(self._physics_client)


if __name__ == '__main__':
    env = RobotEnv(use_gui=True)
    state = env.observation_spec()
    print(state)
    obs = env.reset()
    # for _ in range(50):
    #     simple_action = np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
    #     res = env.step(simple_action)
    #     print(res.reward)
    #
    # print('hoi')
    # env.reset()
    # for _ in range(50):
    #     simple_action = np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
    #     res = env.step(simple_action)
    #     print(res.reward)
    # print('hoi')




