import random
from time import sleep

import numpy as np
import pybullet as p
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import random
from absl import logging
from copy import copy

from src.reinforcementlearning.environment.robot_env_utils import get_control_point_pos, sphere_2_id, sphere_3_id, \
    get_attractive_force_world, get_target_points, draw_debug_lines, get_repulsive_forces_world, \
    get_normalized_current_angles, get_clipped_state
from src.reinforcementlearning.environment.scenario import scenarios_no_obstacles
from src.simulation.simulation_utils import start_simulated_robot
from src.utils.obstacle import BoxObstacle, SphereObstacle


class RobotEnv(py_environment.PyEnvironment):

    def __init__(self, use_gui=False, raw_obs=False, scenarios=None, is_eval=False, robot_controller=None,
                 angle_control=False):
        super().__init__()
        self._use_gui = use_gui
        self._raw_obs = raw_obs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(20,), dtype=np.float32, minimum=-1, maximum=1, name='observation')

        scenarios = scenarios_no_obstacles if scenarios is None else scenarios

        self._non_completed_scenarios = [scenario.copy() for scenario in scenarios]
        self._completed_scenarios = []
        self._current_scenario_id = 0
        self._doing_already_completed_scenario = False

        self._is_eval = is_eval
        self._eval_scenario_id = 0

        self._update_step_size = 0.02

        self._xyz_update_step_size = 1.5
        self._alpha_beta_gamma_update_step_size = 0.1

        self._max_steps_to_take_before_failure = 200
        self._simulation_steps_per_step = 1
        self._wait_time_per_step = self._simulation_steps_per_step / 240  # Pybullet simulations run at 240HZ
        self._episode_ended = False
        if robot_controller is None:
            self._robot_controller = start_simulated_robot(use_gui)
        else:
            self._robot_controller = robot_controller
        self._physics_client = self._robot_controller.physics_client
        self._start_pose = None
        self._closest_distance_so_far = None
        self._robot_body_id = self._robot_controller.body_id
        self._target_pose = None
        self._steps_taken = 0
        self._current_angles = None
        self._floor = BoxObstacle([1000, 1000, 1], [0, 0, -1], color=[1, 1, 1, 0])
        self._floor.build(self._physics_client)
        self._obstacles = None
        self._target_spheres = None
        self._attr_lines = None
        self._rep_lines = None
        self._current_scenario = None
        self._done = True
        self._externally_set_scenario = None
        self._traveled_distances = []
        self._times_current_scenario_payed = 0
        self._target_reached_distance = 15
        self.rewards = []
        self.distance = []
        self._current_pose = None
        self.angle_control = angle_control

    def set_target_reached_distance(self, val):
        self._target_reached_distance = val

    def disable_max_steps_to_take_before_failure(self):
        self._max_steps_to_take_before_failure = 10000000

    @property
    def current_angles(self):
        return self._current_angles

    @property
    def robot_controller(self):
        return self._robot_controller

    def set_scenario(self, scenario):
        self._externally_set_scenario = scenario

    @staticmethod
    def get_random_id(max_range):
        return random.randint(0, max_range - 1)

    def _pick_eval_scenario(self):
        id = self._eval_scenario_id
        scenario = self._non_completed_scenarios[self._eval_scenario_id % len(self._non_completed_scenarios)]
        self._eval_scenario_id += 1
        return id, scenario

    def _pick_scenario(self):
        if self._is_eval:
            return self._pick_eval_scenario()

        non_completed_scenarios = len(self._non_completed_scenarios)
        completed_scenarios = len(self._completed_scenarios)

        self._doing_already_completed_scenario = False

        if completed_scenarios == 0:
            scenario_id = self.get_random_id(non_completed_scenarios)
            return scenario_id, self._non_completed_scenarios[scenario_id]

        if non_completed_scenarios == 0:
            self._doing_already_completed_scenario = True
            scenario_id = self.get_random_id(completed_scenarios)
            return scenario_id, self._completed_scenarios[scenario_id]

        # pick unsolved scenarios 80% of the time, only train on the completed scenarios in 20% of the time
        if random.random() < 0.8:
            scenario_id = self.get_random_id(non_completed_scenarios)
            return scenario_id, self._non_completed_scenarios[scenario_id]
        else:
            self._doing_already_completed_scenario = True
            scenario_id = self.get_random_id(completed_scenarios)
            return scenario_id, self._completed_scenarios[scenario_id]

    def _generate_obstacles_and_target_pose(self):
        if self._current_scenario is not None:
            self._current_scenario.destroy_scenario(self._physics_client)

        if self._externally_set_scenario is not None:
            self._current_scenario = self._externally_set_scenario
        else:
            self._current_scenario_id, self._current_scenario = self._pick_scenario()

        self._current_scenario.build_scenario(self._physics_client)
        return self._current_scenario.obstacles, self._current_scenario.target_pose, self._current_scenario.start_pose

    def _create_visual_target_spheres(self, target_pose):
        if not self._use_gui:
            return

        if self._target_spheres is not None:
            for target_sphere in self._target_spheres:
                p.removeBody(target_sphere.obstacle_id, physicsClientId=self._physics_client)

        _, target_point_2, target_point_3 = get_target_points(target_pose, self._robot_controller.robot_config.d6)

        target_sphere_2 = SphereObstacle(1, target_point_2.tolist(), color=[1, 1, 0, 1])
        target_sphere_2.build(self._physics_client)
        target_sphere_3 = SphereObstacle(1, target_point_3.tolist(), color=[1, 1, 0, 1])
        target_sphere_3.build(self._physics_client)

        self._target_spheres = [target_sphere_2, target_sphere_3]

        num_joints_on_robot = p.getNumJoints(self._robot_body_id)
        # Disable collisions
        for sphere_id in [sphere.obstacle_id for sphere in self._target_spheres]:
            # p.setCollisionFilterGroupMask(sphere_id, -1, 0, 0)
            for link_id_on_robot in range(-1, num_joints_on_robot):
                p.setCollisionFilterPair(self._robot_body_id, sphere_id, link_id_on_robot, -1, 0,
                                         physicsClientId=self._physics_client)

    def render(self, mode='rgb_array'):
        pass

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

        self._obstacles, self._target_pose, self._start_pose = self._generate_obstacles_and_target_pose()
        self._current_pose = copy(self._start_pose)

        if self._start_pose.x > self._start_pose.x:
            print("flipped scenario: start_pose={} stop_pose={}".format(self._start_pose, self._target_pose))
            self._target_pose, self._start_pose = self._start_pose, self._target_pose

        self._create_visual_target_spheres(self._target_pose)

        self._robot_controller.reset_to_pose(self._start_pose)
        self._advance_simulation()

        self._current_angles = self._robot_controller.get_current_angles()
        observation, self._closest_distance_so_far = self._get_observations()
        self._episode_ended = False
        self._steps_taken = 0
        self._traveled_distances = []
        self._done = False
        return ts.restart(observation)

    @property
    def done(self):
        return self._done

    def close(self):
        if self._current_scenario is not None:
            self._current_scenario.destroy_scenario(self._physics_client)

    def get_info(self):
        return None

    def _update_current_pose_and_clip(self, action):
        xyz_step = self._xyz_update_step_size
        rot_step = self._alpha_beta_gamma_update_step_size

        new_x = self._current_pose.x + xyz_step * action[0]
        new_y = self._current_pose.y + xyz_step * action[1]
        new_z = self._current_pose.z + xyz_step * action[2]

        # very carefully tuned numbers, do not touch!
        y_clip = 17
        if new_z > 26:
            y_clip = 10
        elif new_z > 17.5:
            z_distance_to_base = new_z - 10
            y_clip = np.clip(np.sqrt(17 * 17 - z_distance_to_base * z_distance_to_base), 10, 50)

        self._current_pose.x = np.clip(new_x, -35, 35)
        self._current_pose.y = np.clip(new_y, y_clip, 50)
        self._current_pose.z = np.clip(new_z, 0, 50)
        self._current_pose.alpha = np.clip(self._current_pose.alpha + rot_step * action[3], -0.45 * np.pi, 0.45 * np.pi)
        self._current_pose.gamma = np.clip(self._current_pose.gamma + rot_step * action[4], -0.45 * np.pi, 0.45 * np.pi)

    def _step(self, action):
        if not action.shape == (5,):
            raise ValueError("Action should be of shape (5,)")

        if self._current_angles is None:
            raise ValueError("Please reset the environment before taking steps!")

        if self._done:
            return self.reset()

        if self.angle_control:
            converted_action = np.append(np.append([0], action), [0])  # angles are not 0 indexed and we don't use the last angle
            self._current_angles = get_clipped_state(self._current_angles + self._update_step_size * converted_action)
            self._robot_controller.move_servos(self._current_angles)
        else:
            self._update_current_pose_and_clip(action)
            self._current_angles = self._robot_controller.move_to_pose_and_give_new_angles(self._current_pose)

            _, _, _, _, p6 = self._robot_controller.forward_position_kinematics(self.current_angles)
            self._current_pose.x = p6[0]
            self._current_pose.y = p6[1]
            self._current_pose.z = p6[2]

        self._advance_simulation()

        contact_points = p.getContactPoints(bodyA=self._robot_body_id, physicsClientId=self._physics_client)
        collision = contact_points != ()
        # print("collision: {}".format(collision))

        observation, total_distance = self._get_observations()
        self._traveled_distances.append(total_distance)
        stuck = self._is_stuck(total_distance, self._traveled_distances)

        extra_distance_closed_this_step = 0
        if total_distance < self._closest_distance_so_far:
            extra_distance_closed_this_step = self._closest_distance_so_far - total_distance
            self._closest_distance_so_far = total_distance

        reward = self._get_reward(extra_distance_closed_this_step, total_distance, action)

        self.rewards.append(reward)
        self.distance.append(total_distance)

        self._current_time_step = self._get_current_time_step(collision, observation, total_distance, reward, stuck)
        self._steps_taken += 1
        return self._current_time_step

    @staticmethod
    def _get_reward(extra_distance_closed_this_step, total_distance, action):
        reward = extra_distance_closed_this_step

        # effort = np.sum(action * action) * 0.01
        # reward -= effort

        return reward

    def _get_current_time_step(self, collision, observation, total_distance, reward, stuck):
        if self._steps_taken > self._max_steps_to_take_before_failure:
            self._done = True
            return ts.termination(observation, reward=0)
        elif collision:
            self._done = True
            return ts.termination(observation, reward=-30)
        if stuck:
            self._done = True
            return ts.termination(observation, reward=0)
        elif total_distance < self._target_reached_distance:
            self._done = True
            max_speed_bonus = 5
            speed_bonus = (-max_speed_bonus / self._max_steps_to_take_before_failure) * self._steps_taken \
                          + max_speed_bonus

            total_reward = 10 + speed_bonus

            self._switch_current_scenario_to_done()
            return ts.termination(observation, reward=total_reward)
        else:
            self._done = False
            return ts.transition(observation, reward=reward, discount=1.0)

    def _switch_current_scenario_to_done(self):
        if self._is_eval or self._doing_already_completed_scenario:
            return
        logging.info("completed a scenario!")
        del self._non_completed_scenarios[self._current_scenario_id]
        self._completed_scenarios.append(self._current_scenario)

    @staticmethod
    def _is_stuck(total_distance, traveled_distances):
        if len(traveled_distances) > 250:
            return abs(traveled_distances[-200] - traveled_distances[-1]) < 1
        return False

    def _get_observations(self):
        c1, c2, c3 = self._robot_controller.control_points

        _, target_point_2, target_point_3 = get_target_points(self._target_pose, self._robot_controller.robot_config.d6)

        # Control point 1 is not used for the attractive forces
        attractive_cutoff_dis = 10
        attractive_forces, total_distance = get_attractive_force_world(
            np.array([c1.position, c2.position, c3.position]),
            np.array([None, target_point_2, target_point_3], dtype=object),
            attractive_cutoff_distance=attractive_cutoff_dis)

        # now the attractive foces will go from 1 down to 0 when the robot is within attractive_cutoff_dis of the target
        attractive_forces[1] /= attractive_cutoff_dis
        attractive_forces[2] /= attractive_cutoff_dis

        obstacle_ids = [obstacle.obstacle_id for obstacle in self._obstacles] + [self._floor.obstacle_id]

        repulsive_cutoff_distance = 6
        repulsive_forces = get_repulsive_forces_world(self._robot_body_id, np.array([c1, c2, c3]),
                                                      obstacle_ids, self._physics_client,
                                                      repulsive_cutoff_distance=repulsive_cutoff_distance,
                                                      clip_force=6)

        if self._use_gui:
            self._attr_lines, self._rep_lines = draw_debug_lines(self._physics_client, np.array([c1, c2, c3]),
                                                                 attractive_forces, repulsive_forces,
                                                                 self._attr_lines, self._rep_lines,
                                                                 line_size=6)

        total_observation = []

        # attractive forces[0] is for a control point which is not considered for the attractive forces
        total_observation += self._get_normalized_vector_as_list(attractive_forces[1])
        total_observation += self._get_normalized_vector_as_list(attractive_forces[2])

        total_observation += self._get_normalized_vector_as_list(repulsive_forces[0])
        total_observation += self._get_normalized_vector_as_list(repulsive_forces[1])
        total_observation += self._get_normalized_vector_as_list(repulsive_forces[2])

        total_observation += self._get_normalized_pose()
        # total_observation += get_normalized_current_angles(self._current_angles[1:6])

        # return ts.transition(np.array(observation, dtype=np.float32), reward=reward, discount=1.0)

        return np.array(np.array(total_observation), dtype=np.float32), total_distance

    def _get_normalized_vector_as_list(self, vec):
        if self._raw_obs:
            return vec.tolist()
        norm = np.linalg.norm(vec)
        if norm < 1:  # Only normalize the vector if it's too big
            return vec.tolist()
        normalized_vec = vec / np.linalg.norm(vec)
        return normalized_vec.tolist()

    def _get_normalized_pose(self):
        normalized_x = self._current_pose.x / 40
        normalized_y = self._current_pose.y / 40
        normalized_z = self._current_pose.z / 40
        normalized_alpha = self._current_pose.alpha / np.pi
        normalized_gamma = self._current_pose.gamma / np.pi
        return [normalized_x, normalized_y, normalized_z, normalized_alpha, normalized_gamma]


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

    def get_state(self):
        return self._current_time_step

    def set_state(self, state):
        self._current_time_step = state
