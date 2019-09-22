import random
from time import sleep

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import pybullet as p
import numpy as np
from numpy import pi

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.robot_env_utils import get_control_point_pos, sphere_2_id, sphere_3_id, \
    get_attractive_force_world, get_target_points, draw_debug_lines, get_repulsive_forces_world
from src.reinforcementlearning.environment.scenarios import Scenario
from src.reinforcementlearning.occupancy_grid_util import create_hilbert_curve_from_obstacles

from src.simulation.simulation_utils import start_simulated_robot

from src.utils.obstacle import BoxObstacle, SphereObstacle


class RobotEnv(py_environment.PyEnvironment):

    def __init__(self, use_gui=False, raw_obs=False, no_obstacles=True):
        super().__init__()
        self._use_gui = use_gui
        self._raw_obs = raw_obs
        self._no_obstacles = no_obstacles
        self._hilbert_curve_iteration = 3
        self._grid_len_x = 40
        self._grid_len_y = 40
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        if no_obstacles:
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='observation')
        else:
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(15 + 2**(2*self._hilbert_curve_iteration),),
                dtype=np.float32, minimum=-1, maximum=1, name='observation')
        self._update_step_size = 0.01
        self._simulation_steps_per_step = 1
        self._wait_time_per_step = self._simulation_steps_per_step / 240  # Pybullet simulations run at 240HZ
        self._episode_ended = False
        self._robot_controller = start_simulated_robot(use_gui)
        self._physics_client = self._robot_controller.physics_client
        self._start_pose = None
        self._previous_distance_to_target = 0
        self._robot_body_id = self._robot_controller.body_id
        self._target_pose = None
        self._steps_taken = 0
        self._current_angles = None
        self._floor = BoxObstacle([1000, 1000, 1], [0, 0, -1], color=[1, 1, 1, 1])
        self._floor.build(self._physics_client)
        self._obstacles = None
        self._target_spheres = None
        self._attr_lines = None
        self._rep_lines = None
        self._current_scenario = None
        self.scenario_id = None
        self.reverse_scenario = False

    @property
    def current_angles(self):
        return self._current_angles

    @property
    def robot_controller(self):
        return self._robot_controller

    def _generate_obstacles_and_target_pose(self):
        if self._current_scenario is not None:
            self._current_scenario.destroy_scenario()

        if self.scenario_id is not None:
            self._current_scenario = scenarios[self.scenario_id]
        else:
            if self._no_obstacles:
                scenario_id = random.randint(0, 4)
            else:
                scenario_id = random.randint(0, len(scenarios) - 1)
            self._current_scenario = scenarios[scenario_id]

        self._current_scenario.build_scenario(self._physics_client)
        return self._current_scenario.obstacles, self._current_scenario.target_pose, self._current_scenario.start_pose

    def _create_visual_target_spheres(self, target_pose):
        if not self._use_gui:
            return

        if self._target_spheres is not None:
            for target_sphere in self._target_spheres:
                p.removeBody(target_sphere.obstacle_id)

        _, target_point_2, target_point_3 = get_target_points(target_pose, self._robot_controller.robot_config.d6)

        target_sphere_2 = SphereObstacle(1, target_point_2.tolist(), color=[1, 1, 0, 1])
        target_sphere_2.build(self._physics_client)
        target_sphere_3 = SphereObstacle(1, target_point_3.tolist(), color=[1, 1, 0, 1])
        target_sphere_3.build(self._physics_client)

        for sphere_id in [target_sphere_2.obstacle_id, target_sphere_3.obstacle_id]:
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 2, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 3, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 4, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 5, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 6, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 7, -1, 0)
            p.setCollisionFilterPair(self._robot_body_id, sphere_id, 8, -1, 0)

        self._target_spheres = [target_sphere_2, target_sphere_3]

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
        if self.reverse_scenario:
            self._target_pose, self._start_pose = self._start_pose, self._target_pose

        self._create_visual_target_spheres(self._target_pose)

        self._robot_controller.reset_to_pose(self._start_pose)
        self._advance_simulation()

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

        contact_points = p.getContactPoints(bodyA=self._robot_body_id, physicsClientId=self._physics_client)
        collision = contact_points != ()
        # print("collision: {}".format(collision))

        observation, total_distance = self._get_observations()
        delta_distance = self._previous_distance_to_target - total_distance
        self._previous_distance_to_target = total_distance

        self._steps_taken += 1

        if self._steps_taken > 1000 or collision:
            return ts.termination(np.array(observation, dtype=np.float32), reward=-10)
        elif total_distance < 5:  # target reached
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
        attractive_forces, total_distance = get_attractive_force_world(
            np.array([c1.position, c2.position, c3.position]),
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

        if not self._no_obstacles:
            curve = create_hilbert_curve_from_obstacles(self._obstacles, grid_len_x=self._grid_len_x,
                                                        grid_len_y=self._grid_len_y,
                                                        iteration=self._hilbert_curve_iteration)
            total_observation += curve.tolist()

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

    def show_occupancy_grid_and_curve(self):
        if self._obstacles is None:
            return
        from src.reinforcementlearning.occupancy_grid_util import create_occupancy_grid_from_obstacles

        len_x = 40
        len_y = 40
        curve_iteration = 3

        grid = create_occupancy_grid_from_obstacles(self._obstacles, grid_len_x=len_x, grid_len_y=len_y, grid_size=1)
        curve = create_hilbert_curve_from_obstacles(self._obstacles, grid_len_x=len_x, grid_len_y=len_y, iteration=curve_iteration)

        import matplotlib.pyplot as plt

        plt.set_cmap('hot')

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)

        reshape = 2 ** curve_iteration
        ax1.imshow(curve.reshape(reshape, reshape))
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(grid)

        plt.show()


scenarios = [Scenario([],
                      Pose(-25, 35, 10), Pose(25, 35, 10)),
             Scenario([],
                      Pose(-30, 20, 10), Pose(20, 40, 20)),
             Scenario([],
                      Pose(-35, 15, 10), Pose(25, 30, 30)),
             Scenario([],
                      Pose(0, 20, 10), Pose(0, 30, 40)),
             Scenario([BoxObstacle([20, 25, 40], [0, 35, 0], alpha=np.pi / 4)],
                      Pose(-25, 25, 10), Pose(25, 25, 10)),
             Scenario([BoxObstacle([10, 10, 30], [0, 35, 0], alpha=0),
                       BoxObstacle([10, 20, 20], [10, 35, 0], alpha=np.pi / 4)],
                      Pose(-25, 20, 10), Pose(30, 30, 10)),
             Scenario([BoxObstacle([10, 20, 20], [-10, 35, 0], alpha=-np.pi / 4),
                       BoxObstacle([10, 20, 20], [10, 35, 0], alpha=np.pi / 4)],
                      Pose(-25, 20, 10), Pose(25, 20, 10)),
             Scenario([BoxObstacle([10, 40, 25], [0, 35, 0], alpha=0)],
                      Pose(-25, 30, 10), Pose(25, 30, 10)),
             Scenario([BoxObstacle([10, 30, 20], [0, 30, 0], alpha=np.pi / 8),
                       BoxObstacle([10, 10, 30], [10, 35, 0], alpha=0)],
                      Pose(-25, 30, 10), Pose(25, 30, 10)),
             Scenario([BoxObstacle([10, 30, 20], [0, 35, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 35], [0, 25, 0], alpha=0)],
                      Pose(-25, 30, 10), Pose(25, 30, 10)),
             Scenario([BoxObstacle([20, 20, 20], [-20, 40, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 35], [0, 25, 0], alpha=0)],
                      Pose(-25, 20, 10), Pose(20, 40, 10)),
             Scenario([BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 8),
                       BoxObstacle([10, 10, 35], [-5, 38, 0], alpha=0)],
                      Pose(-25, 40, 10), Pose(20, 20, 10)),
             Scenario([BoxObstacle([10, 10, 40], [5, 30, 0], alpha=0),
                       BoxObstacle([30, 30, 20], [-5, 40, 0], alpha=0)],
                      Pose(-35, 15, 10), Pose(25, 30, 30)),
             Scenario([BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 4),
                       BoxObstacle([10, 40, 20], [-10, 40, 0], alpha=np.pi / 4)],
                      Pose(-35, 15, 10), Pose(25, 30, 30)),

             ]

if __name__ == '__main__':
    env = RobotEnv(use_gui=True)
    state = env.observation_spec()
    print(state)
    env.scenario_id = 8
    obs = env.reset()
    env.show_occupancy_grid_and_curve()
    print("hoi")
