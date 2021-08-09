from src.kinematics.kinematics import jacobian_transpose_on_f
from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.robot_env import RobotEnv
import numpy as np
from time import sleep

from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import Scenario, easy_scenarios, medium_scenarios, hard_scenarios, \
    super_easy_scenarios, sensible_scenarios
from src.utils.obstacle import BoxObstacle

control_point_1_position = 11.2


def reset_to_scenario(env, scenario):
    env.set_scenario(scenario)
    env.reset()


start_pose = Pose(-30, 25, 10)
end_pose = Pose(30, 25, 10)
scenario = Scenario([BoxObstacle([10, 40, 30], [0, 40, 0])],
             start_pose, end_pose)

env = RobotEnvWithObstacles(use_gui=True, scenarios=[scenario], angle_control=True, is_eval=True)



# env.set_scenario(medium_scenarios[3])
# env.set_scenario(hard_scenarios[scenario_id])

state = env.reset()

steps_taken = 0
total_reward = 0

while True:

    raw_observation = state.observation
    observation = raw_observation['observation']

    c1_attr = np.zeros(3)
    c2_attr = np.zeros(3)
    c3_attr = observation[0:3]

    c1_rep = 4 * observation[3:6]
    c2_rep = 4 * observation[6:9]
    c3_rep = 4 * observation[9:12]

    attractive_forces = np.stack((c1_attr, c2_attr, c3_attr))
    repulsive_forces = np.stack((c1_rep, c2_rep, c3_rep))

    forces = attractive_forces + repulsive_forces

    current_angles = env.current_angles

    robot_controller = env.robot_controller

    joint_forces = jacobian_transpose_on_f(forces, current_angles,
                                           robot_controller.robot_config, control_point_1_position)

    absolute_force = np.linalg.norm(joint_forces)

    action = (joint_forces / absolute_force)

    state = env.step(action[1:6])
    # print(state.reward)
    total_reward = total_reward + state.reward
    sleep(0.02)

    if state.step_type == 2:
        print("goal reached!")
        print("steps taken: ", env._steps_taken)
        break
