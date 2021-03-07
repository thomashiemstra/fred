from src.kinematics.kinematics import jacobian_transpose_on_f
from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.robot_env import RobotEnv
import numpy as np

from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import Scenario, easy_scenarios, medium_scenarios, hard_scenarios
from src.utils.obstacle import BoxObstacle

control_point_1_position = 11.2


def reset_to_scenario(env, scenario):
    env.set_scenario(scenario, reverse)
    env.reset()




env = RobotEnvWithObstacles(use_gui=True)
scenario_id = 6

reset_to_scenario(env, easy_scenarios[scenario_id])
# reset_to_scenario(env, medium_scenarios[scenario_id])
# reset_to_scenario(env, hard_scenarios[scenario_id])

steps_taken = 0

state = env.reset()

total_reward = 0


while True:
    raw_observation = state.observation
    observation = raw_observation[0]

    c1_attr = np.zeros(3)
    c2_attr = observation[0:3]
    c3_attr = observation[3:6]

    # if steps_taken < 15:
    #     c2_attr += [0, 0, 0.5]
    #     c3_attr += [0, 0, 0.5]

    c1_rep = 3 * observation[6:9]
    c2_rep = 3 * observation[9:12]
    c3_rep = 3 * observation[12:15]

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

    if state.step_type == 2:
        print("goal reached!")
        print("steps taken: ", env._steps_taken)
        break
