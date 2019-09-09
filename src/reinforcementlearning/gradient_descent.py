from src.kinematics.kinematics import jacobian_transpose_on_f
from src.reinforcementlearning.environment.robot_env import RobotEnv
import numpy as np

control_point_1_position = 11.2

env = RobotEnv(use_gui=True, raw_obs=True)
state = env.reset()

steps_taken = 0

while True:
    observation = state.observation

    c1_attr = np.zeros(3)
    c2_attr = observation[0:3]
    c3_attr = observation[3:6]

    c1_rep = observation[6:9]
    c2_rep = observation[9:12]
    c3_rep = observation[12:15]

    attractive_forces = np.stack((c1_attr, c2_attr, c3_attr))
    repulsive_forces = np.stack((c1_rep, c2_rep, c3_rep))

    forces = attractive_forces # + repulsive_forces

    current_angles = env.current_angles

    robot_controller = env.robot_controller

    joint_forces = jacobian_transpose_on_f(forces, current_angles,
                                           robot_controller.robot_config, control_point_1_position)

    absolute_force = np.linalg.norm(joint_forces)

    action = (joint_forces / absolute_force)

    state = env.step(action[1:7])


