import numpy as np
import matplotlib.pyplot as plt

from src.reinforcementlearning.environment.robot_env_utils import get_de_normalized_current_angles


def generate_data(items):
    input = np.zeros((items, 6))
    output = np.zeros((items, 6))

    for i in range(items):
        normalized_angles = np.random.uniform(-1, 1, 6)

        angles = get_de_normalized_current_angles(normalized_angles)
        input[i] = angles


        # todo
        output[i] = angles

    return input, output


input, output = generate_data(6)

print(input.shape)

print(output)

