import numpy as np
import matplotlib.pyplot as plt


def generate_data(items):
    input = np.zeros((items, 6))
    output = np.zeros((items, 6))

    for i in range(items):
        angles = np.random.uniform(-1, 1, 6)
        input[i] = angles

        # todo
        output[i] = angles

    return input, output


input, output = generate_data(6)

print(input.shape)

print(output)

