from src.global_constants import recommended_max_servo_speed
import numpy as np


def get_recommended_wait_time(current_angles, new_angles):
    recommended_time = 0
    for current_angle, new_angle in zip(current_angles, new_angles):
        delta_angle = current_angle - new_angle
        if delta_angle == 0:
            continue
        delta_angle = np.abs(delta_angle)
        time = delta_angle / recommended_max_servo_speed
        recommended_time = np.maximum(time, recommended_time)

    return recommended_time
