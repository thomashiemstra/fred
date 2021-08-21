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


def servo_1_check(positions):
    if positions[1] > 2500 or positions[1] < 0:
        print()
        print("------------------------------------------------------------------------------")
        ans = input("DANGER servo 1 might be below the 0 degree line, are you sure!? y/n")
        if ans == 'y':
            return True
        else:
            return False
    else:
        return True


def servo_2_check(positions):
    if positions[2] > 2500 or positions[2] < 0:
        print()
        print("------------------------------------------------------------------------------")
        ans = input("DANGER servo 2 might be below the 0 degree line, are you sure!? y/n")
        if ans == 'y':
            return True
        else:
            return False
    else:
        return True
