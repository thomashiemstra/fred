import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
import numpy as np
from numpy import pi


# root window
from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot

root = tk.Tk()
root.geometry('1000x500')
root.resizable(False, False)
root.title('Slider Demo')

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)


# slider current value
current_value = tk.IntVar()


def get_current_value() -> str:
    return '{}'.format(current_value.get())


def slider_changed(event: str) -> None:
    value_label.configure(text=get_current_value())


# label for the slider
slider_label = ttk.Label(
    root,
    text='Slider:'
)

slider_label.grid(
    column=0,
    row=0,
)

#  slider
slider = ttk.Scale(
    root,
    from_=-200,
    to=200,
    orient='horizontal',  # vertical
    command=slider_changed,
    variable=current_value
)

slider.grid(
    column=1,
    sticky='we'
)

# current value label
current_value_label = ttk.Label(
    root,
    text='Current Value:'
)

current_value_label.grid(
    row=2,
    column=1
)

# value label
value_label = ttk.Label(
    root,
    text=get_current_value()
)

value_label.grid(
    row=3,
    column=1
)


def create_button(root_element: tk.Misc, text: str, row: int, column: int,
                  command: Callable[[], object] | None = None) -> None:
    button = ttk.Button(
        root_element,
        text=text,
        command=command  # type: ignore[arg-type]  # the stubs don't allow None, tkinter leaves the option out
    )

    button.grid(
        column=column,
        row=row,
    )


robot = get_robot()
assert isinstance(robot, CombinedRobot), 'this script needs the real robot'

servo = robot.servo2
servo_id = 2
print("offset: {}".format(servo.constant_offset))
all_angles = np.array([0, pi/4, pi/4, pi/2, 0.0, 0.0, 0.0])


def move_to_180() -> None:
    all_angles[servo_id] = pi
    robot.move_servos(all_angles)


create_button(root, "180", 6, 0, command=move_to_180)


def move_to_135() -> None:
    all_angles[servo_id] = (3 / 4) * pi
    robot.move_servos(all_angles)


create_button(root, "135", 6, 1, command=move_to_135)


def move_to_90() -> None:
    all_angles[servo_id] = pi/2
    robot.move_servos(all_angles)


create_button(root, "90", 5, 0, command=move_to_90)


def move_to_45() -> None:
    all_angles[servo_id] = pi/4
    robot.move_servos(all_angles)


create_button(root, "45", 5, 1, command=move_to_45)


def move_to_0() -> None:
    all_angles[servo_id] = 0
    robot.move_servos(all_angles)


create_button(root, "0", 5, 2, command=move_to_0)


def move_to_minus45() -> None:
    all_angles[servo_id] = -pi/4
    robot.move_servos(all_angles)


create_button(root, "-45", 5, 3, command=move_to_minus45)


def move_to_minus_90() -> None:
    all_angles[servo_id] = -pi/2
    robot.move_servos(all_angles)


create_button(root, "-90", 5, 4, command=move_to_minus_90)


def enable() -> None:
    robot.enable_servos()
    robot.move_servos(all_angles)


create_button(root, "enable servos", 6, 3, command=enable)


def set_offset() -> None:
    val = current_value.get()
    print("setting offset = {}".format(val))
    servo.constant_offset = val


create_button(root, "set offset", 4, 0, command=set_offset)


root.mainloop()


