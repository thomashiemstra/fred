import tkinter as tk
from tkinter import ttk
from numpy import pi


# root window
from src import global_constants
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController

root = tk.Tk()
root.geometry('1000x500')
root.resizable(False, False)
root.title('Slider Demo')

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)


# slider current value
current_value = tk.IntVar()


def get_current_value():
    return '{}'.format(current_value.get())


def slider_changed(event):
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
    from_=-100,
    to=100,
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


def create_button(root_element, text, row, column, command=None):
    button = ttk.Button(
        root_element,
        text=text,
        command=command
    )

    button.grid(
        column=column,
        row=row,
    )


def get_servo_config(servo_config_path):
    try:
        with open(servo_config_path, 'r') as servo_config_file:
            return servo_config_file.read()
    except FileNotFoundError:
        print("nope")


servo_config = get_servo_config("resources/servo_config.json")
robot = DynamixelRobotController("COM3", global_constants.dynamixel_robot_config, servo_config)

robot.enable_servos()

servo = robot.servo5
servo_id = 5

# robot.servo1.offset = 1
# robot.move_servo(servo_id, pi)



def move_to_90():
    robot.move_servo(servo_id, pi/2)


create_button(root, "90", 5, 0, command=move_to_90)


def move_to_45():
    robot.move_servo(servo_id, pi/4)


create_button(root, "45", 5, 1, command=move_to_45)


def move_to_0():
    robot.move_servo(servo_id, 0)


create_button(root, "0", 5, 2, command=move_to_0)


def move_to_minus45():
    robot.move_servo(servo_id, -pi/4)


create_button(root, "-45", 5, 3, command=move_to_minus45)


def move_to_minus_90():
    robot.move_servo(servo_id, -pi/2)


create_button(root, "-90", 5, 4, command=move_to_minus_90)


def set_offset():
    val = current_value.get()
    print("setting offset = {}".format(val))
    servo.offset = val


create_button(root, "set offset", 4, 0, command=set_offset)


root.mainloop()


