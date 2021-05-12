from src.simulation.simulation_utils import start_simulated_robot
from numpy import pi

robot_controller = start_simulated_robot(True)

angles = [0, 0, pi/2, 0, 0, 0, 0]
robot_controller.move_servos(angles)
print("hi")
