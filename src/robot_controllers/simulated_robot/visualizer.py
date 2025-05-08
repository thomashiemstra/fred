from src.simulation.simulation_utils import start_simulated_robot
from numpy import pi


robot_controller = start_simulated_robot(True)

robot_controller.move_servos([0, pi/2, pi/2, -0.1, 0, 0, 0])
input("Press Enter to close")