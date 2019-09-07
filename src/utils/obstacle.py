from abc import ABC
import pybullet as p
import numpy as np


class Obstacle(ABC):

    def __init__(self, physics_client):
        self.physics_client = physics_client
        self.obstacle_id = None


class BoxObstacle(Obstacle):
    """
    Class to place an obstacle in the simulated world
    The obstacle will be placed with it's base at the given z height
    """

    def __init__(self, physics_client, dimensions, base_center_position, color=None, alpha=0):
        """

        Args:
            physics_client: id of the pybullet physics client
            dimensions: array of length, width, height of the obstacle
            base_center_position: array of the x,y,z coordinates of the center of the base of the obstacle
            alpha: angle by which to rotate the body around the z-axis
        """
        super().__init__(physics_client)
        collision_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[i/200 for i in dimensions],
                                                  physicsClientId=physics_client)

        base_center_position[2] += dimensions[2]/2

        baseOrientation = [0, 0, np.sin(alpha/2), np.cos(alpha/2)]

        self.obstacle_id = p.createMultiBody(0, collision_box_id, -1,
                                             basePosition=[i/100 for i in base_center_position],
                                             baseOrientation=baseOrientation,
                                             physicsClientId=physics_client)

        if color is not None:
            p.changeVisualShape(self.obstacle_id, -1, rgbaColor=color, physicsClientId=physics_client)


class SphereObstacle(Obstacle):

    def __init__(self, physics_client, radius, center_position, color=None):
        super().__init__(physics_client)

        collision_spere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius/100, physicsClientId=physics_client)

        self.obstacle_id = p.createMultiBody(0, collision_spere_id, -1,
                                             basePosition=[i / 100 for i in center_position],
                                             baseOrientation=[0, 0, 0, 1],
                                             physicsClientId=physics_client)

        if color is not None:
            p.changeVisualShape(self.obstacle_id, -1, rgbaColor=color, physicsClientId=physics_client)


if __name__ == '__main__':
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("urdf/plane.urdf")
    p.setRealTimeSimulation(1)

    BoxObstacle(physics_client, [1000, 1000, 1], [0, 0, -1], color=[1, 1, 1, 1])
    BoxObstacle(physics_client, [10, 10, 20], [0, 35, 0], alpha=np.pi/4)
    # SphereObstacle(physics_client, 10, [0, 40, 10], color=[0, 0, 1, 1])
    print("hoi")
