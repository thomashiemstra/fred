from abc import ABC
import pybullet as p
import numpy as np


class Obstacle(ABC):

    def __init__(self):
        self.obstacle_id = None

    def build(self, physics_client):
        pass


class BoxObstacle(Obstacle):
    """
    Class to place an obstacle in the simulated world
    The obstacle will be placed with it's base at the given z height
    """

    def __init__(self, dimensions, base_center_position, alpha=0, color=None):
        """

        Args:
            physics_client: id of the pybullet physics client
            dimensions: array of length, width, height of the obstacle
            base_center_position: array of the x,y,z coordinates of the center of the base of the obstacle
            alpha: angle by which to rotate the body around the z-axis
        """
        super().__init__()
        self.alpha = alpha
        self.dimensions = dimensions
        self.half_extends = [i/2 for i in dimensions]  # Half extends in centimeters
        self.base_center_position = base_center_position
        self.base_center_position[2] += dimensions[2] / 2
        self.color = color

    def build(self, physics_client):

        collision_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[i/200 for i in self.dimensions],
                                                  physicsClientId=physics_client)

        baseOrientation = [0, 0, np.sin(self.alpha/2), np.cos(self.alpha/2)]

        self.obstacle_id = p.createMultiBody(0, collision_box_id, -1,
                                             basePosition=[i/100 for i in self.base_center_position],
                                             baseOrientation=baseOrientation,
                                             physicsClientId=physics_client)

        if self.color is not None:
            p.changeVisualShape(self.obstacle_id, -1, rgbaColor=self.color, physicsClientId=physics_client)

    def destroy(self, physics_client):
        p.removeBody(self.obstacle_id, physicsClientId=physics_client)


class SphereObstacle(Obstacle):

    def __init__(self, radius, center_position, color=None):
        super().__init__()
        self.radius = radius
        self.center_position = center_position
        self.color = color

    def build(self, physics_client):
        collision_spere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius/100,
                                                    physicsClientId=physics_client)

        self.obstacle_id = p.createMultiBody(0, collision_spere_id, -1,
                                             basePosition=[i / 100 for i in self.center_position],
                                             baseOrientation=[0, 0, 0, 1],
                                             physicsClientId=physics_client)

        if self.color is not None:
            p.changeVisualShape(self.obstacle_id, -1, rgbaColor=self.color, physicsClientId=physics_client)


if __name__ == '__main__':
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("urdf/plane.urdf")
    p.setRealTimeSimulation(1)

    floor = BoxObstacle([1000, 1000, 1], [0, 0, -1], color=[1, 1, 1, 1])
    floor.build(physics_client)

    obstacle1 = BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi/4)
    obstacle1.build(physics_client)

    # SphereObstacle(physics_client, 10, [0, 40, 10], color=[0, 0, 1, 1])
    print("hoi")
