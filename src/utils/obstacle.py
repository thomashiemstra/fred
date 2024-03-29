from abc import ABC
import pybullet as p
import numpy as np
from absl import logging


class Obstacle(ABC):

    def __init__(self):
        self.obstacle_id = None

    def build(self, physics_client):
        raise NotImplementedError("obstacle should have a build function")

    def destroy(self, physics_client):
        raise NotImplementedError("obstacle should have a destroy function")

    def copy(self):
        raise NotImplementedError("obstacle should have a copy function")


class BoxObstacle(Obstacle):
    """
    Class to place an obstacle in the simulated world
    The obstacle will be placed with it's base at the given z height
    """

    def __init__(self, dimensions, raw_base_center_position, alpha=0, color=None):
        """

        Args:
            physics_client: id of the pybullet physics client
            dimensions: array of length, width, height of the obstacle
            raw_base_center_position: array of the x,y,z coordinates of the center of the base of the obstacle
            alpha: angle by which to rotate the body around the z-axis
            color: color components for RED, GREEN, BLUE and ALPHA, each in range [0..1]. Alpha has to be 0 (invisible)
             or 1 (visible) at the moment. Note that TinyRenderer doesn't support transparancy, but the GUI/EGL OpenGL3
             renderer does.
        """
        super().__init__()
        self.alpha = alpha
        self.dimensions = dimensions
        self.half_extends = [i/2 for i in dimensions]  # Half extends in centimeters
        self.raw_base_center_position = raw_base_center_position

        self.base_center_position = [self.raw_base_center_position[0],
                                     self.raw_base_center_position[1],
                                     self.raw_base_center_position[2] + dimensions[2] / 2]
        self.color = color
        self.obstacle_id = None

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
        if self.obstacle_id is None:
            logging.warning("no obstacle to destroy, call build() first")
            return
        p.removeBody(self.obstacle_id, physicsClientId=physics_client)

    def copy(self):
        return BoxObstacle(self.dimensions, self.raw_base_center_position, alpha=self.alpha, color=self.color)


class SphereObstacle(Obstacle):

    def __init__(self, radius, center_position, color=None):
        super().__init__()
        self.radius = radius
        self.center_position = center_position
        self.color = color
        self.obstacle_id = None

    def build(self, physics_client):
        collision_spere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius/100,
                                                    physicsClientId=physics_client)

        self.obstacle_id = p.createMultiBody(0, collision_spere_id, -1,
                                             basePosition=[i / 100 for i in self.center_position],
                                             baseOrientation=[0, 0, 0, 1],
                                             physicsClientId=physics_client)

        if self.color is not None:
            p.changeVisualShape(self.obstacle_id, -1, rgbaColor=self.color, physicsClientId=physics_client)


    def destroy(self, physics_client):
        if self.obstacle_id is None:
            logging.warn("no obstacle to destroy, call build() first")
            return
        p.removeBody(self.obstacle_id, physicsClientId=physics_client)

    def copy(self):
        return SphereObstacle(self.radius, self.center_position, color=self.color)

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
