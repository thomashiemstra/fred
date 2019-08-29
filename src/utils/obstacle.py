from abc import ABC
import pybullet as p


class Obstacle(ABC):

    def __init__(self, physics_client):
        self.physics_client = physics_client
        self.obstacle_id = None


class BoxObstacle(Obstacle):
    """
    Class to place an obstacle in the simulated world
    The obstacle will be placed with it's base at the given z height
    """

    def __init__(self, physics_client, dimensions, base_center_position):
        """

        Args:
            physics_client: id of the pybullet physics client
            dimensions: array of length, width, height of the obstacle
            base_center_position: array of the x,y,z coordinates of the center of the base of the obstacle
        """
        super().__init__(physics_client)
        collision_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[i/2 for i in dimensions],
                                                  physicsClientId=physics_client)

        base_center_position[2] += dimensions[2]/2

        self.obstacle_id = p.createMultiBody(0, collision_box_id, -1, basePosition=base_center_position,
                                             baseOrientation=[0, 0, 0, 1],
                                             physicsClientId=physics_client)


if __name__ == '__main__':
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("urdf/plane.urdf")
    p.setRealTimeSimulation(1)

    BoxObstacle(physics_client, [0.1, 0.1, 0.1], [0, 0, 0])
    print("hoi")
