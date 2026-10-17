import unittest

import numpy as np

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment import scenario as scenario_module
from src.reinforcementlearning.environment.occupancy_grid_util import is_point_in_obstacle
from src.reinforcementlearning.environment.pose_recorder import PoseRecorder, DummyPoseRecorder
from src.reinforcementlearning.environment.scenario import Scenario
from src.utils.obstacle import BoxObstacle

SCENARIO_LISTS = ['scenarios_no_obstacles', 'scenarios_obstacles', 'super_easy_scenarios', 'easy_scenarios',
                  'medium_scenarios', 'hard_scenarios', 'sensible_scenarios', 'train_scenarios', 'test']


class ScenarioTest(unittest.TestCase):

    def test_copy_creates_new_unbuilt_obstacles(self):
        obstacle = BoxObstacle([10, 20, 30], [1, 2, 0], alpha=np.pi / 4, color=[1, 0, 0, 1])
        obstacle.obstacle_id = 7  # pretend it has been built in a simulation
        start_pose, target_pose = Pose(-25, 35, 10), Pose(25, 35, 10)
        original = Scenario([obstacle], start_pose, target_pose)

        copied = original.copy()

        self.assertIs(start_pose, copied.start_pose)
        self.assertIs(target_pose, copied.target_pose)
        self.assertEqual(1, len(copied.obstacles))
        copied_obstacle = copied.obstacles[0]
        self.assertIsNot(obstacle, copied_obstacle)
        self.assertIsNone(copied_obstacle.obstacle_id, "a copied obstacle should not refer to the built body")
        self.assertEqual(obstacle.dimensions, copied_obstacle.dimensions)
        self.assertEqual(obstacle.base_center_position, copied_obstacle.base_center_position)
        self.assertEqual(obstacle.alpha, copied_obstacle.alpha)
        self.assertEqual(obstacle.color, copied_obstacle.color)
        self.assertIsNot(original.obstacles, copied.obstacles)

    def test_predefined_start_and_target_poses_are_not_inside_obstacles(self):
        for list_name in SCENARIO_LISTS:
            for index, scenario in enumerate(getattr(scenario_module, list_name)):
                for label, pose in (('start', scenario.start_pose), ('target', scenario.target_pose)):
                    for obstacle in scenario.obstacles:
                        inside_footprint = is_point_in_obstacle(np.array([pose.x, pose.y]),
                                                                np.array(obstacle.base_center_position),
                                                                obstacle.half_extends, obstacle.alpha)
                        self.assertFalse(inside_footprint and pose.z <= obstacle.dimensions[2],
                                         "{} pose of {}[{}] is inside obstacle {} at {}"
                                         .format(label, list_name, index, obstacle.dimensions,
                                                 obstacle.raw_base_center_position))


class PoseRecorderTest(unittest.TestCase):

    def test_records_copies(self):
        recorder = PoseRecorder()
        pose = Pose(1, 2, 3)

        recorder.add_pose(pose)
        pose.x = 100  # changing the pose afterwards should not change the recording
        recorded = recorder.get_recorded_poses()
        recorded[0].y = 200  # neither should changing the returned poses

        self.assertEqual(1, len(recorded))
        self.assertEqual((1, 2, 3), (recorder.poses[0].x, recorder.poses[0].y, recorder.poses[0].z))

        recorder.clear_poses()
        self.assertEqual([], recorder.get_recorded_poses())

    def test_dummy_recorder_records_nothing(self):
        recorder = DummyPoseRecorder()

        recorder.add_pose(Pose(1, 2, 3))

        self.assertEqual([], recorder.get_recorded_poses())


if __name__ == '__main__':
    unittest.main()
