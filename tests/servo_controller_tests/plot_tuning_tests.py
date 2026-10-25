import os
import tempfile
import unittest
from unittest import mock

import matplotlib

from src.robot_controllers.dynamixel_robot.calibration import plot_tuning
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import SAMPLE_FIELDS, TRIAL_FIELDS, parse_pose
from tests.servo_controller_tests.gain_tuning_tests import session_for

matplotlib.use('Agg')


class PlotTuningTests(unittest.TestCase):
    """the plots of the csv files a simulated session writes"""

    @classmethod
    def setUpClass(cls):
        # the planned moves between poses keep time with the simulation
        session, robot, sim, sample_file, trial_file = session_for(resonance_above=2000, writers=True)
        patchers = [mock.patch('src.utils.movement.sleep', sim.advance),
                    mock.patch('src.utils.movement.perf_counter', lambda: sim.now)]
        for patcher in patchers:
            patcher.start()
        try:
            session.run([parse_pose('elbow_out'), parse_pose('low')])
        finally:
            for patcher in patchers:
                patcher.stop()
        cls.directory = tempfile.TemporaryDirectory()
        cls.trials_path = os.path.join(cls.directory.name, 'tune_joint3_test_trials.csv')
        with open(cls.trials_path, 'w', newline='') as trials:
            trials.write(','.join(TRIAL_FIELDS) + '\n' + trial_file.getvalue())
        with open(cls.trials_path.replace('_trials', '_samples'), 'w', newline='') as samples:
            samples.write(','.join(SAMPLE_FIELDS) + '\n' + sample_file.getvalue())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_the_run_is_read_with_its_baselines_and_steps(self):
        run = plot_tuning.read_run(self.trials_path)
        self.assertEqual(3, run.joint)
        self.assertEqual(1500, run.yaml_p)
        self.assertEqual({1, 2}, set(run.poses))
        self.assertEqual({1, 2}, set(run.baselines), 'one baseline per pose, the first of the two recorded')
        self.assertTrue(any(key[0] == 1 and key[1] == 1500 for key in run.steps))
        self.assertEqual({1: (2540, 1650), 2: (2540, 1650)}, plot_tuning.knees(run))

    def test_the_figures_are_drawn(self):
        run = plot_tuning.read_run(self.trials_path)
        self.assertIsNotNone(plot_tuning.plot_knee([run, run]))
        self.assertIsNotNone(plot_tuning.plot_d([run]))
        self.assertIsNotNone(plot_tuning.plot_steps(run))

    def test_main_writes_the_png_files(self):
        with tempfile.TemporaryDirectory() as output:
            self.assertEqual(0, plot_tuning.main([self.trials_path, '--save', output]))
            self.assertEqual({'joint3_knee.png', 'joint3_d.png', 'joint3_steps_tune_joint3_test.png'},
                             set(os.listdir(output)))
