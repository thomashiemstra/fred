import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from flask import Flask

from src.camera import image_handlers, util
from src.camera_control import camera_control_resource

CAMERA_MATRIX = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
DIST_COEFFS = [[0.1, -0.2, 0.0, 0.0, 0.05]]


def write_calibration(path):
    with open(path, 'w') as calibration_file:
        json.dump({'cameraMatrix': CAMERA_MATRIX, 'distCoeffs': DIST_COEFFS}, calibration_file)


class TestGetCalibrations(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def test_existing_file_is_loaded(self):
        path = os.path.join(self.directory, 'calibration.json')
        write_calibration(path)
        camera_matrix, dist_coeffs = util.get_calibrations(path)
        np.testing.assert_array_equal(CAMERA_MATRIX, camera_matrix)
        np.testing.assert_array_equal(DIST_COEFFS, dist_coeffs)

    def test_missing_file_raises_instead_of_exiting(self):
        path = os.path.join(self.directory, 'calibration_data_charuco.json')
        with self.assertRaises(FileNotFoundError) as context:
            util.get_calibrations(path)
        message = str(context.exception)
        self.assertIn(path, message)
        self.assertIn('aruco_camera_calibration.py', message)
        self.assertIn('charuco_camera_calibration_from_images.py', message)

    def test_missing_chessboard_file_names_its_scripts(self):
        with self.assertRaises(FileNotFoundError) as context:
            util.get_calibrations(os.path.join(self.directory, 'calibration_data.json'))
        self.assertIn('camera_calibration.py', str(context.exception))
        self.assertIn('camera_calibration_from_images.py', str(context.exception))

    def test_calibration_directory_does_not_depend_on_the_working_directory(self):
        self.assertTrue(os.path.isabs(util.CALIBRATION_DIR))
        self.assertTrue(os.path.isfile(os.path.join(util.CALIBRATION_DIR, util.CHESSBOARD_CALIBRATION_FILE)))
        self.assertTrue(os.path.isdir(os.path.join(os.path.dirname(util.CALIBRATION_DIR), 'calibration')))
        # the repository has the chessboard calibration
        util.get_calibrations(os.path.join(util.CALIBRATION_DIR, util.CHESSBOARD_CALIBRATION_FILE))


class TestDefaultCalibrationChoice(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        patcher = mock.patch.object(util, 'CALIBRATION_DIR', self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_chessboard_calibration_without_charuco_calibration(self):
        write_calibration(os.path.join(self.directory, 'calibration_data.json'))
        self.assertEqual(os.path.join(self.directory, 'calibration_data.json'), util.get_default_calibration_path())

    def test_charuco_calibration_when_it_exists(self):
        write_calibration(os.path.join(self.directory, 'calibration_data.json'))
        write_calibration(os.path.join(self.directory, 'calibration_data_charuco.json'))
        self.assertEqual(os.path.join(self.directory, 'calibration_data_charuco.json'),
                         util.get_default_calibration_path())

    def test_default_handlers_load_the_chosen_file(self):
        write_calibration(os.path.join(self.directory, 'calibration_data.json'))
        for get_handler in (image_handlers.get_default_aurco_image_handler,
                            image_handlers.get_default_board_to_board_image_handler):
            handler = get_handler()
            np.testing.assert_array_equal(CAMERA_MATRIX, handler.cameraMatrix)

    def test_default_handlers_raise_when_there_is_no_calibration(self):
        for get_handler in (image_handlers.get_default_aurco_image_handler,
                            image_handlers.get_default_board_to_board_image_handler):
            with self.assertRaises(FileNotFoundError):
                get_handler()


class TestMissingCalibrationInTheApi(unittest.TestCase):

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(camera_control_resource.camera_api, url_prefix='/camera')
        self.client = app.test_client()
        for name, value in (('started', True), ('object_handler', camera_control_resource.Handlers())):
            patcher = mock.patch.object(camera_control_resource, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(camera_control_resource.global_objects, 'get_camera')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_aruco_reports_the_missing_calibration(self):
        error = FileNotFoundError('calibration file x not found')
        with mock.patch.object(camera_control_resource, 'get_default_aurco_image_handler', side_effect=error):
            response = self.client.post('/camera/start_aruco')
        self.assertEqual(200, response.status_code)
        self.assertFalse(response.get_json()['success'])
        self.assertIn('calibration file x not found', response.get_json()['error'])
        self.assertIsNone(camera_control_resource.object_handler.get_aruco_image_handler())

    def test_start_board_to_board_reports_the_missing_calibration(self):
        error = FileNotFoundError('calibration file x not found')
        with mock.patch.object(camera_control_resource, 'get_default_board_to_board_image_handler',
                               side_effect=error):
            response = self.client.post('/camera/start_board_to_board')
        self.assertFalse(response.get_json()['success'])
        self.assertIn('calibration file x not found', response.get_json()['error'])
        self.assertIsNone(camera_control_resource.object_handler.get_board_to_board_handler())


if __name__ == '__main__':
    unittest.main()
