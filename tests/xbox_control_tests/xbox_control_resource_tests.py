import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flask import Flask

from src import global_objects
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils.movement import PoseToPoseMovement
from src.xbox_control import xbox_control_resource, xbox_robot_controller
from src.xbox_control.xbox_control_resource import xbox_api
from src.xbox_control.xbox_robot_controller import XboxRobotController


class XboxControlResourceTests(unittest.TestCase):
    """the endpoints with a mocked xbox robot controller, so no robot or gamepad is needed"""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(xbox_api, url_prefix='/xbox')
        self.client = app.test_client()

        self.controller = mock.Mock()
        self.controller.is_running.return_value = False
        patcher = mock.patch.object(global_objects, 'get_xbox_robot_controller', return_value=self.controller)
        patcher.start()
        self.addCleanup(patcher.stop)
        # make sure nothing creates a real robot, even when the patch above would be bypassed
        robot_patcher = mock.patch.object(global_objects, 'get_robot', side_effect=AssertionError('real robot'))
        robot_patcher.start()
        self.addCleanup(robot_patcher.stop)

    def post(self, path, json=None):
        return self.client.post('/xbox' + path, json=json if json is not None else {})

    def test_start(self):
        self.controller.start.return_value = True

        response = self.post('/start')

        self.assertEqual({'success': True}, response.get_json())
        self.controller.start.assert_called_once_with()

    def test_start_that_fails_is_not_a_success(self):
        self.controller.start.return_value = False

        self.assertEqual({'success': False}, self.post('/start').get_json())

    def test_start_when_already_started_does_not_start_again(self):
        self.controller.is_running.return_value = True

        response = self.post('/start')

        self.assertEqual(b'already started', response.data)
        self.controller.start.assert_not_called()

    def test_stop(self):
        self.controller.stop.return_value = True

        self.assertEqual({'success': True}, self.post('/stop').get_json())
        self.controller.stop.assert_called_once_with()

    def test_stop_when_already_stopped(self):
        self.controller.stop.return_value = False

        self.assertEqual(b'already stopped', self.post('/stop').data)

    def test_find_center_reports_whether_it_was_requested(self):
        for requested in (True, False):
            self.controller.request_find_center_mode.return_value = requested
            self.assertEqual({'success': requested}, self.post('/findcenter').get_json())

    def test_clear_center(self):
        self.assertEqual({'success': True}, self.post('/clearcenter').get_json())
        self.controller.clear_center.assert_called_once_with()

    def test_set_speed_within_range(self):
        for speed in (5, 25, 50):
            self.assertEqual({'success': True}, self.post('/setSpeed', {'speed': speed}).get_json())
            self.controller.set_maximum_speed.assert_called_with(speed)

    def test_set_speed_outside_of_the_range_of_the_d_pad_is_rejected(self):
        self.assertEqual(xbox_robot_controller.MINIMUM_SPEED, xbox_control_resource.MINIMUM_SPEED)
        self.assertEqual(xbox_robot_controller.MAXIMUM_SPEED, xbox_control_resource.MAXIMUM_SPEED)
        for speed in (-1, 0, 4, 51, '0'):
            response = self.post('/setSpeed', {'speed': speed})
            self.assertEqual(400, response.status_code, speed)
            self.assertFalse(response.get_json()['success'])
            self.assertIn('error', response.get_json())
        self.controller.set_maximum_speed.assert_not_called()

    def test_bad_input_is_a_400_not_a_500(self):
        bodies = [{}, {'speed': ''}, {'speed': 'abc'}, {'speed': None}, {'speed': 1.5}, {'speed': True},
                  {'speed': [20]}, {'speed': {'a': 1}}, {'other': 20}]
        for body in bodies:
            response = self.post('/setSpeed', body)
            self.assertEqual(400, response.status_code, body)
            self.assertFalse(response.get_json()['success'])
        self.controller.set_maximum_speed.assert_not_called()

    def test_a_body_that_is_not_a_json_object_is_a_400(self):
        for path in ('/setSpeed', '/setScenarioId', '/speed', '/saveToFile', '/restoreFromFile'):
            for data, mimetype in (('[1, 2]', 'application/json'), ('5', 'application/json'),
                                   ('null', 'application/json'), ('{broken', 'application/json'),
                                   ('speed=20', 'text/plain'), ('', 'application/json')):
                response = self.client.post('/xbox' + path, data=data, content_type=mimetype)
                self.assertEqual(400, response.status_code, (path, data))
                self.assertFalse(response.get_json()['success'])
        self.assertEqual([], self.controller.method_calls)

    def test_set_speed_accepts_a_string(self):
        self.assertEqual({'success': True}, self.post('/setSpeed', {'speed': '20'}).get_json())
        self.controller.set_maximum_speed.assert_called_once_with(20)

    def test_speed_percentage(self):
        self.assertEqual({'success': True}, self.post('/speed', {'percentage': '10'}).get_json())
        self.controller.set_profile_velocity_percentage.assert_called_once_with(10)
        for body in ({'percentage': 0}, {'percentage': 101}, {'percentage': 'x'}, {}):
            self.assertEqual(400, self.post('/speed', body).status_code, body)
        self.controller.set_profile_velocity_percentage.assert_called_once()

    def test_set_scenario(self):
        self.controller.set_scenario.return_value = True

        self.assertEqual({'success': True}, self.post('/setScenarioId', {'scenario_id': '2'}).get_json())
        self.controller.set_scenario.assert_called_once_with(2)

    def test_set_scenario_reports_a_rejected_scenario(self):
        self.controller.set_scenario.return_value = False

        self.assertEqual({'success': False}, self.post('/setScenarioId', {'scenario_id': -1}).get_json())

    def test_set_scenario_without_a_whole_number_is_rejected(self):
        for body in ({}, {'scenario_id': ''}, {'scenario_id': 'a'}, {'scenario_id': 1.5}, {'scenario_id': None}):
            self.assertEqual(400, self.post('/setScenarioId', body).status_code, body)
        self.controller.set_scenario.assert_not_called()

    def test_save_to_file_passes_the_name(self):
        self.controller.save_recorded_moves_to_file.return_value = True

        self.assertEqual({'success': True}, self.post('/saveToFile', {'filename': 'moves'}).get_json())
        self.controller.save_recorded_moves_to_file.assert_called_once_with('moves')

    def test_save_to_file_reports_nothing_saved(self):
        self.controller.save_recorded_moves_to_file.return_value = False

        self.assertEqual({'success': False}, self.post('/saveToFile', {'filename': 'moves'}).get_json())

    def test_file_endpoints_without_a_filename_are_rejected(self):
        for path in ('/saveToFile', '/restoreFromFile'):
            for body in ({}, {'filename': 5}, {'filename': None}, {'filename': ['a']}):
                response = self.post(path, body)
                self.assertEqual(400, response.status_code, (path, body))
                self.assertFalse(response.get_json()['success'])
        self.controller.save_recorded_moves_to_file.assert_not_called()
        self.controller.restore_recorded_moves_from_file.assert_not_called()

    def test_restore_from_file(self):
        self.assertEqual({'success': True}, self.post('/restoreFromFile', {'filename': 'moves'}).get_json())
        self.controller.restore_recorded_moves_from_file.assert_called_once_with('moves')

    def test_restore_of_a_missing_file_is_a_404(self):
        self.controller.restore_recorded_moves_from_file.side_effect = FileNotFoundError('moves.json')

        response = self.post('/restoreFromFile', {'filename': 'moves'})

        self.assertEqual(404, response.status_code)
        self.assertFalse(response.get_json()['success'])

    def test_a_name_or_file_the_controller_rejects_is_a_400(self):
        self.controller.save_recorded_moves_to_file.side_effect = ValueError('bad name')
        self.controller.restore_recorded_moves_from_file.side_effect = ValueError('bad file')

        for path, message in (('/saveToFile', 'bad name'), ('/restoreFromFile', 'bad file')):
            response = self.post(path, {'filename': 'moves'})
            self.assertEqual(400, response.status_code)
            self.assertEqual({'success': False, 'error': message}, response.get_json())

    def test_clear_all(self):
        self.assertEqual({'success': True}, self.post('/clearAll').get_json())
        self.controller.clear_recorded_moves_and_positions.assert_called_once_with()

    def test_endpoints_only_accept_post(self):
        self.assertEqual(405, self.client.get('/xbox/start').status_code)
        self.controller.start.assert_not_called()


class FileEndpointTests(unittest.TestCase):
    """the file endpoints with a real controller, which saves in a temporary directory"""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(xbox_api, url_prefix='/xbox')
        self.client = app.test_client()

        self.outside = Path(tempfile.mkdtemp())
        self.directory = self.outside / 'recorded_moves'
        self.controller = XboxRobotController(RobotConfig(d1=13.92, a2=20, d4=22, d6=12), mock.Mock(), mock.Mock())
        self.controller.recorded_moves = [PoseToPoseMovement([Pose(1, 2, 3), Pose(4, 5, 6)], 2, [1, 2, 3])]
        for patcher in (mock.patch.object(global_objects, 'get_xbox_robot_controller', return_value=self.controller),
                        mock.patch.object(xbox_robot_controller, 'RECORDED_MOVES_DIR', self.directory)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.remove_files)

    def remove_files(self):
        for path in sorted(self.outside.rglob('*'), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        self.outside.rmdir()

    def post(self, path, filename):
        return self.client.post('/xbox' + path, json={'filename': filename})

    def test_bad_names_are_rejected_and_nothing_is_written(self):
        names = ['../escaped', '..\\escaped', '/tmp/escaped', 'C:\\escaped', 'sub/dir', 'a.b', 'a b', '', '.', '..',
                 'moves.json', 'a' * 101, 'name\n', 'na\x00me', 'm\u00f6ves']
        for name in names:
            for path in ('/saveToFile', '/restoreFromFile'):
                response = self.post(path, name)
                self.assertEqual(400, response.status_code, (path, name))
        self.assertEqual([], list(self.outside.rglob('*.json')))
        self.assertFalse(self.directory.exists())

    def test_save_and_restore_in_the_recorded_moves_directory(self):
        self.assertEqual({'success': True}, self.post('/saveToFile', 'my-moves_1').get_json())
        self.assertTrue((self.directory / 'my-moves_1.json').is_file())

        self.controller.recorded_moves = []
        self.assertEqual({'success': True}, self.post('/restoreFromFile', 'my-moves_1').get_json())
        self.assertEqual(1, len(self.controller.recorded_moves))
        self.assertIsInstance(self.controller.recorded_moves[0], PoseToPoseMovement)
        self.assertEqual(4, self.controller.recorded_moves[0].poses[1].x)

    def test_restore_of_a_missing_file_is_a_404(self):
        self.assertEqual(404, self.post('/restoreFromFile', 'nothing').status_code)

    def test_restore_does_not_run_code_from_the_file(self):
        self.directory.mkdir()
        marker = self.outside / 'executed.txt'
        crafted = ('{"py/reduce": [{"py/function": "builtins.eval"}, '
                   '{"py/tuple": ["open(%r, \'w\').close()"]}]}' % str(marker))
        (self.directory / 'crafted.json').write_text(crafted)

        response = self.post('/restoreFromFile', 'crafted')

        self.assertEqual(400, response.status_code)
        self.assertFalse(marker.exists())
        self.assertEqual(1, len(self.controller.recorded_moves))

    def test_restore_only_accepts_a_list_of_movements(self):
        self.directory.mkdir()
        movement = '{"py/object": "src.utils.movement.PoseToPoseMovement", "poses": [], "time": 1, "center": null}'
        contents = {'object': '{"py/object": "src.kinematics.kinematics_utils.Pose", "x": 1}',
                    'dict': '{"a": 1}', 'number': '5', 'mixed': '[%s, 5]' % movement, 'broken': '{nope',
                    'empty': '', 'noposes': '[{"py/object": "src.utils.movement.PoseToPoseMovement", "time": 1}]',
                    'noposes2': '[%s]' % movement, 'badposes': '[%s]' % movement.replace('"poses": []', '"poses": [5, 6]')}
        for name, content in contents.items():
            (self.directory / (name + '.json')).write_text(content)
            response = self.post('/restoreFromFile', name)
            self.assertEqual(400, response.status_code, name)
        old_moves = self.controller.recorded_moves
        self.assertEqual(1, len(old_moves))
        self.assertIsInstance(old_moves[0], PoseToPoseMovement)


if __name__ == '__main__':
    unittest.main()
