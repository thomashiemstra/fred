import unittest
from unittest import mock

from flask import Flask

from src import global_objects
from src.xbox_control.xbox_control_resource import xbox_api


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
        for speed in (0, 25, 50):
            self.assertEqual({'success': True}, self.post('/setSpeed', {'speed': speed}).get_json())
            self.controller.set_maximum_speed.assert_called_with(speed)

    def test_set_speed_outside_of_range_or_missing_is_rejected(self):
        for body in ({'speed': -1}, {'speed': 51}, {}):
            self.assertEqual({'success': False}, self.post('/setSpeed', body).get_json(), body)
        self.controller.set_maximum_speed.assert_not_called()

    def test_set_speed_accepts_a_string(self):
        self.assertEqual({'success': True}, self.post('/setSpeed', {'speed': '20'}).get_json())
        self.controller.set_maximum_speed.assert_called_once_with(20)

    def test_set_scenario(self):
        self.controller.set_scenario.return_value = True

        self.assertEqual({'success': True}, self.post('/setScenarioId', {'scenario_id': '2'}).get_json())
        self.controller.set_scenario.assert_called_once_with(2)

    def test_set_scenario_reports_a_rejected_scenario(self):
        self.controller.set_scenario.return_value = False

        self.assertEqual({'success': False}, self.post('/setScenarioId', {'scenario_id': -1}).get_json())

    def test_set_scenario_without_an_id_is_rejected(self):
        self.assertEqual({'success': False}, self.post('/setScenarioId', {}).get_json())
        self.controller.set_scenario.assert_not_called()

    def test_save_to_file_adds_the_json_extension(self):
        self.controller.save_recorded_moves_to_file.return_value = True

        self.assertEqual({'success': True}, self.post('/saveToFile', {'filename': 'moves'}).get_json())
        self.controller.save_recorded_moves_to_file.assert_called_once_with('moves.json')

    def test_save_to_file_reports_nothing_saved(self):
        self.controller.save_recorded_moves_to_file.return_value = False

        self.assertEqual({'success': False}, self.post('/saveToFile', {'filename': 'moves'}).get_json())

    def test_file_endpoints_without_a_filename_are_rejected(self):
        for path in ('/saveToFile', '/restoreFromFile'):
            self.assertEqual({'success': False}, self.post(path, {}).get_json(), path)
        self.controller.save_recorded_moves_to_file.assert_not_called()
        self.controller.restore_recorded_moves_from_file.assert_not_called()

    def test_restore_from_file(self):
        self.assertEqual({'success': True}, self.post('/restoreFromFile', {'filename': 'moves'}).get_json())
        self.controller.restore_recorded_moves_from_file.assert_called_once_with('moves.json')

    def test_clear_all(self):
        self.assertEqual({'success': True}, self.post('/clearAll').get_json())
        self.controller.clear_recorded_moves_and_positions.assert_called_once_with()

    def test_endpoints_only_accept_post(self):
        self.assertEqual(405, self.client.get('/xbox/start').status_code)
        self.controller.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
