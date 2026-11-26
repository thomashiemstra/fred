import threading
import time
import unittest
from unittest import mock

from flask import Flask

from src import global_objects, server
from src.block_stacking import block_stacking_resource
from src.block_stacking.block_stacker import BlockStacker
from src.block_stacking.block_stacking_resource import block_api
from src.block_stacking.stacking_config import StackingConfigError

STATUS = {
    'state': 'idle', 'activity': 'ready', 'error': None, 'warnings': ['block 2 lies where the arm comes down'],
    'hover_only': False, 'board_visible': True,
    'blocks': [{'id': 1, 'x': 10.0, 'y': 25.0, 'yaw': 12.5, 'frames': 5, 'stable': True}],
    'plan': [{'id': 1, 'level': 0}], 'skipped': {'2': 'its position is not steady in the picture'}, 'tower': [],
    'click_mode': False, 'click_message': None}
TIP = {'tip': [10.5, 24.0, 1.5], 'nearest_block': 1, 'block': [10.0, 25.0], 'difference': [-0.5, 1.0]}


class BlockStackingResourceTests(unittest.TestCase):
    """the endpoints with a fake stacker and a fake camera, so no robot, camera or calibration is needed"""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(block_api, url_prefix='/blocks')
        self.client = app.test_client()

        # the autospec makes the test fail when the resource calls a method the stacker does not have
        self.stacker = mock.create_autospec(BlockStacker, instance=True)
        self.stacker.status.return_value = dict(STATUS)
        self.camera = mock.Mock()
        self.handler = mock.Mock(name='block image handler')
        self.xbox_running = False
        patchers = [
            mock.patch.object(global_objects, 'get_block_stacker', return_value=self.stacker),
            mock.patch.object(global_objects, 'get_camera', return_value=self.camera),
            mock.patch.object(global_objects, 'get_block_image_handler', return_value=self.handler),
            mock.patch.object(global_objects, 'xbox_controller_running', side_effect=lambda: self.xbox_running),
            # nothing may open the real robot
            mock.patch.object(global_objects, 'get_robot', side_effect=AssertionError('real robot')),
            # the handler is added once per process, every test starts without it
            mock.patch.object(block_stacking_resource, '_handler_added', False)]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.get_robot = global_objects.get_robot

    def post(self, path, json=None):
        return self.client.post('/blocks' + path, json=json)

    def assert_error(self, response, status):
        self.assertEqual(status, response.status_code, response.data)
        body = response.get_json()
        self.assertFalse(body['success'])
        self.assertIsInstance(body['error'], str)
        self.assertTrue(body['error'])

    # the camera

    def test_camera_start_adds_the_handler_and_starts_the_camera(self):
        response = self.post('/camera/start')

        self.assertEqual({'success': True}, response.get_json())
        self.camera.add_image_handler.assert_called_once_with(self.handler)
        self.camera.start_camera.assert_called_once_with()

    def test_the_handler_is_added_once_however_often_the_camera_is_started(self):
        for _ in range(3):
            self.assertEqual({'success': True}, self.post('/camera/start').get_json())
        self.post('/camera/stop')
        self.post('/camera/start')

        self.camera.add_image_handler.assert_called_once_with(self.handler)
        self.assertEqual(4, self.camera.start_camera.call_count)

    def test_the_handler_is_added_once_when_the_camera_is_started_at_the_same_time(self):
        self.camera.add_image_handler.side_effect = lambda handler: time.sleep(0.05)
        results = []

        def start():
            results.append(self.post('/camera/start').status_code)

        threads = [threading.Thread(target=start) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual([200] * 6, results)
        self.camera.add_image_handler.assert_called_once_with(self.handler)

    def test_a_missing_calibration_file_is_a_503_and_the_camera_is_not_started(self):
        global_objects.get_block_image_handler.side_effect = FileNotFoundError('camera calibration is missing')

        response = self.post('/camera/start')

        self.assert_error(response, 503)
        self.assertIn('camera calibration is missing', response.get_json()['error'])
        self.camera.add_image_handler.assert_not_called()
        self.camera.start_camera.assert_not_called()

    def test_a_bad_stacking_config_is_a_503(self):
        global_objects.get_block_image_handler.side_effect = StackingConfigError('board.position is missing')

        response = self.post('/camera/start')

        self.assert_error(response, 503)
        self.assertIn('board.position is missing', response.get_json()['error'])
        self.camera.start_camera.assert_not_called()

    def test_the_camera_starts_when_the_calibration_file_is_there_the_next_time(self):
        global_objects.get_block_image_handler.side_effect = FileNotFoundError('camera calibration is missing')
        self.assertEqual(503, self.post('/camera/start').status_code)
        global_objects.get_block_image_handler.side_effect = None

        self.assertEqual(200, self.post('/camera/start').status_code)

        self.camera.add_image_handler.assert_called_once_with(self.handler)
        self.camera.start_camera.assert_called_once_with()

    def test_camera_stop(self):
        self.assertEqual({'success': True}, self.post('/camera/stop').get_json())
        self.camera.stop_camera.assert_called_once_with()
        self.camera.add_image_handler.assert_not_called()

    # the status

    def test_status_is_the_status_of_the_stacker_with_success(self):
        response = self.client.get('/blocks/status')

        self.assertEqual(200, response.status_code)
        self.assertEqual(dict(STATUS, success=True), response.get_json())

    def test_status_does_not_build_the_robot_or_start_the_camera(self):
        for _ in range(3):
            self.client.get('/blocks/status')

        self.get_robot.assert_not_called()
        self.assertEqual([], self.camera.method_calls)
        self.assertEqual(['status'] * 3, [call[0] for call in self.stacker.method_calls])

    def test_status_when_the_stacker_can_not_be_built_is_a_503(self):
        for error in (FileNotFoundError('camera calibration is missing'), StackingConfigError('bad config')):
            with self.subTest(error=error):
                global_objects.get_block_stacker.side_effect = error
                response = self.client.get('/blocks/status')
                self.assert_error(response, 503)
                self.assertIn(str(error), response.get_json()['error'])

    def test_status_only_answers_get(self):
        self.assertEqual(405, self.client.post('/blocks/status').status_code)

    # the actions

    def test_scan_answers_the_status(self):
        self.stacker.scan.return_value = True

        response = self.post('/scan')

        self.assertEqual(200, response.status_code)
        self.assertEqual(dict(STATUS, success=True), response.get_json())
        self.stacker.scan.assert_called_once_with()

    def test_scan_while_busy_is_a_409(self):
        self.stacker.scan.return_value = False

        self.assert_error(self.post('/scan'), 409)

    def test_start(self):
        self.stacker.start.return_value = True

        self.assertEqual({'success': True}, self.post('/start').get_json())

        self.stacker.start.assert_called_once_with(hover_only=False)

    def test_start_without_a_body_or_with_an_empty_object_is_no_rehearsal(self):
        self.stacker.start.return_value = True
        self.assertEqual(200, self.client.post('/blocks/start').status_code)
        self.assertEqual(200, self.post('/start', {}).status_code)
        self.assertEqual(200, self.post('/start', {'hover_only': False}).status_code)

        self.assertEqual([mock.call(hover_only=False)] * 3, self.stacker.start.call_args_list)

    def test_start_as_a_rehearsal(self):
        self.stacker.start.return_value = True

        self.assertEqual({'success': True}, self.post('/start', {'hover_only': True}).get_json())

        self.stacker.start.assert_called_once_with(hover_only=True)

    def test_start_with_a_bad_hover_only_is_a_400(self):
        for value in (1, 0, 'true', 'false', '', None, [True], {'a': 1}, 1.5):
            with self.subTest(value=value):
                self.assert_error(self.post('/start', {'hover_only': value}), 400)
        self.stacker.start.assert_not_called()

    def test_start_with_a_body_that_is_not_a_json_object_is_a_400(self):
        for data, mimetype in (('[true]', 'application/json'), ('true', 'application/json'),
                               ('null', 'application/json'), ('{broken', 'application/json'),
                               ('hover_only=true', 'text/plain')):
            with self.subTest(data=data):
                response = self.client.post('/blocks/start', data=data, content_type=mimetype)
                self.assert_error(response, 400)
        self.stacker.start.assert_not_called()

    def test_start_while_the_gamepad_controls_the_robot_is_a_409(self):
        self.xbox_running = True

        response = self.post('/start', {'hover_only': True})

        self.assert_error(response, 409)
        self.assertEqual('the gamepad is controlling the robot, stop it first', response.get_json()['error'])
        self.stacker.start.assert_not_called()

    def test_lay_down_while_the_gamepad_controls_the_robot_is_a_409(self):
        self.xbox_running = True

        response = self.post('/lay_down')

        self.assert_error(response, 409)
        self.assertEqual('the gamepad is controlling the robot, stop it first', response.get_json()['error'])
        self.stacker.lay_down.assert_not_called()

    def test_the_other_actions_are_not_refused_while_the_gamepad_runs(self):
        self.xbox_running = True
        self.stacker.scan.return_value = True
        self.stacker.stop.return_value = True
        self.stacker.disable_servos.return_value = True
        self.stacker.read_tip.return_value = dict(TIP)

        for path in ('/scan', '/stop', '/disable_servos', '/tip', '/camera/stop'):
            with self.subTest(path=path):
                self.assertEqual(200, self.post(path).status_code)

    def test_start_while_busy_is_a_409(self):
        self.stacker.start.return_value = False

        self.assert_error(self.post('/start'), 409)

    def test_stop_lay_down_and_disable_servos(self):
        for path, method in (('/stop', self.stacker.stop), ('/lay_down', self.stacker.lay_down),
                             ('/disable_servos', self.stacker.disable_servos)):
            with self.subTest(path=path):
                method.return_value = True
                self.assertEqual({'success': True}, self.post(path).get_json())
                method.assert_called_once_with()

    def test_stop_lay_down_and_disable_servos_that_are_refused_are_a_409_with_a_sentence(self):
        for path, method in (('/stop', self.stacker.stop), ('/lay_down', self.stacker.lay_down),
                             ('/disable_servos', self.stacker.disable_servos)):
            with self.subTest(path=path):
                method.return_value = False
                response = self.post(path)
                self.assert_error(response, 409)
                self.assertIn(' ', response.get_json()['error'])

    # the click mode

    def set_click_mode(self, on):
        self.stacker.status.return_value = dict(STATUS, click_mode=on)

    def test_the_status_has_the_click_mode_keys(self):
        self.set_click_mode(True)
        self.stacker.status.return_value['click_message'] = 'block 3 is on the tower'

        body = self.client.get('/blocks/status').get_json()

        self.assertIs(True, body['click_mode'])
        self.assertEqual('block 3 is on the tower', body['click_message'])

    def test_click_mode_on_starts_it(self):
        self.stacker.start_click_mode.return_value = True

        response = self.post('/click_mode', {'on': True})

        self.assertEqual(200, response.status_code)
        self.assertEqual({'success': True}, response.get_json())
        self.stacker.start_click_mode.assert_called_once_with()
        self.stacker.stop.assert_not_called()

    def test_click_mode_on_while_busy_is_a_409(self):
        self.stacker.start_click_mode.return_value = False

        response = self.post('/click_mode', {'on': True})

        self.assert_error(response, 409)
        self.assertEqual('the arm is busy, stop it first', response.get_json()['error'])

    def test_click_mode_on_while_the_gamepad_controls_the_robot_is_a_409_and_nothing_is_called(self):
        self.xbox_running = True

        response = self.post('/click_mode', {'on': True})

        self.assert_error(response, 409)
        self.assertEqual('the gamepad is controlling the robot, stop it first', response.get_json()['error'])
        self.assertEqual([], self.stacker.method_calls)

    def test_click_mode_off_with_the_mode_on_stops_it(self):
        self.set_click_mode(True)
        self.stacker.stop.return_value = True

        response = self.post('/click_mode', {'on': False})

        self.assertEqual({'success': True}, response.get_json())
        self.stacker.stop.assert_called_once_with()
        self.stacker.start_click_mode.assert_not_called()

    def test_click_mode_off_is_not_refused_while_the_gamepad_runs(self):
        self.xbox_running = True
        self.set_click_mode(True)
        self.stacker.stop.return_value = True

        self.assertEqual(200, self.post('/click_mode', {'on': False}).status_code)

    def test_click_mode_off_with_the_mode_off_is_a_409_and_does_not_stop_a_run(self):
        # a normal stacking run is going on, the switch must leave it alone
        self.stacker.status.return_value = dict(STATUS, state='stacking', click_mode=False)

        response = self.post('/click_mode', {'on': False})

        self.assert_error(response, 409)
        self.assertEqual('the click mode is not on', response.get_json()['error'])
        self.stacker.stop.assert_not_called()

    def test_click_mode_off_when_the_mode_ended_before_the_stop_is_a_409(self):
        self.set_click_mode(True)
        self.stacker.stop.return_value = False

        self.assert_error(self.post('/click_mode', {'on': False}), 409)

    def test_click_mode_without_a_usable_body_is_a_400(self):
        self.assert_error(self.client.post('/blocks/click_mode'), 400)
        for data in ('[true]', 'null', '{broken'):
            with self.subTest(data=data):
                response = self.client.post('/blocks/click_mode', data=data, content_type='application/json')
                self.assert_error(response, 400)
        self.assert_error(self.client.post('/blocks/click_mode', data='on=true', content_type='text/plain'), 400)
        self.assert_error(self.post('/click_mode', {}), 400)
        self.assert_error(self.post('/click_mode', {'hover_only': True}), 400)
        self.assertEqual([], self.stacker.method_calls)

    def test_click_mode_with_an_on_that_is_not_a_bool_is_a_400(self):
        for value in (1, 0, 'true', 'false', '', None, [True], {'a': 1}, 1.5):
            with self.subTest(value=value):
                self.assert_error(self.post('/click_mode', {'on': value}), 400)
        self.assertEqual([], self.stacker.method_calls)

    def test_click_mode_is_a_503_when_the_stacker_can_not_be_built(self):
        for error in (FileNotFoundError('camera calibration is missing'), StackingConfigError('bad config')):
            global_objects.get_block_stacker.side_effect = error
            for on in (True, False):
                with self.subTest(error=error, on=on):
                    response = self.post('/click_mode', {'on': on})
                    self.assert_error(response, 503)
                    self.assertIn(str(error), response.get_json()['error'])

    def test_click_mode_only_answers_post(self):
        self.assertEqual(405, self.client.get('/blocks/click_mode').status_code)
        self.assertEqual([], self.stacker.method_calls)

    def test_a_click_mode_post_from_another_web_page_is_refused(self):
        response = server.app.test_client().post('/blocks/click_mode', json={'on': True},
                                                 headers={'Origin': 'http://example.com'})

        self.assertEqual(403, response.status_code)
        self.assertEqual([], self.stacker.method_calls)

    def test_tip(self):
        self.stacker.read_tip.return_value = dict(TIP)

        response = self.post('/tip')

        self.assertEqual(200, response.status_code)
        self.assertEqual(dict(TIP, success=True), response.get_json())

    def test_tip_without_a_block_in_the_picture(self):
        tip = {'tip': [1.0, 2.0, 3.0], 'nearest_block': None, 'block': None, 'difference': None}
        self.stacker.read_tip.return_value = tip

        self.assertEqual(dict(tip, success=True), self.post('/tip').get_json())

    def test_tip_that_can_not_be_read_is_a_409_with_the_message(self):
        self.stacker.read_tip.return_value = {'error': 'the arm is running, read the tip when it is idle'}

        response = self.post('/tip')

        self.assert_error(response, 409)
        self.assertEqual('the arm is running, read the tip when it is idle', response.get_json()['error'])

    def test_every_action_is_a_503_when_the_stacker_can_not_be_built(self):
        global_objects.get_block_stacker.side_effect = StackingConfigError('bad config')
        for path in ('/scan', '/start', '/stop', '/lay_down', '/disable_servos', '/tip'):
            with self.subTest(path=path):
                self.assert_error(self.post(path), 503)

    def test_the_actions_only_answer_post(self):
        for path in ('/camera/start', '/camera/stop', '/scan', '/start', '/stop', '/lay_down', '/disable_servos',
                     '/tip'):
            with self.subTest(path=path):
                self.assertEqual(405, self.client.get('/blocks' + path).status_code)
        self.assertEqual([], self.stacker.method_calls)
        self.assertEqual([], self.camera.method_calls)

    def test_a_post_from_another_web_page_is_refused_before_it_reaches_the_arm(self):
        client = server.app.test_client()

        for path in ('/blocks/start', '/blocks/lay_down', '/blocks/disable_servos', '/blocks/camera/start'):
            with self.subTest(path=path):
                response = client.post(path, json={'hover_only': True}, headers={'Origin': 'http://example.com'})
                self.assertEqual(403, response.status_code)
        self.assertEqual([], self.stacker.method_calls)
        self.assertEqual([], self.camera.method_calls)

    def test_a_post_from_the_page_itself_is_accepted(self):
        self.stacker.stop.return_value = True
        client = server.app.test_client()

        response = client.post('/blocks/stop', headers={'Origin': server.UI_ORIGINS[0]})

        self.assertEqual({'success': True}, response.get_json())


if __name__ == '__main__':
    unittest.main()
