import unittest
from unittest import mock

from src import server


class TestServer(unittest.TestCase):

    def setUp(self):
        self.client = server.app.test_client()

    def test_the_debug_route_that_builds_the_robot_is_gone(self):
        with mock.patch.object(server.global_objects, 'get_robot') as get_robot:
            self.assertEqual(404, self.client.get('/test2').status_code)
        get_robot.assert_not_called()
        self.assertNotIn('/test2', [rule.rule for rule in server.app.url_map.iter_rules()])

    def test_the_pages_are_served(self):
        for url in ('/', '/xbox', '/camera', '/blocks'):
            with self.subTest(url=url):
                self.assertEqual(200, self.client.get(url).status_code)

    def test_the_block_stacking_page_renders_without_building_anything(self):
        with mock.patch.object(server.global_objects, 'get_robot') as get_robot, \
                mock.patch.object(server.global_objects, 'get_camera') as get_camera, \
                mock.patch.object(server.global_objects, 'get_block_stacker') as get_block_stacker:
            response = self.client.get('/blocks')
        get_robot.assert_not_called()
        get_camera.assert_not_called()
        get_block_stacker.assert_not_called()
        page = response.get_data(as_text=True)
        self.assertIn('Block stacking', page)
        for button in ('cameraOn', 'cameraOff', 'look', 'checkTip', 'rehearse', 'stack', 'stop', 'layDown',
                       'torqueOff'):
            self.assertIn('id="{}"'.format(button), page)
        self.assertIn('/static/blocks.js', page)
        # the switch of the click mode, a checkbox that the toggle plugin turns into a switch
        self.assertRegex(page, r'<input id="clickMode" type="checkbox"')
        self.assertIn('Click to stack', page)
        self.assertIn('id="clickMessage"', page)
        # closed, so the file of the static response is not left open
        with self.client.get('/static/blocks.js') as script:
            self.assertEqual(200, script.status_code)

    def test_the_nav_has_the_link_to_the_block_stacking_page(self):
        for url in ('/', '/xbox', '/camera', '/blocks'):
            with self.subTest(url=url):
                self.assertIn('href="/blocks"', self.client.get(url).get_data(as_text=True))
        # only the page of the blocks has it as the active one
        active = r'class="nav-link active"\s+href="/blocks"'
        self.assertRegex(self.client.get('/blocks').get_data(as_text=True), active)
        self.assertNotRegex(self.client.get('/camera').get_data(as_text=True), active)

    def test_the_block_stacking_api_is_under_blocks(self):
        rules = {rule.rule for rule in server.app.url_map.iter_rules()}
        for path in ('/blocks', '/blocks/status', '/blocks/scan', '/blocks/start', '/blocks/stop', '/blocks/lay_down',
                     '/blocks/disable_servos', '/blocks/tip', '/blocks/camera/start', '/blocks/camera/stop',
                     '/blocks/click_mode'):
            self.assertIn(path, rules)

    def test_cors_is_limited_to_the_origins_of_the_ui(self):
        for origin in server.UI_ORIGINS:
            response = self.client.get('/getJsonList', headers={'Origin': origin})
            self.assertEqual(origin, response.headers.get('Access-Control-Allow-Origin'))
        response = self.client.get('/getJsonList', headers={'Origin': 'http://example.com'})
        self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))
        response = self.client.options('/camera/enable_servos', headers={
            'Origin': 'http://example.com', 'Access-Control-Request-Method': 'POST'})
        self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))

    def test_posts_from_other_web_pages_are_refused(self):
        for origin in ('http://example.com', 'null'):
            with self.subTest(origin=origin):
                response = self.client.post('/test', json={}, headers={'Origin': origin})
                self.assertEqual(403, response.status_code)
        with mock.patch.object(server.global_objects, 'get_robot') as get_robot:
            self.assertEqual(403, self.client.post('/xbox/stop', headers={'Origin': 'http://example.com'}).status_code)
        get_robot.assert_not_called()

    def test_posts_from_the_ui_and_from_scripts_are_accepted(self):
        for headers in [{'Origin': origin} for origin in server.UI_ORIGINS] + [{}]:
            with self.subTest(headers=headers):
                self.assertEqual(200, self.client.post('/test', json={}, headers=headers).status_code)

    def test_the_server_only_listens_on_this_machine(self):
        with mock.patch.object(server.global_objects, 'configure'), mock.patch.object(server.app, 'run') as run:
            server.start_server()
        run.assert_called_once()
        self.assertEqual('127.0.0.1', run.call_args.kwargs['host'])


if __name__ == '__main__':
    unittest.main()
