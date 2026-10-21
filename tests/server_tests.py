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
        for url in ('/', '/xbox', '/camera'):
            with self.subTest(url=url):
                self.assertEqual(200, self.client.get(url).status_code)

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
