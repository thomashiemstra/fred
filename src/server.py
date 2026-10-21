import json

from flask import Flask, render_template, request, Response

from src import global_objects
from src.xbox_control.xbox_control_resource import xbox_api
from src.camera_control.camera_control_resource import camera_api

from flask_cors import CORS
from flask_cdn import CDN
# from flask_nav import Nav
from flask import jsonify

HOST = '127.0.0.1'
PORT = 5000
# the pages are served by this app, but their requests go to http://127.0.0.1:5000, so a page opened through
# localhost is another origin. Only these two may call the api from a browser
UI_ORIGINS = [f'http://{HOST}:{PORT}', f'http://localhost:{PORT}']

app = Flask(__name__)
CORS(app, origins=UI_ORIGINS)
app.register_blueprint(xbox_api, url_prefix='/xbox')
app.register_blueprint(camera_api, url_prefix='/camera')
CDN(app)


@app.before_request
def refuse_other_web_pages() -> Response | None:
    # cors only stops another web page from reading the answer, its simple posts still reach the robot; browsers
    # send the origin with every cross-origin post, scripts on this machine send none
    if request.method in ('GET', 'HEAD', 'OPTIONS'):
        return None
    origin = request.headers.get('Origin')
    if origin is not None and origin not in UI_ORIGINS:
        return Response('requests from other web pages are refused', status=403)
    return None


# nav = Nav()
# nav.init_app(app)


@app.route('/')
def index() -> str:
    return render_template('index.html', username='dinges')


@app.route('/xbox')
def xbox() -> str:
    return render_template('xbox_control.html')


@app.route('/camera')
def camera() -> str:
    return render_template('camera_control.html')


@app.route('/getJsonList', methods=['GET'])
def getJsonList() -> Response:
    list = [
       "one", "two", "three"
    ]
    return jsonify(list)


@app.route('/test', methods=['POST'])
def test() -> Response:
    print(request.get_json())
    resp = jsonify(success=True)
    return resp


@app.route('/startcamera', methods=['POST'])
def start_camera() -> Response:
    camera = global_objects.get_camera()
    camera.start_camera()
    return jsonify(success=True)


@app.route('/startcameraRecording', methods=['POST'])
def start_camera_recording() -> Response:
    camera = global_objects.get_camera()
    camera.start_camera_recording()
    return jsonify(success=True)


@app.route('/stopcamera', methods=['POST'])
def stop_camera() -> Response:
    camera = global_objects.get_camera()
    camera.stop_camera()
    return jsonify(success=True)


def start_server() -> None:
    # the server is the one process that shows the simulation, see global_objects.configure
    global_objects.configure(show_simulation=True)
    # only this machine, the api has no authentication
    app.run(debug=False, host=HOST, port=PORT)


