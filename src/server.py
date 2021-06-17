import json

from flask import Flask, render_template, request, Response

from src import global_objects
from src.global_constants import dynamixel_robot_arm_port
from src.global_objects import get_robot
from src.xbox_control.xbox_control_resource import xbox_api

from flask_cors import CORS
from flask_cdn import CDN
from flask_nav import Nav
from flask import jsonify


app = Flask(__name__)
CORS(app)
app.register_blueprint(xbox_api, url_prefix='/xbox')
CDN(app)

nav = Nav()
nav.init_app(app)


@app.route('/')
def index():
    return render_template('index.html', username='dinges')


@app.route('/xbox')
def xbox():
    return render_template('xbox_control.html')

@app.route('/getJsonList', methods=['GET'])
def getJsonList():
    list = [
       "one", "two", "three"
    ]
    return jsonify(list)


@app.route('/test', methods=['POST'])
def test():
    print(request.get_json())
    resp = jsonify(success=True)
    return resp


@app.route('/test2')
def test2():
    robot = global_objects.get_robot('COM5')
    print(robot)
    resp = jsonify(success=True)
    return resp


@app.route('/startcamera', methods=['POST'])
def start_camera():
    camera = global_objects.get_camera()
    camera.start_camera()
    return jsonify(success=True)


@app.route('/startcameraRecording', methods=['POST'])
def start_camera_recording():
    camera = global_objects.get_camera()
    camera.start_camera_recording()
    return jsonify(success=True)


@app.route('/stopcamera', methods=['POST'])
def stop_camera():
    camera = global_objects.get_camera()
    camera.stop_camera()
    return jsonify(success=True)


def start_server():
    app.run(debug=False, host='0.0.0.0')


