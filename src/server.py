from flask import Flask, render_template
from src.xbox_controller.xbox_control_resource import xbox_api

from flask_cors import CORS
from flask_cdn import CDN
from flask_nav import Nav
from flask_nav.elements import Navbar, View
import src.global_objects as globals
from flask import jsonify


app = Flask(__name__)
CORS(app)
app.register_blueprint(xbox_api, url_prefix='/xbox')
# app.config['CDN_DOMAIN'] = 'mycdnname.cloudfront.net'
CDN(app)

nav = Nav()
nav.init_app(app)


@app.route('/')
def index():
    return render_template('index.html', username='dinges')


@app.route('/xbox')
def xbox():
    return render_template('xbox_control.html')


@app.route('/robotstatus')
def get_status():
    return jsonify(status=True)
    # dynamixel_servo_controller = globals.get_robot(globals.dynamixel_robot_arm_port)
    # return jsonify(status=dynamixel_servo_controller.get_status())


@app.route('/test')
def test():
    robot = globals.get_robot('COM5')
    print(robot)
    resp = jsonify(success=True)
    return resp


@app.route('/test2')
def test2():
    robot = globals.get_robot('COM5')
    print(robot)
    resp = jsonify(success=True)
    return resp


@app.route('/startcamera', methods=['POST'])
def start_camera():
    camera = globals.get_camera(0)
    camera.start_camera()
    return jsonify(success=True)


@app.route('/stopcamera', methods=['POST'])
def stop_camera():
    camera = globals.get_camera(0)
    camera.stop_camera()
    return jsonify(success=True)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')


