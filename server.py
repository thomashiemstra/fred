from flask import Flask, render_template
from xbox_controller.xbox_control_resource import xbox_api

from flask_cors import CORS
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View
import globals
from flask import jsonify




app = Flask(__name__)
CORS(app)
bootstrap = Bootstrap(app)
app.register_blueprint(xbox_api, url_prefix='/xbox')

nav = Nav()
nav.init_app(app)


@app.route('/')
def index():
    return render_template('index.html', username='dinges')


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


@nav.navigation()
def mynavbar():
    return Navbar(
        'Robot control centre',
        View('Home', 'index'),
        View('Test', 'test'),
    )


if __name__ == '__main__':
    app.run(debug=True)


