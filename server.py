from flask import Flask, render_template
from xbox_controller.xbox_control_resource import xbox_api
from flask import jsonify
from flask_cors import CORS
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View

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
    resp = jsonify(success=True)
    return resp


@nav.navigation()
def mynavbar():
    return Navbar(
        'Robot control centre',
        View('Home', 'index'),
    )


if __name__ == '__main__':
    app.run(debug=True)
