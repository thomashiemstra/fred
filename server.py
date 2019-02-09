from flask import Flask, render_template
from xbox_controller.xbox_control_resource import xbox_api
from flask import jsonify

app = Flask(__name__)
app.register_blueprint(xbox_api, url_prefix='/xbox')


@app.route('/')
def index():
    return render_template('main_page.html')


@app.route('/test')
def test():
    resp = jsonify(success=True)
    return resp


if __name__ == '__main__':
    app.run(debug=True)
