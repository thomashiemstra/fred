from flask import Flask, render_template
from xbox_controller.xbox_control_resource import xbox_api


app = Flask(__name__)
app.register_blueprint(xbox_api, url_prefix='/xbox')


@app.route('/')
def landing_page():
    return render_template('main_page.html')


if __name__ == '__main__':
    app.run(debug=True)
