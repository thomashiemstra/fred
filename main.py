import os

from absl import app
from absl import flags

from src import global_constants, server

flags.DEFINE_boolean('use_simulation', False, 'Use the simulation instead of the real robot')
FLAGS = flags.FLAGS


def main(_):
    global_constants.use_simulation = FLAGS.use_simulation
    global_constants.root_dir = os.getcwd()
    server.start_server()


if __name__ == '__main__':
    app.run(main)
