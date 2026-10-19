"""
Builds the urdf files of the simulated robot from src/simulation/urdf/fred.xacro with the link lengths of
src/robot_config.yaml (simulated_robot, in cm, the urdf is in metres). Run it from the project root after changing
either file:
    python -m src.simulation.build_urdf
    python -m src.simulation.build_urdf --check    (only reports whether the urdf files are up to date)

It writes fred_with_spheres_new.urdf (loaded by the simulation, with the control point spheres) and fred_new.urdf.
"""
import argparse
import os
import sys

import xacro

from src.robot_config import load_robot_configs

URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf')
XACRO_PATH = os.path.join(URDF_DIR, 'fred.xacro')
URDF_FILES = {'fred_with_spheres_new.urdf': True, 'fred_new.urdf': False}  # file name -> with spheres
LENGTHS = ('d1', 'a2', 'd4', 'd6')


def xacro_mappings(robot_config, spheres):
    """the xacro arguments for the link lengths of a RobotConfig (cm) in metres"""
    mappings = {name: '{:g}'.format(getattr(robot_config, name) / 100.0) for name in LENGTHS}
    mappings['spheres'] = 'true' if spheres else 'false'
    return mappings


def build(robot_config, spheres, xacro_path=XACRO_PATH):
    """returns the urdf text for the link lengths of the RobotConfig"""
    document = xacro.process_file(xacro_path, mappings=xacro_mappings(robot_config, spheres))
    return document.toprettyxml(indent='  ')


def expected_files(robot_config=None):
    """returns a mapping of urdf path to the text it should contain"""
    if robot_config is None:
        robot_config = load_robot_configs()['simulated_robot']
    return {os.path.join(URDF_DIR, name): build(robot_config, spheres) for name, spheres in URDF_FILES.items()}


def outdated_files(robot_config=None):
    """returns the urdf files whose content differs from what the xacro and the yaml produce"""
    outdated = []
    for path, text in expected_files(robot_config).items():
        try:
            with open(path) as urdf_file:
                current = urdf_file.read()
        except OSError:
            current = None
        if current != text:
            outdated.append(path)
    return outdated


def main(argv=None):
    parser = argparse.ArgumentParser(description='build the urdf files from fred.xacro and robot_config.yaml')
    parser.add_argument('--check', action='store_true', help='only report whether the urdf files are up to date')
    args = parser.parse_args(argv)

    if args.check:
        outdated = outdated_files()
        for path in outdated:
            print('{} is not up to date, run python -m src.simulation.build_urdf'.format(os.path.relpath(path)))
        if not outdated:
            print('the urdf files are up to date')
        return 1 if outdated else 0

    for path, text in expected_files().items():
        with open(path, 'w') as urdf_file:
            urdf_file.write(text)
        print('written {}'.format(os.path.relpath(path)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
