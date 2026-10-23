# The workspace limit

Oct 21, 2026. Parked: noticed while choosing the poses of `Tuning the servo gains.md`, to be looked at another time.
On 23 October `radius_max` went from 57 to 60 cm; the question of the shape stays open.

## What it is

`WorkSpaceLimits` in `src/global_constants.py`: the gripper tip has to stay between `radius_min` 10 and `radius_max`
60 cm (57 until 23 October 2026) from the origin on the ground under the base, at `y >= y_min` (d6 + 0.1) and `z >= z_min` (-1), and the wrist
centre at `z >= wrist_z_min` (1.5). `workspace_violation` in `src/utils/workspace.py` is the one check: the planner
runs it on every step of a move, the xbox loop and the playback on every pose, the recording and tuning scripts on
their poses. `src/camera_control/board_to_board_robot_controller.py:36` carries its own copy of the radius check.
The sphere dates from "First attempt at radius based workspace limits" (`3a5543a`, July 2020).

## Why it is the wrong shape

The arm's reach is measured from the shoulder, 15.7 cm (d1) above the origin: the wrist centre can be at most
a2 + d4 = 42 cm from it, the tip 12 cm (d6) beyond. The sphere is measured from the ground.

| Distance of the gripper tip | cm |
|---|---|
| Reach from the shoulder, arm and gripper in line | 54 |
| Stretched out horizontally, from the ground origin | 56.2 |
| The limit, from the ground origin (57 until 23 October 2026) | 60 |
| Folded straight up, from the ground origin | 69.7 |

- Horizontally the limit never bites: at 60 it is 3.8 cm beyond the farthest the tip can get, at 57 it refused
  only poses within a centimetre of the elbow-straight singularity. It protects nothing there: for an unreachable
  target `calculate_ik` (`src/kinematics/kinematics.py`) clamps the elbow to the stretched arm and the planner raises
  "can not be reached" (`src/utils/movement.py:223`).
- Upwards it is a height limit in disguise: the tip may not be higher than 60 cm above the ground while the arm
  reaches 69.7 cm. Whether that is wanted depends on what is above the robot (the camera rig?), nothing in the code
  says.

## What it costs today

- The tuning script: the pose with the upper arm horizontal and the lower arm at 45 degrees, plus 8 degrees on
  joint 3, was refused at 57 (57.0 cm) and passes at 60. Not needed either way, the default poses are inside with
  margin and joint 2's largest lever is reachable with the gripper bent down.
- `record_static_load.py`: the tall poses are refused; they carry almost no load on joints 2 and 3, so little is lost.
- Any Cartesian target higher than 60 cm from the origin is refused even when the arm can reach it.

## When we come back to it

Change the shape rather than the number:

1. Measure the reach from the shoulder: the wrist centre within about 40 cm of the shoulder axis, 2 cm short of a
   straight elbow, which keeps the planner away from the singularity where the two IK solutions merge and a
   straight line needs unbounded joint speed (the speed check of `check_move` catches that too).
2. Add an explicit maximum height (`z_max`) if something hangs above the robot; drop the sphere otherwise.
3. Touches `workspace.py`, `WorkSpaceLimits` and its docstring, `tests/utils_tests/workspace_tests.py`, the copy in
   the camera controller, `TestLimits` in `tests/servo_controller_tests/static_load_scripts_tests.py`, and the pose
   section of `Tuning the servo gains.md`.

Questions to answer first: is anything above the robot; does the xbox control depend on the sphere for anything but
the reach (its start pose is at 30, 22, 4).
