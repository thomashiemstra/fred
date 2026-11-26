# The workspace limit

Oct 21, 2026: noticed while choosing the poses of `Tuning the servo gains.md`. On 23 October `radius_max` went from
57 to 60 cm. Done on 25 October 2026: the sphere is replaced by a reach measured from the shoulder, see "What was
done" at the end; the sections before it describe the sphere as it was.

## What it was

`WorkSpaceLimits` in `src/global_constants.py`: the gripper tip had to stay between `radius_min` 10 and `radius_max`
60 cm (57 until 23 October 2026) from the origin on the ground under the base, at `y >= y_min` (d6 + 0.1) and
`z >= z_min` (-1), and the wrist centre at `z >= wrist_z_min` (1.5). `workspace_violation` in `src/utils/workspace.py`
is the one check: the planner runs it on every step of a move, the xbox loop and the playback on every pose, the
recording and tuning scripts on their poses. `src/camera_control/board_to_board_robot_controller.py` carried its own
copy of the radius check. The sphere dated from "First attempt at radius based workspace limits" (`3a5543a`, July
2020).

## Why it was the wrong shape

The arm's reach is measured from the shoulder, 15.7 cm (d1) above the origin: the wrist centre can be at most
a2 + d4 = 42 cm from it, the tip 12 cm (d6) beyond. The sphere was measured from the ground.

| Distance of the gripper tip | cm |
|---|---|
| Reach from the shoulder, arm and gripper in line | 54 |
| Stretched out horizontally, from the ground origin | 56.2 |
| The limit, from the ground origin (57 until 23 October 2026) | 60 |
| Folded straight up, from the ground origin | 69.7 |

- Horizontally the limit never bit: at 60 it was 3.8 cm beyond the farthest the tip can get, at 57 it refused
  only poses within a centimetre of the elbow-straight singularity. It protected nothing there: for an unreachable
  target `calculate_ik` (`src/kinematics/kinematics.py`) clamps the elbow to the stretched arm and the planner raises
  "can not be reached" (`src/utils/movement.py`).
- Upwards it was a height limit in disguise: the tip could not be higher than 60 cm above the ground while the arm
  reaches 69.7 cm. Nothing hangs above the robot, so that limit protected nothing either.

## What it cost

- The tuning script: the pose with the upper arm horizontal and the lower arm at 45 degrees, plus 8 degrees on
  joint 3, was refused at 57 (57.0 cm) and passed at 60.
- `record_static_load.py`: the tall poses were refused; they carry almost no load on joints 2 and 3, so little was
  lost.
- Any Cartesian target higher than 60 cm from the origin was refused even when the arm could reach it.

## What was done

- `workspace_violation` refuses a wrist centre farther than `wrist_reach_max` = 41.8 cm from the shoulder (the
  origin of joint 2 in `forward_position_kinematics`), and a tip closer than `radius_min` = 10 cm to the origin on
  the ground, which keeps the gripper off the base. `radius_max` is gone; there is no `z_max`, nothing hangs above
  the robot.
- 41.8 rather than the 40 planned here: near the straight elbow the reach hardly changes with the bend
  (reach² = a2² + d4² + 2·a2·d4·cos(bend)), so 2 cm short of 42 means the elbow bent at least 35 degrees. That would
  refuse the `reach` pose of the gain tuning (elbow bent 25 degrees, 17 with its test move: 41.5 cm) and much of the
  arm stretched out low. 41.8 keeps the elbow bent at least about 11 degrees. The joint speed check of `check_move`
  is what refuses a straight line that gets too close to the straight elbow, the reach limit only keeps the poses
  themselves off it.
- The camera controller's own copy keeps the `radius_min` part only: the reach of the wrist depends on the
  orientation, which its pre-filter doesn't look at; the robot's `workspace_violation` in its loop checks the whole
  pose.
- The xbox control depended on the sphere for nothing but the reach: it has no limits of its own, only the loop's
  `workspace_violation`. Its start pose (30, 22, 4) puts the wrist 33.7 cm from the shoulder.
- Updated: `WorkSpaceLimits` and its docstring, the test limits in `tests/utils_tests/workspace_tests.py`,
  `movement_tests.py`, `movement_planning_tests.py`, `tests/xbox_control_tests/xbox_robot_controller_tests.py` and
  `tests/servo_controller_tests/static_load_scripts_tests.py`, the pose comment of `tune_servo_gains.py`, the pose
  section of `Tuning the servo gains.md` and both handovers.
- Tested offline only (the unit tests and the tuning poses checked against the new limits); not yet run on the arm.

## The floor of the gripper tip, 24 November 2026

`z_min` went from -2 to 2 cm (`wrist_z_min` stays 1). The -2 dated from the old wrist, which dropped onto its
fingertips with the tip at -1.3 by the kinematics; the wrist was rebuilt in November and the arm now rests with the
gripper pointing forward, its tip at about 3.7 cm and its wrist centre at about 2.3 cm. Checked that day: joint 2 is
157 mm above the table, joint 2 to 3 is 200 mm, joint 3 to the wrist centre 210 mm, and servos 1 and 2 read within 2
positions of their yaml offsets in the jig, so the heights of the model can be taken as they are. The tip of the
model (d6 14 cm) is the point where the gripper grips and stays; the closed fingers reach 150 mm from the wrist
centre, 1 cm beyond it, so 2 cm keeps the fingertips 1 cm above the table when the gripper points straight down.

`tests/servo_controller_tests/workspace_floor_tests.py` holds what has to keep working under the limit: from rest to
the tuning poses and to the rest pose, the step tests, the feedforward and swing sweeps of every joint, every
default load sweep, the recorder's random poses and the start of the gamepad. The margins: the arm rests with its
tip at 3.7 cm and the gamepad starts at 4.0 cm, so a floor above about 3.7 would stop every move from rest.

The random poses of `record_static_load.py` depend on the floor (a refused pose is skipped and the later ones
shift), so a seed gives other poses than it gave before this change; `--floor -2` gives the old ones.

## The 15 cm gripper and the wrist radius, 26 November 2026

Two changes, both in `WorkSpaceLimits`:

- **`d6` is 15 cm and `z_min` is 1 cm.** The tip of the model is now the end of the closed fingertips (the 150 mm
  measured on 24 November), where it was the point the gripper grips at 14 cm. The floor of the tip went from 2 to
  1 cm with it, so the fingertips keep the floor they had: 1 cm above the table with the gripper pointing straight
  down. `y_min` follows `d6` and is 15.1 cm. The arm rests with its tip at about 3.8 cm by the kinematics (3.7 with
  the 14 cm gripper), its wrist centre at about 2.3 cm. The simulation model was rebuilt for the longer gripper.
- **`wrist_radius_min` is 12.5 cm, new.** The wrist centre stays at least that far from the vertical axis through the
  base, at any height and in any direction. Nothing kept the wrist away from the base before: `radius_min` is for
  the tip, and on this day a tuning pose with the wrist 10 cm from the axis put the back of the wrist and gripper
  assembly against the shoulder housing. Measured with the parts touching: the wrist centre 10.4 cm from the base
  axis (10.7 cm from the shoulder, 13.2 cm above the table); 12.5 is that plus about 2 cm. With the gripper level
  and pointing away from the base the tip can therefore not come closer than 27.5 cm to the axis; with the gripper
  pointing down nothing changes, `y_min` already keeps the wrist farther out.

What moved with them: four default sweeps of `record_load_sweep.py` are a few degrees narrower (three reached
inside the wrist radius, one below `y_min` with the longer gripper; the recordings of 19 to 24 November were made
with the wider spans), the grip height of the block stacking is 1 cm, the forward distances of
`calibrate_board.py sink` start at 30 cm, and the warning height of the pose preview is 2 cm. The gravity and
feedforward numbers in `servo_config.yaml` do not depend on `d6`.

The gamepad's flip (the A button) has a rule of its own since the same day: it only starts with the wrist centre at
least 12 cm above the table (`FLIP_MIN_WRIST_HEIGHT` in `xbox_robot_controller.py`), because the gripper swings down
along the line of the forearm while the wrist turns over.

Still open: `wrist_reach_max` is 41.8 cm, set when a2 + d4 was 42; since the wrist was rebuilt on 17 November it is
41, so that limit no longer keeps the elbow bent. The speed check of the planner still refuses moves through the
straight elbow.
