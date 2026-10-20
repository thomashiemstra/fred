# Handover: Dynamixel robot controller and xbox control rework

What was changed in the robot software from 15 to 19 October 2026, what was measured on the physical robot, and what is
still open. The work is in commits `e46f2af` up to `0fd1b93` on master, plus the uncommitted work of 19 October
(the workspace check on the robot, the merged movement module, the gravity compensation).

Environment: Windows 10 Pro (build 19045), Python 3.12.2 in `venv/` at the repo root (`venv\Scripts\python.exe`,
gitignored), U2D2 on COM4. Every command below assumes that interpreter. The measurements were taken on this machine
during the rework, with the firmware and settings in the table; re-measure after changing the return delay time or the
latency timer. The gravity compensation plan is in `Handover Gravity and Spring Compensation.md`.

## Summary

- One `ServoHandler` drives all 7 servos through one U2D2. Goal positions, profile, pid and goal current go out in one
  sync packet (the servos don't answer a sync write, only a failure to send is detected); torque and operating mode
  are written per servo so every servo confirms. Reads are one sync packet **per bus**: the servos on the RS-485
  connector (1-3) and the TTL connector (4-7) can't hear each other's responses.
- The vendored Dynamixel SDK is replaced by the official `dynamixel-sdk==4.1.0` (`requirements.txt`), with a
  `BlockingPortHandler` that doesn't busy wait.
- The port of the robot is always found automatically: `global_objects.get_robot` sets up the bus with
  `setup_dynamixel_handlers(None, ...)`, which pings servos 1-6 on every adapter with usb id `0403:6014`, and hands the
  port and packet handler to `DynamixelRobotController` (20 October; the tests build it with a fake bus instead of
  patching). The servo power has to be on, otherwise the bus setup raises `DynamixelError`.
- The servo configuration lives in `servo_config.yaml`.
- Safety: failed reads never leave stale positions, targets are clamped, servos are only enabled when the arm is in a
  safe state (the check for servos 1 and 2 is a console prompt, issue 18), hardware errors and temperatures are
  monitored.
- Every move except the stick control is planned and checked completely before anything moves
  (`src/utils/movement.py`, merged from `cartesian_movement.py` and `movement_utils.py` on 19 October): the straight
  lines to and from the start pose, the joint space moves and the splines through recorded poses. The stick control
  still sends the IK result straight to the servos (issue 21).
- The xbox controller: only its control loop thread moves the robot, start/stop are safe to call at any time.
- The gripper is non-blocking and squeezes with up to 400mA.
- 590 tests (20 October), all hardware faked: `venv\Scripts\python.exe -m unittest discover -s tests -t . -p "*test*.py"`.

## The physical robot, measured

| What | Value |
|---|---|
| USB adapter | U2D2 (FTDI FT232H, USB id `0403:6014`), currently COM4 |
| Latency timer | set to 1ms in Device Manager (default 16ms), read at startup by `get_port_handler`, a warning is logged when it is above 1ms |
| Baud rate, protocol | 1 Mbps, protocol 2.0 |
| Buses | RS-485: servos 1, 2, 3 (XM430-W350, XM540-W270, XM430-W350). TTL: 4, 5, 6 (XC430), 7 (XL330-M288 gripper) |
| Model numbers | 1020 XM430-W350, 1120 XM540-W270, 1080 XC430-W240, 1070 XC430-W150, 1200 XL330-M288 |
| Firmware | 45 (servos 1-3), 48 (4-6), 51 (gripper) |
| Read times | 3 servos 4.0ms, 4 servos 5.0ms, whole arm 8ms (two packets), at 4-9% cpu; the vendored sdk busy waited at 95% |
| Return delay time | 250 (500us) on all servos, lowering it saves ~1.5-2ms per bus read; the register is in the eeprom and can only be written with the torque off |
| Supply voltage | arm 11.2-11.6V idle, gripper 4.9-5.2V |
| Temperatures | 24-38C in normal use, servo 5 runs warmest |
| Hardware temperature limit | 80C (servos 1-6), 70C (gripper); the software warns at 45C and stops at 50C (`temperature_warning`/`temperature_limit` in `servo_config.yaml`) |
| Profile velocity after power up | 0 (= unlimited), the controller sets it at startup |
| Servo 2 at rest | position 545. Until 18 October the yaml had `offset: -464` (0 degrees at 560), so this read as 15 positions below the calibrated range and `get_angle_from_position` reported 0 degrees for it (`np.interp` clamps to the range end, the position itself is not clamped, `get_servos_outside_calibrated_range` reports it). Recalibrated in a jig on 18 October: raw 457 at 0 degrees (lower arm horizontal), `offset: -567`; the old zero was 9.1 degrees too high, and the arm rests 7.7 degrees above horizontal. Servo 1 read 2394 at 90 degrees, `offset: 346` (was 354). See `Calibrating the servo offsets.md` |
| Joints 1 and 2 | 3:1 timing belt reduction after the servo gearbox, with the servo's magnetic encoder moved to the joint output shaft. The servo's position loop therefore closes on the joint, positions are joint positions (4096 per joint turn, like the other joints) and the belt is inside the loop; the motor turns 3 times per joint turn and delivers about 3 times its torque at the joint. Until the 2024 rebuild the encoder was on the motor side (`servo_configurations.py` at `9e5eb56`: 6144 positions per 180 degrees, extended position mode) |

Gripper (`gripper_test.py`, see the results in the comment of servo 7 in `servo_config.yaml`):

- State 0 is open, state 100 is closed (positions 1024 to 2048, 10.24 positions per state); the whole position range
  closes freely, 1mA when closed on nothing.
- Blocked by an object, it squeezes with about 2mA per position the target is beyond the object (with the position P
  gain of 500 in `servo_config.yaml`), capped by the goal current. With the target at 2048 that is at most ~580mA.
- No hardware errors and no heating (25 -> 29C) up to 600mA. The goal current is 400mA, the xbox controller clips the
  target to at most `GRIPPER_MAX_LEAD = 25` states (256 positions, ~500mA) beyond the actual state, so the goal
  current limits the squeeze. To squeeze harder raise both: the lead in states has to be about the goal current in mA
  divided by 20.
- The earlier shutdowns of the gripper came from the old code never writing the goal current, so it pushed with the
  full 1750mA.

## Running it

- `venv\Scripts\python.exe main.py` runs Flask on `0.0.0.0:5000` (issue 8); `--use_simulation` uses the pybullet
  robot instead of the real one. The control panel is `http://127.0.0.1:5000/`, the xbox page `/xbox`, the camera
  page `/camera`; the pages post to hard coded `http://127.0.0.1:5000/...` urls.
- The robot is created on the first `POST /xbox/start` by `global_objects.get_robot`: with `use_simulation` it is the
  simulated robot, otherwise it finds the bus and builds the `DynamixelRobotController` first, then starts pybullet
  and returns `CombinedRobot(DynamixelRobotController, simulated robot)`, so pybullet is needed for the real robot
  too; the serial port is only opened then, and when the servos are off nothing is left behind, so the next start
  simply retries. `get_robot()` takes no
  arguments (the port argument went on 20 October, the port is always found automatically) and is
  `lru_cache(maxsize=1)`: one robot per process. It is the one place that builds the robot: the calibration scripts
  and `robot_move_test.py` call it too (20 October) and run against the simulation when `use_simulation` is set.
  Whether the simulation is shown is decided once per process by `global_objects.configure(show_simulation=...)`:
  `start_server` turns the window on (1920x1080, real time simulation), the scripts leave it headless, and
  configuring after the robot is built raises.
- `requirements.txt` pins the packages of the venv (20 October); `tf_agents`, `keyboard` and `Arduino` are not
  installed.
- Close `main.py` and DYNAMIXEL Wizard before running a tool, only one program can open the port. Support the arm
  whenever the torque goes off, it drops.

## Architecture

### Servo layer (`src/robot_controllers/dynamixel_robot/`)

- `servo_config.yaml` + `servo_config.py`: per servo name, model, bus, mode, positions, angles (degrees) or states
  (gripper), offset, goal current, profile and pid, plus the temperature warning (45C) and limit (50C).
  `load_servo_config()` checks the structure (fields, mode name, `[first, last]` pairs, whole numbers,
  `temperature_warning <= temperature_limit`) and returns new `Servo` objects every call. It does not check value
  ranges: positions and offset against 0-4095, pid/profile/goal current against the register sizes (issue 17).
  Recalibrating means editing the yaml. The one other place with raw positions is the safety check
  (`servo_1_check`/`servo_2_check` in `src/utils/robot_controller_utils.py`, issue 18); adjust it when the positions
  or offset of servo 1 or 2 change. The controller tests use `tests/servo_controller_tests/test_servo_config.yaml`;
  `test_robot_starts_with_the_real_servo_configuration` and `servo_config_tests` also load the real file, so an
  invalid real yaml fails the tests.
- `servo_handler.py`: all bus access, behind one `RLock` (`bus_lock`). Sync write/read helpers, adjacent registers in
  one write (D/I/P, acceleration/velocity), signed present position, clamping of goal positions to the calibrated
  range (including the offset) and to 0-4095, `set_operating_mode` only writes when the mode differs (it can only be
  written with torque off), `hold_current_position`, `reboot` pings every 100ms for up to 3s and returns False for
  the servos that did not come back, `read_hardware_errors` (bit names from `HARDWARE_ERRORS` in
  `dynamixel_x_config.py`, only servos with an error are returned), `read_temperatures`,
  `read_temperature_and_voltage`; all reads return `None` when any servo can't be read.
  - Sync writes are not acknowledged: `move_to_angles`, `set_pid`, the profile and goal current writes return `True`
    when the packet was sent, not when a servo accepted it. `read_profile_velocity` is the only read-back.
  - `set_profile_velocity_percentage` (used by `/xbox/speed`) raises `ValueError` outside (0, 100] and never writes
    below 1: a profile velocity of 0 is unlimited.
  - `move_servo_to_pos` / `DynamixelRobotController.set_servo_position` (debug, used by the calibration scripts)
    write the raw position unclamped; only `move_servos`, `move_servo` and `set_gripper` are clamped.
- `dynamixel_utils.py`: `BlockingPortHandler` (1ms read timeout instead of polling; `perf_counter` timeouts because
  the sdk uses `time.time()`, which ticks every 15.6ms on Windows up to Python 3.12; packet timeout from the real
  latency timer with a 10ms minimum because the sdk formula ignores the return delay), latency timer detection
  (Linux sysfs, Windows FTDI registry; when it can't be read the handler assumes 16ms and logs a warning, slower
  timeouts but working), adapter discovery by usb id (other ports are never opened, a port held by another program is
  skipped with a warning), `DynamixelError` instead of `sys.exit`.
- `dynamixel_robot_controller.py`: the facade.
  - `perform_safety_checks` (constructor, default `True`, also what `global_objects.get_robot` uses):
    `initialize_servos` and the first position read raise `DynamixelError` on failure, and `enable_servos` runs
    `safety_check()` first. That check prints DANGER and blocks on `input()` when the raw present position of servo 1
    or 2 is above 2500 or below 0 (about 99 degrees for servo 1 and 170 degrees for servo 2 with the current
    calibration), anything but `y` refuses. It can fire from the server or `XboxRobotController.start()` and then
    waits on stdin (issue 18). The hardware error and temperature checks always run.
  - `enable_servos` refuses when `hold_current_position` can't read the positions, when hardware errors or
    temperatures can't be read, when a servo has a hardware error or is too warm, or when `safety_check()` fails; it
    sets the goal positions to the present positions before enabling the torque and disables all servos again when
    one fails.
  - `move_servos(angles)`: `angles` is a 7-element array indexed by servo id, index 0 unused. It stores the commanded
    angles as `_current_angles`, passes the whole array to `set_target_position_from_angle(angle, all_angles)` of
    servos 1-6 (the hook for the gravity compensation, unused by the plain `Servo` the yaml produces) and writes
    positions 1-6 in one sync write. The gripper is moved separately by `set_gripper`.
  - A reboot (the only way to clear a hardware error) disables the torque and wipes pid, profile and goal current;
    the operating mode is eeprom and stays. `reboot_all_servos` raises `DynamixelError` when a servo stays silent
    and otherwise reruns `initialize_servos`. Call `enable_servos()` again afterwards. Nothing calls it yet.
  - `get_current_pose` (forward kinematics), `get_angle_limits`, `get_gripper_state`, `get_hardware_errors`,
    `get_overheating_servos`, `get_servos_outside_calibrated_range` (a mapping of servo id to the present position
    and the calibrated range; the angle of such a servo is clamped to the range end so it is wrong;
    `robot_move_test` checks this before moving).
  - `print_position_error=True` (constructor) prints `present - target` of servos 1-6 every 11th `move_servos` call;
    it adds a bus read, so leave it off in normal use.

### Kinematics and movement

- Units: cm and radians in the kinematics and the planner, degrees only in `servo_config.yaml`. Link lengths come
  from `src/robot_config.yaml` (`src/robot_config.py` loads it into `global_constants.dynamixel_robot_config` and
  `simulated_robot_config`): d1 = 15.7 ground to shoulder, a2 = 20, d4 = 22, d6 = 12 wrist centre to gripper tip.
  `WorkSpaceLimits.y_min` follows d6. The kinematics tests use their own `RobotConfig(d1=13.92, ...)`.
- Angles: arrays of length 7 in radians, index 0 unused, index 1-6 = servo id 1-6. 1 = base yaw. 2 = shoulder, 0 =
  lower arm horizontal, pi/2 = straight up. 3 = elbow, pi/2 = upper arm in line with the lower arm, 0 = upper arm
  turned 90 degrees towards the base. 4 = wrist roll around the upper arm (no effect on the position when 5 = 0).
  5 = wrist bend, its sign is the flip. 6 = gripper roll, orientation only.
- `Pose`: `alpha` turns around world z, `beta` around y, `gamma` around x; at alpha = beta = gamma = 0 the gripper
  points along +y, so the wrist centre is 12 cm behind the tip (start pose (30, 22, 4): wrist at (30, 10, 4)).
  `Pose.time` is ignored by the planner. `calculate_angles_from_euler_matrix` (behind `forward_kinematics` and
  `get_current_pose`) is ambiguous at beta = +-pi/2: it returns gamma = 0 and puts the whole turn in alpha. The
  matrix is exact, so compare `get_euler_matrix()`, not alpha/gamma.
- `kinematics.forward_kinematics(angles, robot_config) -> Pose` inverts `inverse_kinematics` for reachable poses,
  both flips, to 1e-9 (`tests/kinematics_tests/forward_kinematics_tests.py`); flip = angle 5 negative. Not the
  other way around: the IK only returns the elbow-up branch.
- `inverse_kinematics` never fails: for an unreachable position it clamps the elbow (`d = 1`, `kinematics.py:33-34`)
  and returns the stretched arm pointing at the target. Check the result with `forward_position_kinematics`, as
  `plan_straight_line` does (`MAX_POSITION_ERROR` = 0.1 cm).
- `WorkSpaceLimits` (`src/global_constants.py`): the gripper tip stays 10-57 cm from the origin on the ground under
  the base, at `y_min` = d6 + 0.1 = 12.1 (keeps the wrist at y > 0.1, y = 0 breaks the IK) and at `z_min` = -1
  (the fingers may rest on the table); the wrist centre stays at `wrist_z_min` = 1.5, which keeps the wrist servo
  body (about 1.8 cm around its axis) off the table while the gripper points up. The arm starts from rest with the
  wrist servo lying on the table, so `wrist_z_min` can't be higher than that resting height.
- The check is one place: `robot.check_workspace(angles)` (`AbstractRobotController`, `src/utils/workspace.py`),
  with `robot.workspace_limits` (the class above by default, a robot can be given other limits) and the robot's
  link lengths. `move_servos` checks nothing, so it stays fast; whoever moves the robot checks first. The planner
  checks every step of a straight line, joint space move or spline (`check_move`), the xbox control loop checks
  every pose from the gamepad and holds the current pose while the gamepad asks for one outside (logged once), the
  board to board controller holds its pose, the recording script checks its poses. A pose on a limit is inside
  (1e-6 cm tolerance for the kinematics).
- `movement.plan_move_to_pose(robot, current_angles, target_pose)` plans a straight line with a slerp of
  the orientation and a minimum jerk profile, keeping the current flip (the flip of the target pose is ignored; the
  tip is `d6` = 12 cm from the wrist centre, a flip turns the wrist through straight and swings the tip on that
  radius, 15 cm at the start pose). It takes the robot for its link lengths, its workspace check and its joint
  ranges (`robot.get_angle_limits()`, None skips the range check). It raises `MovementException` when any step
  can't be reached, fails the workspace check, is outside a joint range (1e-6 rad tolerance), is faster than
  `recommended_max_servo_speed`, or doesn't start at the current angles. `plan_move_to_angles(robot,
  current_angles, target_angles)` does the same for a joint space move. Moving through the straight wrist (angle 5
  changing sign) is refused by the speed check on joint 4, not by a flip check.
- `src/utils/movement.py` is the one movement module (19 October, it replaces `cartesian_movement.py` and
  `movement_utils.py`). Every move is a `PlannedMove` (the angles of every step at `steps_per_second`, a
  description and the duration), checked by `check_move` and sent by `execute_move`. Straight lines
  (`plan_move_to_pose`, `plan_straight_line`), joint space moves (`plan_move_to_angles`, `move_between_poses`,
  `move_from_current_angles`, which does nothing within `ALREADY_THERE` = 0.2 rad of the target and raises when
  the angles can't be read) and the splines through recorded poses (`plan_spline`, `SplineMovement`,
  `PoseToPoseMovement`) all go through it. Moves with a given duration (joint space moves, splines, the turn
  towards a centre) are slowed down instead of refused when a joint would be faster than
  `recommended_max_servo_speed`: the duration is stretched up to `MAX_STRETCH` = 4 times (logged), beyond that
  the move is refused. A joint space move between differently flipped poses is such a slow move: joints 4 and 6
  turn by pi through the straight wrist and the tip swings around the wrist, within the workspace limits but
  not on the path the recorded poses suggest (issue 4).
- Planner constants: `steps_per_second` = 100, `recommended_max_servo_speed` = 4 rad/s, `MAX_POSITION_ERROR` =
  0.1 cm, `MAX_START_ANGLE_ERROR` = 0.05 rad, `ANGLE_LIMIT_TOLERANCE` = 1e-6 rad. Defaults `linear_speed` = 5 cm/s,
  `angular_speed` = 0.5 rad/s; duration = max(distance / linear_speed, rotation / angular_speed, 1 s), so even a move
  to the current pose takes 1 s.
- `execute_move(robot, move, should_stop)` sends the steps at `steps_per_second` on an absolute schedule (a slow
  `move_servos` does not accumulate delay). `should_stop` is optional and checked before every step; it returns True
  when the whole move was sent and False when it was stopped (`__control_robot` opens the gripper only then).
- A refused move is a `MovementException`; `XboxRobotController._plan_move_to` logs it and returns None. The first
  step comes from the IK, which always returns the elbow-up branch with angles 4 and 6 in (-pi, pi], so an arm that
  is physically in another configuration (elbow bent the other way, wrist turned past pi) is refused with "away from
  the current angles" until it is moved by hand.

### Xbox control (`src/xbox_control/`)

- `XboxRobotController` owns the running state (`is_running`). `start()` checks the gamepad, plans the move to the
  start pose before enabling the servos, checks the arm wasn't moved meanwhile and starts the control loop.
  `stop()` sets `_stop_requested` and waits for the control loop thread to end. The loop then moves back to the start
  pose in a straight line and disables the servos (`stop_robot`); when the straight line can't be planned the servos
  are disabled where the arm is, with a warning. `stop()` blocks until then, also during a playback move. After a
  crash of the loop the servos keep holding until `stop()` disables them.
- After an overheat stop (and after a crash) the loop has ended, but `is_running()` stays True: `/xbox/start`
  answers `already started` until `/xbox/stop` is called.
- Only the control loop thread moves the robot; requests from other threads that move it are put on a command queue
  the loop drains (`__handle_commands`). Find center is the only queued command. Playback of recorded moves checks
  the stop flag between moves, a running move can't be interrupted.
- Every loop step the pose from the gamepad is checked against the workspace limits of the robot
  (`robot.workspace_violation`); outside them the robot holds the current pose and the loop logs it once. The pose
  updater's own clamp on the tip stays as a first filter.
- Every second (`HARDWARE_CHECK_INTERVAL`) the loop reads the hardware errors and temperatures. A hardware error
  raises `RuntimeError` and crashes the loop on purpose: that servo already switched its torque off, so no move back
  is attempted and the other servos hold until `stop()`. A servo above `temperature_limit` sets `_stop_requested`, a
  normal stop. The check only runs between steps of the stick control; the start move, a playback, the find-center
  reset and the moves back are not interrupted by it.
- Start pose: `start_pose = Pose(30, 22.0, 4)`, with the flip the arm has at that moment. Three things are called
  speed: the sticks move at most `pose_updater.maximum_speed` cm/s, 5 by default, d-pad up/down change it in steps of
  `MAXIMUM_SPEED_STEP = 5` within 5 to 50, `/xbox/setSpeed` sets it directly; recorded moves play at `move_speed = 10`
  cm/s, hard coded; `/xbox/speed` is the profile velocity of the servos in percent. The right stick turns at most
  `maximum_angular_speed` = 0.5 rad/s, independent of the linear speed (20 October). Every axis ramps to the speed
  the stick asks for in `ramp_up_time` = 0.1 s, easing off and releasing ramp the same way, and a step covers the
  time that really passed since the previous one (capped at `MAX_STEP_TIME` = 0.1 s), so the speeds hold at any
  loop pace. The updater does no workspace clipping of its own, the loop's `workspace_violation` check is the only
  one; the platform sign of the right stick lives in `XboxController`.
- Buttons are a table, `_button_actions`, the first pressed button wins. `get_buttons()` resets the momentary buttons
  when read, the bumpers are the only buttons it doesn't reset, and they come first in the table (issue 23).
- Gripper: `lb`/`rb` move the target `gripper_state` (0 open, 100 closed) by `GRIPPER_STEP = 1` every loop step while
  held. `__move_gripper` reads the actual state every step (one bus read) and clips the target to actual +/-
  `GRIPPER_MAX_LEAD = 25`, so a blocked gripper squeezes with at most the goal current and lets go within 25 steps;
  when the read fails the target is only clipped to 0-100. After the start move `open_gripper()` sends
  `gripper_state`, which keeps the value of the previous run (issue 22).
- Recording and playback: `y` appends `current_pose` to `recorded_positions`; `x` turns 2 or more positions into a
  move at `move_speed` (`create_move`: 2 poses -> `PoseToPoseMovement`, more -> `SplineMovement` checked against the
  workspace limits), appends it to `recorded_moves` and keeps only the last position as the start of the next move;
  an impossible move moves straight back to the first position. `start` plays all recorded moves:
  `go_to_start_of_move` (a 4 s joint space move, planned and checked), then each move after checking the arm is
  within 0.4 rad of its first pose; the splines and pose to pose moves are planned and checked completely before
  they start (`plan_spline` and `move_between_poses` in `src/utils/movement.py`), a spline that would turn a joint
  too fast is slowed down, up to 4 times. A
  `MovementException` moves straight back to the start pose, when that fails the loop crashes. `recorded_moves`
  survive `stop()`; save/restore go through `jsonpickle` (issue 9).
- Find center: `/xbox/findcenter` queues `__start_find_center_mode`; the loop clears `recorded_positions` and
  `center`, sets `find_center_mode` and resets the orientation (a 1 s joint space move, `move_between_poses`; a refused turn is logged and the pose stays). In the mode the triggers
  (z) and the gamma stick are ignored. The second `y` sets `center = get_center(pos1, pos2)` and ends the mode. With
  a center the orientation is computed from the position every step (`get_angles_center`), the right stick does
  nothing until `/xbox/clearcenter`; moves created meanwhile keep the center.
- Threads: the control loop thread (`__start_internal`), the daemon gamepad monitor thread started in
  `XboxController.__init__`, and the Flask request threads (`app.run` is threaded). Everything that moves the arm
  (`_move_to`, `playback_recorded_moves`, `reset_orientation`, `handle_buttons`, `__move_gripper`) runs in the loop
  thread only; other threads use `_commands.put`. `lock` (RLock) guards the API methods (`clear_center`,
  `set_center`, save/restore/clear moves, `set_maximum_speed`, `set_scenario`, `set_profile_velocity_percentage`);
  the loop reads `recorded_moves`, `recorded_positions`, `center` and `find_center_mode` without it.
  `_start_stop_lock` serialises `start`/`stop`. `_stop_requested` is checked before each step of the start move, each
  loop iteration and between playback moves, nowhere else. Bus writes from a request thread (`/xbox/speed`) are
  serialised by `servo_handler.bus_lock`.
- `XboxController` survives a missing or unplugged gamepad: the monitor thread retries every `RECONNECT_INTERVAL` =
  1 s, `start()` refuses without a gamepad, unplugging during a run zeroes the sticks and the triggers and releases
  every button, so the arm ramps down and holds with the servos enabled, and resumes when the gamepad is back. On
  Windows a gamepad plugged in for the first time later is found too (through the private
  `inputs.devices._detect_gamepads()`); on Linux it has to be present when the process starts. On Windows
  `inputs.get_gamepad()` only returns after the gamepad state changes, so `connected` is False until a button or
  stick is touched after the controller was created; `wait_until_connected()` gives up after 2 s and `start()` logs
  `no gamepad connected`. Touch the gamepad before `/xbox/start`.
- REST: `server.py` mounts `xbox_control_resource.py` at `/xbox`. All return `{"success": bool}` unless noted:
  `/start` -> `start()` (`already started` as text when running); `/stop` -> `stop()` (`already stopped`), blocks
  until the arm is back; `/findcenter` -> `request_find_center_mode()` (queued, False when not running);
  `/clearcenter`; `/saveToFile`, `/restoreFromFile` `{"filename"}` -> `save_recorded_moves_to_file(name + '.json')`,
  `restore_recorded_moves_from_file`, relative to the working directory of the server; `/clearAll`; `/setSpeed`
  `{"speed": 0-50}` -> `set_maximum_speed`; `/speed` `{"percentage": 1-100}` -> `set_profile_velocity_percentage` (a
  bus write from the request thread); `/setScenarioId` `{"scenario_id"}` -> `set_scenario` (pybullet obstacles, only
  works with the simulated robot or the `CombinedRobot`); `GET /test` returns a fixed pose. The controller, robot,
  gamepad thread and pose updater are created by `global_objects.get_xbox_robot_controller` (`lru_cache`) on the
  first `/xbox` request.
- `CombinedRobot` (`src/robot_controllers/combined_robot/combined_robot.py`) sends every movement to the simulated
  robot first and then to the Dynamixel robot and returns the Dynamixel result; reads (angles, gripper state,
  hardware errors, temperatures, angle limits) come from the Dynamixel robot only; `enable_servos` returns False and
  leaves the simulation disabled when the physical servos refuse; `physics_client`, `body_id` and `control_points`
  come from the simulation. Add every new robot method here too, `/xbox/speed` crashed on a missing one.
- `xbox_robot_controller.py:11` imports `medium_scenarios` from `src.reinforcementlearning.environment.scenario`,
  which imports `pybullet` and `absl`; `set_scenario` builds and destroys their obstacles in `robot.physics_client`.

### Tools (run them yourself, they move or talk to the robot)

| Script | What it does |
|---|---|
| `python -m src.robot_controllers.dynamixel_robot.scan_servos` | lists the adapters and the servos that respond (read only) |
| `python -m src.robot_controllers.dynamixel_robot.robot_move_test` | reads the pose and moves in a straight line to `Pose(30, 25, 5)` (issue 28), asks before moving |
| `python -m src.robot_controllers.dynamixel_robot.gripper_test` | gripper range, squeeze sweep (goal currents up to 1500mA) and lead test, asks before moving, only the gripper is enabled and moves. The port is found by pinging servos 1-6, so the arm must be powered |
| `python -m src.simulation.build_urdf` | rebuilds the urdf files from `fred.xacro` and `robot_config.yaml`, no robot involved |
| `python -m src.robot_controllers.dynamixel_robot.calibration.record_static_load` | enables the servos after asking and visits random poses from both directions to record the static load (sag) of every servo to a csv, see the gravity handover; the arm moves through the ranges it prints |
| `python -m src.robot_controllers.dynamixel_robot.calibration.fit_static_load <csv>` | fits the compensation to a recording and prints the yaml blocks, no robot involved |
| `python -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets` | reads servos 1, 2 and 4 held by the jig (90, 0 and 0 degrees, `--jig 1=90 2=0 4=0`), prints the offsets and writes them to `servo_config.yaml` after asking (`--dry-run` to only print); never enables the torque, see `Calibrating the servo offsets.md` |

Other scripts in `src/robot_controllers/dynamixel_robot/`, none asks before moving: `calibration/angle_check.py`,
`gui_test.py` and `pid_test.py` run at import, call `get_robot()`, enable the servos and
move them (`pid_test.py` leaves the torque on). They use the debug API that still exists (`robot.servoN`,
`move_servo`, `move_servos`, `get_current_positions`, `set_pid_single_servo`, `set_servo_position`,
`get_servo_position`, `Servo.target_position`, `Servo.constant_offset`). `calibration/gravity_compensation.py` and
the four `record_going_*.py` scripts need the `keyboard` package (not installed) and an Arduino on COM5
(`ArduinoAngleReader` opens it in `__init__`); the `record_going_*` scripts move servos 2 and 3 immediately.
`gravity_compensation_with_math_angle2/3.py` are offline curve fits of the 2021 data in `calibration/results`.
`arduino/arduino_experiment.py` needs `Arduino` (arduino-python3), not installed. `position_recorder.py` is broken
(issue 19).

## Open issues

Found while writing the tests and in the code reviews of 15-18 October, not fixed. Issues 2, 13, 14, 19, 20 and 27
were fixed on 20 October and are listed below the table; the issues in the table have no test in the repository. Where = repo-relative `path:line` plus the symbol; the numbers
1-17 are the original ones, 18-28 were added on 18 October; the rows are grouped by area.

| # | Where | Issue | Impact |
|---|---|---|---|
| 17 | `src/robot_controllers/dynamixel_robot/dynamixel_utils.py:62-64` (`to_bytes`), `servo_config.py:140-143` | `to_bytes` masks values that don't fit (70000 in 2 bytes -> 4464) without raising; `_integer` only checks the type, so pid, goal current and profile values from the yaml have no range check, and a sync write has no status packet (`servo_handler.py:293`), so a value the servo rejects isn't reported either | low today, all values in the yaml fit; a typo writes a different gain or current limit, or none, while `set_pid`/`set_goal_current` return True |
| 12 | `src/robot_controllers/dynamixel_robot/servo_handler.py:71-72` (`reboot`) | the result of the reboot command is logged and discarded, a servo that still answers pings counts as rebooted | low, `reboot()` can report success while a servo kept its hardware error; `reboot_all_servos` has no callers outside the tests, `FakeBus.reboot` always succeeds |
| 16 | `src/robot_controllers/dynamixel_robot/servo.py:112,169` (`Servo2`, `Servo3`) | the lookup key is `str(round(angle / pi, 1))`, a small negative angle gives `'-0.0'`, which the min clamp misses, `KeyError('-0.0')`; a rounded value between two table keys also `KeyError`s | none, the classes are unused (`load_servo_config` only builds `Servo`), only `tests/utils_tests/servo_test.py` imports them |
| 5 | `src/robot_controllers/dynamixel_robot/servo_config.py:63` | servo ids are sorted as text (`str(key)`), 10 before 7; `servos` and `buses` keep that order | none with ids 1-7; with ids >= 10 the order of the servos in the sync packets changes, the values stay right |
| 18 | `src/utils/robot_controller_utils.py:18-41` (`servo_1_check`, `servo_2_check`), run by `enable_servos` | the safety check prints DANGER and blocks on `input()` when the raw position of servo 1 or 2 is above 2500 or below 0; `enable_servos` runs in the thread that calls it, so a `/xbox/start` request hangs on stdin (`EOFError` without a console); the thresholds are raw positions, they ignore `positions` and `offset` of the yaml | medium, `start()` hangs instead of failing; recalibrating servo 1 or 2 silently moves the angle at which it fires |
| 1 | `src/kinematics/kinematics_utils.py:73-75` (`Pose.__copy__`), `src/utils/movement.py:169` | `__copy__` drops `euler_matrix` and `plan_move_to_pose` copies the target, so a target given as a matrix is planned with alpha=beta=gamma=0; `Pose.__eq__` (`kinematics_utils.py:81-83`) raises `ValueError` when only one side has a matrix | medium, latent today: the only caller (`xbox_robot_controller.py:277`) never passes matrix poses; the board to board controller builds matrix targets (`board_to_board_robot_controller.py:124`) but sends them through `move_between_poses`/`move_from_current_angles` (`:138-144`), whose IK reads the matrix, so it moves correctly until it is switched to `plan_move_to_pose` |
| 3 | `src/kinematics/kinematics.py:193,199` (same copy in `src/reinforcementlearning/softActorCritic/custom_policy.py:38,44`, unused) | `jacobian_transpose_on_f` uses `c2` instead of `c1` in the joint 2 term of control points 1 and 2 (joints 1 and 3 are right); the test at `tests/kinematics_tests/kinematics_tests.py:181` only checks control point 3 | medium, wrong joint 2 force whenever a force on control point 1 or 2 has an x component: the repulsive forces in obstacle avoidance (`camera_control/obstacle_avoidance.py:202`, `simulation/obstacle_avoidance.py:219`) and the demonstrations the RL agent is cloned from |
| 4 | `src/kinematics/kinematics.py:75-82` | with the wrist exactly straight (angle 5 = 0) `r13`, `r23`, `r31`, `r32` are rounding noise, so angle 4 and angle 6 are both arbitrary and the gripper roll is lost; on either side of angle 5 = 0 they differ by pi when the flip is kept | medium, planned moves refuse the jump (the joint speed check of `check_move`, `src/utils/movement.py:86-91`); a joint space move between differently flipped poses spreads the pi turn over the move and is accepted when it is slow enough (the tip swings around the wrist); the stick control (`xbox_robot_controller.py:217` -> `dynamixel_robot_controller.py:139-143`) sends the pi turn of joints 4 and 6 to the servos unchecked |
| 21 | `src/xbox_control/xbox_robot_controller.py` (`_update_pose_within_the_workspace`, the loop at `:217`) | the stick control checks the workspace limits of every pose before moving (19 October), but still sends the IK result to the servos without the other checks of the planner: an unreachable target becomes the stretched arm (`kinematics.py:33-34`), joints 4 and 6 can turn by pi through the straight wrist (issue 4), joint ranges and speeds aren't checked; `move_to_pose` only lengthens the wait. The playback of recorded moves goes through the planner since 19 October | medium, the wrist spin is reachable with the sticks |
| 6 | `src/xbox_control/xbox_robot_controller.py:406` (`set_scenario`) | accepts `scenario_id == len(scenarios)` (`>` should be `>=`; `medium_scenarios` has 7, so id 7 passes); `:414` stores the id before `:415` raises | medium, `IndexError`, a 500 from `/xbox/setScenarioId`, and every later `set_scenario` fails at `:412` until the server restarts; the arm is not affected |
| 7 | `src/xbox_control/xbox_control_resource.py:92,106,115,131,138`, `src/xbox_control/xbox_robot_controller.py:389` | only `KeyError` is caught: a json body that is not an object or a non-string filename is a `TypeError`, `int()` of a bad value a `ValueError`, a missing restore file a `FileNotFoundError`; a non-json body is a 415 on Flask 3.0.3 | low, 500 errors, also from the UI forms (an empty speed field); the robot is not affected |
| 9 | `src/xbox_control/xbox_control_resource.py:129-133`, `src/xbox_control/xbox_robot_controller.py:383,389-392` | `get_filename` only appends `.json`, save/restore `open()` any path (relative to the working directory of the server, `..` and absolute paths included); `jsonpickle.decode` (3.0.4, `safe=False`) runs `py/reduce`/`py/repr` from the file; the result is stored unchecked | high, overwrite any `*.json` with the recorded moves, code execution from a crafted `*.json` the server can read; reachable from the network and from web pages through issue 8 |
| 10 | `src/xbox_control/xbox_robot_controller.py:279-282` (`stop_robot`, `start_pose` at `:47`) | stops at the start pose `Pose(30, 22.0, 4)` (4 cm above the table) and disables the torque, the arm drops; when the move back can't be planned the torque is disabled wherever the arm is; same in `src/camera_control/board_to_board_robot_controller.py:73-75` | high, every stop (api, overheating, stop during the start move or a playback) lets the arm fall unsupported; needs a rest pose on the table (`z_min = -5` allows it) |
| 15 | `src/xbox_control/xbox360controller/XboxController.py:229-234` (`_left_trigger`, `_right_trigger`) | both write `lr_trigger`, the z input (`xbox_pose_updater.py:61`); the last event wins: releasing one trigger zeroes it while the other is held, pressing the second reverses the direction | medium, z movement ramps to a stop until the held trigger moves again |
| 22 | `src/xbox_control/xbox_robot_controller.py:72,199-200,284-285` (`open_gripper`) | `gripper_state` is set once in the constructor; after the start move `open_gripper` sends it, so a second `start()` sends the last target of the previous run | low, the gripper doesn't open on a restart |
| 23 | `src/xbox_control/xbox_robot_controller.py:77-87,303-309` (`handle_buttons`), `xbox360controller/XboxController.py:119-135` | `get_buttons()` resets the momentary buttons when read and `handle_buttons` handles only the first pressed button in `_button_actions` order; the bumpers come first and are held | low, `a`/`b`/`x`/`y`/`start`/d-pad presses are lost while a bumper is held |
| 24 | `src/xbox_control/xbox_control_resource.py:93` (`/xbox/setSpeed`) | accepts `speed = 0` while the d-pad keeps it at 5-50 | low, the sticks stop moving the arm |
| 8 | `src/server.py:17,86` | listens on 0.0.0.0 with CORS for every origin, no authentication | high, anyone on the network, or any web page, can start, stop or disable the arm; the UI only uses 127.0.0.1 (the fetch urls are hard coded, so the 0.0.0.0 bind gives the UI nothing) |
| 25 | `src/server.py:56-59` (`/test2`) | debug route, calls `get_robot()` and prints it; without the port argument it can no longer evict the running robot, but a stray GET still builds the robot (pybullet window, serial port) before anyone pressed start | low, remove it |
| 26 | `src/camera/image_handlers.py:70`, `src/camera/util.py:37-43` (`get_calibrations`) | `get_default_aurco_image_handler` reads `src/camera/calibration/calibration_data_charuco.json`, which isn't in the repository (`*.json` is gitignored, the calibration scripts write it to a cwd relative `calibration/`); `get_calibrations` calls `sys.exit()` when the file is missing | medium, `/camera/start_aruco` raises `SystemExit` in the request thread before it even reaches issue 2 |
| 11 | `src/reinforcementlearning/environment/robot_env_utils.py:108,134-136` | `get_repulsive_forces_world` allocates `(3, n)` but fills `[i][0..2]` | none today, every caller uses 3 control points; another number raises `IndexError` |
| 28 | `src/robot_controllers/dynamixel_robot/robot_move_test.py:26-27` | `START_POSE = Pose(30, 25.0, 5)` with the comment "the same start pose as the xbox controller", which uses `Pose(30, 22.0, 4)` | low, the script and the xbox controller end 3 cm apart |

Fixed on 20 October, after the type check found most of them again (details in
`Handover Bugs Found By Type Checking.md`):

- 2: the camera code uses the aruco API of OpenCV 4.7+, checked against the legacy code on the recorded images
  (`184ec82`); the 6 aruco tests run again.
- 13: `get_current_gripper_position` returns `None` when the servos can't be read (`129511c`).
- 14: `Movement.to_json` is removed (`a2cd1a1`).
- 19: `ServoPositionRecorder.write` writes the numbers (`b5f7f5a`) and `record` takes servo 1 from index 0.
- 20: `Movement.is_robot_at_start_pose` raises a `MovementException` when the angles can't be read (`129511c`).
- 27: `requirements.txt` pins the packages of the venv; a fresh venv installed from it passes the type check and
  the tests.

Fixed in `6efdd99` (found by the code review of the xbox rework): `/xbox/speed` crashed because `CombinedRobot` had no
`set_profile_velocity_percentage`, now at `src/robot_controllers/combined_robot/combined_robot.py:64`.

## Not done, worth considering

- Hardware position limits in the servos (Min/Max Position Limit, eeprom; `# TODO` at
  `dynamixel_robot_controller.py:89`). Write them once with DYNAMIXEL Wizard (torque off) or implement the TODO in
  `initialize_servos`. Servo 3 is mounted reversed (`positions: [3072, 1024]`), so its min and max are swapped, and
  every servo's `offset` has to be included, as `Servo.clamp_position` does.
- Fast Sync Read (sdk 4.1 `fastSyncRead`); whether firmware 45/48/51 supports it is unchecked.
- Lowering the return delay time of the servos.
- Gravity/spring compensation: see `Handover Gravity and Spring Compensation.md`. The code is in place
  (`static_compensation.py`, the `compensation` field of the yaml, the two calibration scripts) but nothing has
  been recorded on the robot yet, so no servo has a compensation configured and the behaviour is unchanged.
- Raise `temperature_limit` to ~65C once every part around the servos is PETG-CF.
- `calibration/measure_offsets.py` measures the offsets of the servos a jig holds; the direction check (does the
  position go up with the angle) is still by hand.
- `xbox_robot_controller.py:11` imports `medium_scenarios` at import time; `scenario.py` only needs numpy, `Pose` and
  `BoxObstacle`, so this is a dependency direction issue. Inject the scenario list instead.
- A rest pose on the table for `stop_robot` (issue 10).

## Working on it

- Run the tests: `venv\Scripts\python.exe -m unittest discover -s tests -t . -p "*test*.py"` (590 tests on 20
  October, 5 skip: 4 `robot_env_test.py` tests because `tf_agents` is not installed, and `test_repulsive_vectors`
  until the control points of `fred_new.urdf` are checked). The pre-commit hook runs them after mypy; enable it once
  per clone with `git config core.hooksPath .githooks`.
- The tests never open a serial port: `FakeBus` in `tests/servo_controller_tests/dynamixel_robot_controller_tests.py`
  fakes the control table of every servo with the real sdk group classes on top, including the bus split; the xbox
  tests use a `FakeRobot` with real kinematics.
- The link lengths live in `src/robot_config.yaml` (`src/robot_config.py`), the servo calibration in
  `servo_config.yaml`. The urdf files the simulation loads (`src/simulation/urdf/fred_with_spheres_new.urdf`,
  `fred_new.urdf`) are built from `src/simulation/urdf/fred.xacro` with those lengths: after changing either, run
  `venv\Scripts\python.exe -m src.simulation.build_urdf` (`--check` only reports), `tests/simulation_tests` fails
  until the files are rebuilt. `fred.urdf` and `fred_with_spheres.urdf` are the 2021 geometry, only used by the old
  `sim.py` and `obstacle_avoidance.py` demos.
- `.gitignore` ignores `*.yaml`, `*.yml`, `*.json`, `*.png`, `*.avi` and `*.mp4` everywhere. The two servo configs and
  `src/robot_config.yaml` are explicit exceptions; the camera test resources (`tests/camera_tests/resources/`,
  `src/camera/calibration/calibration_data.json`) were force-added. A new config file or test resource of these
  types needs an exception or `git add -f`, `git status` won't list it.
- Before running anything that moves the arm: support the arm when the torque is disabled, it drops.
