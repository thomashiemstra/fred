# Handover: Bugs found by the type check

On 20 October 2026 every function in `src` and `main.py` got type hints, and mypy became part of the pre-commit hook
(commit `477cbaa`). mypy found problems in the existing code along the way. They were first marked with a targeted
`# type: ignore[code]` and a reason instead of fixed, then fixed later that day where no hardware was needed. This
document lists what was fixed and how it was verified, and what is still open. The open items need the camera or the
robot.

Environment: Windows 10 Pro, Python 3.12.2 in `venv/` at the repo root, opencv-python 4.9.0, tensorflow 2.19.
`tf_agents` is imported by the reinforcement learning code and the camera obstacle avoidance but is **not installed**
in the venv, so that code can't run at all right now. `requirements.txt` pins the packages of the venv.

## Summary

| # | Bug | Status |
|---|---|---|
| 1 | cv2.aruco functions removed in OpenCV 4.7, camera control couldn't start | fixed, poses match the legacy code on the recorded images; check with the real camera |
| 2 | `ArduinoAngleReader.stop()` hung | fixed, tested on a loopback port |
| 3 | `create_envs` crashed on its `None` defaults | fixed, type checked only (needs `tf_agents`) |
| 4 | `set_center` crashed on poses that define no center | fixed, tested |
| 5 | `get_current_angles()` returning `None` wasn't handled | fixed, tested |
| 6 | camera calibration lost its result in `json.dump` | fixed, the script is untested (needs the camera) |
| 7 | `board_not_detected` could return `None` | fixed, tested |
| 8 | `aruco_camera_calibration` without frames | fixed, the script is untested (needs the camera) |
| 9 | `findChessboardCorners` flags in the wrong argument | fixed, the scripts are untested (needs the camera) |
| 10 | `Movement.to_json` called a method `Pose` doesn't have | removed, it had no callers |
| 11 | `ServoPositionRecorder.write` raised `TypeError`, `record` was off by one servo | fixed, tested |
| 12 | small things and stale scripts | fixed |
| 13 | camera calibration scripts: image size, termination criteria, where they write | fixed, untested (needs the camera) |

The test suite has 667 tests (21 October), all passing. 5 are skipped: 4 need `tf_agents`, and
`test_repulsive_vectors` waits until the control points of `fred_new.urdf` are checked.

## Still open

### Check the OpenCV port with the real camera

The ported pose code was compared with the legacy code: `opencv-contrib-python==4.6.0.66` in a throwaway venv ran the
unchanged old handlers on the 4 images in `tests/camera_tests/resources`.

- Board poses, the board-to-board pose and the 9 marker poses on `2_boards.png` are identical (difference 0.0).
  `tests/camera_tests/aruco_legacy_reference_test.py` keeps them identical, with the reference numbers in
  `legacy_opencv_reference.py`.
- On `image_with_markers_and_board.png`, the new detector places a corner of markers **17 and 29** 1-2 px differently
  than 4.6 did. That changes marker 17's relative tvec by up to 3.6 and its rotation by about 1.1. The pose math itself reproduces the legacy poses exactly from the legacy corners, so the
  difference is the 4.7+ detector, and no `DetectorParameters` setting restored the old corners. 17 and 29 are small
  markers, and they are the ones `camera_control` uses.

With the camera: put a board and the markers at measured positions and check the reported poses, especially of the
small markers. `cv2.drawFrameAxes`, which replaces `drawAxis`, only runs when drawing is on and hasn't been exercised.

If the camera that made `src/camera/calibration/calibration_data.json` is gone, recalibrate first. The calibration
scripts (`aruco_camera_calibration.py`, `charuco_camera_calibration_from_images.py`, `camera_calibration.py`,
`camera_calibration_from_images.py`) and `aruco_test.py` are ported but untested; they compile and pass mypy.

### Optional checks on the robot

- Fix 5: start a move, cut the servo power during it, and check that it stops with a `MovementException` (or a
  `DynamixelError` from `move_to_pose`) and the robot holds still.
- Fix 2 was only tested on a loopback port. If the Arduino angle sensor (COM5) is used again, check that `stop()`
  returns. The gravity handover suggests a new recording script instead of `calibration/gravity_compensation.py`;
  if that script goes, the Arduino reader can go with it.

## What was fixed

### 1. OpenCV aruco API

All 34 calls to the removed module functions (`DetectorParameters_create`, `detectMarkers`, `drawAxis`,
`estimatePoseSingleMarkers`, `interpolateCornersCharuco`, `estimatePoseCharucoBoard`,
`calibrateCameraCharucoExtended`, `refineDetectedMarkers`) now go through helpers in `src/camera/util.py`:

| Helper | Uses |
|---|---|
| `detect_aruco_markers` | `cv2.aruco.ArucoDetector` |
| `interpolate_charuco_corners` | `cv2.aruco.CharucoDetector.detectBoard` with the already detected markers |
| `estimate_pose_charuco_board` | `board.matchImagePoints` + `cv2.solvePnP`, with the legacy checks: at least 4 corners, not collinear |
| `estimate_pose_single_markers` | `cv2.solvePnP` with the default iterative solver; `SOLVEPNP_IPPE_SQUARE` gave different poses than the legacy function (up to 4.4 in rvec) |
| `calibrate_camera_charuco` | `board.matchImagePoints` + `cv2.calibrateCameraExtended` |

`drawAxis` became `cv2.drawFrameAxes`. The public interface of `ArucoImageHandler` and `BoardToBoardImageHandler` and
the shapes of their vectors are unchanged. The legacy `refineDetectedMarkers` calls threw their results away, so
they did nothing and were dropped. `BoardToBoardImageHandler.detect_board` now interpolates the target board's
corners with the target board instead of `self.board`; both boards have the same geometry, so the result is the
same. The aruco tests that were skipped run again.

### 2. `ArduinoAngleReader.stop()`

There were two bugs: `stop()` set `_running = True` instead of `False`, and it held the lock while joining the
thread, which the reading thread needs to see the flag. It now sets the flag under the lock and joins outside it.
`tests/servo_controller_tests/arduino_angle_reader_tests.py` runs the reader on pyserial's `loop://` port; the stop
test fails on the old code.

### 3. `create_envs`

With obstacles, missing scenarios now raise a `ValueError` instead of `None + None`. The Linux parallel branch also
ignored the scenarios completely, which crashed `soft_actor_critic` with parallel environments; both branches now
build the same environments: training on `scenarios + train_scenarios`, evaluation on `scenarios`.
`behavioral_cloning.main` passes the scenarios `soft_actor_critic` trains on (`sensible_scenarios`,
`train_scenarios`), since that training starts from the behavioral cloning agent. **Untested**: `sac_utils` can't be
imported without `tf_agents`.

### 4. `set_center`

When `get_center` rejects the two recorded poses, `set_center` now drops them, logs a warning and stays in
find-center mode, so two new poses can be recorded. Before, it raised a `TypeError` from the Y button.

### 5. `get_current_angles()` returning `None`

- `Movement.is_robot_at_start_pose` raises a `MovementException` when the angles can't be read, so `Movement.move`
  refuses the move the same way as when the robot isn't at the start pose.
- `DynamixelRobotController.get_current_gripper_position` returns `None` when the servos can't be read.
- `DynamixelRobotController.move_to_pose` reads the current angles when they aren't known yet (only before the first
  move, when the initial read failed without safety checks), and raises a `DynamixelError` if that read fails too.
  The control loops that call it already catch exceptions and hold position.

### 6-9. Camera details

- 6: `camera_calibration.py` writes the matrices with `.tolist()`.
- 7: `board_not_detected` returns `True` when the vectors don't have length 3.
- 8: `aruco_camera_calibration.py` exits with a message when no frame was read.
- 9: the flags of `findChessboardCorners` are passed with `flags=`.

### 10-12. The rest

- 10: `Movement.to_json` is removed; recorded moves are saved with `jsonpickle`.
- 11: `ServoPositionRecorder.write` writes one line per recording: id, current and target position of each servo.
  `record` took servos 2-7 as 1-6 (the gripper as 6); it now starts at index 0, where the controller keeps servo 1.
- 12:
  - The attribute that shadowed `BoardToBoardRobotController.should_filter()` is now `_should_filter`, and the
    control loop reads it through the locked method.
  - `XboxController._start` stores a bool.
  - `Servo2` and `Servo3` raise a `ValueError` without dynamic offsets.
  - `sim.py` calls `create_move` with the right arguments and exits when the arc is outside the workspace.
  - `scratches/json_dumping.py` passes the `SplineMovement` arguments in the right order.
  - `robot_move_test.py` stops with a message when `use_simulation` is set.

### 13. Calibration script details

The two ChArUco scripts passed `gray.shape`, which is (height, width), as the image size; OpenCV expects
(width, height), and their initial camera matrix has its principal point at (989, 505), the middle of a 1920x1080
image. Their termination criteria were `TERM_CRITERIA_EPS & TERM_CRITERIA_COUNT`, which is 0, a bug of the
OpenCV tutorial they come from; now `+`, like the other criteria in those files. The chessboard scripts already
reversed the shape.

`camera_calibration.py` wrote `calibration/calibration_data.txt`, which nothing read. It now writes
`calibration_data.json`, and all calibration scripts write to `src/camera/calibration/` instead of a
`calibration/` relative to where they are started (21 October, issue 26 of the robot controller handover). The
handlers use `calibration_data_charuco.json` when it exists, otherwise `calibration_data.json`.

Every fix removed its `type: ignore`; 53 are left, all library typing gaps or places where mypy can't follow the logic.

## Not bugs

The remaining `type: ignore` lines don't need fixing:

| Reason in the comment | What it is |
|---|---|
| `currentframe never returns none on cpython`, `currentframe is optional in typeshed` | `inspect.currentframe()` is typed optional |
| `missing from the opencv stubs, exists at runtime` | `cv2.VideoWriter_fourcc` |
| `stubs expect a list of marker corners, a charuco board takes one array of corners` | `CharucoBoard.matchImagePoints` |
| `stubs do not accept none for the matrices, opencv does` | `cv2.calibrateCamera(..., None, None)` |
| `str-format` ignores | numpy floats in `'{:.1f}'.format(...)` |
| `matplotlib stubs miss the 3d axes` | `set_xlim3d` and similar |
| `the stubs don't allow None, tkinter leaves the option out` | `ttk.Button(command=None)` |
| `checked for none above, mypy can't narrow a getter` | `camera_control_resource.py`, the value is checked under the lock |
| `set by start before the control loop uses it` | `XboxRobotController.current_pose`, typed `Pose` so the control loop needs no `None` checks |
| `set by reset`, `none until start`, `ids are set once the scenario is built` and similar | attributes that are `None` until `reset()`, `start()` or `build()` runs, which the code always calls first |

## Fixing more

When fixing something marked with a `type: ignore`, remove it in the same commit. `mypy.ini` has
`warn_unused_ignores`, so mypy, and with it the pre-commit hook, fails on an ignore that is no longer needed. The hook
runs mypy and then the full test suite.
