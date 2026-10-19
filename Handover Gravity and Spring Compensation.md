# Handover: Gravity and spring compensation

Oct 16, 2026 · @wew. Revised 18 October 2026 against the code, the servo manuals and the 2021 calibration data; the
corrections to the first version are listed at the end. Code references are to the state after the controller rework
(`Handover Robot Controller Rework.md`).

## Summary

The arm sags at rest because the position loop of the servos runs with I gain 0 (`pid: {i: 0}` for every servo in
`servo_config.yaml`), so holding a load needs a steady position error. The plan is to predict the static holding load
of joints 2, 3 and 5 from a physics model of the pose plus a small residual, and add the equivalent number of positions
to each goal position. The servos stay in position mode; the host adds a slow term inside the existing `move_servos`
path.

- This was done once before, in 2021, with position error tables and fitted cosines for servos 2 and 3 (see "What
  already exists"). It was dropped in the 2024 rebuild. The data from then shows 1-2.6 degrees of sag on servo 3 and
  about 1.8 degrees of hysteresis between moving up and down, so friction, not stiffness, sets the floor.
- Three things change the plan compared to the first version: joints 4-6 are XC430 servos without current sensing,
  so the measured quantity has to be position error, not current; joints 1 and 2 have a 3:1 belt reduction but the
  servo's encoder sits on the joint shaft, so positions are joint positions on every joint and a fit in positions
  needs no gear ratio; the communication settings it asked for are already in place.
- Rigid-body gravity: each link is a first moment (mass times centre of mass) in the arm plane; the holding load of
  every joint is a linear combination of a handful of sin/cos terms of q2, q3, q4, q5. Fitted with linear least
  squares in positions, per joint.
- Spring: it runs from the housing of servo 2 (on the base plate) to the housing of servo 3, which moves with the
  upper arm, so its length depends on q2 and q3 and it loads both joints: one potential of the length, two linear
  columns per joint.
- Residual: a low order polynomial or a grid on (q2, q3) for what the physics misses.
- Expected outcome: static sag below the friction deadband per joint, roughly 0.1-0.3 degrees (1-3 positions) on
  joints 2 and 3 when a pose is approached from a fixed direction, up to half the hysteresis otherwise.

## What already exists

- The 2021 compensation: `ServoWithOffsetFunction` (`src/robot_controllers/dynamixel_robot/servo.py`, still there,
  unused) subtracts an offset in positions from the target of a servo, with one function for moving up and one for
  moving down. It was wired into `servo_configurations.py` for servos 2 and 3 from `785e616` (2021-06-11) until
  `9e115b5` (2024-06-12, the rebuild). The last fitted functions, from `servo_configurations.py` at `e46f2af`
  (`git show e46f2af:src/robot_controllers/dynamixel_robot/servo_configurations.py`), in positions:
  servo 2 up `12.0 cos(q2) + 5.1 cos(q2 + q3 - pi/2)`, down `2.6 cos(q2) - 0.9 cos(q2 + q3 - pi/2)`;
  servo 3 up `29.6 cos(q2 + q3 - pi/2)`, down `9.75 cos(q2 + q3 - pi/2)`.
  `cos(q2 + q3 - pi/2)` is `sin(q2 + q3)`: this is the rigid body model below without the perpendicular and wrist
  terms.
- The recording tools: `calibration/gravity_compensation.py` (`Servo2Manager`, keyboard driven: step servo 2 or 3,
  wait, record `present - target`), `record_going_up.py`, `record_going_down.py` and the `_servo3` variants (scripted
  sweeps, 3 s settle per point), the raw data in `calibration/results` (servo 2 over a grid of q2 and q3, servo 3 over
  q2 + q3, going up and going down, in positions), the fits in `gravity_compensation_with_math_angle2.py` and
  `_angle3.py` (`scipy.optimize.curve_fit`, offline). They need the `keyboard` package (not in the venv) and an
  Arduino angle sensor on COM5 (`arduino/ArduinoAngleReader.py`, an external reference for joint 2). They still call
  the current API (`robot.servo2`, `move_servo`, `get_current_positions`, `Servo.target_position`) but move the arm
  without asking, and the grid is in the 2021 units (angles as fractions of pi, q3 shifted by 0.5).
- The 2021 numbers (servo 2 still had its encoder on the motor side then, 6144 positions per 180 degrees of the
  joint; servo 3 direct, 4096 per 360): servo 3 error up to 30 positions going up and 10 going down (2.6 and 0.9
  degrees), the difference of about 20 positions (1.8 degrees) is the friction and backlash deadband. Servo 2: -17
  going up, +4 to -3 going down (0.5 degrees of the joint at the time). The encoder of servo 2 now sits on the joint
  shaft, so the belt and its backlash are inside the position loop, and the arm was reprinted; re-measure, the shape
  of the curves is what carries over.
- The hook in the code: `Servo.set_target_position_from_angle(angle, all_angles)` gets all joint angles from
  `DynamixelRobotController.move_servos` ("some servos compensate based on the other angles"). The plain `Servo` the
  yaml produces ignores `all_angles`. `Servo2`/`Servo3` (lookup tables) are unused and broken (rework handover
  issue 16).
- Masses: `resources/weights.txt` (2021): "4 screws 9 g, arm 143 g, joint3 1036 g, wrist 360 g". The names are
  ambiguous and predate the 2024 rebuild (PETG-CF lower arm). Servo weights from the manuals: XM430-W350 82 g,
  XM540-W270 165 g, XC430 65 g, XL330-M288 23 g.
- Geometry: `src/kinematics/kinematics.py` is the kinematics to use, with the link lengths from
  `src/robot_config.yaml` (d1 = 15.7, a2 = 20, d4 = 22, d6 = 12 cm, conventions in the rework handover). The URDFs
  `src/simulation/urdf/fred_new.urdf` and `fred_with_spheres_new.urdf` are built from `fred.xacro` with the same
  lengths (`python -m src.simulation.build_urdf`, a test checks the chain against the yaml); their masses and
  inertias are placeholders (0.05-0.3 kg) and their joint axes have different signs from the kinematics; `fred.urdf`
  and `fred_with_spheres.urdf` are the 2021 geometry. Pinocchio is not installed (conda only on Windows) and not
  needed.

## The robot as it is

| Joint | Servo | P gain (yaml) | Current sensing | Current-based position mode | Stall torque, current (12 V) | Gravity load |
|---|---|---|---|---|---|---|
| 1 base | XM430-W350 | 1500 (d 500) | Present Current, 2.69 mA per unit | yes (5) | 4.1 N.m, 2.3 A (1.78 N.m/A) | none, yaw |
| 2 shoulder | XM540-W270 | 1000 | Present Current, 2.69 mA per unit | yes | 10.6 N.m, 4.4 A (2.41 N.m/A) | largest |
| 3 elbow | XM430-W350 | 1500 | Present Current | yes | 4.1 N.m, 2.3 A | large; mounted reversed (`positions: [3072, 1024]`) |
| 4 wrist 1 | XC430-W240 | 1000 | none, address 126 is Present Load (0.1 %); PWM limit instead of a current limit | no (modes 1, 3, 4, 16) | 1.9 N.m, 1.4 A | roll, only through the bent wrist, small |
| 5 wrist 2 | XC430-W240 | 2000 | none | no | 1.9 N.m, 1.4 A | pitch, the gripper |
| 6 wrist 3 | XC430-W150 | 600 | none | no | | roll, none |
| 7 gripper | XL330-M288 | 500 | Present Current, 1 mA per unit | in use (goal current 400 mA) | 0.52 N.m, 1.47 A at 5 V | |

- All arm servos run in position mode (3) with I = 0 and D = 0 (servo 1: D = 500). Manual defaults are P 800 (XM),
  700 (XC), 400 (XL); the configured gains are higher.
- 4096 positions per joint turn on every joint, 0.0879 degrees per position. Joints 1 and 2 have a 3:1 timing belt
  reduction after the servo gearbox with the servo's magnetic encoder moved to the joint output shaft: the position
  loop closes on the joint, the belt is inside the loop, and the motor delivers about 3 times its torque at the
  joint (so for the current cross-check on joints 1 and 2 use 3 times the torque constant, minus belt losses).
  Until the 2024 rebuild the encoder was on the motor side (6144 positions per 180 degrees, extended position
  mode).
- Bus: U2D2 on COM4 (Windows), 1 Mbps, latency timer 1 ms, `GroupSyncRead`/`GroupSyncWrite`, one read packet per
  bus (RS-485: 1-3, TTL: 4-7), 8 ms for the whole arm, return delay time 500 us. `move_servos` runs at 100 Hz from
  `execute_move` and the xbox loop on a `perf_counter` schedule. There is no Present Current read in
  `servo_handler.py` yet; only `gripper_test.py` reads it, with `read2ByteTxRx`.
- Safety: the robot warns at 45 C and stops at 50 C (PLA). The arm drops when the torque goes off.
- Settings the first version asked for: latency timer 1 ms, sync read/write and a fixed period loop are done. 4 Mbps
  is not worth an eeprom change on every servo: a bus read is the latency timer plus the return delays plus about
  0.5 ms of bytes at 1 Mbps, so 4 Mbps saves ~0.35 ms per bus while a lower return delay saves 1.5-2 ms. `chrt` and
  `ttyUSB0` are Linux, the robot runs on Windows. Host-side current control or an OpenRB-150/Teensy are not needed
  for a position offset.

## Why the arm sags

- In position mode the PID output is a PWM duty proportional to the position error (KPP = P / 128 in the manual), so
  a static load needs a static error. The stiffness K (load per position of error) is not a datasheet number: it
  scales with the P gain (1000-2000 here), shifts with the supply voltage (PWM mode), temperature and friction.
  Don't compute it, measure it (Step 5), or avoid it by fitting the compensation directly in positions (Step 2).
- The I gain register exists and is 0 by configuration. A small I would remove the static error but winds up
  against the table, an object or the end of the range, with PLA parts around the servos. Not the main fix; maybe a
  trim later.
- Current-based position mode (5) would make the loop output a current (torque) instead of a PWM duty, so K would
  not depend on the supply voltage. Only joints 1-3 have it.
- Friction and backlash: 2021 measured about 20 positions between approaching a pose from above and from below on
  servo 3. No static model removes that; a direction-dependent offset (what 2021 did) halves it.

## The static model

```
load(q) = gravity(q) + spring(q) + residual(q)
```

Only joints 2, 3 and 5 carry a meaningful gravity load: joint 1 (yaw) none; joints 4 and 6 (rolls) only through the
gripper's off-axis mass when the wrist is bent, a few hundredths of a N.m, ignored.

Gravity in the arm's own convention (`kinematics.py`: q2 = 0 lower arm horizontal, phi = q2 + q3, phi = pi/2 upper arm
horizontal and in line, radial = away from the base axis):

```
joint 3:  tau3 = g [ h3a sin(phi) + h3p cos(phi) + m6 c6 w(q) ]        w = sin(phi) cos(q5) + cos(phi) cos(q4) sin(q5)
joint 2:  tau2 = g [ (h2a + md a2) cos(q2) - h2p sin(q2) ] + tau3
joint 5:  tau5 = g m6 c6 [ cos(phi) sin(q5) + cos(q4) cos(q5) sin(phi) ]
(joint 4: tau4 = g m6 c6 sin(q4) sin(q5) sin(phi), small)
```

`h = m * centre of mass` is the first moment of a link, `a` along the link, `p` perpendicular to it in the arm plane;
`md` is the mass of everything beyond joint 3, `m6 c6` the first moment of the gripper beyond the wrist centre, `w`
its radial reach per cm. Everything beyond joint 3 loads joint 2 with the same terms as joint 3 plus the `a2 cos(q2)`
lever, which is why `tau3` reappears in `tau2`.

So the holding load of each joint is a linear combination of 1-5 sin/cos terms: that is the regressor,
`numpy.linalg.lstsq` solves it, Pinocchio is not needed. With the target in positions instead of torque every
coefficient absorbs `g / K` of its joint, so fit per joint; the terms shared between joints 2 and 3 then differ by
`K3 / K2`, a consistency check. The sign of the offset in positions depends on the mounting (servo 3 is reversed);
fit positions directly and the sign comes out of the fit. The 2021 fit `a cos(q2) + b sin(phi)` is this model without
the perpendicular and wrist terms.

Spring. It runs from the housing of servo 2, on the base plate, to the housing of servo 3. Servo 3 is mounted
reversed: its housing is part of the upper arm and turns with q2 + q3 about the elbow, so the spring length is
`L(q2, q3)` and the spring loads both joints (the readme doesn't mention it; measure the anchors). For a linear
spring the torque on joint j is `k (L - L0) dL/dqj`: two linear columns per joint, `L dL/dqj` and `dL/dqj`, with the
same coefficients `k` and `-k L0` in torque (in positions each joint absorbs its own `1 / K`); a nonlinear or
pre-tensioned spring gets a polynomial in `L` instead. The anchor on the upper arm sits close to the joint 3 axis, so
`dL/dq3` and the joint 3 load are small compared to joint 2, which is why q2 and q3 have to be varied independently
to tell the spring from gravity.

Residual. What the physics leaves over on (q2, q3), maybe q5: a low order polynomial or a
`scipy.interpolate.RegularGridInterpolator` on a q2 x q3 grid. An MLP is more than a few hundred poses support.
Clamp the inputs to the sampled range. A large residual means a wrong attachment point, a missing force or a wrong
sign; fix that instead of fitting it. Payload: add a known mass `mp` at the gripper tip analytically, the same `w`
term with `c = d6`; this needs K.

## Step 1: collect the data

- Target: steady state position error per joint in positions, `present - goal`, as in 2021. It works on every joint
  including the XC430s and needs neither a torque constant nor K. It is tied to the P gains and the supply voltage,
  so log both and refit after changing them. Present Current on joints 1-3 (2.69 mA per unit) is a cross-check;
  Present Load on 4-6 (0.1 %) is a PWM proxy only.
- Reads: add a sync read of address 126 (2 bytes, signed) to `servo_handler.py` for the current, one packet per bus
  like the other reads. `read_current_pos` and `Servo.target_position` give the error (`_print_position_error` in the
  controller prints exactly that).
- Poses: random inside the calibrated ranges, q2 and q3 varied independently (to separate the spring from gravity),
  q4 and q5 varied for the wrist terms, including poses near the range ends. Approach each pose from both
  directions, wait until the arm has settled (2021: 3 s), log angles (radians, the 7 element array), goal and
  present position per servo, current, temperature and input voltage, as csv. The mean of both directions is the
  static load, half the difference the friction deadband. Hold out about 20 % of the poses.
- Warm up with a few minutes of slow sweeps but stay below the 45 C warning (the robot stops at 50 C), and log the
  temperatures.
- Moving: `plan_move_to_pose`/`execute_move` for checked moves, or `move_servo` for one joint at a time (unchecked,
  small steps). Write a new script for this instead of reviving `calibration/gravity_compensation.py`.

## Step 2: fit the gravity model

```python
import numpy as np

def basis(q):
    """regressor rows for the holding load of joints 2, 3 and 5; q is the 7 element angle array in radians
    (index 1-6 = servo id). A factor g / K per joint is absorbed by the fitted coefficients."""
    q2, q3, q4, q5 = q[2], q[3], q[4], q[5]
    phi = q2 + q3                                                        # pi/2 = upper arm horizontal
    w = np.sin(phi) * np.cos(q5) + np.cos(phi) * np.cos(q4) * np.sin(q5)  # radial reach of the gripper
    row3 = [np.sin(phi), np.cos(phi), w]                                 # h3a, h3p, m6 c6
    row2 = [np.cos(q2), np.sin(q2)] + row3                               # h2a + md a2, -h2p, then as joint 3
    row5 = [np.cos(phi) * np.sin(q5) + np.cos(q4) * np.cos(q5) * np.sin(phi)]  # m6 c6
    return {2: np.array(row2), 3: np.array(row3), 5: np.array(row5)}

def spring_columns(q2, q3, base_anchor, arm_anchor, a2=20.0, h=1e-6):
    """{joint: [L dL/dq, dL/dq]} for the spring between the servo 2 housing (anchor relative to joint 2: radial,
    up, in cm) and the servo 3 housing, which moves with the upper arm (anchor relative to joint 3: along the
    upper arm, perpendicular to it); the coefficients are k and -k L0"""
    def length(q2, q3):
        phi = q2 + q3
        along, perp = arm_anchor
        anchor = np.array([a2 * np.cos(q2) + along * np.sin(phi) + perp * np.cos(phi),
                           a2 * np.sin(q2) - along * np.cos(phi) + perp * np.sin(phi)])
        return np.linalg.norm(anchor - np.asarray(base_anchor))
    L = length(q2, q3)
    dL2 = (length(q2 + h, q3) - length(q2 - h, q3)) / (2 * h)
    dL3 = (length(q2, q3 + h) - length(q2, q3 - h)) / (2 * h)
    return {2: [L * dL2, dL2], 3: [L * dL3, dL3]}

# Q: list of angle arrays, E: {servo id: steady state error in positions}, mean of both directions
A = {j: np.array([basis(q)[j] for q in Q]) for j in (2, 3, 5)}
for j in (2, 3):
    A[j] = np.hstack([A[j], np.array([spring_columns(q[2], q[3], base_anchor, arm_anchor)[j] for q in Q])])
coefficients = {}
for j in (2, 3, 5):
    coefficients[j], _, rank, _ = np.linalg.lstsq(A[j], E[j], rcond=None)
    print(j, coefficients[j], 'condition', np.linalg.cond(A[j]), 'rank', rank)
```

- Keep `cond` moderate (below a few hundred); a large value means q2 and q3, or q4 and q5, were not varied enough.
  `w` and `sin(phi)` are only separable when q5 is varied.
- Sanity checks: the coefficients of `sin(phi)` and `cos(phi)` in joint 2 and joint 3 should have the same ratio to
  each other (same physical moments, different K); the `cos(q2)` coefficient of joint 2 should be the largest.
- Check the fit on the held-out poses in positions; the 2021 scatter was 1-2 positions.

## Step 3: the spring

1. Measure the anchor on the servo 2 housing relative to the joint 2 axis (radial and vertical offset in the arm
   plane) and the anchor on the servo 3 housing relative to the joint 3 axis (along the upper arm and perpendicular
   to it), with a ruler or from the CAD.
2. Measure k and L0 offline by hanging weights, as a check on the fitted `k` and `-k L0`.
3. The fitted `k` has to come out the same from the joint 2 and the joint 3 columns up to the ratio `K3 / K2`; if
   not, or if `k` is far from the measured one, suspect the anchor measurement or a sign error, not the spring.

## Step 4: the residual

- Target: measured error minus the model, per joint, on the training poses.
- Inputs: q2 and q3, plus q5 for joint 3 if the wrist terms don't capture it. Base yaw and wrist roll should not
  matter.
- Model: a polynomial of degree 2-3 in (q2, q3), or a `RegularGridInterpolator` on a coarse grid; clamp the inputs
  to the sampled range at runtime.
- If the residual is larger than a few positions, go back to the model.

## Step 5: apply it at runtime

Where it plugs into the code:

- `Servo.set_target_position_from_angle(angle, all_angles)` (`servo.py`): after interpolating the angle to a position
  and adding `constant_offset`, add the compensation in positions computed from `all_angles`. `move_servos` passes
  the commanded angles, so the model input is the target pose, not the measured one, which is what the first version
  asked for. Store the applied compensation on the servo for diagnostics.
- Parameters in the yaml: `servo_config.py` rejects unknown fields; add an optional `compensation` field (the
  coefficients per basis term, the direction split, the magnitude clamp) to `_OPTIONAL_SERVO_FIELDS`, parse it in
  `_parse_servo` and build a `Servo` with it. Keep the basis functions in one module shared with the fit script.
- Clamping: `servo_handler.__clamp_target_position` runs after the hook, in `move_to_angles`. Either clamp the
  uncompensated position first and let the goal exceed the calibrated range by the compensation (the physical limits
  have to allow it), or let the clamp cut the compensation at the range ends. The rework handover asks for the first;
  decide per joint, servo 3's range is [1024, 3072] reversed.
- Reported angles stay truthful: `get_angle_from_position` converts the measured position, and with the
  compensation the measured position is close to the uncompensated goal, so `get_current_angles`,
  `forward_kinematics`, the 0.05 rad start check of `check_move` (33 positions) and `hold_current_position` don't need
  to know about it. Only `_print_position_error` (`present - target`, the target includes the compensation) has to
  subtract it to show the residual.
- Rate: `move_servos` runs every step (100 Hz in `execute_move` and the xbox loop); a few sin/cos per step cost
  nothing. No separate loop, no change to the servos' own 1 kHz loop.
- Direction: keep the 2021 split as an option: fit up and down separately, use the mean for static targets and the
  direction of the last commanded change to pick the side (`ServoWithOffsetFunction` did this with
  `_last_commanded_angle`).
- Safety: clamp the compensation magnitude (60 positions is 5 degrees), enable it one joint at a time with the arm
  supported, and check the sign by reading `present - target` before and after: the error has to shrink. A wrong sign
  doubles it.
- Tests to extend: `tests/utils_tests/servo_test.py` (the compensation in `set_target_position_from_angle`, the clamp
  order), `tests/servo_controller_tests/servo_config_tests.py` (the yaml field), `dynamixel_robot_controller_tests.py`
  (the goal positions written with compensation).

K per joint is only needed for torque-level numbers: the current cross-check and analytic payloads. Measure it with a
known weight at a known lever arm at a few poses: `K = delta load / delta error`, median over poses. Recalibrate after
changing the P gain, the supply voltage or the operating mode.

## Step 6: validate

1. Held-out poses, both directions, per joint, before and after; report positions and degrees.
2. External reference: the servo encoder sits on the output shaft, so `present - goal` is honest about the sag in the
   servo but blind to flex in the printed parts and belts. Options: the 2021 Arduino angle sensor
   (`arduino/ArduinoAngleReader.py`, COM5, joint 2); the camera with an aruco marker on the gripper (`src/camera`,
   once the aruco port of rework issue 2 is done); a phone IMU strapped to the gripper; a dial indicator at the tip
   (57 cm reach, awkward).
3. Repeat with the heaviest expected payload and with warm servos (below 45 C).
4. Expectation: static error below the friction deadband; from one direction 0.1-0.3 degrees (1-3 positions),
   otherwise up to half the hysteresis (about 0.9 degrees on servo 3 in 2021). An error that flips sign with the
   direction of approach is backlash or friction, not the model.

## Motion and friction (second stage)

Static compensation fixes the resting sag. While moving, friction adds about `Fc / K` of tracking error, from the 2021
data about 10 positions (0.9 degrees) on servo 3, as large as the sag it removes. Add this after the static model is in
and its remaining error is measured.

1. Identify friction per joint from slow constant velocity sweeps in both directions at several speeds: half the
   up/down difference is the Coulomb term, the slope against velocity the viscous term. Joint 2 is probably the
   largest.
2. Feedforward `Fc tanh(v / v0) + b v`, in positions, added to the static term. Use the desired velocity (the
   difference of consecutive steps of a `PlannedMove` times `steps_per_second`), not the measured one; v0 a few
   degrees per second.
3. Velocity lag: a PD loop trails a moving target in proportion to speed. Shift the goal ahead along the path, or use
   the Feedforward gain registers of the servos.
4. Trajectories are already smooth: the planner uses a minimum jerk profile and the profile velocity/acceleration
   per servo are set from the yaml.
5. Resting deadband: approach final positions from a consistent direction, or the up/down split of Step 5.

Limit: this is for slow moves. Fast motion or heavy payloads need inverse dynamics with link inertias, which this plan
does not identify.

## Corrections to the first version

| Topic | First version | Corrected |
|---|---|---|
| Servo models | six mostly XM430-W350-R | XM430 (joints 1, 3), XM540 (2), XC430 (4-6), XL330 (gripper) |
| Measured quantity | log Present Current for all joints, fit joints 4-6 first from their currents | XC430 has no Present Current (address 126 is Present Load, 0.1 %) and no current-based position mode; use the position error on all joints, current only as a cross-check on 1-3 |
| Positions per turn | 4096 per revolution | right for every joint: joints 1 and 2 have a 3:1 belt reduction, but the encoder was moved to the joint shaft, so the servo counts joint positions; the reduction only shows up as 3 times the torque per ampere on those joints |
| P gain, K | default 640, K about 15 N.m/rad | manual defaults 800 (XM), 700 (XC), 400 (XL); configured 600-2000; K is measured per joint, not computed |
| Integral term | the position loop has no integral term | the I gain register exists and is configured 0 |
| Spring coupling | joint 2 only, then corrected to joints 2 and 3 | joints 2 and 3 is right: the spring runs from the servo 2 housing to the servo 3 housing, which is part of the upper arm (servo 3 is mounted reversed) |
| Torque constant | 1.0-1.3, then 1.8 N.m/A | 1.78 XM430, 2.41 XM540, 0.35 XL330 from the stall specs; XC430 has no current reading |
| Settings | set the latency timer, use sync read/write and a `perf_counter` loop, 4 Mbps, `chrt` | the first three were done in the rework; 4 Mbps saves ~0.35 ms per bus read, a lower return delay 1.5-2 ms; Windows, no `chrt` or `ttyUSB0` |
| Tooling | Pinocchio regressor from the URDF | not installed, the URDF masses are placeholders; the closed form regressor above with numpy |
| Prior work | none mentioned | the 2021 compensation, its data and its scripts (What already exists) |

## Open items

Information still needed:

- [x] Kinematics: `src/kinematics/kinematics.py`, the URDF is not needed.
- [ ] Masses: `resources/weights.txt` is from 2021 with ambiguous names; re-weigh the lower arm, the upper arm with
  the wrist servos, and the gripper, or skip it: the regressor fits the first moments directly, masses are only
  needed for payloads.
- [ ] Spring: the anchor points on the servo 2 and servo 3 housings, k and L0 (the attachment is known: both joints
  stretch it).
- [ ] Which joints sag most and the payload range: 2021 says servo 3, then 2; re-measure, the gearing changed.

Work to do:

- [ ] Sync read of Present Current (joints 1-3, 7) and Present Load (4-6) in `servo_handler.py`.
- [ ] Pose sampling and logging script: random poses, both directions, settle, csv (replaces
  `calibration/gravity_compensation.py`).
- [ ] Fit the positions per joint on the cos/sin basis, plus the spring columns; check `cond` and the hold-out error.
- [ ] Residual and the up/down split, if the hold-out error asks for it.
- [ ] `Servo` compensation with a yaml field and tests, the clamp order decision, the magnitude clamp.
- [ ] Enable it per joint on the robot with the arm supported, sign check.
- [ ] Validate (Step 6), then the friction stage.
