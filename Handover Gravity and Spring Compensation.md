# Handover: Gravity and spring compensation

Oct 16, 2026 · @wew. Revised 18 October 2026 against the code, the servo manuals and the 2021 calibration data. Code
references are to the state after the controller rework (`Handover Robot Controller Rework.md`). Revised 23 October
2026: the spring was removed from the robot (Step 3) and the friction of joint 2 was measured; the gains come first,
see `Tuning the servo gains.md`.

## Revision of 18 November 2026: pwm units, sweeps, and how large the sag still is

This section is newer than the rest of the document and wins where they disagree. The gains table, the temperature
limits and the "fit in positions" of the sections below are from October.

How large the sag is now. The P gains are 2.5 to 3 times those of October (joint 2: 1000 to 2500, joint 3: 1500 to
4710), and the sag shrinks in proportion. From the tuning recordings of 17 and 18 November, without a payload:

| Joint | P | Sag (positions) | Holding drive (pwm units) | Where |
|---|---|---|---|---|
| 2 shoulder | 2500 | 2.3 | about 45 | pose `reach`, mean of the two step directions |
| 3 elbow | 4710 | 3.2 | 118 | upper arm horizontal, mean of both sweep directions |
| 5 wrist 2 | 760 | about 5 | 26 to 36 | sweep around its tuning pose |

That is 1 to 2 mm per joint at the gripper tip. The lag while moving (5 to 15 positions) is larger. The encoder does
not see flex of the printed parts, so the true sag at the tip is larger than these numbers.

What the literature does, and how this code relates to it:

- The gravity torque of a joint is linear in the first moments (mass times centre of mass) of the links beyond it,
  so it is fitted with linear least squares. `static_compensation.py` is that, with the simplification that the
  gripper's centre of mass lies on the roll axis and joint 6 is at 0 (joint 4 has one term; its drive is about 6 pwm
  units, so this is left as it is).
- Measuring: either at rest at many poses (Ma and Hollerbach, ICRA 1996), or, more common, while one joint moves
  slowly at constant speed in both directions: the mean of the two directions is gravity, half the difference is
  friction. Approaching a pose from both sides and averaging (the 2021 scripts, `record_static_load.py`) is the
  at-rest form of the second.
- Applying: the textbook form is a torque added to a PD loop. For a joint that acts like a spring the equivalent
  is a shifted position reference, `goal = q + g(q) / K` (Tomei 1991, PD control of robots with elastic joints).
  The spring here is the P gain of the servo, and `Servo.set_target_position_from_angle` adds exactly that.

What changed in the code on 18 November:

- The coefficients are the holding drive in pwm units, no longer positions. The servo drives with
  `pwm = P / 128 * (goal - present)`, so `positions = drive * 128 / P`, and `StaticCompensation.positions` takes
  the P gain of the servo. A fit stays valid when a gain is retuned. Checked against the data: joint 3 held at
  -118 pwm measured, and 3.2 positions times 4710 / 128 is 118.
- `fit_static_load.py` fits `error - compensation`, the whole load. Before, a recording made with a compensation
  active would have been fitted as if its remaining sag were the whole load. It converts every row with the P gain
  in its `p` column, and reports the measured pwm against the drive computed from the error.
- `record_static_load.py` logs the gains, the measured pwm, and the mean of several readings per row, and lowers
  the arm to the rest pose at the end.
- New: `calibration/record_load_sweep.py`, the measurement at constant speed (next paragraph).
- 19 November: the model is written out as one formula per joint (`joint_2_drive` to `joint_5_drive` in
  `static_compensation.py`), which `StaticCompensation.drive` calls with the yaml coefficients by name; the fit
  derives its columns from the same formulas. The spring term, kept as an option since October, is gone from the
  code, the yaml and the fit script.

Sweeps. A joint at rest sits somewhere inside its friction band, and where depends on how the move ended: in the
step tests joint 3 came to rest at 3.0 positions from both sides, joint 2 at 3.6 and 1.0. A joint that moves at
constant speed has no such freedom. `record_load_sweep.py --joint N` drives one joint through a span and back at
two slow speeds while the others hold, at a few poses of the others, and writes per 5 degree bin the mean error and
pwm of each direction, in the csv format of `record_static_load.py`, so `fit_static_load.py` reads both. Only the
moving joint is measured; the rows of the others carry their angles and a blank error. From the feedforward sweeps
of 18 November, binned the same way: joint 3 gives -116 to -122 pwm in every bin at 10 and at 30 degrees/s, joint 2
the same slope along the sweep at 5 and 15 degrees/s. Joint 5 gives -36 at 10 degrees/s and -26 at 30: its friction
is not the same in both directions, so the two speeds of a recording are compared and a disagreement of more than
a position is warned about.

Gravity only. Since 19 November the compensation has one set of coefficients per joint and no direction: the
same pose always gets the same correction. The optional `up`/`down` blocks (carried over from the 2021 scripts,
never used on the robot) and `--split` of the fit script are removed, and a yaml with `up` or `down` is refused.
They were a friction compensation, which is the later stage (see "Motion and friction"), to be built on the
commanded speed. The recorders still measure both directions, because the mean of the two is the gravity load; the
fit reports half the difference as the friction it does not compensate.

The procedure for a joint, joint 3 first (its load depends on `phi` only):

1. `python -m src.robot_controllers.dynamixel_robot.calibration.record_load_sweep --joint 3`, with no
   compensation in the yaml. Its summary prints the sag per sweep.
2. `python -m src.robot_controllers.dynamixel_robot.calibration.fit_static_load <the csv> --joints 3`, and paste
   the block under servo 3 in `servo_config.yaml`.
3. Record the same sweeps again. The `error` column is now the sag that remains; the summary should print about 0
   where it printed about 3 positions. A wrong sign shows as twice the sag.
4. The check at rest: `record_static_load.py --yes --poses 30 --seed 1` with `--range` for the joints, then
   `check_static_load.py <the csv>`: the sag that is left and what it was without the compensation, from the one
   recording. Done for joints 2 and 3 on 21 November, see below.

Joint 3, 19 November 2026 (`calibration/tuning/load_sweep_joint3_20261119_before.csv` and `_after.csv`, 8 sweeps
and 148 bins each, P 4710):

- Before: the sag (mean of both directions) was 3.0 to 3.2 positions with the upper arm horizontal and 1.5 at 57
  degrees up or down, the same at 5 and at 10 degrees/s to 0.08 position, and it followed `phi` (with the lower arm
  at 60 degrees the peak sat at joint 3 = 30 degrees). The fit: constant 3.9, sin_phi -108.7, cos_phi 6.7,
  wrist_reach -8.7 pwm units, leaving 2.1 pwm units on the fitted bins and 2.9 on the held out ones (under 0.1
  position). The measured pwm was 1.00 times the drive computed from the error.
- After, with that block in the yaml (`max_positions: 15`): the sag is -0.02 positions on average, 0.21 rms,
  between -0.4 and +0.5. What is left is mostly the rounding of the compensation to whole positions (it steps from
  -2 to -3). The error while moving went from +2.6..+5.4 (up) and +0.1..+1.4 (down) to +0.9..+2.5 and -2.3..-0.9:
  the worst error in either direction halved, 5.4 to 2.5 positions, and the errors are now centred on zero.
- Half the up/down difference did not change (1.7 to 1.8 positions): friction and the lag of the speed, about 1.3
  positions of it friction, and more where the load is larger (1.1 positions at 57 degrees, 1.75 horizontal).
  Friction is now the larger part of the error of joint 3.
- A fit of the second recording (made with the compensation active) gives the same coefficients within their
  standard errors (sin_phi -108.3, cos_phi 7.1, wrist_reach -9.2), so `error - compensation` is handled right.

Joint 5, 19 November 2026 (`calibration/tuning/load_sweep_joint5_20261119_before.csv`, 6 sweeps, 182 bins,
P 760), recorded but NOT compensated:

- Gravity (the mean of both directions) is up to about 30 pwm units, 5 positions; the fit (constant -10.2,
  wrist_pitch -21.7) leaves 0.6 to 0.8 position, and the two speeds disagree by up to 2.3 positions.
- Friction is as large as the load and grows with it: moving up takes about twice the gravity drive (up to 58 pwm
  units, 10 positions), moving down takes almost none at any load (0 to 5). So the joint rests up to 10 positions
  short after moving up and about on its goal after moving down.
- A gravity block would centre that band (worst case about 10 to 6 positions) and make the down direction worse
  than it is. The gripper is 12 cm long, so a position of joint 5 is 0.18 mm at the tip: the whole error is under
  2 mm there. No block has been put in the yaml for joint 5 (whether to is still open); it is the joint to try a
  friction term on later.
- Not explained: after the lower turning point of the sweeps the first 25 degrees upward (5 degrees/s; less at 10)
  took much less drive than the model gives, in two of three poses, with nothing touching the gripper on the
  video. The recorder keeps bins, not samples, so this needs a recording of the raw samples to look into.
- Worth a mechanical look: since November the servo sits directly on the wrist joint. Friction equal to the load
  could be the weight of the gripper hanging on the output shaft of the servo; not checked.

Joint 2, 20 November 2026 (`calibration/tuning/load_sweep_joint2_20261120_p2000.csv`, `_p2000_bent.csv` and the
two `_p2000_after` files; P 2000, see below for why not 2500):

- Before: the sag was up to 3.6 positions, the same at 5 and 10 degrees/s. All default poses of joint 2 have the
  wrist straight, where `wrist_reach` equals `sin_phi` and a fit splits one number over the two; two more sweeps
  with the wrist bent 60 degrees either way (`--pose 90,0,0,0,60,0 90,0,0,0,-60,0 --span=50:110`) separate them.
- The block in the yaml has four terms: cos_q2 52.9, sin_phi 16.2, cos_phi 0.5, wrist_reach 1.6, leaving 2.4 pwm
  units (0.15 position). A constant and sin_q2 make the fit no better (2.1) and come out as +21 and -20: they cancel
  inside the recorded 43 to 118 degrees of joint 2 and not outside it, so they are left out. `fit_static_load.py`
  fits all six and has no option to leave terms out; the four were fitted by hand with `fit_joint(names=...)`.
- Cross-check with joint 3: its sin_phi of 108.7 pwm units, divided by the 3:1 belt and by the larger torque per
  pwm unit of the XM540 (10.6 against 4.1 N.m at stall), predicts about 14 for joint 2; the fit gives 16.2.
- After: the sag is 0.0 on average, 0.26 rms, between -0.5 and +0.7 positions. The errors while moving are centred
  now (-9.1 going up, +9.1 going down at 5 to 10 degrees/s, where they were -10.6 and +7.6). They are the lag of the
  speed and the friction, 8 mm at the tip at full reach: the velocity lead of `Handover Velocity Lead on the PC.md`
  is what addresses that.
- cos_q2 was 48.5 in the recordings of the same afternoon at a looser belt, 52.9 after it was tightened. Gravity
  did not change: about a tenth of the mean of the two directions is friction that is not the same lifting and
  lowering, and it follows the belt. Record joint 2 again after the belt is retensioned.

Joint 2 swings while it moves (20 November, `calibration/tuning/joint2_ringing_20261120*.csv`, raw samples of
sweeps of 10 degrees either side of the folded pose at 10 degrees/s, eight gain settings each): at its own
resonance, whatever the speed, and most going down, where gravity does the work and there is hardly any motor
torque on the belt. The size of the swing going down in positions at P 2500 / 2000 / 1500 (D 250):

| Belt | Resonance | P 2500 | P 2000 | P 1500 |
|---|---|---|---|---|
| as on 17 November | 6.6 Hz | 1.9-2.4 | 1.2 | 0.75 |
| loosened | 6.0 Hz | 2.7-2.8 | 2.4 | 1.0 |
| tightened further (kept) | 7.2 Hz | 1.2-1.5 | 0.66 | 0.34 |

Going up it is 0.5 to 0.7 at the kept tension, about the floor of the measurement. D 0, 250, 640 and 1640 made no
difference at any tension (the swing is one position at 7 Hz, a count every 20 ms: the D term of the servo sees
nothing most of the time and a spike at each count). So the belt tension sets how much P the joint takes, and the
step tests of the gain tuning do not see this. Joint 2 is at P 2000 since. The tighter belt costs friction: about a
fifth more by motor current than on 17 November.

A stall of the PC becomes a kick: on the video of one run both the camera recorder and the arm stopped for about a
quarter second during the first move, after which the arm moved at twice its planned speed and rang at 6 Hz for
0.7 s. `execute_move` (`src/utils/movement.py`) and `stream_sweep` (`tune_servo_gains.py`) sent the steps that were
late back to back. Since 20 November they carry on from where the move is, one period between the steps, as the
gamepad loop does since 18 November: the arm pauses for as long as the PC stalls and the move ends that much
later. Not checked on the robot yet. Open: what stalls (it happened with and without the camera recorder running).

One position of error is about 0.5 mm at the gripper tip for joint 3 (34 cm) and up to 0.9 mm for joint 2 (57 cm).
The cos_phi term of joint 3 is a shift of 3.3 degrees: the load peaks with
the upper arm commanded 3.3 degrees above horizontal. The fit can not tell a centre of mass that sits off the axis
of the link from a zero error of the angle (servo 3 has no offset measured in the jig). The upper arm is bolted
directly to the servo, so a zero error of 3 degrees is not expected there and the term is taken as the weight
sitting below the axis (cables, housings). A level on the upper arm commanded horizontal would confirm it; not
done.

The check at rest, 21 November 2026 (`calibration/tuning/static_load_20261121_check.csv`): 30 random poses with
joints 2 to 5 inside the ranges of the sweeps (`--range 2=45:115 3=-60:60 4=-60:60 5=-60:60 --seed 1`), each
approached from 5 degrees above and below, 3 s of settling, the compensation of joints 2 and 3 active. The report
is `check_static_load.py` (new, it also reads the sweep recordings):

| Joint | Sag without the compensation (rms) | Sag left (rms) | Range of the sag left | Friction band at rest |
|---|---|---|---|---|
| 2 | 1.87 | 0.39 | -0.7 to +1.0 | 1.9 (2.5 at most) |
| 3 | 2.65 | 0.38 | -0.5 to +0.9 | 0.8 (1.6 at most) |

All in positions; the sag is the mean of the two directions of a pose, the friction band half their difference.

- The compensation holds at rest as it did in the sweeps: no pose of either joint is over 1 position. What is left
  is the size of the rounding: the compensation is a whole number of positions and so is every reading.
- Joint 2 has a small trend left along its own angle: about +0.4 below 60 degrees, about -0.3 above 100. Not
  worth a new fit while it stays inside the rounding.
- Friction is now the error at rest: joint 2 stops 2 positions short of its goal from either side at nearly every
  pose (0.18 degrees, up to 1.8 mm at the tip at full reach), joint 3 about 1 short. The largest error of a single
  approach is 3.0 positions on joint 2 and 2.2 on joint 3.
- The joints without a compensation, for the record: joint 4 sags -4 to +5.5 positions with a friction band of 2.5
  (errors up to 10, P 780); joint 5 rests up to 9 positions short when it arrives with its angle rising and on its
  goal when it arrives the other way, at any pose and whichever way the gripper tilts (sag +3 on average). That is
  one-sided: it looks more like a constant force on the joint (the cable to the gripper?) plus friction than like
  gravity. Not looked into. Joint 1 is within 0.8 and joint 6 sits a constant 1 position off.
- Before a recording, `calibration/offline/preview_static_load_poses.py` lists the poses it will visit and how low
  the wrist and the gripper tip come, without the robot: the planner's floor lets the tip down to where the
  fingertips touch the table.

The lag while moving, 21 November 2026: every arm joint has a velocity lead since (`velocity_feedforward` in the
yaml, see `Handover Velocity Lead on the PC.md`). The sweeps of joint 2 recorded again with it
(`calibration/tuning/load_sweep_joint2_20261121_lead.csv`): the errors while moving went from -9.1 and +9.1 to -2.4
and +2.5 positions, the sag is unchanged at 0.27 rms. The recorders measure against the goal without the lead and
log it in a `lead` column.

Still open: joint 4; joint 5 (see above); the wrist model is the simplified
one; what the encoder does not see (flex) needs an outside reference, see Step 6. A webcam with aruco markers on
the arm (`aruco_markers_a4.pdf`) records the runs since 19 November, as a check of what happened, not as a
measurement.

## Summary

The arm sags at rest because the position loop of the servos runs with I gain 0 (`pid: {i: 0}` for every servo in
`servo_config.yaml`), so holding a load needs a steady position error. The plan is to predict the static holding load
of joints 2, 3 and 5 from a physics model of the pose plus a small residual, and add the equivalent number of positions
to each goal position. The servos stay in position mode; the host adds a slow term inside the existing `move_servos`
path.

- This was done once before, in 2021, with position error tables and fitted cosines for servos 2 and 3 (see "What
  already exists"). It was dropped in the 2024 rebuild. The data from then shows 1-2.6 degrees of sag on servo 3 and
  about 1.8 degrees of hysteresis between moving up and down, so friction, not stiffness, sets the floor.
- Three facts of the robot shape the plan: joints 4-6 are XC430 servos without current sensing, so the measured
  quantity has to be position error, not current; joints 1 and 2 have a 3:1 belt reduction but the servo's encoder
  sits on the joint shaft, so positions are joint positions on every joint and a fit in positions needs no gear
  ratio; the communication settings (latency timer, sync read and write) are already in place.
- Rigid-body gravity: each link is a first moment (mass times centre of mass) in the arm plane; the holding load of
  every joint is a linear combination of a handful of sin/cos terms of q2, q3, q4, q5. Fitted with linear least
  squares in positions, per joint.
- Spring: removed from the robot on 23 October 2026. With it servo 3 tripped its overload protection holding the
  upper arm horizontal, without it the same hold ran for minutes and joint 2 holds the stretched arm at 15 % of its
  stall current (Step 3). The model is gravity plus residual; the spring term was taken out of the code on 19
  November 2026.
- Friction: on joint 2 the Coulomb friction is nearly as large as the gravity load (about 100 against 150 units
  of current, 23 October), so the static load has to be recorded from both directions and fitted as their mean,
  never from one direction, and final poses should be approached from a consistent direction.
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
- The recording tools of 2021 are deleted: `calibration/gravity_compensation.py` (`Servo2Manager`, keyboard driven:
  step servo 2 or 3, wait, record `present - target`) and the fits `gravity_compensation_with_math_angle2.py` and
  `_angle3.py` (`scipy.optimize.curve_fit`, offline) on 25 October 2026, the four `record_going_*` scripts (scripted
  sweeps, 3 s settle per point) on 23 October; `record_static_load.py` and `fit_static_load.py` replace them. They
  needed the `keyboard` package and an Arduino angle sensor on COM5 (`arduino/ArduinoAngleReader.py`, still there,
  an external reference for joint 2), moved the arm without asking, and used the 2021 units (angles as fractions of
  pi, q3 shifted by 0.5). The raw data stays in `calibration/results` (servo 2 over a grid of q2 and q3, servo 3 over
  q2 + q3, going up and going down, in positions).
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
| 2 shoulder | XM540-W270 | 1000 | Present Current, 2.69 mA per unit | yes | 10.6 N.m, 4.4 A (2.41 N.m/A) | largest; holds the arm stretched out at about 250 units (0.67 A, 15 % of the stall current, about 4.8 N.m at the joint through the belt) without the spring |
| 3 elbow | XM430-W350 | 1500 | Present Current | yes | 4.1 N.m, 2.3 A | large; mounted reversed (`positions: [3072, 1024]`); holds the upper arm horizontal at about 280 units (0.75 A) without the spring, 300-320 (0.81-0.86 A) with it, which tripped its overload protection in a static hold (23 October) |
| 4 wrist 1 | XC430-W240 | 1000 | none, address 126 is Present Load (0.1 %); PWM limit instead of a current limit | no (modes 1, 3, 4, 16) | 1.9 N.m, 1.4 A | roll, only through the bent wrist, small |
| 5 wrist 2 | XC430-W240 | 2000 | none | no | 1.9 N.m, 1.4 A | pitch, the gripper |
| 6 wrist 3 | XC430-W150 | 600 | none | no | | roll, none |
| 7 gripper | XL330-M288 | 500 | Present Current, 1 mA per unit | in use (goal current 400 mA) | 0.52 N.m, 1.47 A at 5 V | |

- All arm servos run in position mode (3) with I = 0 and D = 0 (servo 1: D = 500). Manual defaults are P 800 (XM),
  700 (XC430-W240), 460 (XC430-W150), 400 (XL); the configured gains are higher. The gains, profile and goal
  current are RAM registers written by `initialize_servos` when the controller is constructed (not at enable), so
  after a power cycle the servos run at the defaults until a script starts the robot, and a gain changed on the fly
  stays until the next start or an explicit `set_pid()`. The servos accept gains of 0-16383 and drop a larger one
  silently (the handler refuses it since 21 October); `ServoHandler.read_pid` reads them back.
- Reads measured on 21 October: `ServoHandler.read_state` (addresses 120-146 in one packet: tick, moving status,
  pwm, current or load, velocity, position, the velocity and position trajectory of the profile, voltage,
  temperature) takes 2.0 ms for one servo, 4.3 ms for the RS-485 bus, 5.0 ms for the TTL bus, 9.2 ms for the arm;
  the same as a 4 byte position read, the latency timer and the return delays dominate. The input voltage reads
  11.2-11.6 V along the chain.
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
- Safety: the robot warns at 45 C and stops at 50 C (PLA). The arm drops when the torque goes off. Servo 3 trips
  its overload protection (Hardware Error Status 32, the overload bit) in a static hold of the upper arm
  horizontal at about 0.8 A of holding current, as it did after the tuning run `tune_joint6_20261023_0039` with the
  spring attached; the torque is off until a reboot, which drops the arm. The threshold isn't documented.
- Settings: latency timer 1 ms, sync read/write and a fixed period loop are done. 4 Mbps
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
  servo 3. No static model removes that; a direction-dependent offset (what 2021 did) halves it. Joint 2 measured
  in current on 23 October, holding a pose with the arm out: about 250 units after moving up and stopping, about
  50 after moving down a little and stopping; the mean, about 150 units (0.4 A), is gravity, half the difference,
  about 100 units (0.27 A), is Coulomb friction of the gearbox and the 3:1 belt. Friction is nearly as large as
  gravity on joint 2, so the direction of approach decides most of its resting error.

## The static model

```
load(q) = gravity(q) + residual(q)
```

(Until 23 October 2026 there was a `spring(q)` term; the spring is gone, see Step 3.)

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

Spring (removed). The spring ran from the housing of servo 2, on the base plate, to the housing of servo 3, which is
part of the upper arm, so its length was `L(q2, q3)` and it loaded both joints. For a linear spring the torque on
joint j is `k (L - L0) dL/dqj`; the `k L dL/dqj` part pulls like a mass and is absorbed by the gravity terms, only
`dL/dqj` is a column of its own (coefficient `-k L0`). That term (`spring_dl`, with a `spring` section in the
yaml) was in `static_compensation.py` until 19 November 2026; a yaml with a `spring` section is now refused.

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
- Poses: random inside the calibrated ranges, q2 and q3 varied independently (they load joints 2 and 3 with
  different levers), q4 and q5 varied for the wrist terms, including poses near the range ends. Approach each pose
  from both directions, wait until the arm has settled (2021: 3 s), log angles (radians, the 7 element array), goal
  and present position per servo, current, temperature and input voltage, as csv. The mean of both directions is
  the static load, half the difference the friction deadband. On joint 2 the deadband is nearly as large as the
  load (23 October), so a recording from one direction would be off by half of it. Hold out about 20 % of the
  poses. Servo 3 overloads in long static holds of the upper arm horizontal (about 0.8 A); the 3 s settles are
  short, but check the hardware error status between poses, which `record_static_load.py` does.
- Warm up with a few minutes of slow sweeps but stay below the 45 C warning (the robot stops at 50 C), and log the
  temperatures.
- Moving: `plan_move_to_pose`/`execute_move` for checked moves, or `move_servo` for one joint at a time (unchecked,
  small steps). `record_static_load.py` is that script; the 2021 `calibration/gravity_compensation.py` is deleted.

## Step 2: fit the gravity model

The code: the model is one explicit formula per joint in `static_compensation.py` (`joint_2_drive` to
`joint_5_drive`, the coefficients are their keyword arguments: `constant`, `cos_q2`, `sin_q2`, `sin_phi`, `cos_phi`,
`wrist_reach`, `wrist_pitch`, `wrist_roll`; `term_values` and `regressor_row` are derived from the formulas) and
`calibration/fit_static_load.py` does what the snippet below shows, per joint, with hold-out poses and standard
errors. The snippet is the idea in 30 lines:

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

# Q: list of angle arrays, E: {servo id: steady state error in positions}, mean of both directions
A = {j: np.array([basis(q)[j] for q in Q]) for j in (2, 3, 5)}
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

## Step 3: the spring (obsolete, the spring was removed on 23 October 2026)

Measured by hand on the robot on 23 October, raw register reads:

- After the tuning run `tune_joint6_20261023_0039` servo 3 reported a hardware error: Hardware Error Status
  (address 70) read 32, the overload bit. It happened during a static hold with the upper arm horizontal and was
  cleared by a reboot.
- Present Current of servo 3 (address 126, 2.69 mA per unit) holding the upper arm horizontal: -300 to -320 with
  the spring (0.81-0.86 A), about -280 without it (0.75 A). With the spring the overload tripped in a static hold;
  without it the same hold ran for several minutes without an error.
- Pulling on the spring by hand did not visibly change the current of servo 3 (its anchor is close to the joint 3
  axis, as the model said), yet the overload only occurred with the spring attached.
- Joint 2 without the spring, arm fully stretched out: about 250 units (0.67 A, about 15 % of the stall current
  of the XM540, about 4.8 N.m at the joint through the 3:1 belt). Comfortable, the spring is not needed.

Decision: the spring is off the robot. The anchor measurements, k and L0 and the consistency check of the fitted
`-k L0` between the two joints are no longer needed. The code kept the term as an option until 19 November 2026,
when it was removed (the `spring` section of the yaml, `--no-spring` of the fit script, `SpringGeometry`).

## Step 4: the residual

- Target: measured error minus the model, per joint, on the training poses.
- Inputs: q2 and q3, plus q5 for joint 3 if the wrist terms don't capture it. Base yaw and wrist roll should not
  matter.
- Model: a polynomial of degree 2-3 in (q2, q3), or a `RegularGridInterpolator` on a coarse grid; clamp the inputs
  to the sampled range at runtime.
- If the residual is larger than a few positions, go back to the model.

## Step 5: apply it at runtime

Where it plugs into the code (implemented on 19 October, see the work list at the end):

- `Servo.set_target_position_from_angle(angle, all_angles)` (`servo.py`): after interpolating the angle to a position
  and adding `constant_offset`, the `compensation` of the servo (a `StaticCompensation`) gives the positions for
  `all_angles` and they are added; `compensation_positions` keeps what was added. `move_servos` passes the
  commanded angles, so the model input is the target pose, not the measured one.
- Parameters in the yaml: the optional `compensation` field of a servo (`max_positions` and `terms`), validated
  by `servo_config.py`, with the coefficients in pwm units since 18 November. The model is one module shared with
  the fit script.
- Clamping: `Servo.clamp_target_position` clamps the uncompensated position to the calibrated range and adds the
  compensation back, so the compensation is kept at the ends of the range and the goal may exceed the calibrated
  range by at most `max_positions`; `servo_handler` then clamps to 0-4095. The physical range has to allow that
  margin, servo 3's range is [1024, 3072] reversed.
- Reported angles stay truthful: `get_angle_from_position` converts the measured position, and with the
  compensation the measured position is close to the uncompensated goal, so `get_current_angles`,
  `forward_kinematics`, the 0.05 rad start check of `check_move` (33 positions) and `hold_current_position` don't need
  to know about it. Only `_print_position_error` (`present - target`, the target includes the compensation) has to
  subtract it to show the residual.
- Rate: `move_servos` runs every step (100 Hz in `execute_move` and the xbox loop); a few sin/cos per step cost
  nothing. No separate loop, no change to the servos' own 1 kHz loop.
- Direction: none. The 2021 split (one set of coefficients for moving up and one for moving down, picked by the
  direction of the last commanded change) was kept as an option until 19 November 2026 and then removed: it is a
  friction compensation, not a gravity one. What it was for still holds: with friction at about two thirds of the
  gravity load on joint 2 (23 October), a mean fit leaves half the band, and a pose approached from the wrong
  direction sits a whole band off. Until the friction stage exists, approach final poses from one consistent
  direction.
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
   up/down difference is the Coulomb term, the slope against velocity the viscous term. Joint 2 is the largest:
   about 100 units of current (0.27 A) of Coulomb friction against 150 of gravity, measured at rest on 23 October.
2. Feedforward `Fc tanh(v / v0) + b v`, in positions, added to the static term. Use the desired velocity (the
   difference of consecutive steps of a `PlannedMove` times `steps_per_second`), not the measured one; v0 a few
   degrees per second.
3. Velocity lag: a PD loop trails a moving target in proportion to speed. Shift the goal ahead along the path, or use
   the Feedforward gain registers of the servos.
4. Trajectories are already smooth: the planner uses a minimum jerk profile and the profile velocity/acceleration
   per servo are set from the yaml.
5. Resting deadband: approach final positions from a consistent direction. (A split of the compensation by
   direction existed as an unused option until 19 November 2026; the friction term of item 2 replaces the idea.)

Limit: this is for slow moves. Fast motion or heavy payloads need inverse dynamics with link inertias, which this plan
does not identify.

## Open items

Information still needed:

- [x] Kinematics: `src/kinematics/kinematics.py`, the URDF is not needed.
- [ ] Masses: `resources/weights.txt` is from 2021 with ambiguous names; re-weigh the lower arm, the upper arm with
  the wrist servos, and the gripper, or skip it: the regressor fits the first moments directly, masses are only
  needed for payloads.
- [x] Spring: removed from the robot on 23 October 2026 (Step 3), nothing to measure.
- [ ] Which joints sag most and the payload range: 2021 says servo 3, then 2; re-measure, the gearing changed.

Gains first (21 October 2026, the full account is in `Tuning the servo gains.md`): the fit is in positions and
every coefficient absorbs 1/K with K proportional to the
P gain and the supply voltage, so the P gains have to be settled before the static load is recorded; a later P
change rescales the coefficients by P_old / P_new (plus a check), a change of operating mode invalidates the fit.
The friction band scales with 1/P too and the feedforward can't remove it, so P sets the floor of the result. The
script `calibration/tune_servo_gains.py` (one joint per run, `--joint`) does that: at a few poses it records a
baseline at rest, then trials at rising P (x1.3 from the yaml value up to 2x, or down when the yaml value fails):
the rest ripple of the position and the pwm, small steps (sized to keep the pwm below a goal pwm clamp of 3 times
the holding pwm, at least half the limit, the full limit on the unloaded joints 1, 4 and 6, applied before the
gains are written) with the profile of the servo off
so the loop itself is seen (overshoot, rings, settle time, steady state error per direction, the measured pwm per
position of error against P / 128), and a streamed planned move with the production profile (tracking against the
position trajectory of the servo). The first trial that overshoots more than 10 %, rings more than once, doesn't
settle in 0.5 s or raises the ripple or pwm noise at rest (per unit of P) is the knee; it recommends 0.65 times
the knee per pose. The smallest of the poses is the P of the joint; at that P it tries D = 0, P/8, P/4, P/2 and
the yaml D at every pose, and takes a D above the smallest passing one only where it lowers the overshoot or the
resting error after the steps by a position without adding pwm noise (joint 6 on 23 October: every D overshot a
position or less, but D 0 rested 3-6 positions from the goal and D 1100 0-3). Live abort rules (temperature, voltage drop, position jump, divergence,
oscillation, pwm at the clamp) and Ctrl+C restore the yaml gains, profile and goal pwm at once; divergence,
oscillation and pwm at the clamp count as the knee, the other three end the run with the finished poses kept;
nothing touches the eeprom. Tested offline against a simulated second order servo with friction
(`tests/servo_controller_tests/gain_tuning_tests.py`) and reviewed; joint 6 ran on 21-23 October without a
false abort (the thresholds: 40 positions or 2.5 times the sag of divergence, 6 positions of oscillation, 90 % of
the clamp, the temperature_limit of the yaml, 1.5 V).
Order: the wrist first (6, 4, 5: the rolls carry no static load, so a mistake is cheap and the sampling, the abort
rules and the restore get checked there), then 3, 1 and 2 last (the belt joints; 2 carries the arm). Distal first
because the distal joints are the load of the proximal ones and have to hold stiffly with their final gains when
those are tuned, while a proximal gain doesn't change a distal loop. I stays 0: the firmware's anti-windup only acts
when the pwm saturates, there is no integrator clamp or deadband and the reset on a new goal is undocumented, so
against the table or an object the integrator ramps to full duty (about 30 N.m at joint 2 through the belt); inside
the friction band an integrator hunts or creeps; and the feedforward has the predictable part. A payload trim, if
ever needed, belongs on the host with a deadband of half the band and a clamp. After the gains: add the gains to
the csv of `record_static_load.py` (it doesn't log them, so a fit can't tell which P it belongs to), then record.

Work to do (the code of 19 October 2026 is in `src/robot_controllers/dynamixel_robot/static_compensation.py`,
`calibration/record_static_load.py`, `calibration/fit_static_load.py` and the `compensation` field of
`servo_config.yaml`; see the revision of 18 November at the top for what changed since and what was recorded):

- [x] 23 October: the spring removed from the robot; 19 November: its term removed from the code.
- [x] `tune_servo_gains.py` ran on joint 6 on 21-23 October: the pwm per position of error matched P / 128, the pwm
  noise criterion was fixed to ignore the pwm that follows an encoder flicker. Servo 3 overloaded in a static hold
  after run `0039` (with the spring); the script checks the hardware error status before every trial but not after
  the last one, see the open item in `Tuning the servo gains.md`.
- [ ] Run `tune_servo_gains.py` on the other joints, 4, 5, 3, 1, 2 in that order, and put the recommended gains in
  the yaml. The default poses and their test moves stay inside the workspace limits (the wrist within 41.8 cm
  of the shoulder since 25 October); `--pose` for others.
- [ ] Log p, i, d per servo in the csv of `record_static_load.py`.
- [x] Sync read of Present Current (joints 1-3, 7) and Present Load (4-6): `ServoHandler.read_present_current`,
  `DynamixelRobotController.get_present_currents`.
- [x] Pose sampling and logging script: `python -m src.robot_controllers.dynamixel_robot.calibration.record_static_load`
  (random poses the planner accepts, both directions with an overshoot, settle, csv; joints 4 and 5 are only sampled
  within plus or minus 90 degrees by default, `--range` and `--floor` to change the ranges and the floor).
- [x] Fit: `python -m src.robot_controllers.dynamixel_robot.calibration.fit_static_load static_load.csv` prints the
  coefficients with their standard errors, the condition number, the hold-out error, the hysteresis, and the yaml
  blocks (since 18 November in pwm units, not negated; `--split` and its up/down blocks went on 19 November).
- [x] `Servo` compensation with the yaml field and tests. Decisions taken: the calibrated range clamps the target
  before the compensation (the goal may exceed it by the compensation, `max_positions` caps that, the handler clamps
  to 0-4095); `hold_current_position` drops the compensation. (The direction tracking for the up/down split was
  removed with the split on 19 November.)
- [ ] Residual model, if the hold-out error asks for it.
- [ ] Record on the robot, fit, put the blocks in the yaml, enable one joint at a time with the arm supported and
  check the sign with `_print_position_error`.
- [ ] Validate (Step 6), then the friction stage.
