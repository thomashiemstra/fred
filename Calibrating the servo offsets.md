# Calibrating the zero of servos 1 and 2

Servos 1 and 2 drive their joints through a 3:1 timing belt, and the magnetic encoder of each servo was moved to the
joint output shaft. The servo therefore counts joint positions (4096 per joint turn), but its zero is wherever the
magnet happens to sit on the shaft. After every reassembly the zero moves and the `offset` of the servo in
`src/robot_controllers/dynamixel_robot/servo_config.yaml` has to be measured again. A jig that holds servo 1 at
90 degrees and servo 2 at 0 degrees makes that a single reading per servo.

## How the offset works in the code

`Servo` (`src/robot_controllers/dynamixel_robot/servo.py`) maps angles to calibrated positions with the `positions` and
`angles` of the yaml and then adds the offset:

```
raw position (what the servo reports and gets)  =  calibrated position  +  offset
calibrated position                              =  interp(angle, angles, positions)
```

So at a joint angle you know, the offset is the raw reading minus the calibrated position of that angle:

```
offset = present position - interp(jig angle, angles, positions)
```

Both servos have `positions: [1024, 3072]` for `angles: [0, 180]`, so the calibrated positions are 1024 at 0 degrees,
2048 at 90 degrees and 3072 at 180 degrees. The angle conventions are the ones of `src/kinematics/kinematics.py`:
servo 1 at 0 degrees points the arm along +x, at 90 degrees straight ahead along +y (where the workspace is);
servo 2 at 0 degrees holds the lower arm horizontal, at 90 degrees straight up.

Worked example, servo 2 in the jig at 0 degrees reading 457:

```
offset = 457 - 1024 = -567
```

Not +567 and not "towards 2048": 2048 is the 90 degree position, and the code adds the offset to the calibrated
position, so the offset is negative when the raw reading is below it. Check: with `offset: -567` the code sends 457
for 0 degrees and 1481 for 90 degrees, and reports 0 degrees when it reads 457.

Servo 1 in the jig at 90 degrees reading `p1`:

```
offset = p1 - 2048
```

Measured on 18 October 2026 with the jig: servo 1 read 2394 at 90 degrees (offset 346, the yaml had 354) and servo 2
read 457 at 0 degrees (offset -567, the yaml had -464, so its zero was 103 positions or 9.1 degrees too high). The
rework handover's rest reading of 545 for servo 2 therefore means the arm rests 7.7 degrees above horizontal, not
1.3 degrees below as the old offset suggested. Homing offsets were 0 on both servos.

## Procedure

Prerequisites: `main.py` and DYNAMIXEL Wizard closed (one program per port), servo power on, torque off (the arm is
limp, the jig carries it), the jig mounted and both joints seated against it.

The short way is the script, which does steps 1, 2, 4 and 5 and asks before writing the yaml. By default it expects
the jig to hold servo 1 at 90 degrees, servo 2 at 0 degrees and servo 4 (wrist roll) at 0 degrees:

```
venv\Scripts\python.exe -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets
venv\Scripts\python.exe -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets --dry-run
venv\Scripts\python.exe -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets --jig 1=90 2=0 --samples 10
```

It reads only the servos in the jig (the others may be unpowered, but servo 4 is on the TTL bus, so that bus has to
be powered for the default), refuses when their torque is on or the readings jitter by more than 2 positions, warns
about a homing offset in the servo, checks that the raw range stays inside 0-4095, and rewrites only the `offset:`
lines of the yaml, comments included. Servo 4 covers a full turn (`positions: [0, 4096]`), so with any offset the
last positions at one end of its range are unreachable; the script says how many and goes on. Then do steps 3, 6
and 7 by hand (for servo 4: turn the gripper by hand towards +90 degrees and the raw position must go up). The long
way:

1. Read the raw positions a few times. From the repo root:

   ```
   venv\Scripts\python.exe -c "from src import global_constants as g; from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController as R; r = R(None, g.dynamixel_robot_config); print([r.get_current_positions() for _ in range(5)])"
   ```

   Constructing the controller finds the port, writes mode, profile and pid, and does not enable the torque. The
   array is indexed by servo id (index 0 unused). The readings should agree within 1-2 positions; if they drift, the
   joint isn't seated against the jig.

2. Read the Homing Offset register of both servos (eeprom, address 20, 4 bytes signed). The servo adds it to every
   present position, so the yaml offset is only valid for the homing offset that was in the servo when it was
   measured. `resources/homing_offsets.txt` (2021) lists -5 for servo 2:

   ```
   venv\Scripts\python.exe -c "from src import global_constants as g; from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController as R; from src.robot_controllers.dynamixel_robot.dynamixel_utils import to_signed; r = R(None, g.dynamixel_robot_config); h = r.servo_handler; print({i: to_signed(h.packet_handler.read4ByteTxRx(h.port_handler, i, 20)[0], 4) for i in (1, 2)})"
   ```

   Set it to 0 in DYNAMIXEL Wizard (torque off) and keep all calibration in the yaml, so git holds the single source
   of truth. Use the register only when a joint's raw range does not fit in 0-4095 (step 4), it is valid from -1024
   to 1024 in position mode.

3. Check the direction. Lift the lower arm by hand a few degrees above the jig and read again: the raw position of
   servo 2 must go up. Turn the base towards +x (away from the workspace) and the raw position of servo 1 must go
   down. A joint that counts the other way needs its `positions` reversed (`[3072, 1024]`, like servo 3) before the
   offset is computed, because the magnet on the output shaft can be read in either sense.

4. Compute the offsets and the raw range of each joint, `[1024 + offset, 3072 + offset]`. The whole range has to lie
   inside 0-4095: a servo in position mode cannot cross its 0/4095 boundary, and `servo_handler` clamps every goal
   to 0-4095 anyway. For servo 2 with -567 the range is 457-2505, fine. For servo 1 the reading at 90 degrees has to
   be between 1024 and 3071 for the range to fit; otherwise rotate the magnet, or move the zero with a Homing Offset
   of at most 1024 in the servo and record it in `homing_offsets.txt`.

5. Edit `servo_config.yaml` (`offset:` of servo 1 and servo 2) and run the tests, they load the real yaml:

   ```
   venv\Scripts\python.exe -m unittest discover -s tests -t . -p "*test*.py"
   ```

6. Verify without moving: read the angles with the jig still mounted. Servo 1 must report 1.5708 and servo 2 0.0
   (within 0.002 rad, one position):

   ```
   venv\Scripts\python.exe -c "from src import global_constants as g; from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController as R; r = R(None, g.dynamixel_robot_config); print(r.get_current_angles())"
   ```

7. Verify with a move: take the jig off, support the arm, and run `robot_move_test`, which prints the current pose,
   plans a straight line to `Pose(30, 25, 5)` and asks before moving:

   ```
   venv\Scripts\python.exe -m src.robot_controllers.dynamixel_robot.robot_move_test
   ```

   The reached pose it prints should be within a few mm of the target; an offset error of 10 positions on servo 2 is
   about 0.9 degrees, 4 mm at the wrist. Check the base points straight ahead at the start pose.

8. Update the "Servo 2 at rest" row of `Handover Robot Controller Rework.md`.

## What else the offset touches

- Before the torque is enabled, `enable_servos` refuses when an arm servo rests more than 5 degrees
  (`CALIBRATED_RANGE_MARGIN_DEGREES`) outside the calibrated range of the yaml, offset included: outside the range
  the angle the controller reports is clamped to the range, so the first move would jump. The check follows the
  offset, nothing else to update. (The 2021 check of the raw position against 0 and 2500 and its translation into
  fixed angles were removed on 23 October 2026, they forbade 99 to 180 degrees of the base.)
- The calibrated range in the yaml (`positions`) stays [1024, 3072]; only the offset moves. The mechanical range of
  the joint still has to cover it.
- The gravity compensation handover assumes joint positions; the offset doesn't change it.
