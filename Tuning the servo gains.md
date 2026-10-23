# Tuning the servo gains

Oct 21, 2026. Written against the code of that day, the ROBOTIS manuals and the registers read from the robot. The
decisions here come from a research and refutation pass over the plan and from an independent review of the code
(41 agents, 32 confirmed findings, all worked in); the script has not run on the robot yet.

## Summary

- The P gains come before the gravity compensation. The compensation of `Handover Gravity and Spring
  Compensation.md` is fitted in encoder positions and every coefficient absorbs 1/K, where K, the load per position
  of error, is proportional to the P gain and the supply voltage. A later P change rescales the coefficients by
  P_old / P_new and needs a check; a change of operating mode invalidates the fit. The friction band scales with 1/P
  too and the feedforward can't remove it, so P sets the floor of what the compensation achieves.
- I stays 0. Not because of the textbook (PD plus gravity compensation is proven stable, not accurate, and the
  theorem doesn't cover joints 1 and 2 at all), but because the servos give the integrator no clamp and no deadband,
  so against the table or an object it ramps to full duty, about 30 N.m at joint 2 through the belt, into PLA; inside
  the friction band an integrator hunts or creeps; and the feedforward already carries the predictable part.
- `src/robot_controllers/dynamixel_robot/calibration/tune_servo_gains.py` tunes P and D of one joint per run: small
  steps with the profile of the servo switched off under a goal pwm clamp, rising P until the first sign of ringing,
  ripple or pwm noise, 0.65 times that knee as the recommendation, D only where the step overshoots. Order: the
  wrist first (6, 4, 5), then 3, 1 and 2.
- The script is tested against a simulated servo with friction and reviewed, not run on the robot. The abort
  thresholds are guesses until joint 6 has run.

## Why the gains come first

The static load recording (`record_static_load.py`) measures `present - goal` at rest, in positions. With I = 0 that
error is load / K. K is not a datasheet number: the position controller outputs a pwm duty of P / 128 per position
of error (the manual gives the formula but no unit), the torque is the duty times the supply voltage divided by the
winding resistance, and the resistance rises with temperature. So the fitted coefficients belong to the P gain, the
operating mode and the supply they were recorded with. The recording csv doesn't log the gains yet (an open item),
so an old fit would go stale silently.

What a later gain change costs: a P change is a rescale by P_old / P_new of every coefficient plus a check at a
few poses, because the sag and the friction band both scale with 1/P. D, the feedforward gains and the profile don't
enter the static error at all (at rest the D and feedforward terms are zero), so they can be tuned after the fit. A
change of operating mode resets the gains and changes what the loop output is, so it means recording again.

Two more reasons to settle P first: the hysteresis band (2021: about 20 positions on servo 3, 1.8 degrees) is the
part no model removes, and only P shrinks it; and the up/down split of the compensation needs a final approach that
doesn't ring, since an overshoot flips the direction the pose was approached from.

## Why I stays 0

The textbook line "a robot arm only needs PD" is Takegaki and Arimoto's result: PD plus exact gravity compensation
is globally asymptotically stable for setpoint regulation. It assumes rigid, frictionless, collocated joints and an
exact gravity model. It says nothing about accuracy when K drifts by 10-20 % with temperature and voltage, and joints
1 and 2 have the encoder behind the belt, so the theorem doesn't apply there. The reasons that do hold on this arm:

- Contact. The firmware has an anti-windup gain Ka "that cannot be modified by users", and the diagram shows it
  acting only once the output saturates at the pwm limit or the goal pwm. There is no integrator clamp, no deadband,
  and whether the integrator is cleared on a new goal or on torque enable is undocumented. Against the table, an
  object or the end of a range the integrator ramps to 100 % duty: 10.6 N.m at servo 2, about 30 N.m at the joint
  through the 3:1 belt. The XC430 wrists have no current sensing to even see it.
- The friction band. After the feedforward the remaining error sits inside the stiction band. An integrator inside
  the band produces stick-slip hunting (Olsson and Astrom) or a slow creep, and an integrator with a deadband wider
  than half the band never fires there. It buys nothing where the feedforward stops.
- Moves. An integrator accumulates the tracking lag during a move and returns it as overshoot at the end, which
  also corrupts the up/down direction logic of the compensation.

ROBOTIS itself recommends P, then D, then a small I last for residual error under load, so I = 0 is a design choice,
not a theorem. What would change it is payload: the feedforward has no payload term. A trim for unknown masses in the
gripper belongs on the host, with a deadband of half the friction band, a clamp of about 10 positions, and frozen
while moving or near the pwm limit. Current based position mode (5) would make K independent of the supply voltage
and bound the torque through Goal Current, but it is an eeprom switch with the torque off, it resets the gains and
the profile, it needs its own tuning, and the XC430s can't do it. Not pursued.

## The robot as read on 21 October

The registers were read with the torque off after a power cycle, before any script had started the controller.

| Register | Servos 1-3 (XM) | 4, 5 (XC430-W240) | 6 (XC430-W150) | 7 (XL330, mode 5) |
|---|---|---|---|---|
| Position P, I, D | 800, 0, 0 | 700, 0, 0 | 460, 0, 0 | 250, 0, 600 |
| Feedforward 1st, 2nd | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| Profile velocity, acceleration | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| PWM limit, goal pwm | 885 | 885 | 885 | 885 |
| Current limit | 1193 (XM430), 2047 (XM540) | none | none | 1750 |
| Velocity limit | 200 (XM430), 128 (XM540) | 306 | 460 | 445 |
| Return delay time | 250 (500 us) | 250 | 250 | 250 |
| Input voltage | 11.6, 11.4, 11.3 V | 11.3 V | 11.2 V | 5.1 V |

So the gains, profile and feedforward are RAM registers at their firmware defaults until `initialize_servos` writes
the yaml values when `DynamixelRobotController` is constructed (not at `enable_servos`, as the gravity handover
said before). A gain changed on the fly stays until the next start of the robot, a `reboot_all_servos` or an explicit
`set_pid()`. The servos accept gains of 0-16383 and drop a larger value silently in a sync write; the handler and
the yaml validation refuse those since 21 October. The manual default for the XC430-W150 on joint 6 is 460, not the
700 the handover table had.

Read timings, one `GroupSyncRead` per bus of addresses 124-143 (20 bytes: pwm, current or load, velocity, position,
velocity trajectory, position trajectory), 50 rounds each:

| Read | Mean | Max |
|---|---|---|
| One servo | 2.0 ms | 2.1 ms |
| RS-485 bus, servos 1-3 | 4.3 ms | 5.0 ms |
| TTL bus, servos 4-7 | 5.0 ms | 5.8 ms |
| Whole arm | 9.2 ms | 10.0 ms |

A 4 byte position read costs the same: the latency timer and the return delays dominate, the bytes don't. The
`ServoHandler.read_state` of the code reads 27 bytes from 120 (plus the realtime tick, moving status, voltage and
temperature); it was tested on the fake bus only, the robot was off by then.

## The method

One joint per run. The other joints hold their position at their yaml gains throughout. Per pose:

1. Move there with the planner at the yaml gains, and record a baseline at rest for 5 s: the holding pwm, the
   ripple of the position and the pwm, the sag (`position trajectory - position`).
2. Trials at P = yaml P times 1.3^k up to twice the yaml P (`--factor`, `--max-factor`), or downwards from the
   yaml P when that one fails. A trial:
   - sets the goal pwm clamp of the servo: 3 times the holding pwm, at least half of 885, never above 885,
     written and read back before the gains change, so the output is bounded from that moment;
   - writes the gains to this servo only and reads them back (sync writes aren't acknowledged), re-sends the goal;
   - records 3 s of rest: ripple, pwm noise, sag. The holding pwm of this rest may exceed the baseline's, because
     the servo sits on the other side of the friction band; the clamp is widened then;
   - sizes the step: `(0.8 * clamp - holding pwm) * 128 / P` positions, between 8 and 60, so the pwm stays below
     the clamp if the servo really outputs P / 128 per position (the measured slope goes to the csv);
   - switches the profile of the servo off (velocity and acceleration 0, a step instruction) and steps the goal:
     up, back, down, back, twice; each step is held 0.8 s;
   - restores the profile and the goal pwm and streams a planned 8 degree move of the joint away and back with the
     production profile, sampling the tracking against the position trajectory of the servo;
   - restores the yaml gains and reads everything back.
3. The first trial that violates a criterion is the knee; 0.65 times the knee is the recommended P of the pose.
4. When every pose has a recommendation, the smallest is the P of the joint. At that P, at every pose, D = 0,
   P/8, P/4, P/2 (and the yaml D) are tried unless the step at D = 0 doesn't overshoot; the D with the least overshoot
   and rings whose pwm noise stays below 1.5 times that of D = 0 is recommended.

Everything is measured against the final value of a step, not the goal: with I = 0 the servo settles at the goal
minus the sag, so "settled within 2 positions of the goal" never happens on a loaded joint.

The criteria of a trial (`Criteria` in `gain_tuning.py`):

| Flag | Rule |
|---|---|
| overshoot | beyond the final value by more than 10 % of the step and more than 2 positions |
| ringing | more than one crossing of the final value after the first |
| settle | not within max(2 positions, 5 % of the step) of the final value within 0.5 s |
| ripple | position peak to peak at rest above max(2, twice the baseline) |
| pwm_noise | the pwm std at rest that the position doesn't explain (the residual after fitting the pwm against the position error, P / 128 per count), divided by P / yaml P, above max(5 units, 1.5 times the baseline's residual). A joint resting on an encoder edge flickers between two counts and its pwm steps with it; the raw std of that is half a step and flagged every stable joint at high P (joint 6, 23 October) |
| stuck | no step moved at least half its size, the trial says nothing |

The division by P / yaml P matters: a dither of one position gives a pwm of P / 128 units, so a fixed dither would
fail the noise rule at 1.69 times the yaml P and every sweep would end there.

The live abort rules (`AbortLimits`), checked every 5 samples; a violation restores the yaml gains, the profile
and the goal pwm at once:

| Reason | Rule | Counts as |
|---|---|---|
| divergence | after 0.5 s of grace, error against the trajectory above max(3 steps, 40 positions, 2.5 times the sag expected at these gains); during a streamed move above max(120, 2.5 times the sag) | the knee |
| oscillation | after the grace, position peak to peak over 0.2 s above max(6, 3 times the baseline) | the knee |
| pwm at the clamp | the pwm within 10 % of the clamp for 0.1 s | the knee |
| temperature | 50 C, the stop limit of the robot (raised from 42 on 23 October); the 45 C warning check between trials stops the run earlier, and a servo is not tried within 3 C of the abort limit | ends the run |
| voltage | 1.5 V below the voltage at the start of the trial (0.7 until 23 October, which the dips of the supply reached) | ends the run |
| position jump | more than 50 positions between two samples (encoder glitch, slipped belt) | ends the run |

The sag expected at the gains of a trial is the sag of the baseline times yaml P / P; the baseline itself runs with
the divergence rule off, since its sag isn't known yet. A run that ends keeps the results of the finished poses.

What the method deliberately doesn't do: hunt for sustained oscillation (Ziegler-Nichols style). On a belt
coupled, spring loaded PLA arm an overload shutdown cuts the torque, drops the arm and wipes the gains, so the sweep
stops at the first flag and backs off from there.

## Order of joints

Distal first, as the literature on independent joint control has it: the distal joints are the load of the proximal
ones and must hold stiffly at their final gains when the proximal joints are tuned, while a proximal gain changes
nothing in a distal loop. The order for this arm: 6 and 4 first (the rolls carry no static load, so the sampler, the
abort rules and the restore are checked where a mistake is cheap), then 5, 3, 1, and 2 last (it carries the arm).
Joints 1 and 2 are the belt joints: the belt compliance is inside the loop, a resonance there rather than damping
limits P, and joint 1's yaml D of 500 is the only D in use; compare it with 0 before raising P there. Joint 2 also
has Coulomb friction of about 100 units of current against a gravity load of about 150 (holding the arm out after
moving up: 250 units, after moving down: 50; 23 October), so its steps will show a wide friction band and its
resting error depends mostly on the direction it came from; judge its overshoot and settle against the final value
as the script does, and expect the up and down steady state errors to differ by about the band.

## Poses

The workspace limits are a sphere of 60 cm around the base (57 until 23 October 2026), so an arm stretched out
horizontally sits near its edge (an upper arm horizontal with the lower arm at 45 degrees puts the tip at 56.6 cm).
The named poses keep 8 degrees of margin on every joint for the test moves, in degrees of joints 1-6:

| Name | Pose | Why |
|---|---|---|
| elbow_out | 90, 90, 0, 0, 0, 0 | lower arm vertical, upper arm and gripper horizontal: the largest load on joints 3 and 5 |
| reach | 90, 25, 65, 0, 90, 0 | lower arm 25 degrees up, upper arm horizontal, gripper bent: the largest lever on joint 2, the gripper's mass off the roll axes |
| low | 90, 60, 0, 0, 0, 0 | upper arm 30 degrees below horizontal: less load on 3, another lever on 2 |

Defaults: joint 1 elbow_out and low; 2 reach, elbow_out, low; 3 elbow_out, low; 4 reach, elbow_out; 5 and 6
elbow_out, reach. Both elbow_out and reach hold the upper arm horizontal, the largest load on servo 3: it holds
that at about 280 units of present current (0.75 A) and tripped its overload protection there in a static hold on
23 October while the spring was still on (300-320 units); the spring is gone since, and the same hold then ran for
minutes. Keep the holds at these poses short between runs. `--pose` takes names or 6 comma separated degrees; a pose the planner refuses (the pose itself or
the joint moved 8 degrees either way) is skipped with the reason. A truly retracted pose doesn't exist inside the
limits: the arm folded upwards is 70 cm tall. The sphere is measured from the ground rather than the shoulder, which
is why it cuts tall poses and not much else; that is parked in `Workspace limits.md`.

## Running it

```
python -m src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains --joint 6
```

With the arm powered, its torque off, nothing in its way, foam under the elbow and the wrist for the first runs, and a
hand near the power switch. It prints the plan and asks before enabling the servos (`--yes` skips the question).
Options: `--pose`, `--factor` 1.3, `--max-factor` 2.0, `--no-d`, `--d-fractions 0,0.125,0.25,0.5`, `--repeats` 2,
`--hold` 0.8 s, `--rest` 3 s, `--baseline-rest` 5 s, `--move-degrees` 8 (0 disables the streamed move), `--speed`
0.5 rad/s, `--pwm-clamp` (overrides the clamp, refused below 1.5 times the holding pwm), `--max-temperature` 42,
`--floor`, `--output-dir`, `--plot`.

Output in `--output-dir`: `tune_joint<N>_<time>_samples.csv` (every sample: pose, trial, p, d, segment, repeat,
step index, direction, time, tick, goal, position, position trajectory, velocity, velocity trajectory, pwm, current,
moving status, voltage, temperature) and `..._trials.csv` (per trial: gains, clamp, step, rest ripple and pwm,
sag, max overshoot, rings, settle and rise times, the steady state error per direction, the measured pwm per
position, the tracking of the move, flags, warnings, abort reason, temperatures, voltage). The console shows every
trial as it ends and a summary per pose with the yaml line to copy:

```
    pid: {p: 1650, i: 0, d: 0}
```

Ctrl+C stops after the current read and restores the servo; a second Ctrl+C kills the script. After a kill the trial
gains and profile stay in the servo until the robot is started again (any script that calls `get_robot()` rewrites
the yaml values), and a goal pwm clamp stays until the servos are rebooted or power cycled; the script also resets a
leftover clamp when it starts. Nothing touches the eeprom.

First run, on joint 6 with the arm supported: watch that the baseline shows the sag you expect, that the clamp and
the step sizes are sensible (the console prints them), that a trial ends in the restore (the gains print back), and
try one Ctrl+C. Then the other joints. Put the recommended gains in `servo_config.yaml` and record the static load
again.

## The code

- `src/robot_controllers/dynamixel_robot/calibration/gain_tuning.py`: the arithmetic, pure functions on lists of
  samples: `pwm_clamp`, `step_size`, `rest_metrics`, `step_metrics`, `tracking_metrics`, `pwm_per_position`,
  `evaluate`, `abort_reason`, `environmental_abort`, the sweep values and `choose_p`, `choose_d`.
- `src/robot_controllers/dynamixel_robot/calibration/tune_servo_gains.py`: `TuningSession` (snapshot, restore,
  the planned moves through `record_static_load.move_to`, sampling with the abort rules, the trial, the P phase per
  pose, the D phase at the joint P), `recommend`, `print_summary`, `plot_results`, `main`.
- `ServoHandler` (`servo_handler.py`): `read_state` (one 27 byte packet per bus, addresses 120-146, a `ServoState`
  per servo), `read_pid`, `read_profile`, `read_goal_pwm`, `set_profile(servo_id, velocity, acceleration)`,
  `set_goal_pwm`; the gain range check in `__write_pid`; `read_temperature_and_voltage` in one packet per bus;
  `__sync_read_fields` picks 1, 2 or 4 byte fields out of a block with the sdk's `getData`, which the old
  `__sync_read` couldn't (it passed the block length to `getData`, which returns 0 for anything but 1, 2 or 4).
- `dynamixel_x_config.py`: the addresses of the state block, the feedforward gains, the pwm limit and goal pwm,
  `MAX_GAIN` 16383, `MAX_PWM` 885, the moving status bits. `servo_config.py` bounds the yaml gains to `MAX_GAIN`.
- Tests: `tests/servo_controller_tests/gain_tuning_tests.py` (the metrics on synthetic traces, every abort rule,
  the choices, and the whole session on a simulated second order servo with coulomb friction, a damping that drops
  above a P threshold so it rings and, with a steep enough drop, oscillates with a bounded amplitude; the knee, the
  abort, the sweep downwards, the D phase at the joint P, an environmental stop, a warm servo, a leftover clamp, a
  flaky read, a refused pose) and the block read tests at the end of `servo_handler_tests.py`.
- Removed: `calibration/pid_test.py` (2021: a quarter turn step of servo 1 at import, torque left on).

## What the review changed, and what is still unverified

Confirmed by the review and worked in: the clamp was sized from one side of the friction band and the rest phase
was checked against the wrong register, so the saturation rule would have aborted every trial; the divergence limit
of 40 positions ignored the sag, which a loaded joint exceeds; the pwm noise criterion grew with P; the streamed
move was planned from the measured angles, so every joint lost its holding error; the D sweep ran at a different P
per pose, so no D could be compared; an abort for heat or a voltage dip became the knee; a stop lost the results of
the finished poses; the goal pwm and profile writes weren't read back; a clamp left by a killed run was adopted as
the original; one failed read ended the run.

Assumptions the plan still rests on, each a one line measurement on joint 6:

- The pwm output of the loop is P / 128 units per position of error. The manual gives the formula without a unit;
  the trials csv has the measured slope (`pwm_per_position`). Confirmed on joint 6 on 21 October: 4.69, 6.00 and
  7.89 at P 600, 780 and 1010 (P / 128 is 4.69, 6.09, 7.89). That run also showed servo 6 at 10.5 V with dips to
  9.8 V, 0.7 V below the morning's reading, hence `calibration/check_supply_voltage.py`.
- Position Trajectory is the setpoint of the loop also when the host streams goals at 100 Hz (the servo re-plans
  "based on present movement velocity"), so the tracking error of the move test is meaningful; log the host goal
  next to it, which the csv does.
- The abort thresholds: 40 positions or 2.5 times the sag of divergence, 6 positions of oscillation, 90 % of the
  clamp for 0.1 s, 50 C, 1.5 V, 50 positions of jump. They are guesses; the joint 6 runs of 21-23 October did
  not trip them falsely at rest once the temperature and voltage limits were raised.
- The loop rate of the servos (about 1 kHz is assumed everywhere, nothing in the manual says so), the behaviour of
  the integrator across goals (irrelevant while I = 0), and the Bus Watchdog as a dead man (untested, not used).

Open items:

- The script only watches the servo under test, so it missed servo 3 overloading during run
  `tune_joint6_20261023_0039` (Hardware Error Status 32 on servo 3 after a static hold with the upper arm
  horizontal, with the spring still on). It reads the hardware error status of all servos before every trial
  (`check_servos`), but not after the last trial of a pose or of the run, nor while the arm waits at the final
  prompt, and the samples log the servo under test only. To add: read the Hardware Error Status of all servos
  after every trial as well and before the final prompt, stop the run on any error, and log the present pwm and
  current of all servos (one whole-arm read per segment, or at a lower rate than the 2 ms of the servo under test,
  which a whole-arm read of 9 ms would otherwise slow down four times).
- Log p, i, d per servo in the csv of `record_static_load.py`.
- The Following Error bit of the moving status as a contact detector; Fast Sync Read (supported by every servo of
  the arm) and a lower return delay time for faster reads.
- The friction stage of the gravity handover after the gains and the static load are in; joint 2 will dominate it.

## What the literature says

- Independent joint control: with a high gear ratio every joint is a second order plant (motor inertia plus the
  reflected load, back-EMF damping) and is tuned as one loop, for critical damping, with the loop bandwidth under
  half the lowest structural resonance. M. W. Spong, S. Hutchinson, M. Vidyasagar, Robot Modeling and Control,
  Wiley 2006, chapter 6; the bandwidth rule is from R. P. Paul, Robot Manipulators: Mathematics, Programming, and
  Control, MIT Press 1981. On this arm the belts of joints 1-2 and the PLA are the resonance, and the distal first
  order follows from the distal joints being the load of the proximal ones.
- PD plus gravity compensation: M. Takegaki, S. Arimoto, "A new feedback method for dynamic control of
  manipulators", ASME J. Dynamic Systems, Measurement, and Control 103 (1981) 119-125; P. Tomei, "Adaptive PD
  controller for robot manipulators", IEEE Trans. Robotics and Automation 7 (1991) 565-570, the compensation
  evaluated at the desired configuration, which is what `move_servos` does with the commanded angles; R. Kelly, "PD
  control with desired gravity compensation of robot manipulators: a review", Int. J. Robotics Research 16(5)
  (1997) 660-672. Stability with an exact model; nothing about accuracy under a wrong K or a payload.
- Integral action done right: R. Kelly, "Global positioning of robot manipulators via PD control plus a class of
  nonlinear integral actions", IEEE Trans. Automatic Control 43(7) (1998) 934-938: a saturated integral with gain
  conditions, the principled version of a host side trim with a clamp.
- Friction: B. Armstrong-Helouvry, P. Dupont, C. Canudas de Wit, "A survey of models, analysis tools and
  compensation methods for the control of machines with friction", Automatica 30(7) (1994) 1083-1138 (a PD loop
  with Coulomb friction rests anywhere inside a band, hence the steps in both directions); H. Olsson, K. J. Astrom,
  "Friction generated limit cycles", IEEE Trans. Control Systems Technology 9(4) (2001) 629-636 (integral action
  plus stiction gives hunting).
- Tuning practice: J. G. Ziegler, N. B. Nichols, "Optimum settings for automatic controllers", Trans. ASME 64
  (1942) 759-768, finds the ultimate gain by driving the loop to oscillation, which this arm can't afford; K. J.
  Astrom, T. Hagglund, "Automatic tuning of simple regulators with specifications on phase and amplitude margins",
  Automatica 20(5) (1984) 645-651, and PID Controllers: Theory, Design, and Tuning, ISA 1995, the relay method that
  bounds the amplitude but needs access to the torque command, which the firmware's position mode doesn't give;
  G. Ellis, Control System Design Guide, 4th edition, Butterworth-Heinemann 2012, the servo drive practice the
  script follows: small steps in the linear region, raise P to the first ripple or overshoot and back off, D only
  where the step overshoots, feedforward for the tracking lag.
- Dynamixel specifically: no peer reviewed tuning guidance. The ROBOTIS e-manual pages of the XM430-W350,
  XM540-W270, XC430-W240/W150 and XL330-M288 (the position controller diagram: the error feeds P, I and D in
  parallel, the feedforward gains act on the trajectory velocity and acceleration, the output is limited by the goal
  pwm) and ROBOTIS's general advice: P for speed, D for damping, a small I last. The LeRobot Koch arm runs its
  elbow at P 1500, I 0, D 600, close to this yaml. The firmware's D acts on the error, so with the profile off it
  adds a derivative kick on a step and with the profile on it is a velocity tracking term; the velocity feedforward
  (Feedforward 1st Gain) is outside the loop and can't destabilise it, which makes it the parameter for the tracking
  lag in the friction stage later.
