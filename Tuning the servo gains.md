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
  steps with the profile of the servo switched off under a goal pwm clamp, rising P until the first sign of ringing
  or ripple, 0.65 times that knee as the recommendation, then the whole D grid at that P, a D above the
  smallest passing one only where it lowers the overshoot or the resting error after the steps by a position.
  Order: the wrist first (6, 4, 5), then 3, 1 and 2.
- The feedforward stage (`--ff`, `--ff-only`) measures the velocity lead of a joint from the lag of a streamed sweep
  at three speeds and verifies it with the lead on; since 21 November 2026 all six joints have one in the yaml. See
  "The feedforward stage".
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

What the step tests of the script do not see (20 November 2026): joint 2 swings at its own resonance, 6 to 7 Hz,
while it moves, most going down, and how much depends on P and on the belt tension, not on D. The knee of the step
tests put it at P 2500; sweeps with raw samples at several gains showed P 2000 halves the swing, and a tighter belt
raises the resonance and lowers the swing further. The numbers are in the revision at the top of
`Handover Gravity and Spring Compensation.md` and in the yaml comment of servo 2. For a belt joint: check the belt
tension first, then judge P by a slow sweep with raw samples as well as by the steps.

The swing stage of the script does that sweep since 20 November: `--swing` after the step tests, `--swing-only` on
its own, for example
`python -m src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains --joint 2 --swing-only`. At the
first pose it sweeps the joint 10 degrees either side at 10 degrees/s (`--swing-degrees`, `--swing-speed`), twice
there and back, once per gain setting: the yaml gains, 0.8 and 0.6 times the yaml P (`--swing-p` for others,
`--swing-d` for other D gains at the yaml P), and the yaml gains again, whose difference from the first run is the
spread of a repeat. Every sample goes to `<prefix>_swing.csv`. Per setting and direction of the angle it prints the
strongest component of the lag between 3 and 15 Hz, its size in positions (the swing) and its frequency, and it
names the largest P tried whose swing stays within `--swing-limit` (1 position) both ways; it does not write the
yaml. The yaml gains are put back on every exit. Checked against the raw recording of joint 2 of that day, the
stage's analysis gives the numbers of the experiment (a few percent higher, it reads the peak between the lines of
the spectrum). Joint 1, 20 November (run `tune_joint1_20261120_2214`): 0.33 to 0.46 positions at P 2000, 1600 and
1200 alike, the same both ways and without a frequency of its own, so the base does not swing and P 2000 stays.

Two more reasons to settle P first: the hysteresis band (2021: about 20 positions on servo 3, 1.8 degrees) is the
part no model removes, and only P shrinks it; and where a pose comes to rest inside that band depends on the
direction it was approached from, which an overshoot flips.

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
  also changes the side of the friction band the joint comes to rest on.

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

So the gains and the profile are RAM registers at their firmware defaults until `initialize_servos` writes
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

One joint per run. The other joints hold their position at their yaml gains throughout. P and D are tuned
together (since 25 October 2026): damping adds phase margin, so it raises the P where a loop starts to overshoot,
and a P sweep at one fixed D only measures the knee at that D. Joint 2 showed it on 25 October: at the `elbow_out`
pose the knee was 2000 with D = 0 and 2600 with D = 500. So P is swept at every D of a grid, D by D: 0, then
250 and steps of 1.6 (400, 640, 1020, 1640, ...; `--d-start`, `--d-factor`), with the yaml D added, every pose at a
D before the next D. The steps are a factor apart rather than fractions of P because near critical damping the D a
loop needs grows with the square root of its P (Spong, Hutchinson and Vidyasagar, *Robot Modeling and Control*,
independent joint control: P = ω²J/K, D = (2ζωJ - B)/K). The grid stops at `--d-max`, 2000 by default (so 0, 250,
400, 640, 1020, 1640; the largest D in `servo_config.yaml` is 550, the pwm kick per encoder flip grows with D, and
joint 2 gained nothing from D 250 to 2620); `--d-max` raises it up to the register's 16383. Too much D shows in the
trials themselves, as pwm noise (D multiplies the encoder's one-count flicker; reported as a warning, and a D that
adds it is held back when the candidates are compared), a slow settle (overdamped), or, on the belt joints 1 and 2, where the encoder sits on the joint shaft
behind the belt, as oscillation (the belt delays what the encoder sees, and at its resonance D can feed the
oscillation instead of damping it). The grid stops when 2 D gains in a row (`--d-patience`) are no progress. A D gain
is progress when it raises the P of the joint above the best before it, or, at the same P of the joint, when the
overshoot of the trial at the knee fell clearly. The knee is that of the limiting pose (the pose whose recommended P
is the P of the joint, the one with the larger overshoot when several tie): its failing trial of the P sweep at that
D, when it is flagged for overshoot (a knee from ringing, settling or ripple alone says nothing about the overshoot).
The overshoot has fallen clearly when it is at least the larger of 1 position (`KNEE_OVERSHOOT_DROP`) and 10 %
(`KNEE_OVERSHOOT_DROP_FRACTION`) of the lowest one recorded below it for the same pose and knee P since the P of the
joint was last raised (the lowest, not the previous D gain: 11.7, 12.5, 11.0 is no progress); a knee at another pose
or P is recorded and only compared from then on. A D that lowers the P of the joint, or has none, is no progress
whatever its overshoot. The first grid run only stopped on a lower P: joint 2 had the same P of the joint from D 250
to 2620, so it walked on until servo 3 (holding the upper arm) passed its temperature warning after 13 minutes
(`tune_joint2_20261025_1509`). The overshoot rule came on 18 November 2026: P moves in steps of 1.3, so D can help
steadily without the knee moving a whole step. Joint 4 had, at the pose that limits it, a knee at P 1010 at D 250,
400 and 640, with an overshoot of the failing trial of 11.7, 10.0 and 8.7 positions (the limit is 6): the grid
stopped after D 640 although the overshoot fell steadily, and never tried D 1020, at which the run the day before had
moved the knee up. `--d-values` sweeps a
given list instead, `--no-d` the yaml D only (the method before 25 October).

Per pose and D:

1. Move there with the planner at the yaml gains, and record a baseline at rest for 5 s: the holding pwm, the
   ripple of the position and the pwm, the sag (`position trajectory - position`).
2. Trials at P times 1.3^k up to twice the yaml P (`--factor`, `--max-factor`), starting at the yaml P at the
   first D and at the highest P that passed at the previous D after that, or downwards from the start when that
   one fails for too much gain. A start that only fails `settle` (or `stuck`) is too slow, not too stiff, and the
   sweep doesn't go down from it: at the low pose joint 2 failed `settle` at P 1000 and D 0 on 25 October, and
   P 770, 590 and 460 all oscillated 8 to 10 positions in the play of the belt and gears, which the small gravity
   load there doesn't take up. A trial:
   - sets the goal pwm clamp of the servo: 3 times the holding pwm, at least half of 885, never above 885,
     written and read back before the gains change, so the output is bounded from that moment. Joints 1, 4 and 6
     (the base yaw and the two wrist rolls) carry no static load and run at the full 885 by default since 23
     October: their clamp sat at the floor of 442 and starved the steps, and a run of joint 6 needed
     `--pwm-clamp 885` to measure anything;
   - writes the gains to this servo only and reads them back (sync writes aren't acknowledged), re-sends the goal;
   - records 3 s of rest: ripple, pwm noise, sag. The holding pwm of this rest may exceed the baseline's, because
     the servo sits on the other side of the friction band; the clamp is widened then;
   - sizes the step: `(0.8 * clamp - holding pwm) * 128 / P` positions, between 8 and a cap of 60 (15 on joints 1 and
     2, `--max-step`, see below), so the pwm stays below the clamp if the servo really outputs P / 128 per position
     (the measured slope goes to the csv). At the full clamp of joints 4 and 6 that is 60 positions up to a P of
     about 1500 and 34 at 2600, twice what the floor of 442 allowed;
   - switches the profile of the servo off (velocity and acceleration 0, a step instruction) and steps the goal:
     up, back, down, back, twice; each step is held 0.8 s. The steps are written with `step_servo_to_pos`, which
     writes the profile registers 0, 0 and the goal in one packet, also for a servo with a `move_time` (whose registers
     mean the time of the move and the acceleration time in ms: its yaml registers are `(move_time, 0)`, and that is
     what the restore writes and checks);
   - restores the profile and the goal pwm and streams a planned 8 degree move of the joint away and back with the
     production profile, sampling the tracking against the position trajectory of the servo;
   - restores the yaml gains and reads everything back.
3. The first trial that violates a criterion is the knee; 0.65 times the knee is the recommended P of the pose.
4. The smallest recommendation of the poses at a D is the P of the joint at that D. The D with the highest P of
   the joint wins: the most margin to the knee. One step of the sweep (a factor 1.3) is its resolution and counts.
   D gains that share the highest P of the joint have their knees in the same step and can't be told apart by
   margin, typically an unloaded joint that passes up to the top of the sweep at every D; they are compared at that
   P at every pose. A D counts there when it passed and its pwm noise stays below 1.5 times that of the smallest D
   compared (at least 5 units). The reference is the smallest D, or the smallest that passed when it failed; a
   larger D replaces it only when it lowers the overshoot or the mean resting error after the steps
   (`mean_abs_error`, the mean of |final - goal|) by at least a position, or removes a ring, without being a
   position worse in the other; of those the one with the least resting error wins, then the least overshoot,
   rings, and the smaller D. Per joint a D is judged on its worst pose. Until 25 October 2026 P was swept at the
   yaml D only and D = 0, P/8, P/4, P/2 and the yaml D were then tried at the resulting P, so the D in the yaml
   decided how far P got, and a run that stopped before the end of the P phase tried no D at all. The resting error is what tells the D values apart on a joint that
   doesn't overshoot: joint 6 at P 2600 on 23 October (`tune_joint6_20261023_1822`) passed with D 0, 320, 650, 1100
   and 1300, all with an overshoot of a position or less, and the old rule picked 1100 on a 0.3 position
   difference. What really differed was where the steps came to rest: D 0 at +4 +5 +3 +5 after the steps up and
   -6 -3 -6 -3 after the steps down, D 1100 at +0 +2 +0 +2 and -3 +0 -2 +0 at the same pose. Without damping the
   joint runs into the far side of its friction band, with damping it arrives at the setpoint. On a loaded joint
   the sag dominates the resting error (the band sits around it), so there the choice falls back to the overshoot.

Everything is measured against the final value of a step, not the goal: with I = 0 the servo settles at the goal
minus the sag, so "settled within 2 positions of the goal" never happens on a loaded joint.

The criteria of a trial (`Criteria` in `gain_tuning.py`):

| Flag | Rule |
|---|---|
| overshoot | beyond the final value by more than 10 % of the step and more than 2 positions |
| ringing | more than one crossing of the final value after the first |
| settle | not within max(2 positions, 5 % of the step) of the final value within 0.5 s |
| ripple | position peak to peak at rest above max(2, twice the baseline) |
| stuck | no step moved at least half its size, the trial says nothing |

The pwm noise is reported as a warning (`pwm_noise`), not as a flag: the pwm std at rest that the position doesn't
explain (the residual after fitting the pwm against the position error, P / 128 per count), divided by P / yaml P,
above max(5 units, 1.5 times the baseline's residual). A trial with only this warning passes and is not the knee; the
warning shows in the console line, the summary and the `warnings` column of the trials csv, and a D that adds it is
still held back when the candidates are compared. A joint resting on an encoder edge flickers between two counts and
its pwm steps with it; the raw std of that is half a step and flagged every stable joint at high P (joint 6,
23 October). On 13 November 2026 (joint 3) the residual equalled the raw std in every trial, because the pwm and
position readings of one packet don't line up at the time scale of the flicker, and it followed how long the joint
sat on the neighbouring count, which depends on where the sag parks it relative to an encoder edge: the baseline at
unchanged gains ranged from 1.3 to 5.3 over eight visits against the floor of 5.0, and one sweep got its knee at the
yaml P from a reading of 5.23. Oscillation at rest is caught by the ripple flag and the live oscillation abort,
instability by overshoot, ringing and settle.

The division by P / yaml P matters: a dither of one position gives a pwm of P / 128 units, so a fixed dither would
exceed the noise level at 1.69 times the yaml P and warn on every trial from there.

The live abort rules (`AbortLimits`), checked every 5 samples; a violation restores the yaml gains, the profile
and the goal pwm at once, and the run then waits 1.5 s (`ABORT_SETTLE`) for the arm to move back to the pose:

| Reason | Rule | Counts as |
|---|---|---|
| divergence | after 0.5 s of grace, error against the trajectory above max(3 steps, 40 positions, 2.5 times the sag expected at these gains); during a streamed move above max(120, 2.5 times the sag) | the knee |
| oscillation | after the grace, the position goes both up and down by more than max(6, 3 times the baseline) within the last 1 s (the window never reaches back into the 0.5 s of grace); a creep in one direction is not oscillation | the knee |
| pwm at the clamp | the pwm within 10 % of the clamp for 0.1 s | the knee |
| temperature | the `temperature_limit` of `servo_config.yaml`, the stop limit of the robot (`--max-temperature` overrides it; a fixed 50 C until 23 October, which ignored the yaml); the `temperature_warning` check between trials stops the run earlier, and a servo is not tried within 3 C of the abort limit | ends the run |
| voltage | 1.5 V below the voltage at the start of the trial (0.7 until 23 October, which the dips of the supply reached) | ends the run |
| position jump | more than 50 positions between two samples (encoder glitch, slipped belt) | ends the run |

The oscillation rule used to be the position's peak to peak over the last 0.2 s. On joint 1 on 17 November 2026
(runs tune_joint1_20261117_2150 and _2156) a step of 60 positions with the arm stretched out left the joint creeping
towards the goal in one direction 0.5 s later (8, 4 and 1 positions from it at 300, 350 and 400 ms, which is 7
positions in 0.2 s), so the trial was aborted for 'oscillation' although nothing swung back and forth. The abort
restored the gains and the next trial started while the arm was still settling from it, and was aborted in its
rest phase for the same reason. Now the rule needs the position to go both ways by more than the threshold, over
a longer window (1 s, so slow swings are seen as well as fast ones), the run waits 1.5 s after every abort, and
the base and the shoulder step by at most 15 positions, since a 60 position step with the profile off shakes the
whole structure.

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

The workspace limits keep the wrist centre within 41.8 cm of the shoulder (since 25 October 2026, a sphere of 57,
later 60 cm around the base before that), which keeps the elbow bent at least about 11 degrees. The named poses
keep 8 degrees of margin on every joint for the test moves, in degrees of joints 1-6; the closest to the limit is
`reach` with joint 3 moved 8 degrees towards straight, the wrist 41.5 cm from the shoulder:

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
the joint moved 8 degrees either way) is skipped with the reason. There is no height limit any more: tall poses
are allowed as long as the elbow stays bent, only the arm stretched straight up (70 cm tall) is refused, for its
straight elbow; `Workspace limits.md` has why the sphere went.

## Running it

```
python -m src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains --joint 6
```

With the arm powered, its torque off, nothing in its way, foam under the elbow and the wrist for the first runs, and a
hand near the power switch. It prints the plan and asks before enabling the servos (`--yes` skips the question).
Options: `--pose`, `--factor` 1.3, `--max-factor` 2.0, `--no-d`, `--d-start` 250, `--d-factor` 1.6, `--d-max`
2000 (up to 16383), `--d-values` (e.g. `0,250,500`), `--d-patience` 2, `--repeats` 2,
`--hold` 0.8 s, `--rest` 3 s, `--baseline-rest` 5 s, `--move-degrees` 8 (0 disables the streamed move), `--speed`
0.5 rad/s, `--pwm-clamp` (overrides the clamp on every joint, refused below 1.5 times the holding pwm; by default
3 times the holding pwm, at least 442, and the full 885 on joints 1, 4 and 6, which carry no static load),
`--max-step` (the largest step of the trials in positions, at least 8; default 15 for joints 1 and 2, which swing
the whole arm, and 60 for the others; a smaller step is a gentler but coarser test, the overshoot criterion only
counts above 2 positions),
`--max-temperature` (default the yaml `temperature_limit`), `--floor`, `--output-dir`, `--plot`, and the options of
the feedforward stage (see there): `--ff`, `--ff-only`, `--ff-degrees`, `--ff-speeds`, `--ff-lead`,
`--ff-max-positions`.

Output in `--output-dir`: `tune_joint<N>_<time>_samples.csv` (every sample: pose, trial, p, d, segment, repeat,
step index, direction, time, tick, goal, position, position trajectory, velocity, velocity trajectory, pwm, current,
moving status, voltage, temperature) and `..._trials.csv` (per trial: gains, clamp, step, rest ripple and pwm,
sag, max overshoot, rings, the mean resting error after the steps (`mean_abs_error`), settle and rise times, the
steady state error per direction, the measured pwm per
position, the tracking of the move, flags, warnings, abort reason, temperatures, voltage). The console shows every
trial as it ends and a summary per pose with the yaml line to copy:

```
    pid: {p: 1650, i: 0, d: 0}
```

Ctrl+C stops after the current read and restores the servo; a second Ctrl+C kills the script. After a kill the trial
gains and profile stay in the servo until the robot is started again (any script that calls `get_robot()` rewrites
the yaml values), and a goal pwm clamp stays until the servos are rebooted or power cycled; the script also resets a
leftover clamp when it starts. Nothing touches the eeprom.

The run ends with the arm lowered at 0.3 rad/s to `record_static_load.REST_POSE` (where it lies on the table with the
torque off, straight ahead and a bit over 1 cm higher) and the torque switched off there, before the plot. If the servo under test
doesn't have its yaml gains again, a servo has a hardware error, the planner refuses the move or Ctrl+C comes during
it, the arm is not lowered and the prompt to support the arm and press enter is shown instead.

First run, on joint 6 with the arm supported: watch that the baseline shows the sag you expect, that the clamp and
the step sizes are sensible (the console prints them), that a trial ends in the restore (the gains print back), and
try one Ctrl+C. Then the other joints. Put the recommended gains in `servo_config.yaml` and record the static load
again.

## The feedforward stage

Since 21 November 2026 the stage measures and verifies the velocity lead of the PC (the `velocity_feedforward` block
of a servo in `servo_config.yaml`, `velocity_feedforward.py`). Until then it set the servo's own feedforward gain FF1,
which turned out unusable: the servo applies it as a pulse per position of its trajectory (`Handover Velocity Lead on
the PC.md` has the measurements); that gain is gone from the code and the yaml since. `--ff` runs the stage after the
P and D tuning, `--ff-only` runs it alone. It needs the time-based profile (a `move_time` in the yaml); otherwise it
says why and skips the stage.

What the lead does. The position controller outputs P / 128 pwm units per position of error. A joint that moves
needs drive that grows with its speed, mostly the motor's back voltage, and that drive can only come from P times the
error, so the joint trails its trajectory by a lag that grows with its speed. The lead shifts every goal by that lag:
`lead in positions = pwm_per_degree_per_second * degrees/s * 128 / P`. The coefficient is the drive the motor needs per
degree/s of the joint, in pwm units, so it does not depend on P: joint 2 gave 14.29 at P 2500 on 18 November and
13.97 at P 2000 on 21 November. What does not depend on the speed stays: the friction (a lag of 1 to 5 positions in
the direction of the move) and the sag under a static load, which the averaging over both directions takes out of the
measurement. A delay that is the same on every joint (the servo's interpolation between the goals) is not cancelled
either: it makes the arm a little late without bending the path.

The method, at the first accepted pose only:

1. Before anything moves, every sweep is built and checked with the planner's `check_move`, and every sweep must have
   cruise parts in both directions. A refused sweep prints why and skips the stage. Joints 1 and 2 (behind the 3:1
   belts) sweep 10 degrees either way at 5, 10 and 15 degrees/s by default and are refused above 15; the others
   sweep 15 degrees at 10, 20 and 30.
2. A sweep at one speed moves only the joint under test: from the pose to +A, to -A, to +A, to -A and back, every leg
   rest to rest with the speed ramping up over 0.2 s, cruising and ramping down. The angles go out with `move_servos`
   every 1 / `steps_per_second` s against the clock, with the velocity of every step (`movement.step_velocities`),
   and the servo is read as often as fits between two writes.
3. A sample counts as cruise when the commanded speed has been at its full value for at least 0.15 s and still is.
   Per speed, over the cruise samples, separately for the position going up and going down and then averaged over
   the two: the error `(position trajectory - position) * direction`, the pwm times the direction, the lead that was
   added, and the lag behind the intended trajectory, which is the error less the lead (the servo's trajectory
   follows the goals that were sent, lead included).
4. Round 1, without a lead (the servo's own `velocity_feedforward` is set aside): one sweep per speed, and two
   straight lines against the speed in degrees/s. From the lag: `error = friction + slope * speed`, coefficient
   `slope * P / 128`. From the measured pwm: `pwm = friction + k * speed`, the coefficient directly. Their ratio
   is 1.0 when the factor 128 and the reasoning hold (0.97 to 1.03 on joints 1 to 5; 0.51 on joint 6, see below).
5. Round 2, with a lead: the coefficient of `--ff-lead` when given, else that of the yaml, else the one fitted from
   the lag in round 1. When the lag behind the intended trajectory still changes by more than 1 position between the
   slowest and the fastest speed, the coefficient is corrected once by the leftover slope and measured again. A lead
   that reached its clamp is not corrected: the stage says to raise `--ff-max-positions`.
6. The servo has its yaml gains and its own `velocity_feedforward` again afterwards, whatever ended the stage, and the
   arm is lowered at the end.

Options: `--ff`, `--ff-only`, `--ff-degrees`, `--ff-speeds` (three or more), `--ff-lead` (pwm per degree/s),
`--ff-max-positions` (the clamp of the lead when the yaml has no `velocity_feedforward`, default 30).

Output: a table per round (speed, samples, pwm, error, lead, lag behind the intended trajectory, the std of the error
and of the pwm), the two fits and their ratio, the friction part, whether the lag is flat with the lead, whether the
lead made the joint rougher, and the `velocity_feedforward` block for the yaml. The samples go to
`tune_joint<N>_<time>_ff.csv` with the columns p, lead and lead_gain since 21 November.
`calibration/fit_velocity_lead.py` reads such files without the robot and prints the same numbers; it refuses the
recordings of 18 November, which lack those columns.

Results of 21 November 2026 (all in the yaml; lag in positions, without and with the lead):

| Joint | Coefficient | Speeds (degrees/s) | Lag without | Lag with |
|---|---|---|---|---|
| 1 | 10.54 | 5, 10, 15 | 4.4, 7.8, 11.2 | 1.5, 1.0, 1.4 |
| 2 | 13.97 | 5, 10, 15 | 6.6, 10.9, 15.6 | 2.6, 1.9, 2.6 |
| 3 | 3.07 | 10, 20, 30 | 1.9, 2.6, 3.5 | 0.9, 0.6, 0.5 |
| 4 | 1.45 | 10, 20, 30 | 6.8, 9.3, 11.5 | 4.4, 4.3, 4.5 |
| 5 | 2.50 | 10, 20, 30 | 5.8, 10.4, 14.2 | 1.9, 2.5, 1.2 |
| 6 | 4.45 | 10, 20, 30 | 6.2, 11.5, 16.1 | 0.6, 1.4, 1.0 |

No joint needed a correction, and none was rougher by the stage's test. Joint 2 was checked with the swing stage
afterwards: 0.65 to 0.74 positions with the angle rising and 0.82 to 0.86 falling at P 2000, under the limit of 1; on
20 November without a lead the falling figure was about 0.7, and the std of the error in the lead round was 0.1
position higher at 10 and 15 degrees/s, so the lead costs about a tenth of a position of swing there.

The friction round, since 23 November 2026. What is left with the lead does not depend on the speed: friction, the
drive the joint needs to keep sliding. The block has a third, optional value for it, `pwm_for_friction`, and the
drive becomes `pwm_per_degree_per_second * speed + pwm_for_friction * direction`, the direction being the commanded
speed divided by `FRICTION_FULL_SPEED` (1 degree/s) and clipped to -1 and 1, so the shift is whole from 1 degree/s
and zero at rest. One value per joint, the same in both directions. After the lead rounds the stage sweeps once more
with the lead and the friction drive: `--ff-friction` when given, else the yaml's, else the offset of the lag fit of
round 1 times P / 128. When the mean lag is then more than 0.5 position from zero it corrects once (the value plus
that mean times P / 128) and measures again. The csv has a `lead_friction` column, and `fit_velocity_lead.py` prints
the value with the block.

Results of 23 November 2026 (all in the yaml; lag in positions without feedforward, with the lead, with both):

| Joint | pwm_for_friction | Speeds (degrees/s) | Nothing | Lead | Lead and friction |
|---|---|---|---|---|---|
| 1 | 17.1 | 5, 10, 15 | 4.5, 7.8, 11.2 | 1.5, 1.0, 1.4 | 0.5, 0.0, 0.4 |
| 2 | 31.6 | 5, 10, 15 | 6.6, 11.0, 15.7 | 2.6, 2.0, 2.7 | 0.6, -0.1, 0.7 |
| 3 | 48.4 | 10, 20, 30 | 2.1, 2.7, 3.5 | 1.1, 0.8, 0.6 | 0.0, -0.2, -0.4 |
| 4 | 24.8 | 10, 20, 30 | 7.0, 9.3, 11.5 | 4.4, 4.2, 4.4 | 0.1, 0.2, 0.4 |
| 5 | 13.2 | 10, 20, 30 | 6.2, 10.8, 14.5 | 2.3, 2.8, 1.5 | 0.3, -0.2, -0.5 |
| 6 | none | 10, 20, 30 | 5.1, 11.5, 16.1 | -0.8, 0.4, -0.9 | |

- Joint 4 needed the correction (29 from the fit left it 0.7 position ahead, 24.8 is the corrected value); on the
  others the fitted value held. No joint was rougher by the stage's test; joint 4 at 10 degrees/s is the nearest
  to it (error std 1.1 without feedforward, 1.8 with both terms).
- Joint 6 got no friction term: the fit of that day put the lag at no speed at zero, and its lead coefficient went
  from 4.45 to 4.97 instead. At 10 degrees/s the joint moves in a stick-slip way (error std 2 positions, 0.7 at the
  higher speeds) and its mean lag there was 6.2 on 21 November and 5.1 on 23 November, so on this joint the split
  between the part that grows with the speed and the constant part is uncertain by about a position.
- Joint 2 with both terms in the swing stage: 0.47 to 0.56 positions with the angle rising and 0.60 to 0.68
  falling at P 2000, no more than with the lead alone two days before (0.65 to 0.74 and 0.82 to 0.86).
- At rest the term adds nothing, but the joints stop closer to their goals, see the check at rest of 23 November
  in `Handover Gravity and Spring Compensation.md`.
- Not measured: below 5 degrees/s (10 on joints 3 to 6), where friction is larger and the term is only partly on
  under 1 degree/s; a second pose or load per joint.

Where the stage can mislead: it measures the lag against the Position Trajectory of the servo, so it assumes the
trajectory is what the loop follows when goals stream (see "Assumptions the plan still rests on"). On joint 6 (the
XC330) the measured pwm is half of P / 128 times the error at every speed, on 18 and on 21 November; the lead uses
the coefficient from the lag, which cancels the lag by construction, but the "pwm units" of that joint are twice the
real ones and the reason is not known. The stage measures at one pose; that the coefficient does not depend on the
pose is assumed, not checked. The lead is rounded to whole positions, which leaves up to half a position.

## The code

- `src/robot_controllers/dynamixel_robot/calibration/gain_tuning.py`: the arithmetic, pure functions on lists of
  samples: `pwm_clamp`, `step_size`, `rest_metrics`, `step_metrics`, `tracking_metrics`, `pwm_per_position`,
  `evaluate`, `abort_reason`, `environmental_abort`, the sweep values (`p_values`, `p_values_down`, `d_ladder`) and
  `choose_p`, `choose_d`.
- `src/robot_controllers/dynamixel_robot/calibration/tune_servo_gains.py`: `TuningSession` (snapshot, restore,
  the planned moves through `record_static_load.move_to`, sampling with the abort rules, the trial, `run_sweep` and
  `sweep_pose` for the P sweep at a D, `compare_d` for the D gains that share the best P, `run` for the grid),
  the grid functions `d_grid`, `swept_d_values`, `joint_p_at`, `d_candidates`, `knee_trial`, `joint_knee_at`, `d_progress`, `grid_is_done`, `compared_d`, and
  `recommend`, `print_summary`, `plot_results`, `main`. `PoseResult.sweeps` holds a `Sweep` (knee, recommended P,
  highest P that passed) per D.
- `src/robot_controllers/dynamixel_robot/calibration/plot_tuning.py`: the figures from the csv files: the knee and
  the recommended P against D (`plot_grid`), the metrics of the P sweep per D (`plot_knee`), the comparison of the
  D gains (`plot_d`), the step responses per D (`plot_steps`).
- The feedforward stage: in `gain_tuning.py` `ff_sweep` (the steps of a sweep: time, joint offset in degrees,
  direction, cruise flag, leg), `fit_line` and `rougher_speeds`; in `fit_velocity_lead.py` the statistics per speed
  and the two fits (`speed_statistics`, `fit_group`, `analyse_group`), shared with the offline script; in
  `tune_servo_gains.py` `feedforward_refusal` (the checks before anything moves), `TuningSession.run_feedforward`,
  `run_lead_rounds`, `feedforward_sweep` and `measure_feedforward`, `print_feedforward`. The lead itself is
  `VelocityFeedforward` (`velocity_feedforward.py`), added in `Servo.set_target_position_from_angle`.
- `ServoHandler` (`servo_handler.py`): `read_state` (one 27 byte packet per bus, addresses 120-146, a `ServoState`
  per servo), `read_pid`, `read_profile`, `read_goal_pwm`, `set_profile(servo_id, velocity, acceleration)`,
  `set_goal_pwm`; the gain range check in `__write_pid`; `read_temperature_and_voltage` in one packet per bus;
  `__sync_read_fields` picks 1, 2 or 4 byte fields out of a block with the sdk's `getData`, which the old
  `__sync_read` couldn't (it passed the block length to `getData`, which returns 0 for anything but 1, 2 or 4).
- `dynamixel_x_config.py`: the addresses of the state block, the pwm limit and goal pwm,
  `MAX_GAIN` 16383, `MAX_PWM` 885, the moving status bits. `servo_config.py` bounds the yaml gains to `MAX_GAIN`.
- Tests: `tests/servo_controller_tests/gain_tuning_tests.py` (the metrics on synthetic traces, every abort rule,
  the choices, and the whole session on a simulated second order servo with coulomb friction, a damping that drops
  above a P threshold so it rings and, with a steep enough drop, oscillates with a bounded amplitude, and a friction
  band that narrows with D; the knee, the abort, the sweep downwards, a D that raises the knee winning, a noisy D
  ending the grid, D gains that share the best P compared and chosen for the band they rest in, the full clamp of an unloaded joint, an environmental stop, a warm servo, a leftover clamp, a
  flaky read, a refused pose; the feedforward stage on a simulated servo whose lag grows with the speed and whose
  trajectory ramps over the move time: the gain that makes the lag flat, the single correction, the clip, an abort and
  a stop during a sweep, a servo without `move_time`, the options of `main`) and the block read tests at the end of
  `servo_handler_tests.py`.
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
- The abort thresholds: 40 positions or 2.5 times the sag of divergence, 6 positions of oscillation (both up and down within 1 s), 90 % of the
  clamp for 0.1 s, the yaml temperature limit, 1.5 V, 50 positions of jump. They are guesses; the joint 6 runs of 21-23 October did
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
  where the step overshoots (the script also takes a D for the resting error, see the method), feedforward for the
  tracking lag.
- Dynamixel specifically: no peer reviewed tuning guidance. The ROBOTIS e-manual pages of the XM430-W350,
  XM540-W270, XC430-W240/W150 and XL330-M288 (the position controller diagram: the error feeds P, I and D in
  parallel, the feedforward gains act on the trajectory velocity and acceleration, the output is limited by the goal
  pwm) and ROBOTIS's general advice: P for speed, D for damping, a small I last. The LeRobot Koch arm runs its
  elbow at P 1500, I 0, D 600, close to this yaml. The firmware's D acts on the error, so with the profile off it
  adds a derivative kick on a step and with the profile on it is a velocity tracking term; the velocity feedforward
  (Feedforward 1st Gain) is outside the loop and can't destabilise it, which would make it the parameter for the
  tracking lag, but on these servos it pulses the drive and is not used (the register stays at its default of 0,
  the code does not know it): the velocity lead of the PC does that job, see the feedforward stage above.

## History of the values in servo_config.yaml

Moved here from the comments of `servo_config.yaml` on 21 November 2026, word for word, when that file was
trimmed to the values and the notes that matter for using them. Newer results are in the sections above and
in the handovers; `git log -p` of the yaml has every earlier state.

### The header as it was

```
Configuration of the dynamixel servos of the robot, loaded by servo_config.py

Every servo has:
  name                        what the servo does
  model                       the dynamixel model, for reference
  bus                         the connector of the usb adapter, servos on different connectors can't hear each
                              other's responses, so they are read separately
  mode                        position, extended_position or current_based_position
  positions: [first, last]    the servo positions (4096 per turn) of the ends of the range, last can be smaller than
                              first when the servo is mounted the other way around
  angles: [first, last]       the joint angles in degrees at those positions, or for the gripper:
  states: [first, last]       the gripper states at those positions
  offset                      added to the position, to calibrate the zero point (optional, default 0)
  goal_current                the maximum current in mA in current_based_position mode (optional)
  profile                     velocity and acceleration of the movement profile of the servo. With the optional
                              move_time (ms) the servo uses the time-based profile: every streamed goal step
                              is interpolated over that time, and velocity and acceleration then only limit a
                              goal that is far away. The drive mode of the servo is set to match when the robot
                              starts, which needs the torque of that servo off
  pid                         the gains p, i and d of the position controller of the servo
  compensation                the static load compensation of the joint (optional, joints 2-5): the holding drive
                              in pwm units for gravity at the commanded pose. It is converted to positions with
                              the P gain of the servo (positions = drive * 128 / p) and added to every target.
                              Fitted by calibration/fit_static_load.py from the data of
                              calibration/record_static_load.py, see static_compensation.py and Handover Gravity
                              and Spring Compensation.md:
                                max_positions: 60                           clip of the compensation, in positions
                                terms: {constant: 1.0, cos_q2: 12.0, ...}   coefficient per term in pwm units
                              the terms: joint 2 constant, cos_q2, sin_q2, sin_phi, cos_phi, wrist_reach;
                              joint 3 constant, sin_phi, cos_phi, wrist_reach; joint 4 constant, wrist_roll;
                              joint 5 constant, wrist_pitch
  velocity_feedforward        the velocity lead of the joint (optional): the drive in pwm units the motor needs
                              per degree/s of the joint to turn at the commanded speed. It is converted to
                              positions with the P gain (positions = this * degrees/s * 128 / p) and added
                              to every target that is sent with a velocity (planned moves, the gamepad and the
                              sweeps of the calibration scripts), so a moving joint does not trail its
                              trajectory by its speed; zero at rest. Fitted by calibration/fit_velocity_lead.py
                              from the recordings of calibration/tune_servo_gains.py --ff-only, which verifies it,
                              see velocity_feedforward.py and Handover Velocity Lead on the PC.md:
                                max_positions: 30                  clip of the lead, in positions
                                pwm_per_degree_per_second: 14.3    the drive per degree/s of the joint
The robot is printed in PLA, which softens around 55-65C. A servo that is warmer than the warning temperature is
logged, and the robot is stopped while all servos still work when one is warmer than the limit. The servos measure
their temperature inside, the plastic around them is cooler. The servos shut down by themselves at their hardware
temperature limit, 80C (70C for the gripper), and then just go limp.
```

### Servo 1 (base)

`offset: 346`: measured in the jig at 90 degrees, raw 2394

17 November 2026, before the belt was tightened (runs tune_joint1_20261117_2150 and _2239): D 0, 250 and 640 gave
the same result, D makes no difference on this belt joint. With steps of 15 positions the joint rang from P 2800
with the arm folded (pose 90 90 0 0 0 0) and from 2150 with the arm out (pose 90 60 0 0 0 0), and overshot from
1270 at full reach (pose 90 25 65 0 90 0), so P was 980.

After the belt was tightened that evening (run tune_joint1_20261117_2255): no ringing up to 2800 with the arm
folded, ringing from 2800 at the other two poses, and 2150 passed at all three (1.9 positions of overshoot at
full reach, the limit is 2). The friction went up: the drive the joint rests at without moving rose from 10 to
20-27 pwm units, so at P 980 it stopped 3 to 7 positions short. P 2000 depends on the belt tension: with a slack
belt it is above the 1270 at which full reach overshot, so check the belt first when the base rings.

20 November 2026, the swing stage (run tune_joint1_20261120_2214): sweeping at 10 degrees/s the lag swings by
0.33 to 0.46 positions at P 2000, 1600 and 1200 alike, the same both ways and at no frequency of its own. The
base does not swing while it turns as the shoulder did, P 2000 stays

21 November 2026 (run tune_joint1_20261121_1455, sweeps at 5, 10 and 15 degrees/s with the arm folded): without
a lead the joint trails by 4.4, 7.8 and 11.2 positions, with this one by 1.5, 1.0 and 1.4 (what is left is
friction), no rougher. The recording of 18 November gave 10.49, the measured pwm 10.84

### Servo 2 (shoulder)

`offset: -550`: measured in the jig at 0 degrees, raw 474

17 November 2026 (run tune_joint2_20261117_2158): steps overshoot from P 3080 with the arm at reach and with the
elbow out, not up to 4000 with the arm lowered, at D 250, 640 and 1640 alike; P 2000 passed at all three poses

After the belt was tightened that evening (run tune_joint2_20261117_2259, D 250 only): P 3080 passed at all
three poses and 3820 overshot with the elbow out only (2.9 positions, the limit is 2); the friction and the
resting error did not change. 0.65 times 3820 is 2480, rounded up to 2500

20 November 2026: the step tests above did not see that the joint swings while it moves, most going down
(gravity then does the work and there is hardly any motor torque on the belt): at its own resonance, whatever
the speed. Sweeps of 10 degrees/s at the folded pose (calibration/tuning/joint2_ringing_20261120*.csv), the
swing going down in positions at P 2500 / 2000 / 1500, D 250:

```
  belt as on 17 November   6.6 Hz   1.9-2.4 / 1.2  / 0.75
  belt loosened            6.0 Hz   2.7-2.8 / 2.4  / 1.0
  belt tightened further   7.2 Hz   1.2-1.5 / 0.66 / 0.34   (the tension it has now)
```

Going up it is 0.5 to 0.7 at this tension. D 0, 250, 640 and 1640 made no difference at any tension, only
noisier drive at 1640. So the belt tension sets how much P the joint takes: check the tension before the gains
when it swings again. P 2000 makes going down as smooth as going up; the lag while moving is a quarter larger
than at 2500 (9.9 / 12.0 positions at 10 degrees/s)

20 November 2026, from sweeps at 5 and 10 degrees/s at P 2000 and the belt tension of that evening
(calibration/tuning/load_sweep_joint2_20261120_p2000.csv and _p2000_bent.csv, 116 bins: the elbow at 0, 45 and
-30 degrees and the wrist bent 60 degrees either way): the sag was up to 3.6 positions, the same at both speeds,
and this fit leaves 2.4 pwm units (0.15 position). A fit with a constant and sin_q2 as well is no better (2.1)
and gives +21 and -20 for them, which cancel inside the recorded 43 to 118 degrees and not outside it, so they
are left out. The recordings of that afternoon at the earlier belt tension gave 48.5 for cos_q2: record again
after the belt is retensioned

21 November 2026 (run tune_joint2_20261121_2213, sweeps at 5, 10 and 15 degrees/s with the arm folded): without
a lead the joint trails by 6.6, 10.9 and 15.6 positions, with this one by 2.6, 1.9 and 2.6 (what is left is
friction, 2.1 positions), no rougher. The recording of 18 November at P 2500 gave 14.29, the measured pwm 14.21

### Servo 3 (elbow)

17 November 2026, with the heavier wrist: 8 position steps overshoot by more than 2 positions from P 7240,
at D 350, 640 and 1020 alike

19 November 2026, from sweeps at 5 and 10 degrees/s at four poses (calibration/tuning/
load_sweep_joint3_20261119_before.csv, 148 bins): the sag was 3.0 to 3.2 positions with the upper arm
horizontal and 1.5 at 57 degrees up or down, the same at both speeds, and the fit leaves 2 to 3 pwm units
(under 0.1 position). The largest drive is 114 pwm units, 3.1 positions at P 4710, so the clip is far away

21 November 2026 (run tune_joint3_20261121_2216, sweeps at 10, 20 and 30 degrees/s with the elbow out):
without a lead the joint trails by 1.9, 2.6 and 3.5 positions, with this one by 0.9, 0.6 and 0.5, no rougher.
The lead is small here, 3 positions at 30 degrees/s. The recording of 18 November gave 2.60, the pwm 2.98

### Servo 4 (wrist 1)

`offset: 3`: measured in the jig at 0 degrees, raw 2051

17 November 2026, new bearing holder: steps overshoot from P 920 at D 400-550, from 1200 at D 1020.

18 November 2026, with the time-based profile (runs tune_joint4_20261118_1356 and _1418): the same limits, P 1010
passes at D 1020 with the overshoot exactly at its limit and D 1640 rings, so P 780 and D 1020 stay.

21 November 2026 (run tune_joint4_20261121_2217, sweeps at 10, 20 and 30 degrees/s at the reach pose): without
a lead the joint trails by 6.8, 9.3 and 11.5 positions, with this one by 4.4, 4.3 and 4.5, no rougher. What is
left is friction, 4.5 positions at this P. The recording of 18 November gave 1.27, the measured pwm 1.46

### Servo 5 (wrist 2)

since November 2026 the servo sits directly on the wrist joint instead of driving it through a belt, which reversed
its direction (hence the reversed positions). No offset, the horn is mounted at its zero. The offset has not been
checked in the jig since. The pid gains tuned with the belt (P 3300, D 500) were far too high for the direct drive:
on 17 November 2026 (run tune_joint5_20261117_1739) the joint oscillated at P 3300 with the gripper bent up and
still rang at P 1500. A second run from P 900 (tune_joint5_20261117_1744, gripper bent up, the pose with the least
damping) passed at 900 and overshot and rang at 1170 at D 400, 640 and 1020 alike, so P 760 with D 640

21 November 2026 (run tune_joint5_20261121_2219, sweeps at 10, 20 and 30 degrees/s with the elbow out): without
a lead the joint trails by 5.8, 10.4 and 14.2 positions, with this one by 1.9, 2.5 and 1.2, no rougher. The
recording of 18 November gave 2.42, the measured pwm 2.46

### Servo 6 (wrist 3)

an XC330-T288-T on the 12 V bus since 11 November 2026 (rated 6.5-12 V, current limit 910 mA, 1 mA per unit,
1.0 N.m at 12 V), bolted to servo 5. Before it an XL330-M288 on a 5 V step-down converter, which switched off when
the servo braked (its energy pushed the 5 V line up) and later got 11.7 V from a mis-set converter, which broke its
motor driver. Current-based position mode, so it never draws more than the goal current; the goal current and the
profile are those of the XL330 until calibration/wrist_roll_limits.py has measured the XC330. The gains: with the
P of 1200 it came with and D 0 it swung +-115 positions around every target without stopping (11 November); without
D it already oscillates from P 600, with D 400 it rests within 3 positions of the target up to P 800 at 5 mA.
Tuned on the arm on 17 November 2026 with tune_servo_gains (run tune_joint6_20261117_1726): the P at which a step
starts to overshoot rises with D (about 350 at D 250-400, 460 at D 640, 770 at D 1020, 1300 at D 1640, the largest
D tried), so P 840 with D 1640; the old 600 / 400 was above its knee.
Later that day the bearing in front of the servo was changed: the current the joint needs to keep turning went from
24 mA to 34 mA with a first new holder, 16-17 mA without the bearing and 21 mA with the holder that stayed (runs
tune_joint6_20261117_1959, _2006 and _2127). With less friction the knee at D 1640 came down to 1100 in the last
two, so P 720.
A run with more D (tune_joint6_20261117_2142) then put the knee at 2070 with D 2620: P 1220 passed there with one
position of overshoot, no chatter at rest, and moves ending 1 to 3 positions off instead of 5 to 7; the drive is
noisier while moving (pwm std about 38 against 26 at D 1640). So P 1220 with D 2620, above the default 2000 cap of
the D grid of tune_servo_gains.
18 November 2026: position mode like the other arm joints, only the gripper stays current-based. The output of
the position controller is a pwm duty now and no longer a current, so the gains above (P 1220, D 2620) don't
carry over, and nothing limits the torque of the roll any more (the goal current of 300 mA did).
Tuned in position mode that day (run tune_joint6_20261118_1720, from P 400): 60 position steps overshoot by at
most 1 position at every P up to 1490, the largest tried, at D 0, 250 and 400 alike, at both poses; the joint
rests within 1 position at any P, so a higher P has nothing to win. In this mode the motor damps itself, where in
current-based mode it oscillated without D from P 600. So P 1150 without D.
The positions have to be calibrated for the new mounting

21 November 2026 (run tune_joint6_20261121_2220, sweeps at 10, 20 and 30 degrees/s with the elbow out): without
a lead the joint trails by 6.2, 11.5 and 16.1 positions, with this one by 0.6, 1.4 and 1.0, no rougher. On
this servo the measured pwm is half of P / 128 times the error (2.25 against 4.45 here, the same on 18
November), so its "pwm units" are twice the real ones; the value is the one from the lag, which is what the
lead works through. Not looked into why (the XC330 is the only one of its model among the arm joints)

### Servo 7 (gripper)

an XC330-T288-T on the 12 V bus since 11 November 2026, before it an XL330-M288.
NOT CALIBRATED: the positions are still those of the XL330 and the new servo sits far outside them (3999 on 11
November), so don't open or close the gripper until its open and closed positions are measured.
0 is open and 100 is closed. The XC330 makes about 3 times the torque of the XL330 per mA (1.136 against 0.354
N.m/A), so 150 mA squeezes about as hard as the 400 mA of the XL330 did (0.17 against 0.14 N.m); its current
limit is 910 mA. With the XL330: when an object blocked the gripper it squeezed with about 2 mA per position the
target was beyond the object (with p=500), up to the goal current, see GRIPPER_MAX_LEAD of the xbox controller

`pid: {p: 600, i: 0, d: 400}`: the bench gains of servo 6, not tuned on the arm
