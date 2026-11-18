# Handover: velocity lead on the PC instead of the servo's feedforward gain

Written 18 November 2026. Not a priority: it only matters for how closely the arm follows a trajectory while it
moves fast. The end point of a move is not affected.

## The problem

A joint that moves trails the trajectory of its servo, and the faster it moves the further it trails. The servo's
position controller has no other source for the drive a turning motor needs (mostly its back voltage) than P times
the error, so the error has to grow with the speed. Measured on 18 November with
`tune_servo_gains.py --ff-only` (the feedforward stage, sweeps at three speeds, at the yaml gains):

| Joint | P | Lag at no speed (positions) | Lag grows by (ms of travel) | Lag at 30 degrees/s |
|---|---|---|---|---|
| 1 base | 2000 | 1.4 | 59 | 11.5 at 15 degrees/s |
| 2 shoulder | 2500 | 1.2 | 64 | 12.3 at 15 degrees/s |
| 3 elbow | 4710 | 1.5 | 6 | 3.6 |
| 4 wrist 1 | 780 | 5.5 | 18 | 11.8 |
| 5 wrist 2 | 760 | 2.4 | 36 | 14.5 |
| 6 wrist 3 | 1150 | about 0 | 50 (poor fit, see below) | 16.0 |

"Grows by 59 ms" means: the joint is where its trajectory was 59 ms ago, plus the part that does not depend on
the speed (friction). One position is 0.088 degrees. Joints 1 and 2 were measured at 5, 10 and 15 degrees/s with
the arm folded, the others at 10, 20 and 30 degrees/s. The slopes of joints 1 and 2 are large because the motor
turns three times as fast as the joint behind the 3:1 belt; converted to drive they are three times the back
voltage of their motors, as they should be.

Joint 6 does not follow a straight line well (4.6, 11.3 and 16.0 positions at the three speeds, with a rough,
stick-slip like motion at the slowest), so its 50 ms is a rough number.

## Why the servo's own feedforward gain is not the fix

The servos have a velocity feedforward gain (Feedforward 1st Gain, address 90, `ff1` in `servo_config.yaml`). It is
0 on every joint and should stay 0. What was found:

- The servo adds `FF1 / 4` pwm units for one control cycle each time its trajectory advances one position. At the
  speeds of this arm that is less than once per millisecond, so the term is a train of pulses with the right
  average, not a steady drive. The measured drive noise fits that exactly.
- Joints 4 and 5: it works (FF1 585 and 850 made the lag the same at every speed, about 2 and 3.5 positions), but
  the standard deviation of the pwm went from about 10 to 50-100.
- Joint 3: FF1 915 worked but only wins about 2 positions, for a pwm noise of 80-120 instead of 25.
- Joints 1 and 2: the gain that would flatten the lag (about 3700 and 5000) asks for pulses above full drive
  (885 pwm units), which the servo can not give; at 4000 the drive jumps between nothing and full about 170
  times a second and the joints were very noisy. Joint 2's lag still grew with the speed.
- Joint 6 (XC330): gains of 1796 and 3255 gave the same lag to 0.1 position. The XC330 limits how fast its pwm may
  change (PWM Slope, address 62, already at its fastest), which clips the pulses.

## The plan: lead the goal on the PC

Send every goal a fixed time ahead along the planned path: `goal + lead_time * velocity`. The P term of the servo
then supplies the same drive the feedforward would have, but smoothly, with no pulses and no ceiling, and it works
the same on every servo model and gear ratio. The lead time per joint is the "grows by" column above.

Where it goes:

- A new optional key per servo in `servo_config.yaml`, for example `lead_time: 59` (ms), next to `move_time` in
  `profile`; parsed in `servo_config.py`, stored on `Servo` (`servo.py`).
- The lead is added where the target position is made, `Servo.set_target_position_from_angle`, or where the goals
  are written, `ServoHandler.__write_goals` (`servo_handler.py`). The first has the unrounded position, which
  matters (next point).
- The velocity has to come from unrounded positions. The goal is an integer and steps by 1 or 2 positions per
  10 ms at low speed; a lead computed from those steps jumps by whole multiples of `lead_time / 10 ms` positions
  and would shake the joint. Keep the previous target as a float (before `np.rint`) and the time of the previous
  write, and use their difference; or have the planner hand the velocity over (it knows it exactly, the path is a
  minimum jerk curve in `src/utils/movement.py`).
- The lead is zero when the joint stands still, so the end point of a move does not change. When the goals stop,
  the last write still carries the lead of the last step: write the final goal once more without it, or let the
  lead decay over a few steps.
- Clamp the lead (a few tens of positions) and clamp the result to the calibrated range, as the static load
  compensation is (`max_positions`). The distance for `Servo.goal_timing` (the far-goal rule of the time-based
  profile) should be taken from the goal that is really sent.
- The static load compensation also adds positions to the target; the two just add up.

How to measure and check it, with what exists:

- `tune_servo_gains.py --ff-only --joint N` already measures the slope and prints it ("grows by X ms per
  position/ms"). The stage could print the lead time and verify with the lead applied instead of writing FF1; its
  sweep, cruise statistics and line fit (`gain_tuning.py`: `ff_sweep`, `cruise_statistics`, `fit_line`) stay as
  they are.
- The check is the same as for FF1: with the lead, the lag while cruising is the same at all three speeds, and
  the standard deviation of the lag and of the pwm are no larger than without it.
- Joints 1 and 2: keep the sweeps slow and small (`--pose elbow_out --ff-degrees 10 --ff-speeds 5 10 15`).

What it will not do: remove the part of the lag that does not depend on the speed (friction, 1 to 5 positions).
That would take a friction term, a step in the direction of motion, which is a separate question.

## State of the code on 18 November

- All six arm servos use the time-based profile (`move_time: 15`): the servo interpolates between the goals the PC
  streams every 10 ms, and every goal is written with its time (`ServoHandler.__write_goals`,
  `Servo.goal_timing`). The gripper is unchanged.
- `ff1` exists as a yaml key and is written at start-up (`ServoHandler.set_feedforward`); it is absent, so 0, on
  every servo. The feedforward stage of `tune_servo_gains.py` (`--ff`, `--ff-only`) works and recommends an FF1;
  its recommendation is not to be used, for the reasons above.
- The recordings are in `src/robot_controllers/dynamixel_robot/calibration/tuning/`:
  `tune_joint4_20261118_1356_ff.csv` and `_1418_ff.csv`, and the `--ff-only` runs of joints 6, 5, 3, 1 and 2 of
  that evening (`tune_joint<N>_20261118_*_ff.csv`).
- `Tuning the servo gains.md` has the section on the feedforward stage and its formulas.
