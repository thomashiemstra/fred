# Handover: Gravity & Spring Compensation for 6-DoF XM430 Arm

Oct 16, 2026 · @wew

## Summary

The arm sags up to about 1° at rest because the XM430 position loop has no integral term, so holding a load needs a steady position error. The plan is to predict the static holding torque of every joint from a fitted physics model plus a small learned residual, then add the equivalent offset to each goal position.

- **Rigid-body gravity:** each link is a point mass at its centre of mass (COM). Masses are weighed; COM positions are fitted by linear least squares.
- **Springs:** a spring from the base to the lower arm loads joints 2 and 3 together, so it is modelled as one potential that depends on spring length.
- **Residual:** a small MLP (or similar) on q2 and q3 absorbs whatever the physics misses.
- **Application:** `goal = target + tau_static(q) / K`, with K calibrated per joint. Servos stay in normal position mode and the host only updates a slow term.
- **Expected outcome:** sag from about 1° down to roughly 0.1-0.3°. Friction and backlash set the floor.

## System and constraints

A Python host talks to six mostly XM430-W350-R servos through a U2D2, and realistic loop rates are 200-400 Hz at best, so the servos' own 1 kHz loop should keep doing the fast work.

**Settings to change first**

1. Set the FTDI latency timer to 1 ms (Linux: write 1 to `/sys/bus/usb-serial/devices/ttyUSB0/latency_timer`; Windows: port Advanced settings in Device Manager).
2. Use 4 Mbps if the servos and cabling allow it, and avoid USB hubs between the U2D2 and the PC.
3. Use `GroupSyncRead` and `GroupSyncWrite` and read only the registers needed.
4. Run a fixed-period loop with `time.perf_counter()`; on Linux consider `chrt -f 50`.

Compensation is applied as a position offset because loads change slowly, so 50-100 Hz host updates suffice and the servo's own 1 kHz loop does the fast work. Host-side current control would replace that with a 200-400 Hz Python loop; if torque-level control is needed later, move the loop to an OpenRB-150 or Teensy 4.x (not the Uno-based Dynamixel Shield, whose software serial is too slow).

## Why the arm sags

At standstill the servo's holding torque is proportional to its position error (τ ≈ K·e), so any steady load needs a steady error. With the default P gain of 640, K is roughly 15 N·m/rad, so 1° of sag is about 0.25 N·m. Treat that as an order of magnitude: K scales with the P gain and shifts with supply voltage, temperature and friction, so calibrate it per joint (Step 5).

The offset that cancels a known static torque is therefore:

```latex
\text{goal} = q_{target} + \frac{\tau_{static}(q)}{K}
```

Current-based position mode (mode 5) makes K independent of supply voltage and is worth considering if K proves hard to calibrate.

## Strategy: three layers, one static torque function

Spring and gravity loads depend only on pose, so together they form a single function of the joint angles, and friction is removed from the data rather than modelled.

```latex
\tau_{static}(q) = \underbrace{\tau_{grav}(q;\,m,\,c)}_{\text{physics}} + \underbrace{\tau_{spring}(q;\,p)}_{\text{joints 2, 3}} + \underbrace{r_\theta(q_2, q_3, \dots)}_{\text{learned residual}}
```

| Layer | What it captures | Parameters | How it is fitted |
| --- | --- | --- | --- |
| Rigid-body gravity | Weight of every link | One COM (3 numbers) per link; masses are known from weighing | Linear least squares on static torque data |
| Spring | Base-to-lower-arm spring acting on joints 2 and 3 | Anchor points, stiffness, free length (about 8 numbers) | Nonlinear least squares, jointly with the COMs of joints 2-3 |
| Residual | Model errors, spring nonlinearity, anything unmodelled | Small MLP or Gaussian process on q2, q3 (maybe the wrist pitch) | Supervised fit on what the first two layers leave over |

Friction is not part of this static function. It is averaged out of the data in Step 1 and handled separately in the motion section below.

## Step 1: Collect static torque data

The dataset is holding current versus pose, with friction cancelled by approaching every pose from both directions. Moving the robot is cheap, so collect generously.

1. **Warm up first.** Run the arm for a few minutes so friction and winding resistance have settled; temperature drift is real.
2. **Pick random poses** across the whole workspace, including awkward ones near joint limits, rather than a coarse grid. Vary q2 and q3 independently so spring and gravity effects can be told apart.
3. **Approach each pose from two directions** (or do slow constant-velocity sweeps up and down). Wait until the arm has settled, then log Present Position and Present Current for all joints.
4. **Average the two directions.** The mean is the static torque with friction removed; half the difference is Coulomb friction.
5. **Log with the gains you will actually run**, and record the supply voltage. If steady-state position error is used as the target instead of current, the data is tied to those gains.
6. **Hold out a share of poses** (about 20%) for validation in Step 6.

**Which output to learn.** Holding current is independent of the P gain, so retuning later does not invalidate the data, but it needs the current-to-torque factor and K. Steady-state position error skips both but is tied to the gains and supply voltage. Holding current is the recommended default.

## Step 2: Identify link centres of mass

Gravity torque is linear in the first moments of the links, so the COMs come from an ordinary least-squares solve with no local minima. A rigid body under gravity behaves exactly like a point mass at its COM, and inertia tensors do not matter at standstill.

```latex
\tau_g(q) = Y(q)\,h, \qquad h_k = m_k\,c_k \;\Rightarrow\; c_k = h_k / m_k
```

With the masses known from weighing, only three numbers per link are unknown. Pinocchio builds the regressor Y from a URDF whose kinematics (axes and link offsets) are right; the inertia values in the URDF can be placeholders.

```python
import numpy as np, pinocchio as pin

model = pin.buildModelFromUrdf("arm.urdf")
data = model.createData()
nb = model.nv

def regressor(q):
    Y = pin.computeJointTorqueRegressor(model, data, q, np.zeros(nb), np.zeros(nb))
    # standard 10 params per body: [m, mx, my, mz, Ixx, ...]; at v=a=0 only m and m*c matter
    Ym = np.stack([Y[:, 10*k] for k in range(nb)], axis=1)
    Yh = np.stack([Y[:, 10*k+1:10*k+4] for k in range(nb)], axis=1).reshape(nb, -1)
    return Ym, Yh

A = np.vstack([regressor(q)[1] for q in Q])
b = np.concatenate([tau - regressor(q)[0] @ masses for q, tau in zip(Q, TAU)])
h, *_ = np.linalg.lstsq(A, b, rcond=None)
com = h.reshape(-1, 3) / masses[:, None]
```

**Order of fitting**

1. Fit the distal links (joints 4-6) first from the torques of joints 4-6 only. The spring does not reach them, so this data is clean.
2. Subtract their contribution, then fit the upper-arm and lower-arm COMs together with the spring parameters (Step 3) on the joint 2 and joint 3 data.

**Torque units.** Measurements are current, not torque. Either calibrate each joint's current-to-torque factor with a known weight on a known lever arm, or fit it jointly: because the masses are known, the scale is observable, so solve for the COMs and a factor Kt per joint with `scipy.optimize.least_squares`.

**Sanity checks.** Keep `np.linalg.cond(A)` moderate (the wrist roll link is the usual weak spot) and fitted COMs inside each link's physical envelope; otherwise suspect a frame or sign error in the URDF or a wrong mass.

## Step 3: Model the spring on joints 2 and 3

The spring runs from the base to the lower arm, so it loads joints 2 and 3 together and leaves joints 4-6 untouched. Both torques come from one potential that depends on the spring length L(q2, q3), which ties them together and cuts the number of free parameters.

```latex
\tau_{spring} = \frac{\partial U}{\partial q}, \qquad U = U_{spring}\big(L(q_2, q_3)\big)
```

Three ways to model it, in order of preference:

| Option | Model | Use when |
| --- | --- | --- |
| A. Parametric (try first) | Two 3D anchor points, stiffness k, free length L0 (about 8 numbers); torque from the gradient of U | It is an ordinary extension spring. Measure k and L0 offline by hanging weights; take anchors from CAD or a ruler |
| B. Learned force curve | Anchors from geometry, force law F(L) as a small polynomial or spline | Anchors are trusted but the spring is nonlinear or pre-tensioned in an unknown way |
| C. Potential MLP | MLP outputs scalar U(q2, q3); torques are its autograd gradient (PyTorch or JAX) | Geometry is not trusted; gradient form keeps both joints' torques consistent |

Fit the COMs of links 2-3 and the spring parameters jointly with `scipy.optimize.least_squares` on the joint 2 and 3 torques, after subtracting the distal links. Spring torques are the numerical gradient of U with respect to q.

**Identifiability.** Spring and gravity look alike if q2 and q3 only move together, so the pose sampling must vary them independently. Compare the fitted k with the offline measurement, and check that the anchors land on the physical mounting points; odd values usually mean a frame or sign error.

## Step 4: Fit a small residual model

What the physics and spring layers leave over is smooth and small, so a tiny learner is enough and the physics keeps carrying extrapolation.

- **Target:** measured static torque minus the model prediction, per joint (joints 2 and 3 matter most).
- **Inputs:** only joints that change the static load, likely q2, q3 and the wrist pitch. Base yaw and wrist roll should not matter; dropping them shrinks the problem.
- **Model choices** with 3-4 inputs: a two-layer MLP of 64 units, a Gaussian process, or a gridded interpolator (`scipy.interpolate.RegularGridInterpolator`). A polynomial in q2 and q3 may already suffice.
- **Clamp inputs** to the sampled range at runtime. A network can misbehave outside the data it has seen.
- **Keep the payload out.** Do not add held objects as network inputs unless data with several payloads was collected; add the mass analytically.

If the residual turns out large, that points at a wrong mass, a wrong frame or a missing force (for example the spring wrapping around a part), and it is better to fix that than to let the network hide it.

## Step 5: Apply the compensation at runtime

The offset is computed from the target pose and added to the goal position, at 50-100 Hz, while the servos keep running their own position loop.

```python
def gravity_compensated_goal(q_target):          # radians, URDF convention
    tau = tau_gravity(q_target) + spring_torque(q_target, p) + residual(q_target)
    return q_target + tau / K                    # K per joint, N·m/rad
```

1. Convert the result to servo ticks (4096 per revolution) and send it with `GroupSyncWrite` as Goal Position.
2. Use the target pose, not the measured pose, as the model input. This avoids feeding sensor noise back into the command.
3. Add the gripper and any carried object as a point mass on the last link before computing the gravity term.

**Calibrating K per joint**

- With the gains set as they will be run, command several poses and wait until the arm settles.
- K = τ\_model / sag, where sag is commanded minus measured position in radians. Take the median over poses.
- Cross-check: at rest, Present Current × torque constant should roughly equal the model torque. A mismatch means wrong masses or COMs, or a wrong torque constant.
- Recalibrate K whenever the P gain, supply voltage or operating mode changes.

**Zero offsets and signs.** The URDF's joint zero and direction must match the servo's. 2048 ticks is not necessarily the URDF zero, and some joints are mirrored. A wrong sign makes the sag worse, so enable the compensation one joint at a time and watch the direction of the change.

**Limits.** Keep the offset inside the Current Limit capability and clamp its magnitude, so a model error near a joint limit cannot command a violent move.

## Step 6: Validate against an external reference

Success is measured as remaining sag on held-out poses, compared against the roughly 1° baseline, using a reference that does not depend on the servo encoders.

1. Run the compensation on the held-out poses and measure the remaining error per joint.
2. Use an external reference where possible: a dial indicator, or an IMU on the end effector. This separates real sag from encoder offsets.
3. Compare joint by joint with the uncompensated baseline, and repeat for approaches from both directions.
4. Re-test with the heaviest expected payload and with warm servos.

**Expected result.** Sag falls from about 1° to roughly 0.1-0.3° per joint. The remainder is mostly the friction deadband and gearbox backlash, which no static model removes. If the error flips sign with the direction of approach, that is backlash.

## Motion and friction

Static compensation fixes resting sag, but while moving, friction adds about ±Fc/K of tracking error; 0.2 N·m against K ≈ 15 N·m/rad is roughly 0.8°, as large as the original sag. Add the terms below as a second stage, once the static model is in and the remaining error is measured.

1. **Identify friction per joint** from slow constant-velocity sweeps in both directions at several speeds. Half the up/down torque difference is Coulomb friction Fc; the slope against velocity is the viscous term b. Joint 2 is likely the largest.
2. **Friction feedforward:** add `Fc·tanh(v_des / v0) + b·v_des` to the static torque before dividing by K. Use the desired velocity, not the measured one, to avoid noise-driven chatter; v0 is a few °/s.
3. **Velocity lag:** a PD loop trails a moving target in proportion to speed. Shift the goal ahead along the path, or use the servo's Feedforward gain registers.
4. **Smooth trajectories:** use Profile Velocity and Acceleration or host-generated paths; abrupt jumps excite stiction and overshoot.
5. **Resting deadband (±Fc/K):** approach final positions from a consistent direction, or add a small Position I gain as trim.

**Limit.** This is adequate for slow moves. Faster motion or heavy payloads need inverse dynamics (Pinocchio `rnea` with desired velocity and acceleration), which requires link inertias this plan does not identify.

## Pitfalls and corrections made during planning

Two earlier statements were wrong and are corrected here, so the rest of this doc should be read with these in mind.

| Topic | Earlier statement | Corrected |
| --- | --- | --- |
| Spring coupling | The spring loads joint 2 only | The spring connects to the lower arm, so it loads joints 2 and 3; modelled as one potential (Step 3) |
| Torque constant | About 1.0-1.3 N·m/A | About 1.8 N·m/A at the output shaft from stall specs; still calibrate per joint |

## Open items and next steps

The next blocker is inputs: kinematics, masses and spring geometry are needed before any fitting can start.

**Information still needed**

- [ ] URDF, or joint axes and link offsets, for all six joints
- [ ] Weighed mass of every link, plus gripper
- [ ] Spring attachment points on the base/upper arm and the lower arm, and its measured stiffness k and free length L0
- [ ] Which joints sag most, and the payload range expected

**Work to do**

- [ ] Apply the latency-timer, baud rate and Return Delay Time settings (System and constraints)
- [ ] Write the pose-sampling and logging script (random poses, two-direction approach, averaging)
- [ ] Fit distal-link COMs from joints 4-6, then joints 2-3 with the spring (Steps 2 and 3)
- [ ] Fit the residual model and calibrate K per joint (Steps 4 and 5)
- [ ] Validate against an external reference (Step 6)
