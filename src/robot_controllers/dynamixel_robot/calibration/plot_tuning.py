"""
Plots the recordings of tune_servo_gains.py so the knee of a sweep can be seen: no robot, only the csv files.
    python -m src.robot_controllers.dynamixel_robot.calibration.plot_tuning tune_joint6_20261023_1822_trials.csv [more ...]

Every trials csv is read with the samples csv next to it (the same name with _samples). Three figures per joint:
1. The knee: every metric of the P sweep against P on a log axis, one panel per metric, with the limit of the
   acceptance criterion drawn in the same panel (overshoot against its limit of max(2 positions, 10 % of the
   step), rings, settle time, rest ripple, pwm noise per unit of P, and the resting error band, the mean |error|
   after the steps). Failed trials have hollow markers, the knee of every pose is a vertical line and the
   recommended P a dotted one. Poses are colours, runs (csv files) are marker shapes.
2. The D sweep at the P of the joint: overshoot and the resting error band against D, per pose.
3. The step responses: the first step up of every P trial per pose, as a fraction of the step against time,
   darker blue for higher P, so overshoot and ringing can be seen directly.

--save <dir> writes png files instead of opening windows.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np

from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import Criteria, Sample, choose_p, rest_metrics

# the reference palette of the charting guide: poses in fixed order, limits in the secondary text colour, a
# light-to-dark blue ramp for the P gain of the step responses
POSE_COLOURS = ('#2a78d6', '#eb6834', '#1baf7a')
LIMIT_COLOUR = '#52514e'
GRID_COLOUR = '#e6e5e1'
BLUE_RAMP = ('#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95',
             '#104281', '#0d366b')
RUN_MARKERS = ('o', 's', '^', 'D', 'v')


class Trial(NamedTuple):
    run: str
    pose: int
    pose_degrees: str
    phase: str
    p: int
    d: int
    step: int
    overshoot: float
    ring_count: int
    settle_time: float | None
    rest_pp: int
    pwm_residual_std: float
    error_band: float  # mean |steady state error| of the steps
    passed: bool
    flags: str


class Run(NamedTuple):
    name: str
    joint: int
    yaml_p: int
    trials: list[Trial]
    baselines: dict[int, tuple[int, float]]  # pose to (rest position pp, pwm residual std) at the yaml gains
    steps: dict[tuple[int, int, int], list[Sample]]  # (pose, p, d) to the samples of the first step up
    poses: dict[int, str]  # pose number to degrees


def _float(value: str) -> float | None:
    return float(value) if value not in ('', None) else None


def read_run(trials_path: str) -> Run:
    """the trials and the samples of one run of tune_servo_gains.py"""
    name = os.path.basename(trials_path).replace('_trials.csv', '')
    with open(trials_path, newline='') as trials_file:
        rows = list(csv.DictReader(trials_file))
    if not rows:
        raise ValueError('{} has no trials'.format(trials_path))
    trials = []
    poses = {}
    for row in rows:
        errors = [abs(e) for e in (_float(row['error_up']), _float(row['error_down'])) if e is not None]
        trials.append(Trial(run=name, pose=int(row['pose']), pose_degrees=row['pose_degrees'], phase=row['phase'],
                            p=int(row['p']), d=int(row['d']), step=int(row['step']),
                            overshoot=float(row['max_overshoot']), ring_count=int(row['max_ring_count']),
                            settle_time=_float(row['max_settle_time']), rest_pp=int(row['rest_pp']),
                            pwm_residual_std=float(row.get('rest_pwm_residual_std') or row['rest_pwm_std']),
                            error_band=float(np.mean(errors)) if errors else 0.0,
                            passed=not row['flags'] and not row['aborted'],
                            flags=row['aborted'] or row['flags']))
        poses[int(row['pose'])] = row['pose_degrees']
    joint = int(rows[0]['joint'])

    samples_path = trials_path.replace('_trials.csv', '_samples.csv')
    baselines: dict[int, tuple[int, float]] = {}
    steps: dict[tuple[int, int, int], list[Sample]] = {}
    yaml_p = trials[0].p
    if os.path.exists(samples_path):
        baseline_rows: dict[int, list[Sample]] = {}
        baseline_p: dict[int, int] = {}
        baseline_trial: dict[int, str] = {}  # the D phase records a second baseline at a pose, keep the first
        with open(samples_path, newline='') as samples_file:
            for row in csv.DictReader(samples_file):
                sample = Sample(time=float(row['time']), tick=int(row['tick']), goal=int(row['goal']),
                                position=int(row['position']), position_trajectory=int(row['position_trajectory']),
                                velocity=int(row['velocity']), velocity_trajectory=int(row['velocity_trajectory']),
                                pwm=int(row['pwm']), current=int(row['current']),
                                moving_status=int(row['moving_status']), voltage=float(row['voltage']),
                                temperature=int(row['temperature']))
                pose = int(row['pose'])
                if row['segment'] == 'baseline':
                    if baseline_trial.setdefault(pose, row['trial']) != row['trial']:
                        continue
                    baseline_rows.setdefault(pose, []).append(sample)
                    baseline_p[pose] = int(row['p'])
                elif row['segment'] == 'step' and row['repeat'] == '0' and row.get('step_index', '0') == '0':
                    steps.setdefault((pose, int(row['p']), int(row['d'])), []).append(sample)
        for pose, samples in baseline_rows.items():
            metrics = rest_metrics(samples, baseline_p[pose])
            baselines[pose] = (metrics.position_pp, metrics.pwm_residual_std)
        if baseline_p:
            yaml_p = next(iter(baseline_p.values()))
    return Run(name, joint, yaml_p, trials, baselines, steps, poses)


def knees(run: Run, factor: float = 1.3) -> dict[int, tuple[int | None, int | None]]:
    """pose to (knee P, recommended P) of the P sweep of the run"""
    result = {}
    for pose in sorted(run.poses):
        sweep = [(t.p, t.passed) for t in run.trials if t.pose == pose and t.phase in ('p', 'p_down')]
        if sweep:
            recommended, knee = choose_p(sweep, factor)
            result[pose] = (knee, recommended)
    return result


def _style(axis: Any) -> None:
    axis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)


def plot_knee(runs: Sequence[Run], criteria: Criteria = Criteria()) -> Any:
    """the metrics of the P sweeps against P, with their limits, per pose"""
    import matplotlib.pyplot as plt
    joint = runs[0].joint
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    panels = {
        'overshoot': (axes[0][0], 'overshoot, positions (dashed: its limit)'),
        'rings': (axes[0][1], 'rings after the first crossing'),
        'settle': (axes[1][0], 'settle time, s'),
        'ripple': (axes[1][1], 'rest ripple, positions peak to peak'),
        'noise': (axes[2][0], 'pwm noise per unit of P, pwm units'),
        'band': (axes[2][1], 'resting error band, mean |error| after a step'),
    }
    poses = sorted({pose for run in runs for pose in run.poses})
    for run_index, run in enumerate(runs):
        marker = RUN_MARKERS[run_index % len(RUN_MARKERS)]
        for pose in poses:
            colour = POSE_COLOURS[(pose - 1) % len(POSE_COLOURS)]
            trials = sorted((t for t in run.trials if t.pose == pose and t.phase in ('p', 'p_down')),
                            key=lambda t: t.p)
            if not trials:
                continue
            p = [t.p for t in trials]
            baseline_pp, baseline_noise = run.baselines.get(pose, (0, 0.0))
            series: dict[str, list[float]] = {
                'overshoot': [t.overshoot for t in trials],
                'rings': [float(t.ring_count) for t in trials],
                'settle': [t.settle_time if t.settle_time is not None else np.nan for t in trials],
                'ripple': [float(t.rest_pp) for t in trials],
                'noise': [t.pwm_residual_std / (t.p / run.yaml_p) for t in trials],
                'band': [t.error_band for t in trials],
            }
            limits: dict[str, list[float]] = {
                'overshoot': [max(criteria.min_overshoot_positions, criteria.max_overshoot_fraction * t.step)
                              for t in trials],
                'rings': [criteria.max_ring_count + 0.5] * len(trials),
                'settle': [criteria.max_settle_time] * len(trials),
                'ripple': [max(criteria.max_rest_pp, criteria.rest_pp_factor * baseline_pp) + 0.5] * len(trials),
                'noise': [max(criteria.pwm_std_floor, criteria.pwm_std_factor * baseline_noise)] * len(trials),
            }
            label = 'pose {} ({})'.format(pose, run.poses[pose]) + (' ' + run.name if len(runs) > 1 else '')
            for key, (axis, _) in panels.items():
                axis.plot(p, series[key], color=colour, linewidth=2, marker=marker, markersize=7, label=label,
                          markerfacecolor=colour)
                failed = [(t.p, value) for t, value in zip(trials, series[key]) if not t.passed]
                if failed:
                    axis.plot([f[0] for f in failed], [f[1] for f in failed], linestyle='none', marker=marker,
                              markersize=9, markerfacecolor='white', markeredgecolor=colour, markeredgewidth=2)
                if key in limits:
                    axis.plot(p, limits[key], color=LIMIT_COLOUR, linewidth=1.2, linestyle='--')
            knee, recommended = knees(run).get(pose, (None, None))
            for key, (axis, _) in panels.items():
                if knee is not None:
                    axis.axvline(knee, color=colour, linewidth=1, alpha=0.6)
                if recommended is not None:
                    axis.axvline(recommended, color=colour, linewidth=1, linestyle=':', alpha=0.8)
    for key, (axis, title) in panels.items():
        axis.set_title(title, fontsize=10, loc='left')
        axis.set_xscale('log')
        _style(axis)
    for axis in axes[2]:
        axis.set_xlabel('P gain (log scale); solid line: knee, dotted: recommended, hollow marker: failed')
    axes[0][0].legend(fontsize=8, frameon=False)
    figure.suptitle('joint {}: the P sweep'.format(joint))
    figure.tight_layout()
    return figure


def plot_d(runs: Sequence[Run]) -> Any | None:
    """overshoot and the resting error band against D at the P of the joint, per pose; None without D trials"""
    import matplotlib.pyplot as plt
    d_trials = [(run_index, t) for run_index, run in enumerate(runs) for t in run.trials if t.phase == 'd']
    if not d_trials:
        return None
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    poses = sorted({t.pose for _, t in d_trials})
    for run_index, run in enumerate(runs):
        marker = RUN_MARKERS[run_index % len(RUN_MARKERS)]
        for pose in poses:
            trials = sorted((t for t in run.trials if t.phase == 'd' and t.pose == pose), key=lambda t: t.d)
            if not trials:
                continue
            colour = POSE_COLOURS[(pose - 1) % len(POSE_COLOURS)]
            label = 'pose {} at P {}'.format(pose, trials[0].p) + (' ' + run.name if len(runs) > 1 else '')
            d = [t.d for t in trials]
            for axis, values in ((left, [t.overshoot for t in trials]), (right, [t.error_band for t in trials])):
                axis.plot(d, values, color=colour, linewidth=2, marker=marker, markersize=7, label=label)
                failed = [(t.d, v) for t, v in zip(trials, values) if not t.passed]
                if failed:
                    axis.plot([f[0] for f in failed], [f[1] for f in failed], linestyle='none', marker=marker,
                              markersize=9, markerfacecolor='white', markeredgecolor=colour, markeredgewidth=2)
    left.set_title('overshoot, positions', fontsize=10, loc='left')
    right.set_title('resting error band, mean |error| after a step', fontsize=10, loc='left')
    for axis in (left, right):
        axis.set_xlabel('D gain (hollow marker: failed)')
        _style(axis)
    left.legend(fontsize=8, frameon=False)
    figure.suptitle('joint {}: the D sweep'.format(runs[0].joint))
    figure.tight_layout()
    return figure


def plot_steps(run: Run) -> Any | None:
    """the first step up of every P trial per pose as a fraction of the step, darker for a higher P"""
    import matplotlib.pyplot as plt
    poses = sorted(run.poses)
    keys = [key for key in run.steps if any(t.pose == key[0] and t.p == key[1] and t.d == key[2]
                                            and t.phase in ('p', 'p_down') for t in run.trials)]
    if not keys:
        return None
    figure, axes = plt.subplots(1, len(poses), figsize=(6 * len(poses), 4), squeeze=False)
    for axis, pose in zip(axes[0], poses):
        pose_keys = sorted((key for key in keys if key[0] == pose), key=lambda key: key[1])
        ramp = BLUE_RAMP[-len(pose_keys):] if len(pose_keys) <= len(BLUE_RAMP) else BLUE_RAMP
        for index, key in enumerate(pose_keys):
            samples = run.steps[key]
            goal = samples[-1].goal
            before = [s for s in samples if s.goal != goal]
            start = before[-1].position if before else samples[0].position
            after = [s for s in samples if s.goal == goal]
            step = goal - start
            if not after or step == 0:
                continue
            t0 = after[0].time
            times = [s.time - t0 for s in after]
            fraction = [(s.position - start) / step for s in after]
            colour = ramp[min(index * len(ramp) // max(len(pose_keys), 1), len(ramp) - 1)]
            axis.plot(times, fraction, color=colour, linewidth=1.6, label='P {} (step {})'.format(key[1], step))
        axis.axhline(1.0, color=LIMIT_COLOUR, linewidth=1, linestyle='--')
        axis.set_title('pose {} ({}): the first step up'.format(pose, run.poses[pose]), fontsize=10, loc='left')
        axis.set_xlabel('s after the step')
        axis.set_ylabel('fraction of the step (dashed: the goal)')
        axis.legend(fontsize=8, frameon=False)
        _style(axis)
    figure.suptitle('joint {}: step responses of {}'.format(run.joint, run.name))
    figure.tight_layout()
    return figure


def print_knees(runs: Iterable[Run]) -> None:
    for run in runs:
        print('{} (joint {}, yaml P {}):'.format(run.name, run.joint, run.yaml_p))
        for pose, (knee, recommended) in knees(run).items():
            print('  pose {} ({}): knee {}, recommended {}'.format(pose, run.poses[pose], knee, recommended))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='plot the recordings of tune_servo_gains.py')
    parser.add_argument('trials', nargs='+', help='the _trials.csv files of one joint, the _samples.csv next to them')
    parser.add_argument('--save', default=None, help='write png files to this directory instead of showing them')
    parser.add_argument('--no-steps', action='store_true', help='skip the step response figure')
    args = parser.parse_args(argv)
    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [read_run(path) for path in args.trials]
    joints = {run.joint for run in runs}
    if len(joints) > 1:
        print('the files are of different joints: {}'.format(sorted(joints)))
        return 1
    print_knees(runs)
    figures = {'knee': plot_knee(runs), 'd': plot_d(runs)}
    if not args.no_steps:
        for run in runs:
            figures['steps_' + run.name] = plot_steps(run)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, figure in figures.items():
            if figure is not None:
                path = os.path.join(args.save, 'joint{}_{}.png'.format(runs[0].joint, name))
                figure.savefig(path, dpi=120)
                print('wrote {}'.format(path))
    else:
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
