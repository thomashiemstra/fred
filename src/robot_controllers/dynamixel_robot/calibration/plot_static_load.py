"""
Shows how well the static load formula of a joint follows the measured holding drive: no robot, only the csv files of
record_static_load.py and servo_config.yaml.
    python -m src.robot_controllers.dynamixel_robot.calibration.plot_static_load FILE.csv [FILE.csv ...] --joint 4
        [--fit] [--without TERM [TERM ...]] [--save DIR]

The recordings are pooled and every pose in which the joint has a measured drive (the mean of the two directions the
pose was approached from, in pwm units, see fit_static_load.py) is a sample. The measured drive is drawn with up to
three models of it, always in the same colours and line styles:
    yaml          blue, dashed: the compensation the joint has in servo_config.yaml now (left out when it has none)
    fit           orange, solid, with --fit: all terms of the joint fitted to these recordings
    fit without   aqua, solid, with --without: the same fit with the named terms left out, to show what a term buys
The measured points are neutral grey dots. Per model the coefficients and the rms of measured minus model are
printed, in pwm units and in positions at the P of the joint in the recordings (the mean when there are several).

Two figures, each only when there are samples of its kind (rows without a mode column count as at rest):
1. Sweeps (mode sweep: one joint swept slowly through its range at a few fixed poses of the other joints, binned by
   its angle): one panel per pose of the other joints, ordered by their angles, all panels on the same y axis. The
   dots are the measured bins, every model is a line through its values at the measured poses. A model that follows the
   load runs through the dots in every panel; the grey text in a panel gives the rms every model leaves there in
   positions, so one sees at which pose a model is off. The title of a panel names the joints that differ between
   the panels.
2. At rest (mode static: random poses, every one different): the measured drive against the drive of each model at the
   same pose, with the diagonal y = x. A pose on the diagonal is one the model has right; a model that is off at some
   poses spreads away from the diagonal there.
--save <dir> writes static_load_joint<N>_sweeps.png and static_load_joint<N>_rest.png instead of opening windows.
"""
import argparse
import math
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from src.robot_controllers.dynamixel_robot.calibration.fit_static_load import Sample, fit_joint, gains_of_joints, \
    read_rows, samples_from_rows
from src.robot_controllers.dynamixel_robot.dynamixel_x_config import PWM_PER_POSITION_PER_P
from src.robot_controllers.dynamixel_robot.static_compensation import JOINT_DRIVES, StaticCompensation, term_names

# the reference palette of the charting guide
BACKGROUND_COLOUR = '#fcfcfb'
GRID_COLOUR = '#e1e0d9'
AXIS_COLOUR = '#c3c2b7'
SECONDARY_COLOUR = '#52514e'  # tick labels, small annotations and the measured points
INK_COLOUR = '#0b0b0b'  # titles and the text of the legend
# the models in fixed order, by kind and never by how many are drawn; the yaml one is dashed so that identity is not
# carried by colour alone
YAML, FIT, WITHOUT = 'yaml', 'fit', 'fit without'
MODEL_COLOURS = {YAML: '#2a78d6', FIT: '#eb6834', WITHOUT: '#1baf7a'}
MODEL_LINESTYLES = {YAML: '--', FIT: '-', WITHOUT: '-'}
MODEL_MARKERS = {YAML: 'o', FIT: 's', WITHOUT: '^'}  # at rest, where there are no lines
LINE_WIDTH = 2.0
DOT_SIZE = 4.5  # points across
PANEL_WIDTH = 3.2  # inches
PANEL_HEIGHT = 3.4
MAX_COLUMNS = 4
HEADER_HEIGHT = 1.3  # inches above the panels: title, subtitle and legend
MIN_FIGURE_WIDTH = 7.0
SAVE_DPI = 150
SWEEP_MODE = 'sweep'


class Model(NamedTuple):
    name: str  # as in the legend
    kind: str  # YAML, FIT or WITHOUT: decides the colour and the line style
    drive: Callable[[np.ndarray], float]  # the holding drive in pwm units at the angles of a pose (7 elements, radians)
    coefficients: dict[str, float]


class Recording(NamedTuple):
    """the samples of the joint of one kind of recording and the P gains of the joint in its rows"""
    samples: list[Sample]
    gains: list[float]


class Panel(NamedTuple):
    angles: tuple[int, ...]  # the angles of the other joints in whole degrees, in the order of other_joints
    title: str
    samples: list[Sample]  # sorted by the angle of the joint


def is_sweep(row: dict[str, str]) -> bool:
    """whether a row is of a sweep; a row without a mode counts as at rest"""
    return (row.get('mode') or '').strip().lower() == SWEEP_MODE


def split_rows(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """the rows of sweeps and the rows at rest (every other mode and rows without a mode)"""
    return [row for row in rows if is_sweep(row)], [row for row in rows if not is_sweep(row)]


def recording_of_joint(rows: Sequence[dict[str, str]], joint: int) -> Recording:
    """
    the samples of the rows in which the joint has a measured drive, and the P gains of the joint in the rows
    :raises ValueError: for a row with an error and without a P gain
    """
    samples = [sample for sample in samples_from_rows(rows) if joint in sample.static]
    return Recording(samples, gains_of_joints(rows).get(joint, []))


def select_samples(rows: Sequence[dict[str, str]], joint: int) -> tuple[Recording, Recording]:
    """the (sweep, at rest) recordings of the joint: the samples of each mode, split before the poses are built"""
    sweep_rows, rest_rows = split_rows(rows)
    return recording_of_joint(sweep_rows, joint), recording_of_joint(rest_rows, joint)


def other_joints(joint: int) -> list[int]:
    """the joints with a static load, without the joint itself: the ones that make the pose of a sweep"""
    return [other for other in sorted(JOINT_DRIVES) if other != joint]


def degrees_of(sample: Sample, joint: int) -> float:
    return math.degrees(float(sample.angles[joint]))


def group_panels(samples: Sequence[Sample], joint: int) -> list[Panel]:
    """
    the samples of a sweep grouped by the pose of the other joints (their angles in whole degrees), ordered by those
    angles. The title names the joints whose angle differs between the panels, or all of them when none does.
    """
    others = other_joints(joint)
    groups: dict[tuple[int, ...], list[Sample]] = {}
    for sample in samples:
        key = tuple(int(round(math.degrees(float(sample.angles[other])))) for other in others)
        groups.setdefault(key, []).append(sample)
    keys = sorted(groups)
    varying = [index for index in range(len(others)) if len({key[index] for key in keys}) > 1]
    named = varying if varying else list(range(len(others)))
    return [Panel(key, '\n'.join('joint {} at {} degrees'.format(others[index], key[index]) for index in named),
                  sorted(groups[key], key=lambda sample: degrees_of(sample, joint))) for key in keys]


def yaml_compensation(joint: int) -> StaticCompensation | None:
    """the compensation of the joint in servo_config.yaml; None when it has none or the yaml can not be read"""
    from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
    try:
        servo = load_servo_config().servos.get(joint)
    except ValueError as e:
        print('servo_config.yaml can not be read: {}'.format(e))
        return None
    return servo.compensation if servo is not None else None


def check_terms(joint: int, names: Sequence[str]) -> str | None:
    """a message when a name is not a term of the joint, or when no term would be left; None when they are fine"""
    terms = term_names(joint)
    unknown = [name for name in names if name not in terms]
    if unknown:
        return 'joint {} has no terms {}, it has {}'.format(joint, ', '.join(unknown), ', '.join(terms))
    if names and not [name for name in terms if name not in names]:
        return 'a fit without {} has no terms left'.format(', '.join(names))
    return None


def fitted_model(joint: int, samples: Sequence[Sample], names: Sequence[str] | None, name: str, kind: str) -> Model:
    """the terms `names` (default all) of the joint fitted to all samples, without a hold-out"""
    fit = fit_joint(joint, samples, holdout=0, names=names)
    coefficients = {term: float(value) for term, value in fit.coefficients.items()}
    return Model(name, kind, StaticCompensation(joint, coefficients, 1).drive, coefficients)


def build_models(joint: int, samples: Sequence[Sample], compensation: StaticCompensation | None, fit: bool,
                 without: Sequence[str]) -> list[Model]:
    """
    the models to draw, in the order yaml, fit, fit without
    :param compensation: the compensation of the joint in the yaml, None for none
    :param fit: add the fit of all terms
    :param without: the terms to leave out of one more fit, none for no such fit
    :raises ValueError: for a term the joint does not have, or too few samples for the fit
    """
    message = check_terms(joint, without)
    if message is not None:
        raise ValueError(message)
    models = []
    if compensation is not None:
        models.append(Model(YAML, YAML, compensation.drive, dict(compensation.terms)))
    if fit:
        models.append(fitted_model(joint, samples, None, FIT, FIT))
    if without:
        names = [term for term in term_names(joint) if term not in without]
        models.append(fitted_model(joint, samples, names, '{} {}'.format(WITHOUT, ', '.join(without)), WITHOUT))
    return models


def residuals(model: Model, samples: Sequence[Sample], joint: int) -> np.ndarray:
    """measured minus model in pwm units, per sample"""
    return np.array([sample.static[joint] - model.drive(sample.angles) for sample in samples], dtype=np.float64)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float('nan')


def rms_of_models(models: Sequence[Model], samples: Sequence[Sample], joint: int) -> dict[str, float]:
    """the rms of measured minus model in pwm units, per model name"""
    return {model.name: rms(residuals(model, samples, joint)) for model in models}


def rms_of_panels(models: Sequence[Model], panels: Sequence[Panel], joint: int) -> list[dict[str, float]]:
    """the rms of measured minus model in pwm units in every panel, per model name"""
    return [rms_of_models(models, panel.samples, joint) for panel in panels]


def mean_gain(gains: Sequence[float]) -> float | None:
    return float(np.mean(gains)) if len(gains) else None


def to_positions(drive: float, p: float | None) -> float | None:
    """a drive in pwm units as positions at the P gain; None without a P"""
    return drive / (p * PWM_PER_POSITION_PER_P) if p else None


def subtitle_text(count: int, noun: str, gains: Sequence[float]) -> str:
    """e.g. '320 bins, 1 position = 6.1 pwm units at P 780'; with several P gains the mean is used"""
    text = '{} {}{}'.format(count, noun, '' if count == 1 else 's')
    p = mean_gain(gains)
    if p is None:
        return text
    several = len(set(gains)) > 1
    return '{}, 1 position = {:.1f} pwm units at P {}{}'.format(
        text, p * PWM_PER_POSITION_PER_P, '{:.0f}'.format(p) if several else '{:g}'.format(p),
        ' (the mean of the P gains of the recordings)' if several else '')


def describe_models(joint: int, kind: str, recording: Recording, models: Sequence[Model]) -> list[str]:
    """the lines printed per model: coefficients and the rms of measured minus model for the samples of one kind"""
    p = mean_gain(recording.gains)
    rms_by_name = rms_of_models(models, recording.samples, joint)
    lines = ['{} ({} samples)'.format(kind, len(recording.samples))]
    for model in models:
        positions = to_positions(rms_by_name[model.name], p) if p else None
        lines.append('  {}: rms {:.1f} pwm units{}'.format(
            model.name, rms_by_name[model.name],
            '' if positions is None or not p else ' = {:.2f} positions at P {:g}'.format(positions, p)))
    return lines


def _style(axis: Any) -> None:
    """the axes of the charting guide: light grid behind the data, no top or right spine"""
    axis.set_facecolor(BACKGROUND_COLOUR)
    axis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(AXIS_COLOUR)
    axis.tick_params(colors=SECONDARY_COLOUR, labelsize=8, color=AXIS_COLOUR)
    axis.xaxis.label.set_color(SECONDARY_COLOUR)
    axis.yaxis.label.set_color(SECONDARY_COLOUR)


def _header(figure: Any, width: float, height: float, title: str, subtitle: str, handles: Sequence[Any]) -> None:
    """the title, the subtitle and the one legend above the panels, then the tight layout of the panels below them"""
    margin = 0.15 / width
    figure.text(margin, 1 - 0.12 / height, title, ha='left', va='top', fontsize=12, color=INK_COLOUR)
    figure.text(margin, 1 - 0.42 / height, subtitle, ha='left', va='top', fontsize=9, color=SECONDARY_COLOUR)
    figure.legend(handles=list(handles), loc='upper center', bbox_to_anchor=(0.5, 1 - 0.72 / height),
                  ncol=min(len(handles), 4), frameon=False, fontsize=9, labelcolor=INK_COLOUR)
    figure.tight_layout(rect=(0, 0, 1, 1 - HEADER_HEIGHT / height))


def _measured_handle() -> Any:
    from matplotlib.lines import Line2D
    return Line2D([], [], linestyle='none', marker='o', markersize=DOT_SIZE, markerfacecolor=SECONDARY_COLOUR,
                  markeredgecolor=BACKGROUND_COLOUR, markeredgewidth=0.6, label='measured')


def _model_handle(model: Model, line: bool) -> Any:
    from matplotlib.lines import Line2D
    colour = MODEL_COLOURS[model.kind]
    if line:
        return Line2D([], [], color=colour, linewidth=LINE_WIDTH, linestyle=MODEL_LINESTYLES[model.kind],
                      label=model.name)
    return Line2D([], [], linestyle='none', marker=MODEL_MARKERS[model.kind], markersize=6, markerfacecolor=colour,
                  markeredgecolor=BACKGROUND_COLOUR, markeredgewidth=0.6, label=model.name)


def _rms_text(models: Sequence[Model], rms_by_name: dict[str, float], p: float | None) -> str:
    """the small text of a panel: the rms of every model in positions, or in pwm units without a P"""
    lines = ['rms, positions' if p else 'rms, pwm units']
    for model in models:
        positions = to_positions(rms_by_name[model.name], p)
        value = rms_by_name[model.name] if positions is None else positions
        lines.append('{} {:.1f}'.format(model.name, value))
    return '\n'.join(lines)


def plot_sweeps(joint: int, panels: Sequence[Panel], models: Sequence[Model], panel_rms: Sequence[dict[str, float]],
                gains: Sequence[float]) -> Any | None:
    """
    one panel per pose of the other joints with the measured bins as dots and a line per model, all on the same y
    axis; None without panels
    :param panel_rms: per panel the rms of every model in pwm units, see rms_of_panels
    :param gains: the P gains of the joint in the recordings
    """
    import matplotlib.pyplot as plt
    if not panels:
        return None
    columns = min(len(panels), MAX_COLUMNS)
    rows = math.ceil(len(panels) / MAX_COLUMNS)
    width, height = max(PANEL_WIDTH * columns, MIN_FIGURE_WIDTH), PANEL_HEIGHT * rows + HEADER_HEIGHT
    figure, axes = plt.subplots(rows, columns, figsize=(width, height), sharey=True, squeeze=False,
                                facecolor=BACKGROUND_COLOUR)
    p = mean_gain(gains)
    for index, axis in enumerate(axes.flat):
        if index >= len(panels):
            axis.set_visible(False)
            continue
        panel = panels[index]
        x = [degrees_of(sample, joint) for sample in panel.samples]
        axis.plot(x, [sample.static[joint] for sample in panel.samples], linestyle='none', marker='o',
                  markersize=DOT_SIZE, markerfacecolor=SECONDARY_COLOUR, markeredgecolor=BACKGROUND_COLOUR,
                  markeredgewidth=0.6, zorder=2)
        for model in models:
            axis.plot(x, [model.drive(sample.angles) for sample in panel.samples], color=MODEL_COLOURS[model.kind],
                      linestyle=MODEL_LINESTYLES[model.kind], linewidth=LINE_WIDTH, zorder=3)
        axis.set_title(panel.title, fontsize=9, loc='left', color=INK_COLOUR)
        axis.set_xlabel('angle of joint {} (degrees)'.format(joint), fontsize=9)
        if index % columns == 0:
            axis.set_ylabel('holding drive (pwm units)', fontsize=9)
        _style(axis)
    # room above the data for the text of the panels: one line per model and the heading
    low, high = axes[0][0].get_ylim()
    headroom = min(0.07 * (len(models) + 1), 0.6)
    axes[0][0].set_ylim(low, high + (high - low) * headroom / (1 - headroom))
    for axis, panel, values in zip(axes.flat, panels, panel_rms):
        axis.text(0.04, 0.96, _rms_text(models, values, p), transform=axis.transAxes, ha='left', va='top',
                  fontsize=7, color=SECONDARY_COLOUR, linespacing=1.3)
    count = sum(len(panel.samples) for panel in panels)
    _header(figure, width, height, 'joint {}: measured holding drive and its models, sweeps'.format(joint),
            subtitle_text(count, 'bin', gains), [_measured_handle()] + [_model_handle(m, True) for m in models])
    return figure


def plot_rest(joint: int, samples: Sequence[Sample], models: Sequence[Model], rms_by_name: dict[str, float],
              gains: Sequence[float]) -> Any | None:
    """
    the measured drive against the drive of every model at the same pose, with the diagonal y = x; None without
    samples
    :param rms_by_name: the rms of every model in pwm units, see rms_of_models
    :param gains: the P gains of the joint in the recordings
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not samples:
        return None
    width, height = MIN_FIGURE_WIDTH, 5.8 + HEADER_HEIGHT
    figure, axis = plt.subplots(figsize=(width, height), facecolor=BACKGROUND_COLOUR)
    measured = [sample.static[joint] for sample in samples]
    predicted = {model.name: [model.drive(sample.angles) for sample in samples] for model in models}
    everything = measured + [value for values in predicted.values() for value in values]
    margin = 0.05 * (max(everything) - min(everything) or 1.0)
    low, high = min(everything) - margin, max(everything) + margin
    axis.plot([low, high], [low, high], color=SECONDARY_COLOUR, linewidth=1.0, zorder=1)
    for model in models:
        axis.plot(predicted[model.name], measured, linestyle='none', marker=MODEL_MARKERS[model.kind],
                  markersize=DOT_SIZE + 1, markerfacecolor=MODEL_COLOURS[model.kind],
                  markeredgecolor=BACKGROUND_COLOUR, markeredgewidth=0.6, zorder=2)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('drive of the model (pwm units)', fontsize=9)
    axis.set_ylabel('measured holding drive (pwm units)', fontsize=9)
    _style(axis)
    p = mean_gain(gains)
    axis.text(0.97, 0.04, _rms_text(models, rms_by_name, p), transform=axis.transAxes, ha='right', va='bottom',
              fontsize=7, color=SECONDARY_COLOUR, linespacing=1.3)
    diagonal = Line2D([], [], color=SECONDARY_COLOUR, linewidth=1.0, label='measured = model')
    _header(figure, width, height, 'joint {}: measured holding drive against its models, at rest'.format(joint),
            subtitle_text(len(samples), 'pose', gains), [_model_handle(m, False) for m in models] + [diagonal])
    return figure


def print_summary(joint: int, sweep: Recording, rest: Recording, models: Sequence[Model]) -> None:
    for model in models:
        print('{}: {}'.format(model.name, ', '.join('{} {:.3f}'.format(name, value)
                                                    for name, value in model.coefficients.items())))
    for kind, recording in (('sweeps', sweep), ('at rest', rest)):
        if recording.samples:
            for line in describe_models(joint, kind, recording, models):
                print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='plot the measured static load of a joint against its models')
    parser.add_argument('files', nargs='+', help='csv files of record_static_load.py')
    parser.add_argument('--joint', type=int, required=True, choices=sorted(JOINT_DRIVES))
    parser.add_argument('--fit', action='store_true', help='add the fit of all terms to these recordings')
    parser.add_argument('--without', nargs='+', default=[], metavar='TERM',
                        help='add the fit without these terms, to show what they buy')
    parser.add_argument('--save', default=None, help='write png files to this directory instead of showing them')
    args = parser.parse_args(argv)
    joint: int = args.joint

    message = check_terms(joint, args.without)
    if message is not None:
        print(message)
        return 1
    missing = [path for path in args.files if not os.path.isfile(path)]
    if missing:
        print('no such file: {}'.format(', '.join(missing)))
        return 1
    try:
        rows = read_rows(args.files)
        sweep, rest = select_samples(rows, joint)
    except (ValueError, KeyError) as e:
        print('can not read the recordings: {}'.format(e))
        return 1
    samples = sweep.samples + rest.samples
    if not samples:
        print('no samples with a measured drive of joint {} in {}'.format(joint, ', '.join(args.files)))
        return 1

    compensation = yaml_compensation(joint)
    if compensation is None:
        print('joint {} has no compensation in servo_config.yaml: no yaml model'.format(joint))
    try:
        models = build_models(joint, samples, compensation, args.fit, args.without)
    except ValueError as e:
        print(e)
        return 1
    if not models:
        print('no model to draw: only the measured drive')
    print('joint {}: {} bins in sweeps, {} poses at rest'.format(joint, len(sweep.samples), len(rest.samples)))
    print_summary(joint, sweep, rest, models)

    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = {}
    panels = group_panels(sweep.samples, joint)
    figures['sweeps'] = plot_sweeps(joint, panels, models, rms_of_panels(models, panels, joint), sweep.gains)
    figures['rest'] = plot_rest(joint, rest.samples, models, rms_of_models(models, rest.samples, joint), rest.gains)
    made = {name: figure for name, figure in figures.items() if figure is not None}
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, figure in made.items():
            path = os.path.join(args.save, 'static_load_joint{}_{}.png'.format(joint, name))
            figure.savefig(path, dpi=SAVE_DPI)
            plt.close(figure)
            print('wrote {}'.format(path))
    elif made:
        plt.show()
    return 0 if made else 1


if __name__ == '__main__':
    sys.exit(main())
