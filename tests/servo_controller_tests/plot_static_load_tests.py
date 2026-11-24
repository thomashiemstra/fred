import contextlib
import csv
import io
import os
import tempfile
import unittest
from unittest import mock

import matplotlib
import numpy as np
from matplotlib.colors import to_hex

from src.robot_controllers.dynamixel_robot.calibration import plot_static_load
from src.robot_controllers.dynamixel_robot.calibration.fit_static_load import Sample
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP
from src.robot_controllers.dynamixel_robot.calibration.plot_static_load import Model, Recording
from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation, term_names

matplotlib.use('Agg')

JOINT = 4
FIELDS = ['time', 'pose', 'direction', 'servo', 'angle', 'goal_position', 'compensation', 'lead', 'present_position',
          'error', 'mode', 'p']
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'


def angles_of(q2: float = 90, q3: float = 20, q4: float = 0, q5: float = 0) -> np.ndarray:
    """the angles of a pose, given in degrees for joints 2 to 5, as the 7 element array in radians"""
    angles = np.zeros(7)
    angles[2:6] = np.radians([q2, q3, q4, q5])
    return angles


def sample_at(angles: np.ndarray, static: float) -> Sample:
    return Sample(angles, {JOINT: static}, {})


def true_terms() -> dict[str, float]:
    """a coefficient for every term the joint has, whichever they are"""
    return {name: 12.0 * (index + 1) * (-1) ** index for index, name in enumerate(term_names(JOINT))}


def true_compensation() -> StaticCompensation:
    return StaticCompensation(JOINT, true_terms(), 15)


def make_row(pose: int, direction: str, servo: int, angle: float, error: float | None, p: float, mode: str | None
             ) -> dict[str, str]:
    row = {'time': '1.0', 'pose': str(pose), 'direction': direction, 'servo': str(servo),
           'angle': '{:.6f}'.format(angle), 'goal_position': '2048', 'compensation': '0', 'lead': '0',
           'present_position': '', 'error': '' if error is None else '{:.6f}'.format(error), 'p': '{:g}'.format(p)}
    if mode is not None:
        row['mode'] = mode
    return row


def pose_rows(pose: int, angles: np.ndarray, mode: str | None, p: float = 780.0, friction: float = 3.0
              ) -> list[dict[str, str]]:
    """
    the rows of one pose like record_static_load writes them, for the joint only (the other servos have an angle and no
    error, like in a sweep): the drive is that of the true compensation plus friction with the sign of the direction.
    The servo drives with P / 128 * (goal sent - present), so the error is minus the drive in positions.
    """
    drive = true_compensation().drive(angles)
    rows = []
    for direction, sign in ((UP, 1), (DOWN, -1)):
        for servo in range(1, 7):
            error = -(drive + sign * friction) * 128 / p if servo == JOINT else None
            rows.append(make_row(pose, direction, servo, float(angles[servo]), error, p, mode))
    return rows


def sweep_rows(first_pose: int = 0, bends: tuple[float, ...] = (-60, 0, 60), mode: str | None = 'sweep',
               p: float = 780.0) -> list[dict[str, str]]:
    """a sweep of the joint from -75 to 75 degrees (11 bins) at every bend of joint 5"""
    rows = []
    pose = first_pose
    for bend in bends:
        for q4 in np.linspace(-75, 75, 11):
            rows.extend(pose_rows(pose, angles_of(90, 20, q4, bend), mode, p))
            pose += 1
    return rows


def rest_rows(count: int = 20, first_pose: int = 0, mode: str | None = 'static', seed: int = 1, p: float = 780.0
              ) -> list[dict[str, str]]:
    """poses at rest, every one different"""
    rng = np.random.default_rng(seed)
    rows = []
    for pose in range(first_pose, first_pose + count):
        angles = angles_of(rng.uniform(40, 140), rng.uniform(-40, 60), rng.uniform(-80, 80), rng.uniform(-80, 80))
        rows.extend(pose_rows(pose, angles, mode, p))
    return rows


def write_csv(directory: str, name: str, rows: list[dict[str, str]]) -> str:
    path = os.path.join(directory, name)
    fields = [field for field in FIELDS if field in rows[0]]
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def constant_model(name: str, kind: str, value: float) -> Model:
    return Model(name, kind, lambda angles: value, {})


class TearDownFigures(unittest.TestCase):
    def tearDown(self) -> None:
        import matplotlib.pyplot as plt
        plt.close('all')


class SamplesTests(TearDownFigures):

    def test_rows_are_split_by_mode(self):
        rows = sweep_rows(bends=(0,)) + rest_rows(3, first_pose=100)
        sweep, rest = plot_static_load.split_rows(rows)
        self.assertEqual({'sweep'}, {row['mode'] for row in sweep})
        self.assertEqual({'static'}, {row['mode'] for row in rest})
        self.assertEqual(len(rows), len(sweep) + len(rest))

    def test_rows_without_a_mode_count_as_at_rest(self):
        rows = rest_rows(3, mode=None)
        self.assertNotIn('mode', rows[0])
        sweep, rest = plot_static_load.split_rows(rows)
        self.assertEqual([], sweep)
        self.assertEqual(rows, rest)
        blank = [dict(row, mode='') for row in rows] + [dict(row, mode=' Sweep ') for row in rows]
        sweep, rest = plot_static_load.split_rows(blank)
        self.assertEqual(len(rows), len(sweep), 'the mode is compared without case and spaces')
        self.assertEqual(len(rows), len(rest), 'a blank mode is at rest')

    def test_samples_are_selected_per_mode_with_the_gains(self):
        rows = sweep_rows(bends=(0, 60), p=780.0) + rest_rows(5, first_pose=100, p=600.0)
        sweep, rest = plot_static_load.select_samples(rows, JOINT)
        self.assertEqual(22, len(sweep.samples))
        self.assertEqual({780.0}, set(sweep.gains))
        self.assertEqual(5, len(rest.samples))
        self.assertEqual({600.0}, set(rest.gains))
        # the measured drive is the mean of the two directions: friction cancels
        for sample in sweep.samples + rest.samples:
            self.assertAlmostEqual(true_compensation().drive(sample.angles), sample.static[JOINT], places=2)

    def test_poses_where_the_joint_has_no_drive_are_left_out(self):
        rows = sweep_rows(bends=(0,))
        self.assertEqual(([], []), tuple(plot_static_load.select_samples(rows, 5)[0]))
        self.assertEqual(11, len(plot_static_load.recording_of_joint(rows, JOINT).samples))


class PanelTests(TearDownFigures):

    def test_panels_are_ordered_by_the_pose_of_the_other_joints_and_sorted_by_the_angle(self):
        samples = [sample_at(angles_of(90, 20, q4, bend), 0.0)
                   for bend in (60, -30, 0, -60, 30) for q4 in (50, -50, 0)]
        panels = plot_static_load.group_panels(samples, JOINT)
        self.assertEqual([(90, 20, bend) for bend in (-60, -30, 0, 30, 60)], [panel.angles for panel in panels])
        self.assertEqual(['joint 5 at {} degrees'.format(bend) for bend in (-60, -30, 0, 30, 60)],
                         [panel.title for panel in panels])
        for panel in panels:
            self.assertEqual([-50, 0, 50], [round(plot_static_load.degrees_of(s, JOINT)) for s in panel.samples])

    def test_the_angles_are_rounded_to_whole_degrees(self):
        samples = [sample_at(angles_of(90.2, 19.8, 0, 59.6), 0.0), sample_at(angles_of(89.9, 20.4, 10, 60.4), 0.0)]
        panels = plot_static_load.group_panels(samples, JOINT)
        self.assertEqual(1, len(panels))
        self.assertEqual((90, 20, 60), panels[0].angles)
        self.assertEqual(2, len(panels[0].samples))

    def test_the_title_names_only_the_joints_that_differ(self):
        samples = [sample_at(angles_of(q2, 20, 0, bend), 0.0) for q2 in (80, 100) for bend in (-60, 60)]
        panels = plot_static_load.group_panels(samples, JOINT)
        self.assertEqual([(80, 20, -60), (80, 20, 60), (100, 20, -60), (100, 20, 60)], [p.angles for p in panels])
        self.assertEqual('joint 2 at 80 degrees\njoint 5 at -60 degrees', panels[0].title)
        self.assertNotIn('joint 3', panels[0].title, 'joint 3 is the same in every panel')

    def test_a_single_pose_is_named_by_all_other_joints(self):
        panels = plot_static_load.group_panels([sample_at(angles_of(90, 20, 0, 30), 0.0)], JOINT)
        self.assertEqual(['joint 2 at 90 degrees\njoint 3 at 20 degrees\njoint 5 at 30 degrees'],
                         [panel.title for panel in panels])

    def test_no_samples_no_panels(self):
        self.assertEqual([], plot_static_load.group_panels([], JOINT))

    def test_other_joints_leave_out_the_joint(self):
        self.assertEqual([2, 3, 5], plot_static_load.other_joints(4))
        self.assertEqual([3, 4, 5], plot_static_load.other_joints(2))


class ModelTests(TearDownFigures):

    def setUp(self) -> None:
        rows = sweep_rows() + rest_rows(30, first_pose=100)
        sweep, rest = plot_static_load.select_samples(rows, JOINT)
        self.samples = sweep.samples + rest.samples

    def test_the_yaml_model_is_the_compensation_of_the_joint(self):
        models = plot_static_load.build_models(JOINT, self.samples, true_compensation(), False, [])
        self.assertEqual(['yaml'], [model.name for model in models])
        self.assertEqual(plot_static_load.YAML, models[0].kind)
        self.assertEqual(true_terms(), models[0].coefficients)
        angles = self.samples[0].angles
        self.assertAlmostEqual(true_compensation().drive(angles), models[0].drive(angles))

    def test_without_a_compensation_there_is_no_yaml_model(self):
        self.assertEqual([], plot_static_load.build_models(JOINT, self.samples, None, False, []))
        models = plot_static_load.build_models(JOINT, self.samples, None, True, [])
        self.assertEqual(['fit'], [model.name for model in models])

    def test_the_fit_has_all_terms_and_recovers_the_data(self):
        models = plot_static_load.build_models(JOINT, self.samples, true_compensation(), True, [])
        self.assertEqual(['yaml', 'fit'], [model.name for model in models])
        fit = models[1]
        self.assertEqual(plot_static_load.FIT, fit.kind)
        self.assertEqual(list(term_names(JOINT)), list(fit.coefficients))
        for name, value in true_terms().items():
            self.assertAlmostEqual(value, fit.coefficients[name], delta=0.05, msg=name)
        rms = plot_static_load.rms_of_models(models, self.samples, JOINT)
        self.assertLess(rms['fit'], 0.05)

    def test_without_leaves_the_named_terms_out_of_the_fit(self):
        left_out = term_names(JOINT)[-1]
        models = plot_static_load.build_models(JOINT, self.samples, None, True, [left_out])
        self.assertEqual(['fit', 'fit without {}'.format(left_out)], [model.name for model in models])
        without = models[1]
        self.assertEqual(plot_static_load.WITHOUT, without.kind)
        self.assertEqual([name for name in term_names(JOINT) if name != left_out], list(without.coefficients))
        rms = plot_static_load.rms_of_models(models, self.samples, JOINT)
        self.assertGreater(rms[without.name], rms['fit'], 'a term that is in the data buys something')

    def test_without_works_without_the_full_fit(self):
        left_out = term_names(JOINT)[1]
        models = plot_static_load.build_models(JOINT, self.samples, true_compensation(), False, [left_out])
        self.assertEqual(['yaml', 'fit without {}'.format(left_out)], [model.name for model in models])

    def test_several_terms_are_left_out_together_in_one_model(self):
        left_out = list(term_names(JOINT)[:2])
        models = plot_static_load.build_models(JOINT, self.samples, None, False, left_out)
        self.assertEqual(['fit without {}'.format(', '.join(left_out))], [model.name for model in models])

    def test_an_unknown_term_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'no terms nonsense'):
            plot_static_load.build_models(JOINT, self.samples, None, False, ['nonsense'])
        self.assertIsNone(plot_static_load.check_terms(JOINT, []))
        self.assertIn('no terms', plot_static_load.check_terms(JOINT, ['nonsense']))

    def test_removing_every_term_is_an_error(self):
        self.assertIn('no terms left', plot_static_load.check_terms(JOINT, list(term_names(JOINT))))

    def test_a_fit_needs_samples(self):
        with self.assertRaises(ValueError):
            plot_static_load.build_models(JOINT, self.samples[:1], None, True, [])


class RmsTests(TearDownFigures):

    def test_rms_of_measured_minus_model(self):
        samples = [sample_at(angles_of(), value) for value in (12.0, 8.0, 13.0)]
        models = [constant_model('yaml', plot_static_load.YAML, 10.0),
                  constant_model('fit', plot_static_load.FIT, 12.0)]
        residuals = plot_static_load.residuals(models[0], samples, JOINT)
        self.assertEqual([2.0, -2.0, 3.0], list(residuals))
        rms = plot_static_load.rms_of_models(models, samples, JOINT)
        self.assertAlmostEqual(np.sqrt((4 + 4 + 9) / 3), rms['yaml'])
        self.assertAlmostEqual(np.sqrt((0 + 16 + 1) / 3), rms['fit'])

    def test_rms_per_panel(self):
        samples = [sample_at(angles_of(90, 20, 0, 0), 12.0), sample_at(angles_of(90, 20, 10, 0), 8.0),
                   sample_at(angles_of(90, 20, 0, 30), 16.0)]
        panels = plot_static_load.group_panels(samples, JOINT)
        model = constant_model('yaml', plot_static_load.YAML, 10.0)
        rms = plot_static_load.rms_of_panels([model], panels, JOINT)
        self.assertEqual(2, len(rms))
        self.assertAlmostEqual(2.0, rms[0]['yaml'])  # bend 0: residuals 2 and -2
        self.assertAlmostEqual(6.0, rms[1]['yaml'])  # bend 30: residual 6

    def test_positions_are_pwm_divided_by_p_over_128(self):
        self.assertAlmostEqual(3.0, plot_static_load.to_positions(6.0, 256.0))
        self.assertAlmostEqual(1.0, plot_static_load.to_positions(780 / 128, 780.0))
        self.assertIsNone(plot_static_load.to_positions(6.0, None))

    def test_without_a_p_the_panel_text_is_in_pwm_units(self):
        model = constant_model('yaml', plot_static_load.YAML, 10.0)
        self.assertEqual('rms, pwm units\nyaml 2.0', plot_static_load._rms_text([model], {'yaml': 2.0}, None))
        self.assertEqual('rms, positions\nyaml 1.0', plot_static_load._rms_text([model], {'yaml': 2.0}, 256.0))

    def test_the_subtitle_gives_the_count_and_what_a_position_is_worth(self):
        self.assertEqual('320 bins, 1 position = 6.1 pwm units at P 780', plot_static_load.subtitle_text(320, 'bin',
                                                                                                         [780.0] * 5))
        self.assertEqual('1 pose', plot_static_load.subtitle_text(1, 'pose', []))
        mixed = plot_static_load.subtitle_text(3, 'pose', [640.0, 640.0, 1280.0])
        self.assertIn('P 853', mixed)
        self.assertIn('mean', mixed)

    def test_the_printed_lines_give_pwm_and_positions(self):
        samples = [sample_at(angles_of(), value) for value in (12.0, 8.0)]
        lines = plot_static_load.describe_models(
            JOINT, 'sweeps', Recording(samples, [256.0]), [constant_model('yaml', plot_static_load.YAML, 10.0)])
        self.assertEqual(['sweeps (2 samples)', '  yaml: rms 2.0 pwm units = 1.00 positions at P 256'], lines)
        lines = plot_static_load.describe_models(
            JOINT, 'at rest', Recording(samples, []), [constant_model('yaml', plot_static_load.YAML, 10.0)])
        self.assertEqual('  yaml: rms 2.0 pwm units', lines[1])


class FigureTests(TearDownFigures):

    def setUp(self) -> None:
        sweep, rest = plot_static_load.select_samples(sweep_rows(bends=(-60, -30, 0, 30, 60)) + rest_rows(12, 200),
                                                      JOINT)
        self.sweep, self.rest = sweep, rest
        # models that are off by different amounts, in the order yaml, fit, fit without
        self.models = [Model('yaml', plot_static_load.YAML, lambda a: true_compensation().drive(a) + 3.0, {}),
                       Model('fit', plot_static_load.FIT, lambda a: true_compensation().drive(a) + 1.0, {}),
                       Model('fit without x', plot_static_load.WITHOUT, lambda a: 0.0, {})]
        self.panels = plot_static_load.group_panels(sweep.samples, JOINT)

    def sweep_figure(self, models=None):
        models = self.models if models is None else models
        figure = plot_static_load.plot_sweeps(JOINT, self.panels, models,
                                              plot_static_load.rms_of_panels(models, self.panels, JOINT),
                                              self.sweep.gains)
        self.assertIsNotNone(figure)
        return figure

    def test_one_panel_per_pose_sharing_the_y_axis(self):
        figure = self.sweep_figure()
        visible = [axis for axis in figure.axes if axis.get_visible()]
        self.assertEqual(5, len(visible))
        self.assertEqual(8, len(figure.axes), 'two rows of four, the empty ones are hidden')
        for axis in visible[1:]:
            self.assertTrue(visible[0].get_shared_y_axes().joined(visible[0], axis))
        self.assertEqual(['joint 5 at {} degrees'.format(bend) for bend in (-60, -30, 0, 30, 60)],
                         [axis.get_title(loc='left') for axis in visible])
        self.assertEqual('angle of joint 4 (degrees)', visible[0].get_xlabel())
        self.assertEqual('holding drive (pwm units)', visible[0].get_ylabel())

    def test_the_figure_is_sized_to_its_panels(self):
        width, height = self.sweep_figure().get_size_inches()
        self.assertAlmostEqual(3.2 * 4, width)
        self.assertAlmostEqual(3.4 * 2 + plot_static_load.HEADER_HEIGHT, height)
        one = plot_static_load.plot_sweeps(JOINT, self.panels[:1], self.models[:1], [{'yaml': 1.0}], [780.0])
        self.assertAlmostEqual(3.4 + plot_static_load.HEADER_HEIGHT, one.get_size_inches()[1])

    def test_the_measured_bins_are_neutral_dots_and_every_model_is_a_line(self):
        figure = self.sweep_figure()
        axis = figure.axes[0]
        dots, *lines = axis.get_lines()
        self.assertEqual('None', dots.get_linestyle())
        self.assertEqual('o', dots.get_marker())
        self.assertEqual(4.5, dots.get_markersize())
        self.assertEqual('#52514e', to_hex(dots.get_markerfacecolor()))
        self.assertEqual('#fcfcfb', to_hex(dots.get_markeredgecolor()))
        self.assertEqual(11, len(dots.get_xdata()))
        self.assertEqual([BLUE, ORANGE, AQUA], [to_hex(line.get_color()) for line in lines])
        self.assertEqual(['--', '-', '-'], [line.get_linestyle() for line in lines])
        self.assertEqual([2.0] * 3, [line.get_linewidth() for line in lines])
        for line in lines:
            self.assertEqual(sorted(line.get_xdata()), list(line.get_xdata()), 'sorted by the angle')
            self.assertEqual(list(dots.get_xdata()), list(line.get_xdata()))
        # the line of a model is its values at the measured poses
        panel = self.panels[0]
        self.assertAlmostEqual(self.models[0].drive(panel.samples[3].angles), lines[0].get_ydata()[3])

    def test_the_colour_belongs_to_the_model_and_not_to_how_many_are_drawn(self):
        figure = self.sweep_figure(self.models[1:])
        colours = [to_hex(line.get_color()) for line in figure.axes[0].get_lines()[1:]]
        self.assertEqual([ORANGE, AQUA], colours)
        figure = self.sweep_figure(self.models[:1])
        self.assertEqual([BLUE], [to_hex(line.get_color()) for line in figure.axes[0].get_lines()[1:]])

    def test_one_legend_for_the_figure_in_ink_colours(self):
        figure = self.sweep_figure()
        self.assertEqual(1, len(figure.legends))
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
        legend = figure.legends[0]
        self.assertEqual(['measured', 'yaml', 'fit', 'fit without x'], [text.get_text() for text in legend.get_texts()])
        self.assertEqual({'#0b0b0b'}, {to_hex(text.get_color()) for text in legend.get_texts()})

    def test_title_subtitle_and_the_rms_text_of_a_panel(self):
        figure = self.sweep_figure()
        texts = {text.get_text(): text for text in figure.texts}
        title = 'joint 4: measured holding drive and its models, sweeps'
        self.assertIn(title, texts)
        self.assertEqual('#0b0b0b', to_hex(texts[title].get_color()))
        subtitle = '55 bins, 1 position = 6.1 pwm units at P 780'
        self.assertIn(subtitle, texts)
        self.assertEqual('#52514e', to_hex(texts[subtitle].get_color()))
        rms = plot_static_load.rms_of_panels(self.models, self.panels, JOINT)[0]
        in_panel = [text.get_text() for text in figure.axes[0].texts]
        self.assertEqual(1, len(in_panel))
        expected = ['rms, positions'] + ['{} {:.1f}'.format(m.name, rms[m.name] * 128 / 780) for m in self.models]
        self.assertEqual(expected, in_panel[0].split('\n'))
        self.assertEqual('#52514e', to_hex(figure.axes[0].texts[0].get_color()))

    def test_the_axes_follow_the_charting_guide(self):
        figure = self.sweep_figure()
        self.assertEqual('#fcfcfb', to_hex(figure.get_facecolor()))
        axis = figure.axes[0]
        self.assertEqual('#fcfcfb', to_hex(axis.get_facecolor()))
        self.assertFalse(axis.spines['top'].get_visible())
        self.assertFalse(axis.spines['right'].get_visible())
        self.assertEqual('#c3c2b7', to_hex(axis.spines['left'].get_edgecolor()))
        self.assertTrue(axis.get_axisbelow())
        self.assertEqual('#e1e0d9', to_hex(axis.xaxis.get_gridlines()[0].get_color()))

    def test_without_panels_there_is_no_sweep_figure(self):
        self.assertIsNone(plot_static_load.plot_sweeps(JOINT, [], self.models, [], []))

    def test_the_rest_figure_has_the_diagonal_and_a_marker_colour_per_model(self):
        rms = plot_static_load.rms_of_models(self.models, self.rest.samples, JOINT)
        figure = plot_static_load.plot_rest(JOINT, self.rest.samples, self.models, rms, self.rest.gains)
        self.assertIsNotNone(figure)
        self.assertEqual(1, len(figure.axes))
        axis = figure.axes[0]
        diagonal, *points = axis.get_lines()
        self.assertEqual(list(diagonal.get_xdata()), list(diagonal.get_ydata()), 'y = x')
        measured = [sample.static[JOINT] for sample in self.rest.samples]
        self.assertEqual(3, len(points))
        for line, model, colour in zip(points, self.models, (BLUE, ORANGE, AQUA)):
            self.assertEqual(colour, to_hex(line.get_markerfacecolor()))
            self.assertEqual('None', line.get_linestyle())
            self.assertEqual(measured, list(line.get_ydata()), 'measured on y')
            self.assertEqual([model.drive(s.angles) for s in self.rest.samples], list(line.get_xdata()))
        self.assertLessEqual(diagonal.get_xdata()[0], min(measured))
        self.assertGreaterEqual(diagonal.get_xdata()[1], max(measured))
        self.assertEqual(1, len(figure.legends))
        self.assertEqual(['yaml', 'fit', 'fit without x', 'measured = model'],
                         [text.get_text() for text in figure.legends[0].get_texts()])
        self.assertIn('12 poses, 1 position = 6.1 pwm units at P 780', [text.get_text() for text in figure.texts])
        self.assertEqual('measured holding drive (pwm units)', axis.get_ylabel())

    def test_without_samples_there_is_no_rest_figure(self):
        self.assertIsNone(plot_static_load.plot_rest(JOINT, [], self.models, {}, []))


class MainTests(TearDownFigures):

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = os.path.join(self.directory.name, 'png')

    def run_main(self, args, compensation=true_compensation()):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), \
                mock.patch.object(plot_static_load, 'yaml_compensation', return_value=compensation):
            code = plot_static_load.main(args)
        return code, buffer.getvalue()

    def files(self, sweeps=True, rest=True, mode_column=True):
        paths = []
        if sweeps:
            paths.append(write_csv(self.directory.name, 'sweep.csv', sweep_rows()))
        if rest:
            mode = 'static' if mode_column else None
            paths.append(write_csv(self.directory.name, 'rest.csv', rest_rows(30, mode=mode)))
        return paths

    def test_the_two_figures_are_written_as_non_empty_png_files(self):
        code, printed = self.run_main(self.files() + ['--joint', '4', '--fit', '--save', self.output])
        self.assertEqual(0, code, printed)
        self.assertEqual({'static_load_joint4_sweeps.png', 'static_load_joint4_rest.png'}, set(os.listdir(self.output)))
        for name in os.listdir(self.output):
            path = os.path.join(self.output, name)
            self.assertGreater(os.path.getsize(path), 5000, name)
            with open(path, 'rb') as png:
                self.assertEqual(b'\x89PNG', png.read(4))
        self.assertIn('33 bins in sweeps, 30 poses at rest', printed)
        self.assertIn('yaml: ', printed)
        self.assertIn('fit: ', printed)
        self.assertIn('rms', printed)
        self.assertIn('positions at P 780', printed)
        for name in term_names(JOINT):
            self.assertIn(name, printed)

    def test_the_figure_of_a_kind_without_samples_is_not_made(self):
        code, printed = self.run_main(self.files(rest=False) + ['--joint', '4', '--save', self.output])
        self.assertEqual(0, code, printed)
        self.assertEqual({'static_load_joint4_sweeps.png'}, set(os.listdir(self.output)))
        other = os.path.join(self.directory.name, 'other')
        code, printed = self.run_main(self.files(sweeps=False) + ['--joint', '4', '--save', other])
        self.assertEqual(0, code, printed)
        self.assertEqual({'static_load_joint4_rest.png'}, set(os.listdir(other)))

    def test_a_csv_without_a_mode_column_is_at_rest(self):
        paths = self.files(sweeps=False, mode_column=False)
        code, printed = self.run_main(paths + ['--joint', '4', '--save', self.output])
        self.assertEqual(0, code, printed)
        self.assertEqual({'static_load_joint4_rest.png'}, set(os.listdir(self.output)))

    def test_without_a_yaml_compensation_a_line_says_so(self):
        code, printed = self.run_main(self.files() + ['--joint', '4', '--fit', '--save', self.output],
                                      compensation=None)
        self.assertEqual(0, code, printed)
        self.assertIn('no compensation in servo_config.yaml', printed)
        self.assertNotRegex(printed, r'(?m)^ *yaml:')
        self.assertIn('fit: ', printed)

    def test_without_names_the_terms_left_out(self):
        left_out = term_names(JOINT)[-1]
        code, printed = self.run_main(self.files() + ['--joint', '4', '--without', left_out, '--save', self.output])
        self.assertEqual(0, code, printed)
        self.assertIn('fit without {}'.format(left_out), printed)

    def test_an_unknown_term_is_a_message_and_return_code_1(self):
        code, printed = self.run_main(self.files() + ['--joint', '4', '--without', 'nonsense', '--save', self.output])
        self.assertEqual(1, code)
        self.assertIn('nonsense', printed)
        self.assertFalse(os.path.exists(self.output))

    def test_a_missing_file_is_a_message_and_return_code_1(self):
        code, printed = self.run_main([os.path.join(self.directory.name, 'nothing.csv'), '--joint', '4'])
        self.assertEqual(1, code)
        self.assertIn('no such file', printed)

    def test_no_samples_for_the_joint_is_a_message_and_return_code_1(self):
        # the rows have a drive for joint 4 only
        code, printed = self.run_main(self.files() + ['--joint', '5', '--save', self.output])
        self.assertEqual(1, code)
        self.assertIn('no samples', printed)
        self.assertFalse(os.path.exists(self.output))

    def test_a_row_with_an_error_and_without_p_is_a_message_and_return_code_1(self):
        rows = rest_rows(5)
        for row in rows:
            row['p'] = ''
        path = write_csv(self.directory.name, 'nop.csv', rows)
        code, printed = self.run_main([path, '--joint', '4', '--save', self.output])
        self.assertEqual(1, code)
        self.assertIn('no usable p', printed)

    def test_too_few_samples_for_the_fit_is_a_message_and_return_code_1(self):
        path = write_csv(self.directory.name, 'few.csv', rest_rows(1))
        code, printed = self.run_main([path, '--joint', '4', '--fit', '--save', self.output])
        self.assertEqual(1, code)
        self.assertIn('at least', printed)

    def test_the_real_yaml_is_read_without_hardware(self):
        # the compensation of joint 4 in servo_config.yaml, or none: either way a StaticCompensation or None
        compensation = plot_static_load.yaml_compensation(JOINT)
        self.assertTrue(compensation is None or isinstance(compensation, StaticCompensation))


if __name__ == '__main__':
    unittest.main()
