import contextlib
import csv
import io
import os
import unittest
from unittest import mock

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import check_supply_voltage as csv_check
from src.robot_controllers.dynamixel_robot.calibration.check_supply_voltage import (
    CHAIN, FIELDS, Reading, Sampler, VoltageStats, against_reference, chain_drops, conclusions, format_phase,
    hop_drops, largest_hop, out_of_range, statistics)
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from tests.servo_controller_tests.dynamixel_robot_controller_tests import FakeBus

TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_servo_config.yaml')
MODELS = {1: 'XM430-W350', 2: 'XM540-W270', 3: 'XM430-W350', 4: 'XC430-W240', 5: 'XC430-W240', 6: 'XC430-W150',
          7: 'XL330-M288'}


def readings(phase, voltages, pose='', temperature=30, count=3):
    """count readings per servo with the given voltages (servo id to volts)"""
    rows = []
    for index in range(count):
        for servo, voltage in voltages.items():
            rows.append(Reading(phase, pose, 0.01 * index, servo, voltage, 0, temperature))
    return rows


def stats_of(voltages, spread=0.0):
    """statistics with the mean at the given voltages, min and max spread around it"""
    return {servo: VoltageStats(servo, 10, voltage - spread, voltage, voltage + spread)
            for servo, voltage in voltages.items()}


REFERENCE = {1: 11.6, 2: 11.4, 3: 11.3, 4: 11.3, 5: 11.3, 6: 11.2, 7: 5.1}


class StatisticsTests(unittest.TestCase):

    def test_statistics_per_servo(self):
        rows = readings('rest', {1: 11.6, 2: 11.4}) + [Reading('rest', '', 0.1, 1, 11.2, 0, 30),
                                                      Reading('rest', '', 0.1, 2, 11.8, 0, 30)]
        stats = statistics(rows)
        self.assertEqual([1, 2], list(stats))
        self.assertEqual(VoltageStats(1, 4, 11.2, 11.5, 11.6), stats[1])
        self.assertEqual(VoltageStats(2, 4, 11.4, 11.5, 11.8), stats[2])
        self.assertEqual({}, statistics([]))

    def test_chain_and_hop_drops(self):
        stats = stats_of({1: 11.6, 2: 11.4, 3: 11.3, 4: 11.3, 5: 11.0, 6: 10.9, 7: 5.1})
        drops = chain_drops(stats)
        self.assertEqual([2, 3, 4, 5, 6], list(drops), 'the gripper is not in the chain')
        self.assertAlmostEqual(0.2, drops[2])
        self.assertAlmostEqual(0.7, drops[6])
        hops = hop_drops(stats)
        self.assertAlmostEqual(0.3, hops[(4, 5)])
        self.assertEqual(((4, 5), hops[(4, 5)]), largest_hop(hops))
        self.assertEqual({}, chain_drops(stats_of({7: 5.1})))
        self.assertEqual({}, chain_drops({}))
        self.assertIsNone(largest_hop({}))
        # a servo missing from the chain: the hops skip it
        self.assertEqual([(1, 3), (3, 4)], list(hop_drops(stats_of({1: 11.6, 3: 11.3, 4: 11.2}))))

    def test_out_of_range(self):
        stats = stats_of({1: 11.0, 4: 9.0, 6: 10.0, 7: 5.0}, spread=0.5)
        self.assertEqual({}, out_of_range(stats, MODELS))
        stats = stats_of({1: 10.4, 4: 6.9, 6: 10.0, 7: 3.4}, spread=0.5)
        flags = out_of_range(stats, MODELS)
        self.assertEqual([1, 4, 7], sorted(flags))
        self.assertIn('9.9 V is below the rated 10.0-14.8 V of the XM430-W350', flags[1])
        self.assertIn('below', flags[4])
        self.assertIn('above', out_of_range(stats_of({6: 12.5}), MODELS)[6])
        self.assertEqual({}, out_of_range(stats_of({9: 1.0}), MODELS), 'an unknown model is not judged')

    def test_against_reference(self):
        deltas = against_reference(stats_of({1: 11.0, 7: 5.2, 9: 1.0}), REFERENCE)
        self.assertAlmostEqual(-0.6, deltas[1])
        self.assertAlmostEqual(0.1, deltas[7])
        self.assertNotIn(9, deltas)


class ConclusionTests(unittest.TestCase):

    def test_a_healthy_chain(self):
        rest = stats_of({1: 11.6, 2: 11.4, 3: 11.3, 4: 11.3, 5: 11.3, 6: 11.2, 7: 5.1})
        hold = stats_of({1: 11.5, 2: 11.3, 3: 11.2, 4: 11.2, 5: 11.2, 6: 11.1, 7: 5.1}, spread=0.1)
        lines = conclusions({'rest': rest, 'hold': hold}, MODELS, REFERENCE)
        self.assertEqual(3, len(lines))
        self.assertIn('the supply is fine at rest', lines[0])
        self.assertIn('the chain itself is fine', lines[1])
        self.assertIn('largest hop is servo 1 to 2', lines[1])
        self.assertIn('gripper rail reads 5.1-5.1 V', lines[2])

    def test_a_low_supply(self):
        rest = stats_of({1: 10.9, 2: 10.8, 3: 10.7, 4: 10.7, 5: 10.7, 6: 10.6, 7: 5.1})
        lines = conclusions({'rest': rest}, MODELS, REFERENCE)
        self.assertIn('0.7 V below the reference', lines[0])
        self.assertIn('the supply or its lead', lines[0])
        self.assertIn('0.3 V at rest', lines[1])

    def test_a_lead_that_loses_voltage_under_load(self):
        rest = stats_of({1: 11.6, 2: 11.4, 3: 11.3, 4: 11.3, 5: 11.3, 6: 11.2})
        # the XM430 of joint 3 is rated down to 10 V, the XC430s of the wrist down to 6.5 V
        motion = stats_of({1: 11.5, 2: 11.4, 3: 10.3, 4: 10.2, 5: 10.1, 6: 10.0}, spread=0.5)
        lines = conclusions({'rest': rest, 'motion': motion}, MODELS, REFERENCE)
        self.assertIn('grows from 0.4 V at rest to 1.5 V during motion', lines[1])
        self.assertIn('leads or the connectors', lines[1])
        self.assertIn('largest hop is servo 2 to 3 with 1.1 V', lines[1])
        self.assertIn('dips to 9.8 V at servo 3 during motion', lines[2])

    def test_without_a_rest_phase(self):
        hold = stats_of({1: 11.5, 2: 11.3, 6: 11.0})
        lines = conclusions({'hold': hold}, MODELS, REFERENCE)
        self.assertEqual(1, len(lines))
        self.assertIn('no rest phase to compare with', lines[0])
        self.assertEqual([], conclusions({}, MODELS, REFERENCE))

    def test_format_phase_marks_the_flags(self):
        stats = stats_of({1: 11.6, 6: 6.2, 7: 5.1}, spread=0.1)
        text = format_phase('hold', 'reach', stats, MODELS, 5.0, REFERENCE)
        self.assertIn('hold at reach: 5.0 s, 10 samples per servo', text)
        self.assertIn('OUTSIDE THE RATED RANGE', text)
        self.assertIn('5 V rail', text)
        self.assertNotIn('OUTSIDE', text.split('\n')[2], 'servo 1 is fine')


class FakeBusTests(unittest.TestCase):
    """the sampler on the fake bus: one packet per bus, the csv rows, the statistics and the stops"""

    def setUp(self):
        self.config = load_servo_config(TEST_SERVO_CONFIG_PATH)
        self.bus = FakeBus(list(self.config.servos.values()), self.config.buses)
        self.handler = ServoHandler(dict(self.config.servos), cfg, mock.Mock(), self.bus, self.config.buses)
        self.now = 0.0
        self.set_voltages({1: 116, 2: 114, 3: 113, 4: 113, 5: 113, 6: 112, 7: 51})
        for servo_id in self.config.servos:
            self.bus.memory[servo_id][cfg.ADDR_PRESENT_TEMPERATURE] = 30 + servo_id
            self.bus.memory[servo_id][cfg.ADDR_PRESENT_CURRENT:cfg.ADDR_PRESENT_CURRENT + 2] = \
                (10 * servo_id).to_bytes(2, 'little')
        self.output = io.StringIO()
        writer = csv.DictWriter(self.output, fieldnames=FIELDS)
        self.sampler = Sampler(self.handler, self.config.models, self.config.temperature_warning, writer,
                               clock=self.clock)

    def clock(self):
        self.now += 0.009
        return self.now

    def set_voltages(self, tenths):
        for servo_id, value in tenths.items():
            self.bus.memory[servo_id][cfg.ADDR_PRESENT_INPUT_VOLTAGE:cfg.ADDR_PRESENT_INPUT_VOLTAGE + 2] = \
                value.to_bytes(2, 'little')

    def test_one_read_is_one_packet_per_bus_with_every_servo(self):
        readings = self.sampler.read('rest')
        self.assertEqual([('sync_read', cfg.ADDR_REALTIME_TICK, [1, 2, 3]),
                          ('sync_read', cfg.ADDR_REALTIME_TICK, [4, 5, 6, 7])], self.bus.packets)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], [r.servo for r in readings])
        self.assertEqual([11.6, 11.4, 11.3, 11.3, 11.3, 11.2, 5.1], [r.voltage for r in readings])
        self.assertEqual([31, 32, 33, 34, 35, 36, 37], [r.temperature for r in readings])
        rows = list(csv.DictReader(io.StringIO(self.output.getvalue()), fieldnames=FIELDS))
        self.assertEqual(7, len(rows))
        self.assertEqual({'phase': 'rest', 'pose': '', 'servo': '2', 'voltage': '11.4', 'current_raw': '20',
                          'current': '53.80', 'current_unit': 'mA', 'temperature': '32'},
                         {key: rows[1][key] for key in rows[1] if key != 'time'})
        self.assertEqual('load_0.1%', rows[3]['current_unit'], 'the XC430 has a load, not a current')

    def test_capture_and_statistics(self):
        self.sampler.start_phase()
        self.sampler.capture('rest', '', 0.1)
        self.set_voltages({6: 60})
        self.sampler.start_phase()
        self.sampler.capture('hold', 'reach', 0.1)
        rest = self.sampler.stats('rest')
        hold = self.sampler.stats('hold', 'reach')
        self.assertGreater(rest[1].samples, 5)
        self.assertEqual((11.2, 11.2), (rest[6].minimum, rest[6].maximum))
        self.assertAlmostEqual(11.2, rest[6].mean)
        self.assertEqual(6.0, hold[6].minimum)
        self.assertEqual({6: '6.0 V is below the rated 6.5-12.0 V of the XC430-W150'}, out_of_range(hold, MODELS))
        self.assertEqual({}, out_of_range(rest, MODELS))
        self.assertAlmostEqual(5.6, chain_drops(hold)[6])
        self.assertEqual({}, self.sampler.stats('motion'))

    def test_a_warm_servo_stops_the_sampling(self):
        self.bus.memory[5][cfg.ADDR_PRESENT_TEMPERATURE] = self.config.temperature_warning + 1
        with self.assertRaises(csv_check.Stop) as raised:
            self.sampler.read('rest')
        self.assertIn('servo 5 is at 46C', str(raised.exception))
        self.assertEqual(7, len(self.sampler.readings), 'the reading that showed it is kept')

    def test_a_failed_read_is_retried_and_counted(self):
        original = self.handler.read_state
        calls = []

        def flaky(servo_ids=None):
            calls.append(1)
            return None if len(calls) in (2, 5, 6) else original(servo_ids)
        self.handler.read_state = flaky
        self.sampler.start_phase()
        for _ in range(4):
            self.sampler.read('hold', 'reach')
        self.assertEqual(4 * 7, len(self.sampler.readings))
        self.assertEqual({'hold': 3}, self.sampler.failed_reads)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            csv_check.print_report(self.sampler, self.config.models, [('hold', 'reach', 0.1)])
        self.assertIn('3 of 7 reads failed during hold and were retried', output.getvalue())

    def test_repeated_failures_and_a_stop_request(self):
        self.bus.unresponsive = {4}
        with self.assertRaises(csv_check.Stop) as raised:
            self.sampler.read('rest')
        self.assertIn('3 times in a row', str(raised.exception))
        self.assertEqual({'rest': 3}, self.sampler.failed_reads)
        self.bus.unresponsive = set()
        self.sampler.stop.requested = True
        with self.assertRaises(csv_check.Stop):
            self.sampler.read('rest')

    def test_the_report_includes_the_motion_phase(self):
        # the motion readings are labelled with the move, the report looks them up by phase alone
        self.sampler.start_phase()
        self.sampler.capture('rest', '', 0.05)
        self.set_voltages({4: 103, 5: 102, 6: 101})
        self.sampler.start_phase()
        for _ in range(5):
            self.sampler.read('motion', 'elbow_out-reach')
        for _ in range(5):
            self.sampler.read('motion', 'reach-elbow_out')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            csv_check.print_report(self.sampler, self.config.models, [('rest', '', 0.05), ('motion', '', 0.1)])
        text = output.getvalue()
        self.assertIn('motion: 0.1 s, 10 samples per servo', text)
        self.assertIn('grows from 0.4 V at rest to 1.5 V during motion', text)
        self.assertIn('largest hop is servo 3 to 4 with 1.0 V', text)

    def test_the_rest_phase_warns_when_the_torque_is_on(self):
        self.bus.memory[2][cfg.ADDR_TORQUE_ENABLE] = 1
        with self.assertLogs(csv_check.logger, 'WARNING') as logs:
            csv_check.run_rest(self.sampler, 0.05)
        self.assertIn('servos [2] are holding', logs.output[0])
        self.assertTrue(self.sampler.stats('rest'))
