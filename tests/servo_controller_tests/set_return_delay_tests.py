import contextlib
import io
import unittest
from unittest import mock

import dynamixel_sdk as dxl
import serial

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot import set_return_delay


class FakePacketHandler:
    """a bus with servos: {id: {'model': n, 'torque': 0/1, 'delay': n}}, nothing touches hardware"""

    def __init__(self, servos, fail_write=(), ignore_write=(), fail_read=()):
        self.servos = servos
        self.fail_write = set(fail_write)  # ids that answer a write with an error
        self.ignore_write = set(ignore_write)  # ids that acknowledge a write but keep their value
        self.fail_read = set(fail_read)  # ids that answer the ping but not a read
        self.pings = []
        self.writes = []

    def getTxRxResult(self, result):
        return 'comm result %d' % result

    def getRxPacketError(self, error):
        return 'packet error %d' % error

    def ping(self, port, servo_id):
        self.pings.append(servo_id)
        if servo_id in self.servos:
            return self.servos[servo_id]['model'], dxl.COMM_SUCCESS, 0
        return 0, dxl.COMM_RX_TIMEOUT, 0

    def read1ByteTxRx(self, port, servo_id, address):
        if servo_id in self.fail_read:
            return 0, dxl.COMM_RX_TIMEOUT, 0
        servo = self.servos[servo_id]
        return servo['torque'] if address == cfg.ADDR_TORQUE_ENABLE else servo['delay'], dxl.COMM_SUCCESS, 0

    def write1ByteTxRx(self, port, servo_id, address, value):
        self.writes.append((servo_id, address, value))
        servo = self.servos[servo_id]
        if servo_id in self.fail_write or servo['torque']:
            return dxl.COMM_SUCCESS, 0x07
        if servo_id not in self.ignore_write:
            servo['delay' if address == cfg.ADDR_RETURN_DELAY_TIME else 'torque'] = value
        return dxl.COMM_SUCCESS, 0


class SetReturnDelayTests(unittest.TestCase):

    def run_main(self, argv, buses=None, answer='yes', open_error=None):
        """
        run main with the fake adapters, buses is a list of FakePacketHandler, one per adapter, returns
        (exit code, output)
        """
        buses = buses if buses is not None else []
        ports = [mock.Mock(name='port%d' % i) for i in range(len(buses))]
        adapters = [('FAKE%d' % i, 'U2D2') for i in range(len(buses))]
        get_port_handler = mock.Mock(side_effect=open_error or ports)
        self.ports = ports
        self.get_port_handler = get_port_handler
        self.input = mock.Mock(return_value=answer)
        with mock.patch.object(set_return_delay, 'list_dynamixel_adapters', return_value=adapters), \
                mock.patch.object(set_return_delay, 'get_port_handler', get_port_handler), \
                mock.patch.object(set_return_delay.dynamixel, 'PacketHandler', side_effect=buses), \
                mock.patch('builtins.input', self.input), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = set_return_delay.main(argv)
        return code, out.getvalue()

    @staticmethod
    def servo(delay=250, torque=0, model=1080):
        return {'model': model, 'torque': torque, 'delay': delay}

    def test_default_value_is_zero(self):
        bus = FakePacketHandler({1: self.servo(), 2: self.servo()})
        code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(0, code)
        self.assertEqual([(1, cfg.ADDR_RETURN_DELAY_TIME, 0), (2, cfg.ADDR_RETURN_DELAY_TIME, 0)], bus.writes)
        self.assertIn('250 (500 us) -> 0 (0 us)', out)
        self.assertIn('XC430-W240', out)

    def test_given_value_is_written(self):
        bus = FakePacketHandler({1: self.servo()})
        code, _ = self.run_main(['--value', '100', '--yes'], [bus])
        self.assertEqual(0, code)
        self.assertEqual([(1, cfg.ADDR_RETURN_DELAY_TIME, 100)], bus.writes)
        self.assertEqual(100, bus.servos[1]['delay'])

    def test_value_out_of_range_is_rejected(self):
        for value in ('255', '-1', '1000', 'abc'):
            with self.subTest(value=value):
                bus = FakePacketHandler({1: self.servo()})
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    self.run_main(['--value', value, '--yes'], [bus])
                self.assertEqual(2, raised.exception.code)
                self.assertEqual([], bus.writes)
        for value in ('0', '254'):  # the edges are fine
            with self.subTest(value=value):
                bus = FakePacketHandler({1: self.servo()})
                self.assertEqual(0, self.run_main(['--value', value, '--yes'], [bus])[0])

    def test_only_servos_that_answer_are_touched(self):
        bus = FakePacketHandler({2: self.servo(), 5: self.servo()})
        code, _ = self.run_main(['--yes', '--max-id', '6'], [bus])
        self.assertEqual(0, code)
        self.assertEqual([1, 2, 3, 4, 5, 6], bus.pings)
        self.assertEqual([2, 5], [servo_id for servo_id, _, _ in bus.writes])

    def test_max_id_default_is_20(self):
        bus = FakePacketHandler({1: self.servo()})
        self.run_main(['--dry-run'], [bus])
        self.assertEqual(list(range(1, 21)), bus.pings)

    def test_servo_already_at_the_value_is_not_written(self):
        bus = FakePacketHandler({1: self.servo(delay=0), 2: self.servo()})
        code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(0, code)
        self.assertEqual([2], [servo_id for servo_id, _, _ in bus.writes])
        self.assertIn('already 0', out)

    def test_all_servos_at_the_value_needs_no_prompt(self):
        bus = FakePacketHandler({1: self.servo(delay=0)})
        code, out = self.run_main([], [bus])
        self.assertEqual(0, code)
        self.input.assert_not_called()
        self.assertEqual([], bus.writes)
        self.assertIn('nothing to write', out)

    def test_servo_with_torque_on_is_skipped_and_exit_code_is_1(self):
        bus = FakePacketHandler({1: self.servo(torque=1), 2: self.servo()})
        code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(1, code)
        self.assertEqual([2], [servo_id for servo_id, _, _ in bus.writes])
        self.assertEqual(250, bus.servos[1]['delay'])
        self.assertIn('disable the torque first', out)

    def test_only_servo_has_torque_on(self):
        bus = FakePacketHandler({1: self.servo(torque=1)})
        code, _ = self.run_main(['--yes'], [bus])
        self.assertEqual(1, code)
        self.assertEqual([], bus.writes)

    def test_dry_run_writes_nothing(self):
        bus = FakePacketHandler({1: self.servo(), 2: self.servo(torque=1)})
        code, out = self.run_main(['--dry-run'], [bus])
        self.assertEqual(0, code)
        self.assertEqual([], bus.writes)
        self.input.assert_not_called()
        self.assertIn('250 (500 us)', out)
        self.assertIn('torque on', out)

    def test_answering_no_writes_nothing(self):
        for answer in ('no', '', 'y'):
            with self.subTest(answer=answer):
                bus = FakePacketHandler({1: self.servo()})
                code, out = self.run_main([], [bus], answer=answer)
                self.assertEqual(0, code)
                self.assertEqual([], bus.writes)
                self.assertIn('not writing', out)
                self.input.assert_called_once()

    def test_answering_yes_writes(self):
        bus = FakePacketHandler({1: self.servo()})
        code, _ = self.run_main([], [bus], answer=' Yes ')
        self.assertEqual(0, code)
        self.assertEqual(1, len(bus.writes))

    def test_failed_write_gives_exit_code_1(self):
        bus = FakePacketHandler({1: self.servo(), 2: self.servo()}, fail_write=[1])
        with self.assertLogs('src.robot_controllers.dynamixel_robot.dynamixel_utils', 'WARNING'):
            code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(1, code)
        self.assertIn('writing failed', out)
        self.assertEqual(0, bus.servos[2]['delay'])  # the other servo is still done

    def test_read_back_mismatch_gives_exit_code_1(self):
        bus = FakePacketHandler({1: self.servo()}, ignore_write=[1])
        code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(1, code)
        self.assertIn('read back 250 (500 us) instead of 0', out)

    def test_unreadable_servo_is_skipped_and_exit_code_is_1(self):
        bus = FakePacketHandler({1: self.servo(), 2: self.servo()}, fail_read=[1])
        with self.assertLogs('src.robot_controllers.dynamixel_robot.dynamixel_utils', 'WARNING'):
            code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(1, code)
        self.assertEqual([2], [servo_id for servo_id, _, _ in bus.writes])
        self.assertIn('could not read', out)

    def test_torque_register_is_never_written(self):
        bus = FakePacketHandler({1: self.servo(), 2: self.servo(torque=1), 3: self.servo(delay=0)}, ignore_write=[1])
        self.run_main(['--yes'], [bus])
        self.run_main(['--value', '10', '--yes'], [bus])
        self.assertTrue(bus.writes)
        self.assertNotIn(cfg.ADDR_TORQUE_ENABLE, [address for _, address, _ in bus.writes])
        self.assertEqual({cfg.ADDR_RETURN_DELAY_TIME}, {address for _, address, _ in bus.writes})

    def test_no_adapter_returns_1(self):
        code, out = self.run_main([], [])
        self.assertEqual(1, code)
        self.assertIn('no dynamixel usb adapter', out)
        self.get_port_handler.assert_not_called()

    def test_no_servo_answering_returns_1_and_closes_the_port(self):
        bus = FakePacketHandler({})
        code, out = self.run_main(['--yes'], [bus])
        self.assertEqual(1, code)
        self.assertIn('no servo', out)
        self.ports[0].closePort.assert_called_once()

    def test_port_that_cannot_be_opened_is_skipped(self):
        bus = FakePacketHandler({1: self.servo()})
        port = mock.Mock()
        code, out = self.run_main(['--yes'], [bus, bus], open_error=[serial.SerialException('busy'), port])
        self.assertEqual(0, code)
        self.assertIn('could not open the port', out)
        self.assertEqual(1, len(bus.writes))
        port.closePort.assert_called_once()

    def test_port_that_cannot_be_opened_alone_returns_1(self):
        bus = FakePacketHandler({1: self.servo()})
        code, _ = self.run_main(['--yes'], [bus], open_error=[serial.SerialException('busy')])
        self.assertEqual(1, code)

    def test_servos_of_two_adapters(self):
        buses = [FakePacketHandler({1: self.servo()}), FakePacketHandler({2: self.servo()})]
        code, _ = self.run_main(['--yes'], buses)
        self.assertEqual(0, code)
        self.assertEqual([[(1, 9, 0)], [(2, 9, 0)]], [bus.writes for bus in buses])
        for port in self.ports:
            port.closePort.assert_called_once()

    def test_port_is_closed_after_a_normal_run(self):
        bus = FakePacketHandler({1: self.servo()})
        self.run_main(['--yes'], [bus])
        self.ports[0].closePort.assert_called_once()

    def test_port_is_closed_when_the_prompt_is_refused(self):
        bus = FakePacketHandler({1: self.servo()})
        self.run_main([], [bus], answer='no')
        self.ports[0].closePort.assert_called_once()

    def test_port_is_closed_when_something_fails(self):
        bus = FakePacketHandler({1: self.servo()})
        bus.write1ByteTxRx = mock.Mock(side_effect=RuntimeError('boom'))
        with self.assertRaises(RuntimeError):
            self.run_main(['--yes'], [bus])
        self.ports[0].closePort.assert_called_once()

    def test_all_ports_are_closed_when_a_later_adapter_fails(self):
        first = FakePacketHandler({1: self.servo()})
        second = FakePacketHandler({})
        second.ping = mock.Mock(side_effect=RuntimeError('boom'))
        with self.assertRaises(RuntimeError):
            self.run_main(['--yes'], [first, second])
        for port in self.ports:
            port.closePort.assert_called_once()
        self.assertEqual([], first.writes)


if __name__ == '__main__':
    unittest.main()
