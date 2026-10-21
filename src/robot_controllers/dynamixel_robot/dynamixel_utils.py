import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from types import ModuleType
from typing import Any

import dynamixel_sdk as dynamixel
import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

# usb (vendor id, product id) of the adapters for dynamixel servos
DYNAMIXEL_ADAPTERS = {
    (0x0403, 0x6014): 'U2D2',
}


class DynamixelError(RuntimeError):
    """raised when the dynamixel servos cannot be set up or controlled"""


def set_torque(packet_handler: Any, port_handler: Any, config: ModuleType, servo_id: int, enable: bool) -> bool:
    """Enable or disable the torque of a single servo, returns True on success"""
    if enable:
        val = config.TORQUE_ENABLE
    else:
        val = config.TORQUE_DISABLE

    comm_result, error = packet_handler.write1ByteTxRx(port_handler, servo_id, config.ADDR_TORQUE_ENABLE, val)
    if not check_result(packet_handler, servo_id, comm_result, error):
        return False
    if enable:
        logger.info("Dynamixel#%d has been successfully connected", servo_id)
    return True


def write_1_byte(packet_handler: Any, port_handler: Any, address: int, servo_id: int, mode: int) -> bool:
    """Write a single byte to a servo, returns True on success"""
    comm_result, error = packet_handler.write1ByteTxRx(port_handler, servo_id, address, mode)
    return check_result(packet_handler, servo_id, comm_result, error)


def read_1_byte(packet_handler: Any, port_handler: Any, address: int, servo_id: int) -> int | None:
    """Read a single byte from a servo, returns None on failure"""
    value, comm_result, error = packet_handler.read1ByteTxRx(port_handler, servo_id, address)
    if not check_result(packet_handler, servo_id, comm_result, error):
        return None
    return value


def check_result(packet_handler: Any, servo_id: int, comm_result: int, error: int = 0) -> bool:
    """log the communication or servo error if there is one, returns True if there was none"""
    if comm_result != dynamixel.COMM_SUCCESS:
        logger.warning("[ID:%03d] %s", servo_id, packet_handler.getTxRxResult(comm_result))
        return False
    if error != 0:
        logger.warning("[ID:%03d] %s", servo_id, packet_handler.getRxPacketError(error))
        return False
    return True


def to_bytes(value: float, length: int) -> list[int]:
    """
    convert a (possibly negative) value to the little endian byte list the servos expect
    :raises ValueError: if the value fits neither as signed nor as unsigned number in this many bytes
    """
    bits = 8 * length
    number = int(value)
    if not -(1 << (bits - 1)) <= number < 1 << bits:
        raise ValueError('{} does not fit in {} bytes'.format(value, length))
    return list((number & ((1 << bits) - 1)).to_bytes(length, 'little'))


def to_signed(value: int, length: int) -> int:
    """the sdk returns all values as unsigned, convert it back to a signed value"""
    bits = 8 * length
    if value >= 1 << (bits - 1):
        return value - (1 << bits)
    return value


class BlockingPortHandler(dynamixel.PortHandler):
    """dynamixel port handler that waits for responses without busy waiting

    The sdk opens the serial port without a read timeout and polls it in a loop while waiting for a response,
    which keeps a cpu core (and the GIL) busy. With a read timeout the read blocks until the requested bytes have
    arrived, so the response is handled just as fast, and only a missing response waits for the timeout.

    The packet timeouts are based on the actual latency timer of the usb adapter and use a high resolution monotonic
    clock, the sdk assumes a latency timer of 16ms and uses the wall clock, which jumps when the system time is
    synchronized.
    """

    # how long a single read waits, the sdk checks the packet timeout in between reads
    READ_TIMEOUT = 0.001  # seconds
    # the sdk timeout doesn't include the return delay time of the servos (500us each by default), with a low
    # latency timer a sync read of a few servos takes longer than the sdk timeout
    MIN_PACKET_TIMEOUT = 10.0  # milliseconds

    def __init__(self, port_name: str, latency_timer: float = dynamixel.port_handler.LATENCY_TIMER) -> None:
        """
        :param port_name: the serial port the usb adapter is connected to
        :param latency_timer: the latency timer of the usb adapter in milliseconds
        """
        super().__init__(port_name)
        self.latency_timer = latency_timer

    def setupPort(self, cflag_baud: int) -> bool:
        result = super().setupPort(cflag_baud)
        # setting the timeout reconfigures the port, so it is only done once when the port is opened
        self.ser.timeout = self.READ_TIMEOUT
        return result

    def setPacketTimeout(self, packet_length: int) -> None:
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = max((self.tx_time_per_byte * packet_length) + (self.latency_timer * 2.0) + 2.0,
                                  self.MIN_PACKET_TIMEOUT)

    def getCurrentTime(self) -> float:
        """the current time in milliseconds

        perf_counter is used because time.monotonic only has a resolution of 15.6ms on windows, which is longer than
        the packet timeouts and makes packets time out too early.
        """
        return time.perf_counter() * 1000.0


def get_latency_timer(port: str) -> int | None:
    """returns the latency timer of the usb adapter on the given port in milliseconds, or None if it is unknown"""
    try:
        if sys.platform.startswith('linux'):
            return _get_linux_latency_timer(port)
        if sys.platform == 'win32':
            return _get_windows_latency_timer(port)
    except OSError as e:
        logger.debug("could not read the latency timer of %s: %s", port, e)
    return None


def _get_linux_latency_timer(port: str, sysfs_root: str = '/sys/bus/usb-serial/devices') -> int:
    device = os.path.basename(os.path.realpath(port))
    with open(os.path.join(sysfs_root, device, 'latency_timer')) as latency_file:
        return int(latency_file.read().strip())


def _get_windows_latency_timer(port: str) -> int | None:
    """the ftdi driver stores the latency timer per device in the registry, next to the com port name"""
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Enum\FTDIBUS') as ftdi_bus:
        for device in _registry_subkeys(ftdi_bus):
            with winreg.OpenKey(ftdi_bus, device) as device_key:
                for instance in _registry_subkeys(device_key):
                    try:
                        with winreg.OpenKey(device_key, instance + r'\Device Parameters') as parameters:
                            port_name, _ = winreg.QueryValueEx(parameters, 'PortName')
                            if port_name.upper() == port.upper():
                                latency_timer, _ = winreg.QueryValueEx(parameters, 'LatencyTimer')
                                return int(latency_timer)
                    except OSError:
                        continue
    return None


def _registry_subkeys(key: Any) -> Iterator[str]:
    import winreg

    index = 0
    while True:
        try:
            yield winreg.EnumKey(key, index)
        except OSError:
            return
        index += 1


def get_port_handler(port: str, baud_rate: int) -> BlockingPortHandler:
    latency_timer = get_latency_timer(port)
    if latency_timer is None:
        latency_timer = dynamixel.port_handler.LATENCY_TIMER
        logger.warning("could not determine the latency timer of the usb adapter on %s, assuming %dms",
                       port, latency_timer)
    elif latency_timer > 1:
        logger.warning("the latency timer of the usb adapter on %s is %dms, every read from the servos waits for "
                       "it. Set it to 1ms for faster communication", port, latency_timer)

    port_handler = BlockingPortHandler(port, latency_timer)
    if not port_handler.openPort():
        raise DynamixelError("failed to open port %s" % port)
    logger.info("Succeeded to open the port")
    if not port_handler.setBaudRate(baud_rate):
        raise DynamixelError("failed to set the baudrate of port %s to %d" % (port, baud_rate))
    logger.info("Succeeded to change the baudrate")
    return port_handler


def list_dynamixel_adapters() -> list[tuple[str, str]]:
    """returns a list of (port, adapter name) of the connected usb adapters for dynamixel servos

    The adapters are recognised by their usb vendor and product id, other serial ports are never opened. Opening
    the port of e.g. an arduino resets it.
    """
    adapters = []
    for port in serial.tools.list_ports.comports():
        name = DYNAMIXEL_ADAPTERS.get((port.vid, port.pid))
        if name is not None:
            adapters.append((port.device, name))
    return sorted(adapters)


def ping_servos(port: str, config: ModuleType, servo_ids: Iterable[int]) -> dict[int, int]:
    """returns a mapping of servo id to model number of the servos that respond on the given port

    raises a serial.SerialException when the port can't be opened, e.g. because it is in use by another program
    """
    port_handler = get_port_handler(port, config.BAUDRATE)
    try:
        packet_handler = dynamixel.PacketHandler(config.PROTOCOL_VERSION)
        models = {}
        for servo_id in servo_ids:
            model_number, comm_result, error = packet_handler.ping(port_handler, servo_id)
            if comm_result == dynamixel.COMM_SUCCESS and error == 0:
                models[servo_id] = model_number
        return models
    finally:
        port_handler.closePort()


def find_dynamixel_port(config: ModuleType, servo_ids: Sequence[int]) -> str:
    """returns the port of the usb adapter on which at least one of the servos responds"""
    adapters = list_dynamixel_adapters()
    if not adapters:
        ports = [port.device for port in serial.tools.list_ports.comports()]
        raise DynamixelError("no dynamixel usb adapter found, the connected serial ports are: %s" % ports)

    for port, name in adapters:
        try:
            if ping_servos(port, config, servo_ids):
                logger.info("found the dynamixel servos on %s (%s)", port, name)
                return port
            logger.warning("none of the servos %s respond on %s (%s), is the power on?", list(servo_ids), port, name)
        except serial.SerialException as e:
            logger.warning("could not open %s (%s), is it in use by another program? %s", port, name, e)

    raise DynamixelError("none of the servos %s respond on the dynamixel usb adapters %s"
                         % (list(servo_ids), [port for port, _ in adapters]))


def setup_dynamixel_handlers(port: str | None, config: ModuleType,
                             servo_ids: Sequence[int]) -> tuple[BlockingPortHandler, Any]:
    """Initialize and return the dynamixel port and packet handler for reading and writing to and from servos

    :param port: the serial port of the usb adapter, or None to find the port on which the servos respond
    :param config: a python file containing dynamixel config values, see dynamixel_x_config.py
    :param servo_ids: the servo ids to look for when finding the port
    """
    if port is None:
        port = find_dynamixel_port(config, servo_ids)
    port_handler = get_port_handler(port, config.BAUDRATE)
    packet_handler = dynamixel.PacketHandler(config.PROTOCOL_VERSION)
    return port_handler, packet_handler
