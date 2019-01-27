import dynamixel_sdk as dynamixel


def set_torque(packet_handler, port_handler, config, servo_id, enable):
    if enable:
        val = config.TORQUE_ENABLE
    else:
        val = config.TORQUE_DISABLE

    comm_result, error = packet_handler.write1ByteTxRx(port_handler, servo_id, config.ADDR_TORQUE_ENABLE, val)
    if comm_result != dynamixel.COMM_SUCCESS:
        print("%s" % packet_handler.getTxRxResult(comm_result))
    elif error != 0:
        print("%s" % packet_handler.getRxPacketError(error))
    else:
        print("Dynamixel#%d has been successfully connected" % servo_id)


def add_group_write(group_bulk_write, servo_id, address, address_len, value):
    """Add the byte array to the group bulk write for a given servo based on the position"""
    byte_array = [dynamixel.DXL_LOBYTE(dynamixel.DXL_LOWORD(value)),
                  dynamixel.DXL_HIBYTE(dynamixel.DXL_LOWORD(value)),
                  dynamixel.DXL_LOBYTE(dynamixel.DXL_HIWORD(value)),
                  dynamixel.DXL_HIBYTE(dynamixel.DXL_HIWORD(value))]

    res = group_bulk_write.addParam(servo_id, address, address_len, byte_array)
    if not res:
        raise RuntimeError("[ID:%03d] groupBulkWrite add param failed" % servo_id)


def get_port_handler(port, baud_rate):
    port_handler = dynamixel.PortHandler(port)
    if port_handler.openPort():
        print("Succeeded to open the port")
        if port_handler.setBaudRate(baud_rate):
            print("Succeeded to change the baudrate")
            return port_handler
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        quit()


def setup_dynamixel_handlers(port, config):
    """Initialize and return all the dynamixel handlers for reading and writing to and from servos"""
    port_handler = get_port_handler(port, config.BAUDRATE)
    packet_handler = dynamixel.PacketHandler(config.PROTOCOL_VERSION)
    group_bulk_write = dynamixel.GroupBulkWrite(port_handler, packet_handler)
    group_bulk_read = dynamixel.GroupBulkRead(port_handler, packet_handler)
    return port_handler, packet_handler, group_bulk_write, group_bulk_read
