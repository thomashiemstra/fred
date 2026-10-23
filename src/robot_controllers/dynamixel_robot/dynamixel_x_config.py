# settings for the x-series dynamixel servos

# Control table address
ADDR_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_LED_RED            = 65
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY   = 112
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
ADDR_P                  = 84
ADDR_I                  = 82
ADDR_D                  = 80
ADDR_GOAL_CURRENT       = 102
ADDR_OPERATING_MODE     = 11
ADDR_HOMING_OFFSET      = 20  # eeprom, added by the servo to every position it reports
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_PRESENT_CURRENT    = 126
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_PWM_LIMIT          = 36   # eeprom, the most pwm the servo outputs, 885 is 100%
ADDR_FEEDFORWARD_2ND_GAIN = 88  # acceleration feedforward of the position controller, gain / 4
ADDR_FEEDFORWARD_1ST_GAIN = 90  # velocity feedforward of the position controller, gain / 4
ADDR_GOAL_PWM           = 100  # in position mode a limit on the pwm output, -PWM limit to PWM limit
ADDR_REALTIME_TICK      = 120  # ms counter of the servo, wraps at 32767
ADDR_MOVING             = 122
ADDR_MOVING_STATUS      = 123
ADDR_PRESENT_PWM        = 124
ADDR_PRESENT_VELOCITY   = 128
ADDR_VELOCITY_TRAJECTORY = 136  # the velocity the profile of the servo asks for
ADDR_POSITION_TRAJECTORY = 140  # the position the profile of the servo asks for, the setpoint of the position loop

# Data Byte Length
LEN_LED_RED             = 1
LEN_PROFILE_ACCELERATION= 4
LEN_PROFILE_VELOCITY    = 4
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4
LEN_PID                 = 2
LEN_GOAL_CURRENT        = 2
LEN_OPERATING_MODE      = 1
LEN_HOMING_OFFSET       = 4
LEN_HARDWARE_ERROR_STATUS = 1
LEN_PRESENT_CURRENT     = 2
LEN_PRESENT_INPUT_VOLTAGE = 2
LEN_PRESENT_TEMPERATURE = 1
LEN_PWM_LIMIT           = 2
LEN_FEEDFORWARD_GAIN    = 2
LEN_GOAL_PWM            = 2
LEN_REALTIME_TICK       = 2
LEN_MOVING              = 1
LEN_MOVING_STATUS       = 1
LEN_PRESENT_PWM         = 2
LEN_PRESENT_VELOCITY    = 4
LEN_VELOCITY_TRAJECTORY = 4
LEN_POSITION_TRAJECTORY = 4
LEN_STATE_BLOCK         = 27  # addresses 120 to 146, read in one packet by ServoHandler.read_state

# the position and feedforward gains are 0 to 16383, a servo drops a larger value silently in a sync write
MAX_GAIN                = 16383
MAX_PWM                 = 885  # 100% duty, 0.113% per unit
# bits of the moving status register
MOVING_STATUS_IN_POSITION = 1 << 0
MOVING_STATUS_PROFILE_ONGOING = 1 << 1
MOVING_STATUS_FOLLOWING_ERROR = 1 << 3  # the servo doesn't follow its trajectory

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting
BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
MAX_POSITION                = 4095
MIN_POSITION                = 0

# bits of the hardware error status, a servo with a hardware error disables its torque until it is rebooted
HARDWARE_ERRORS = {
    0: 'input voltage error',
    2: 'overheating',
    3: 'motor encoder error',
    4: 'electrical shock',
    5: 'overload',
}
