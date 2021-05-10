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

# Data Byte Length
LEN_LED_RED             = 1
LEN_PROFILE_ACCELERATION= 4
LEN_PROFILE_VELOCITY    = 4
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4
LEN_PID                 = 2
LEN_GOAL_CURRENT        = 2

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting
BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
MAX_POSITION                = 4095
MIN_POSITION                = 0
