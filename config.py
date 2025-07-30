# Configuration settings for drowsiness detection

# EAR (Eye Aspect Ratio) threshold
EAR_THRESHOLD = 0.21

# Drowsiness detection settings
DROWSY_TIME_THRESHOLD = 2.0  # seconds

# Video capture settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Model settings
MODEL_PATH = "models/eye_state_model.h5"
USE_CNN_MODEL = False  # Set to True if using CNN model instead of EAR

# Display settings
SHOW_EAR_VALUE = True
SHOW_LANDMARKS = True
SHOW_STATUS = True

# Audio settings
ALARM_FILE = "alarm.wav"
ALARM_DURATION = 3  # seconds

# Colors (BGR format for OpenCV)
COLORS = {
    'GREEN': (0, 255, 0),
    'RED': (0, 0, 255),
    'BLUE': (255, 0, 0),
    'YELLOW': (0, 255, 255),
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0)
}