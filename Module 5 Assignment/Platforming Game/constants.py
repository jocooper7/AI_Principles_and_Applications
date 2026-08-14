# ---------------------------
# Screen settings
# ---------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TITLE = "Platformer Proof of Concept"

# ---------------------------
# Colors (RGB)
# ---------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLUE = (30, 30, 200)
YELLOW = (230, 220, 40)
GRAY = (100, 100, 100)
BROWN = (101, 67, 33)
SKY_BLUE = (135, 206, 235)

# ---------------------------
# Player settings
# ---------------------------
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
PLAYER_SPEED = 5
JUMP_STRENGTH = -15
GRAVITY = 0.8
MAX_FALL_SPEED = 20

# ---------------------------
# Life / health settings
# ---------------------------
MAX_LIFE = 4
STARTING_LIFE = 4
INVINCIBILITY_DURATION_MS = 1500

# Slow blink while invincible: player alternates visible/hidden
# every BLINK_INTERVAL_MS milliseconds.
BLINK_INTERVAL_MS = 200

# ---------------------------
# Knockback settings
# ---------------------------
KNOCKBACK_X = 6
KNOCKBACK_Y = -8

# ---------------------------
# Platform settings
# ---------------------------
PLATFORM_COLOR = GREEN
GROUND_HEIGHT = 40

# ---------------------------
# Boundary wall settings
# ---------------------------
WALL_WIDTH = 20
WALL_COLOR = BROWN

# ---------------------------
# Hazard settings
# ---------------------------
HAZARD_COLOR = GRAY
HAZARD_SIZE = 24
HAZARD_SPEED = 2

# ---------------------------
# Goal / Crown settings
# ---------------------------
CROWN_SIZE = 20
CROWN_COLOR = YELLOW

# ---------------------------
# Level / scoring settings
# ---------------------------
LEVEL_TIME_LIMIT_SECONDS = 30
INTRO_MESSAGE_DURATION_MS = 2500
INTRO_MESSAGE_TEXT = "Reach the Gold Crown"
GOOD_JOB_TEXT = "Good Job!"
TIME_UP_TEXT = "Time's Up!"
RETRY_PROMPT_TEXT = "Press R to Retry"
EXIT_PROMPT_TEXT = "Press ESC to Exit"

# ---------------------------
# UI settings
# ---------------------------
UI_FONT_SIZE = 28
UI_BIG_FONT_SIZE = 48
UI_MARGIN = 10
UI_TEXT_COLOR = BLACK
