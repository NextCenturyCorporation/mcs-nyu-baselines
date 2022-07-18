# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

STICKY_MOVE_AHEAD = [
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead'
]
INITIAL_MOVE_RIGHT_SEQ = [
    'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight',
    'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight'
]
INITIAL_MOVE_LEFT_SEQ = [
    'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft',
    'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft'
]
OCCLUDER_AHEAD_SEQ = [
    'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'RotateLeft', 'RotateLeft',
    'RotateLeft', 'RotateLeft', 'RotateLeft', 'RotateLeft', 'LookDown'
]
PICK_UP_SEQUENCE = [
    'PickupObject'
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ACTION_MOVE_AHEAD = ['MoveAhead']
ACTION_MOVE_LEFT = ['MoveLeft']
ACTION_MOVE_RIGHT = ['MoveRight']
ACTION_ROTATE_RIGHT = ['RotateRight']
ACTION_ROTATE_LEFT = ['RotateLeft']
ACTION_PICK_UP_OBJ = ['PickupObject']
ACTION_PASS = ['Pass']
OCCLUDER = 'occluder'
SPORTS_BALL = 'sports ball'
RIGHT_WIDTH = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

SCENE_FILE_PATH = "/Users/Metarya/codebase/NYU/Spring22/MCS " \
                  "Project/baseline_spatial_elimination_eval_5/baseline_spatial_elimination_02_a3.json"
MODEL_WEIGHTS_FILE_PATH = "/Users/Metarya/codebase/MCS/best.pt"
threshold = 0.40
OCCLUDER_IN_FRONT = False
SCENE_HAS_SOCCER_BALL = False
MOVE_AHEAD_OBSTRUCTED = False
FIRST_ACTION = True
LEFT_RIGHT_CUSHION = 20
TOP_BOTTOM_CUSHION = 20
CENTER_TOP_CUSHION = 70
CENTER_LEFT_CUSHION = 70
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

YMAX = 0