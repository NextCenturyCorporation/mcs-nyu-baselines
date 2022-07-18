#!/usr/bin/env python
# coding: utf-8

# In[1]:


STICKY_MOVE_AHEAD = [
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead'
]

STICKY_MOVE_AHEAD_2 = [
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
]
STICKY_MOVE_AHEAD_3 = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',]

INITIAL_MOVE_RIGHT_SEQ = [
    'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight',
    'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight'
]
INITIAL_MOVE_LEFT_SEQ = [
    'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft',
    'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft'
]
INITIAL_ROTATE_RIGHT_SEQ = ['RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                           'RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                           'RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                           'RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                           'RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                           'RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                           'RotateRight','RotateRight','RotateRight','RotateRight','LookDown']
OCCLUDER_AHEAD_SEQ = [
    'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveAhead',
    'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'RotateLeft', 'RotateLeft',
    'RotateLeft', 'RotateLeft', 'RotateLeft', 'RotateLeft', 'LookDown'
]
PICK_UP_SEQUENCE = [
    'PickupObject'
]
PUSH_OBJ_SEQUENCE = [
    'PushObject'
]
PULL_OBJ_SEQUENCE = [
    'PullObject'
]
LOOK_DOWN_SEQ_3 = ['LookDown','LookDown','LookDown']
LOOK_UP_SEQ_3 = ['LookUp','LookUp','LookUp']
LOOK_UP_SEQ_2 = ['LookUp','LookUp']
LOOK_DOWN_SEQ = ['LookDown','LookDown','LookDown','LookDown','LookDown']
LOOK_UP_SEQ = ['LookUp','LookUp','LookUp','LookUp','LookUp']
INITIAL_MOVE_RIGHT_SEQ_1 = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead','MoveAhead', 'MoveAhead','RotateRight','RotateRight']
#12 moveahead 2 rotate right
INITIAL_MOVE_LEFT_SEQ_1 = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
    'MoveAhead','MoveAhead', 'MoveAhead','RotateLeft','RotateLeft']
#12 moveahead 2 rotate left

MOVE_RIGHT_SEQ_1 = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead','RotateRight']
MOVE_LEFT_SEQ_1 = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead','RotateLeft']
MOVE_RIGHT_SEQ_3 = ['MoveRight', 'MoveRight', 'MoveRight','RotateRight','RotateRight','RotateRight']

ROTATE_LEFT_SEQ_5 = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft']
ROTATE_LEFT_SEQ_7 = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft']
ROTATE_LEFT_SEQ_9 = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft']
ROTATE_LEFT_SEQ_10 = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft']
ROTATE_RIGHT_SEQ = ['RotateRight','RotateRight','RotateRight','RotateRight']
ROTATE_RIGHT_SEQ_3 = ['RotateRight','RotateRight','RotateRight']
ROTATE_RIGHT_SEQ_5 = ['RotateRight','RotateRight','RotateRight','RotateRight','RotateRight']
ROTATE_RIGHT_SEQ_9 = ['RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','MoveLeft']
ROTATE_RIGHT_SEQ_10 = ['RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight']
ROTATE_LEFT_SEQ = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft']
ROTATE_LEFT_SEQ_2 = ['RotateLeft','RotateLeft']
ROTATE_RIGHT_SEQ_2 = ['RotateRight','RotateRight']

RAMP_SEQ = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft', 'RotateLeft','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
           'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
           'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead']
ON_RAMP_SEQ_L = ['MoveRight', 'MoveRight','RotateRight','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead']
ON_RAMP_SEQ_R = ['MoveLeft','MoveLeft','RotateLeft','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead']
LAVA_SEQ_L = ['MoveRight', 'MoveRight','MoveRight','RotateRight','RotateRight']
LAVA_SEQ_R = ['MoveLeft','MoveLeft','MoveLeft','RotateLeft','RotateLeft']

ROTATE_LEFT_AHEAD = ['RotateLeft','MoveAhead','MoveAhead','MoveAhead']
ROTATE_RIGHT_AHEAD = ['RotateRight', 'MoveAhead','MoveAhead','MoveAhead']

ROTATE_LEFT_AHEAD_LEFT = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft',
                         'MoveAhead','MoveAhead','MoveAhead','MoveLeft']

ROTATE_RIGHT_AHEAD_LEFT = ['RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                         'MoveLeft','MoveLeft','MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveAhead']
ROTATE_LEFT_AHEAD_RIGHT = ['RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft','RotateLeft',
                         'MoveAhead','MoveAhead','MoveAhead','MoveRight']
REVERSE_ACTION = ['LookUp','LookUp','LookUp','LookUp','LookUp','RotateRight','RotateRight','RotateRight','RotateRight','RotateRight',
                  'RotateRight','RotateRight','RotateRight','RotateRight','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead',
                 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead']

MOVE_LEFT_SEQ_5 = ['MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveAhead', 'MoveAhead']
MOVE_LEFT_SEQ_3 = ['MoveLeft','MoveLeft','MoveLeft']
MOVE_LEFT_SEQ_10 = ['MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft']
MOVE_RIGHT_SEQ_10 = ['MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight']
MOVE_RIGHT_SEQ_5 = ['MoveRight','MoveRight','MoveRight','MoveRight','MoveRight']
MOVE_LEFT_AHEAD = ['MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  , 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead']
MOVE_LEFT_AHEAD_2 = ['MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                     ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                     ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                  ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                    ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                    ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead'
                    ,'MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead','MoveAhead', 'MoveAhead']
MOVE_BACK_LEFT = ['MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveLeft','MoveLeft','MoveLeft']
MOVE_BACK_RIGHT = ['MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveRight','MoveRight','MoveRight']
MOVE_LEFT_BACK = ['MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveBack','MoveBack','MoveBack']
MOVE_BACK = ['MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveBack','MoveBack', 'MoveRight','MoveRight']

MOVE_AHEAD_RIGHT = ['MoveAhead','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight']

TEST_OBSTRUCT = ['MoveRight','MoveRight', 'MoveLeft','MoveLeft', 'MoveAhead','MoveAhead', 
                 'MoveBack','MoveBack', 'MoveLeft','MoveLeft','MoveRight', 'MoveRight',
                 'MoveBack','MoveBack', 'MoveAhead','MoveAhead']

TEST_OBSTRUCT_1 = ['MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight',
                  'MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveBack','MoveBack','MoveBack','MoveBack',
                  'MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft']
TEST_OBSTRUCT_2 = ['MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft','MoveLeft',
                  'MoveAhead','MoveAhead','MoveAhead','MoveAhead','MoveBack','MoveBack','MoveBack','MoveBack',
                  'MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight','MoveRight']

STICKY_PASS = ['Pass','Pass','Pass','Pass','Pass','Pass','Pass','Pass','Pass','Pass',
              'Pass','Pass','Pass','Pass','Pass','Pass','Pass','Pass','Pass','Pass',]
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ACTION_MOVE_AHEAD = ['MoveAhead']
ACTION_MOVE_LEFT = ['MoveLeft']
ACTION_MOVE_RIGHT = ['MoveRight']
ACTION_MOVE_BACK = ['MoveBack']
ACTION_ROTATE_RIGHT = ['RotateRight']
ACTION_ROTATE_LEFT = ['RotateLeft']
ACTION_PICK_UP_OBJ = ['PickupObject']
ACTION_LOOK_DOWN = ['LookDown']
ACTION_LOOK_UP = ['LookUp']
ACTION_PASS = ['Pass']
INTERACT = ['InteractWithAgent']

OCCLUDER = 'occluder'
NON_AGENT = 'non-agent'
AGENT = 'agent'
SPORTS_BALL = 'ball'
TOOL = 'tool'
RAMPS = 'ramp'

RIGHT_WIDTH = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

SCENE_FILE_PATH = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                   "MCS/scenes/baseline_moving_target_prediction_eval_5/baseline_moving_target_prediction_01_b1.json"
SCENE_FILE_PATH_RAMP = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                         "MCS/scenes/baseline_ramps_eval_5/baseline_ramps_01_k1.json"
SCENE_FILE_PATH_TOOL = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                         "MCS/scenes/baseline_tool_use_eval_5/baseline_tool_use_02_a1.json"
MODEL_WEIGHTS_FILE_PATH_AGENT = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                   "MCS/scenes/baseline_agent_identification/model/best.pt"
MODEL_WEIGHTS_FILE_PATH_MOVING_BALL = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                   "MCS/scenes/baseline_moving_target_prediction_eval_5/model/best.pt"
MODEL_WEIGHTS_FILE_PATH_RAMP = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                   "MCS/scenes/baseline_moving_target_prediction_eval_5/model/best_v11.pt"
MODEL_WEIGHTS_FILE_PATH_TOOL = "/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/"                   "MCS/scenes/baseline_moving_target_prediction_eval_5/model/best_v9.pt"

threshold = 0.40
OCCLUDER_IN_FRONT = False
SCENE_HAS_SOCCER_BALL = False
SCENE_HAS_AGENT = False
SCENE_HAS_LAVA = False
SCENE_HAS_TOOL = False
RAMP_DETECTED = False
MOVE_AHEAD_OBSTRUCTED = False
MOVE_LEFT_OBSTRUCTED = False
MOVE_RIGHT_OBSTRUCTED = False
MOVE_BACK_OBSTRUCTED = False
FIRST_ACTION = True
FIRST = True
LEFT_RIGHT_CUSHION = 30
TOP_BOTTOM_CUSHION = 30
CENTER_TOP_CUSHION = 70
CENTER_LEFT_CUSHION = 70
ROTATE_COUNT = 0 
ROAM_COUNT =0
BRUTE_COUNT = 0
SCAN_BALL = False
SCAN_RAMP = False
DIRECTION = ''
LEVEL_BALL_SCANNED = False
LEVEL_RAMP_SCANNED = False
REVERSE = False
SCAN_LAVA = False
SCAN_TOOL = False
ROAM = False
TOOL_REACHED =False
CORNER_COUNT = []
RAMP = False
SCAN_EDGE = False
HORIZONTAL_EDGE = False
TEST = False
FRESH_OFF = False
SCAN_RAMP_EDGE = False
MOVE_RIGHT_OBSTRUCTED = False
BRUTE = False
NAVIGATE_TOOL = False
PREVIOUS = []
TOOL_OUT_OF_REACH = False
PULL_ADJ = False
TOOL_REACH_BALL_FOUND = False
PUSH_OR_PULL = False
TOOL_OBSTRUCTED = False
TOOL_BEHIND= False
TOOL_AHEAD= False
TOOL_LEFT= False
TOOL_RIGHT= False
FINAL = False
LAVA = False
NAVIGATE_BALL=False
TEST_TOOL = False
MOVE_TOOL_RIGHT = False
MOVE_TOOL_LEFT = False
TEST_FIRST = False
BALL=False
TEST_TOOL_1=False
TEST_TOOL_2=False
FINAL_COUNT=False
COUNT = 0


# # # # # # # # # # # # # # # # # # # # # #


# In[ ]:




