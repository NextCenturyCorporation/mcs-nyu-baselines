from curses.panel import top_panel
from shutil import move
from tracemalloc import stop
from zmq import CONFLATE
import import_ipynb
import constant as const
import copy
import sys
import argparse

import cv2 as cv
import numpy as np
import torch
from PIL import Image
import copy
import glob
import os

# import spatial_scene.constants as const
import machine_common_sense as mcs

##################


# Please read these instructions if you want to continue developing these codes

# The basic idea is to moving along the wall, around the platform(room) to search for ball, (or ramp, if ball not detected)
# Another interesting point
# The AI is transferring between seven modes
# 1. Init: moving backwards into the wall and adjust the direction to make sure you are leaning on the wall exactly
# 2. Find ball: Move couter clockwise on four edges of the platform, if ball found, enter Navigate to Ball mode, otherwise, enter Find Ramp mode
# 3. Find Ramp: Move clockwise until ramp found, enter Navigate to Ramp mode if ramp found, enther Leaving Rmap mode if you are leaving from previous ramp
# 4. Navigate to Ball: moving towards the ball until picked up. If the ball is one the same height with AI or higher than AI, enter Find Ramp mode. If the ball
# is under the height with AI, enter change Ramp mode
# 5. Navigate to Ramp: moving towards the ramp until going up (no guarantee for success, succeed with about 85% probability), keep moving, if losing the track of
# the ramp, just keep moving ahead for many steps until obstructed
# 6. Change Ramp: This mode means that the ball is on the other platform, and you should leave this platform through ramp right now
# 7. Leaving Ramp: This mode means that we are finding ramp and suddenly leave from another ramp, this time AI should wait for some time before finding ramp again in
# case navigate to the same ramp and stuck in unstopping loop.

# The problem existing and suggestions:
# In short run: 
# 1. Ramp model tends to judge some shadow of platforms as ramp, possibly because the model is looking for something with similar shape. (Run whiskey_0008_02)
#  I will share my ramp dataset with Prof Fergus, add more data and train your model on it.
# 2. My codes are having a hard time working with one platform on another platform (up, ramp needed 2 or down ramp needed 2). This is because: 1) change_ramp mode
#  is likely to make AI lean on the higher platform, and therefore consider itself as already changing the platform while actually not; (run whiskey_0008_12) 
#  2) when  AI walk around the platform, it might suddenly go down from ramp (because it always move_back to make sure it leans on the wall) and therefore 
#  leave the platform before finding the ball on the platform.
# 3. Navigate to Ramp algorihtm cannot succeed 100 percent. It might get stuck at the last step on the ramp or fall down from ramp, this is because my model 
#  is insenstive when AI is already on the ramp, so I just let it walk straight ahead. But this is not good, I suggest train another model to teach the AI 
#  how to correct its direction while it is on the ramp.

# In long run:
# use some model of automated driving (especially segmentation) to replace object detection model, so AI can adjust its direction, have a more accurate 
# understanding of its circumstances
##################

ball_threshold = 0.3
ramp_threshold = 0.4
# each rotation = 10 degree. rotate_direction can help correct direction whenever needed
rotate_direction = 0
# this is the signal that tells how many edges AI has already walked through
obstructed_times = 0
mode = "Init"
# used in navigate to ball mode, prevent ball_model being unstable, whenever losing sight of the ball, move ahead for 20 times and check again
first = True
ball_x = 300
ball_y = 200
# used in find ramp, to prevent AI fall into the dead loop by considering itself leaving ramp when it actually not (run whiskey_0017_03)
stop_off_ramp = False

# keep moving back until obstructed, return True if the AI keeps moving backward for 20 steps, then considered leaving from a platform through ramp
def move_back(controller):
    global rotate_direction, obstructed_times
    return_status = "OK"
    times_back = 0
    while return_status != "OBSTRUCTED":
        return_status = controller.step("MoveBack").return_status
        times_back += 1
    return times_back >= 20

# rotate right (or left) and update rotate_direction
def rotate_right(controller):
    global rotate_direction, obstructed_times
    controller.step("RotateRight")
    rotate_direction = (rotate_direction + 1) % 36

def rotate_left(controller):
    global rotate_direction, obstructed_times
    controller.step("RotateLeft")
    rotate_direction = (rotate_direction + 35) % 36

# used in find_ball
def move_couter_clockwise(controller):
    global rotate_direction, obstructed_times
    for _ in range(2):
        controller.step("MoveRight")
    if controller.step("MoveRight").return_status == "OBSTRUCTED":
        for _ in range(9):
            controller.step("RotateLeft")
        obstructed_times += 1
        for _ in range(5):
            controller.step("MoveRight")
    return move_back(controller)

# used in find_ramp, leave_ramp, change_ramp
def move_clockwise(controller):
    global rotate_direction, obstructed_times
    for _ in range(2):
        controller.step("MoveLeft")
    if controller.step("MoveLeft").return_status == "OBSTRUCTED":
        for _ in range(9):
            controller.step("RotateRight")
        obstructed_times += 1
        for _ in range(5):
            controller.step("MoveLeft")
    return move_back(controller)

# threshold helps to set up ball_threshold to self_defined value
def find_item(img, model, item, threshold = 0):
    global ball_threshold, ramp_threshold, rotate_direction, obstructed_times
    predictions = model(img, size=640)
    loc_result = predictions.pandas().xyxy[0]
    print("Loc Result:",loc_result)
    if threshold != 0:
        ball_threshold = threshold
    found = False
    top_left = (0, 0)
    bottom_right = (0, 0)
    cur_prob = 0
    for idx, res in loc_result.iterrows():
        if (res['name'] == item) and ((item == 'Ball' and (res['confidence'] > ball_threshold)) or (item == 'ramp' and (res['confidence'] > ramp_threshold))) and (res['confidence'] > cur_prob):
            found = True
            top_left = (int(res['xmin']), int(res['ymin']))
            bottom_right = (int(res['xmax']), int(res['ymax']))
            cur_prob = res['confidence']
    if threshold != 0:
        ball_threshold = 0.3
    return found, top_left, bottom_right

# Used to adjust AI's direction during navigation, but notice that if we are finding ramp, moving left or right will be added in the steps
def navigate(controller, top_left, bottom_right, cw, ch, ball = True):
    global rotate_direction, obstructed_times
    left_border = top_left[0]
    right_border = bottom_right[0]
    if (cw < left_border + 50 and not ball) or (cw < left_border and ball):
        print(left_border, right_border, "1")
        if not ball:
            controller.step("MoveRight")
            controller.step("MoveRight")
            controller.step("MoveRight")
        rotate_right(controller)
    if (cw > right_border - 50 and not ball) or (cw > right_border and ball):
        print(left_border, right_border, "2")
        if not ball:
            controller.step("MoveLeft")
            controller.step("MoveLeft")
            controller.step("MoveLeft")
        rotate_left(controller)

# use rotate_direction to make sure its direction being either 0, 90, 180, 270 degrees with original direction
def simple_init(controller):
    global rotate_direction
    print(rotate_direction)
    while rotate_direction % 9 != 0:
        rotate_left(controller)
        print(rotate_direction)
    for _ in range(15):
        controller.step("MoveRight")
    move_back(controller)

# Init mode: look down and rotate around to find ball. If found, directly navigate to it, otherwise, starting to find ball
def init(controller, model):
    global rotate_direction, obstructed_times
    for i in range(2):
        controller.step("LookDown")
    for i in range(12):
        rotate_left(controller)
        rotate_left(controller)
        rotate_left(controller)
        img = np.array(controller.step("Pass").image_list[0])
        found, _, _ = find_item(img, model, 'Ball')
        if found:
            for i in range(2):
                controller.step("LookUp")
            return "Navigate to Ball"
    for i in range(2):
        controller.step("LookUp")
    simple_init(controller)
    move_back(controller)
    return "Find Ball"

# ball found, try to pick up; if unable to pick up, then use the distance between AI and ball to decide how many steps shold AI try to repeat this process again
def decide_move_ahead_step(controller, top_left, bottom_right, cw, ch):
    global first, ball_x, ball_y
    irst = True
    ball_x = (top_left[0] + bottom_right[0])/2
    ball_y = (top_left[1] + bottom_right[1])/2
    params = {"objectImageCoordsX": ball_x , "objectImageCoordsY": ball_y}
    print((top_left[0] + bottom_right[0])/2, (top_left[1] + bottom_right[1])/2)
    if controller.step("PickupObject", **params).return_status == "SUCCESSFUL":
        print("INFO: Picked Up soccer ball. Ending scene! :)")
        controller.end_scene()
        exit(0)
    navigate(controller, top_left, bottom_right, cw, ch)
    move_ahead_step = 5
    if (bottom_right[0] - top_left[0])*(bottom_right[1] - top_left[1]) >= 4500:
        move_ahead_step = 1
    return move_ahead_step

# Navigate to ball mode
def navigate_to_ball(controller, model):
    global rotate_direction, obstructed_times, first, ball_x, ball_y
    img = np.array(controller.step("Pass").image_list[0])
    found, top_left, bottom_right = find_item(img, model, 'Ball', 0.1)
    cw = int(img.shape[1] / 2)
    ch = int(img.shape[0] / 2)
    # the ball is found
    if found:
        move_ahead_step = decide_move_ahead_step(controller, top_left, bottom_right, cw, ch)
        for _ in range(move_ahead_step):
            # obstructed and found the ball meaning the ball is on the other platform with the same height or smaller height, change the ramp
            if controller.step("MoveAhead").return_status == "OBSTRUCTED":
                for _ in range(2):
                    controller.step("LookDown")
                found, _, _ = find_item(img, model, 'Ball')
                for _ in range(2):
                    controller.step("LookUp")
                simple_init(controller)
                if found:
                    # the place you obstructed must be the edge of a platform, turn around and lean on it
                    for _ in range(18):
                        rotate_left(controller)
                    return "Change Ramp"
                else:
            # otherwise, the ball is on higher ramp, find ramp on current platform
                    return "Find Ramp"
        return "Navigate to Ball"
    # sometimes the ball is close to you, or for some other reason, you need to look down to locate it
    else:
        for _ in range(2):
            controller.step("LookDown")
        found, top_left, bottom_right = find_item(img, model, 'Ball')
        if found:
            # reapeat the same steps but this time remember to change your eyesight to horizontal view (controller.step("LookUp"))
            move_ahead_step = decide_move_ahead_step(controller, top_left, bottom_right, cw, ch)
            for _ in range(move_ahead_step):
                if controller.step("MoveAhead").return_status == "OBSTRUCTED":
                    simple_init(controller)
                    for _ in range(2):
                        controller.step("LookUp")
                    for _ in range(18):
                        rotate_left(controller)
                    return "Change Ramp"
            for _ in range(2):
                controller.step("LookUp")
            return "Navigate to Ball"
        else:
            # sometimes the model is not working when the ball is right under AI's foot, this time AI should trust the previous coordinates and 
            # use them to make another try
            params = {"objectImageCoordsX": ball_x , "objectImageCoordsY": ball_y}
            if controller.step("PickupObject", **params).return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)
            # to prevent the temporal bad performance of the model, if this is the first time model cannot detect the ball, keep moving ahead for some time
            # and try again, if the next round the ball is still not detected, then start to find ramp on current platform
            if not first:
                first = True
                while rotate_direction % 9 != 0:
                    rotate_right(controller)
                move_back(controller)
                for _ in range(2):
                    controller.step("LookUp")
                return "Find Ramp"
            else:
                for _ in range(20):
                    controller.step("MoveAhead")
                first = False
                for _ in range(2):
                    controller.step("LookUp")
                return "Navigate to Ball"

# Navigate to Ramp mode
def navigate_to_ramp(controller, model, ball_model):
    global rotate_direction, obstructed_times
    img = np.array(controller.step("Pass").image_list[0])
    found, top_left, bottom_right = find_item(img, model, 'ramp')
    cw = int(img.shape[1] / 2)
    ch = int(img.shape[0] / 2)
    if found:
        navigate(controller, top_left, bottom_right, cw, ch, False)
        for _ in range(8):
            if controller.step("MoveAhead").return_status == "OBSTRUCTED":
                for i in range(9):
                    rotate_left(controller)
                move_back(controller)
                return "Find Ramp"
        return "Navigate to Ramp"
    # I assume the AI has already been on the ramp and heading almost right to the platform, then just let it move ahead, but twist its direction a little bit
    # to prevent getting stuck at the corner (85% success ramp up rate instead of 100% is due to this else)
    else:
        for _ in range(15):
            controller.step("MoveAhead")
        rotate_right(controller)
        for _ in range(15):
            controller.step("MoveAhead")
        rotate_right(controller)
        for _ in range(15):
            controller.step("MoveAhead")
        rotate_left(controller)
        while controller.step("MoveAhead").return_status != "OBSTRUCTED":
            controller.step("MoveAhead")
        for _ in range(9):
            rotate_right(controller)
        return "Init"

# Find Ball mode
# lean on the back of the platform and wait for obstructed for 4 times (I originally plan to set it to 3 times, but later I find that AI will waste one
# time because of its terrible initial direction, run whiskey_0006_06)
# Also to prevent the ball is right under AI's foot, if the ball is not found, look down and find again
def find_ball(controller, model):
    global rotate_direction, obstructed_times
    obstructed_times = 0
    while obstructed_times < 4:
        for _ in range(5):
            move_couter_clockwise(controller)
        image = controller.step("Pass").image_list[0]
        found, _, _ = find_item(image, model, 'Ball')
        if found:
            return "Navigate to Ball"
        if not found:
            for _ in range(2):
                controller.step("LookDown")
            found, _, _= find_item(image, model, 'Ball')
            for _ in range(2):
                controller.step("LookUp")
            if found:
                return "Navigate to Ball"
    return "Find Ramp"

# Find Ramp mode, just remeber to switch to Navigate to Ramp mode.
def find_ramp(controller, model, ball_model):
    global rotate_direction, obstructed_times, stop_off_ramp
    while True:
        off_ramp = False
        for _ in range(5):
            if move_clockwise(controller):
                off_ramp = True
        output = controller.step("Pass")
        if output is None:
            controller.end_scene()
            exit()
        image = output.image_list[0]
        ramp_found, _, _ = find_item(image, model, 'ramp')
        # check the initial setup of stop_off_ramp, I have commented there
        if off_ramp and not stop_off_ramp:
            stop_off_ramp = True
            return "Leaving Ramp"
        if ramp_found:
            for _ in range(5):
                stop_off_ramp = False
                controller.step("MoveAhead")
            return "Navigate to Ramp"

# Change Ramp mode, just move until left the the platform (or the AI has wasted too much time on switching the platform (obstructed_times >= 5))
def change_ramp(controller):
    global rotate_direction, obstructed_times
    obstructed_times = 0
    while not move_clockwise(controller) and obstructed_times < 5:
        continue
    return "Init"

# Leave Ramp mode, when the AI is finding ramp and suddenly move back for a long time, then it is just falling from platform, prevent it 
# entering the same platform again
def leave_ramp(controller):
    global rotate_direction, obstructed_times
    move_back(controller)
    obstructed_times = 0
    while obstructed_times < 1:
        move_clockwise(controller)
    for _ in range(10):
        move_clockwise(controller)
    return "Find Ramp"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    parser.add_argument(
        '--unity_path',
        type=str,
        default='/home/ubuntu/unity_app/MCS-AI2-THOR-Unity-App-v0.5.7.x86_64'
    )
    args = parser.parse_args()
    fn = args.scene_path
    controller = mcs.create_controller(config_file_or_dict='../sample_config.ini', unity_app_file_path=args.unity_path)
    if os.path.exists(fn):
        scene_data = mcs.load_scene_json_file(fn)

    output = controller.start_scene(scene_data)

    # _, params = output.action_list[0]
    model_ramp = torch.hub.load('ultralytics/yolov5', 'custom', path='./best (17).pt')
    model_ball = torch.hub.load('ultralytics/yolov5', 'custom', path='./best10.pt')

    epoch = 0
    while epoch <= 500:
        epoch += 1
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print(mode)
        if mode == "Init":
            mode = init(controller, model_ball)
        elif mode == "Find Ball":
            mode = find_ball(controller, model_ball)
        elif mode == "Find Ramp":
            mode = find_ramp(controller, model_ramp, model_ball)
        elif mode == "Navigate to Ball":
            mode = navigate_to_ball(controller, model_ball)
        elif mode == "Navigate to Ramp":
            mode = navigate_to_ramp(controller, model_ramp, model_ball)
        elif mode == "Change Ramp":
            mode = change_ramp(controller)
        elif mode == "Leaving Ramp":
            mode = leave_ramp(controller)
    
    controller.end_scene()
