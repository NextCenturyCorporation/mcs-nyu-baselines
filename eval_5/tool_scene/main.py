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
import pandas as pd
from PIL import Image
import copy
import glob
import os

# import spatial_scene.constants as const
import machine_common_sense as mcs


# one enhancement might be made: check codes of lava, use openCV to replace Yolo to detect lava

# The tool codes are highly dependent on model, if you want to visit my roboflow workspace for my dataset, email me at pl2285@nyu.edu
# some class type introduction:
# 1. lava: this bounding box is supposed to cover the entire lava completely. If the lava is slanted, or rhombic, from the agent's point of view, then
#          the bouding box would rather contains something irrelevant
# 2. valid lava: I originally planned to design a class to inform the agent that the lava within lava class bounding box is rhombic, so it is actually safe to move
#                ahead, but this valid lava is not accurate at all so it slowly becomes a complement for lava class
# 3. occluder: you might also call it guildrails, this is the only class in this scene that might not be unique in one frame
# 4. ball
# 5. good tool: because I only let agent to hold the short side of the tool to poke the ball out of lava, so good tool means it is the tool and this is the place
#               I should hold to poke the ball.
# 6. bad tool: this is the long side of the tool, the agent should not push this side of the tool to poke the ball out of lava, but the existence of this class is 
#              a good instruction for the agent to decide how to find good tool
# 7. tool end: this is the class I used to adjust agent's direction, tool end is the other end of the tool when the agent is pushing the tool, namely, this is the side
#              that collides with the ball so that the ball can roll out of lava

mode = "Analysis"
threshold = 0.2
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_tool21.pt')
left = True
w = 600
h = 400
rotate = False
i = 0
steps = 3
first = False


# The codes are long, but many of them are repetitive, so reading some lines of codes can makes understanding others easy


# There are four modes in this task:
# 1. Analysis: analyze whether if we need tools, if no, just enther navigate to ball mode; if yes, analyze which direction should we navigate to tool (from left 
#    or right, update global variable left):
#    2 submodes:
#        1) analyze tool needed
#        2) analyze locate tool
# 2. Navigate to Tool: the goal is to stand right in front of good tool class, right towards tool end class. The agent can stand some inaccuracy in direction, but
#                      you'd better control this inaccuracy in 20 degrees
#    3 submodes:
#        1) walk around lava, tool, occluders until good tool object is in front of you and you can fetch it without being occluded by occluders (guildrails), lava
#        2) move to good tool object so that the agent can stand right in front of the tool
#        3) adjust directions, rotate, move left of right until the agent thinks he faces right towards tool end object
# 3. Push the tool: rotate the tool until it points directly to the ball, start to push, after you made 10 successful pushes, stop because usually the ball should
#                   have enough momentum to roll outside the lava, and finally, move away from lava and occluders and start to find ball
#    3 submodes: 
#        1) rotate tool
#        2) push tool
#        3) move away from lava according to some logic
# 4. Navigate to Ball: exactly same as Navigate to Tool, but this time: 1) picking up ball is necessary, 2) good tool becomes obstacles and ball becomes objections

def save_img(img):
    global i
    print(i)
    cv.imwrite("./dataset/" + str(i) + ".png", cv.cvtColor(np.array(img), cv.COLOR_BGR2RGB))
    i += 1

# run model
def run_model(output):
    global model
    img = output.image_list[0]
    #save_img(img)
    predictions = model(img, size=640)
    loc_result = predictions.pandas().xyxy[0]
    print("Loc Result:",loc_result)
    return loc_result

# threshold helps to set up ball_threshold to self_defined value
def find_item(loc_result, item, unique = True):
    # unique = True means the object must be unique, the only non unique in this task is occluders
    global threshold, w, h
    found = False
    if unique:
        top_left = (w + 1, h + 1)
        bottom_right = (-1, -1)
        cur_prob = 0
        left = w + 1
        right = -1
        up = h + 1
        down = -1
        for _, res in loc_result.iterrows():
            # we do not want to fall into lava, so combine all lava's bounding box together and generate a huge bounding box
            if (item == "lava" or item == "valid lava") and res['name'] == item and res['confidence'] > threshold:
                found = True
                left = min(left, res['xmin'])
                right = max(right, res['xmax'])
                up = min(up, res['ymin'])
                down= max(down, res['ymax'])
            if res['name'] == item and res['confidence'] > threshold and res['confidence'] > cur_prob:
                found = True
                top_left = (int(res['xmin']), int(res['ymin']))
                bottom_right = (int(res['xmax']), int(res['ymax']))
                cur_prob = res['confidence']
        if item == 'lava' or item == "valid lava":
            top_left = (left, up)
            bottom_right = (right, down)
        return found, top_left, bottom_right
    # if this is occluders, return an array of bounding box
    else:
        top_left = []
        bottom_right = []
        for _, res in loc_result.iterrows():
            if res['name'] == item and res['confidence'] > threshold:
                found = True
                top_left.append((int(res['xmin']), int(res['ymin'])))
                bottom_right.append((int(res['xmax']), int(res['ymax'])))
        return found, top_left, bottom_right

# when you get obstructed, move around to get out from obstruction
# steps means how much you want to move aroun d
# direction = "right_right", move left move right move left
# direction = "ahead_back", then just move ahead
def move_obstructed(controller, steps = 1, direction = 'left_right'):
    if direction == 'left_right':
        for _ in range(steps):
            controller.step("MoveLeft")
        for _ in range(2 * steps):
            controller.step("MoveRight")
        for _ in range(steps):
            controller.step("MoveLeft")
    else:
        for _ in range(steps):
            controller.step("MoveAhead")

def compute_area(top_left, bottom_right):
    return (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])

# Giving you a frame, tell me whether you need tools to fetch the ball
def one_frame_tool_needed(controller):
    output = controller.step("Pass")
    loc_result = run_model(output)
    ball_found, ball_top_left, ball_bottom_right = find_item(loc_result, "ball")
    lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
    valid_lava_found, valid_lava_top_left, valid_lava_bottom_right = find_item(loc_result, "valid lava")
    lava_bottom_right = (max(lava_bottom_right[0], valid_lava_bottom_right[0]), max(lava_bottom_right[1], valid_lava_bottom_right[1]))
    lava_top_left = (min(lava_top_left[0], valid_lava_top_left[0]), min(lava_top_left[1], valid_lava_top_left[1]))
    lava_found = lava_found or valid_lava_found
    occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder", False)
    print(lava_found, occluder_found, ball_found, lava_top_left, lava_bottom_right, ball_top_left, ball_bottom_right)
    # if we find ball and lava in the same frame, the check whether the ball is surrounded by lava
    # the ball is too far away, we cannot see it clearly
    if compute_area(ball_top_left, ball_bottom_right) < 300:
        for _ in range(8):
            controller.step("MoveAhead")
        return "Keep moving ahead"
    # the bottom border of ball is between lava's upper and bottom border, the ball's top border > lava's top border, and the ball's right border < lava's right border
    # then the ball is in lava, we need tool
    if ball_found and lava_found:
        if ball_top_left[0] + 10 >= lava_top_left[0] and ball_bottom_right[0] - 10 <= lava_bottom_right[0]:
            if lava_top_left[1] - 20 < ball_bottom_right[1] < lava_bottom_right[1] + 20:
                return "Navigate to Tool"
        # or the occluder exists along with lava and ball
        elif occluder_found:
            if occluder_bottom_right[0][0] < lava_top_left[0] or occluder_top_left[0][0] > lava_bottom_right[0]:
                return "Navigate to Ball"
            return "Navigate to Tool"
        # otherwise, navigate to the ball
        return "Navigate to Ball"
    # if you find ball but no lava, navigate to ball
    if ball_found:
        return "Navigate to Ball"
    # if nothing is found, just keep rotating
    else:
        return "Keep Rotating"

def analysis_tool_needed(controller):
    # use the direction given by one_frame_tool_needed to make the decision
    for _ in range(9):
        frame_res = "Keep moving ahead"
        while frame_res == "Keep moving ahead":
            frame_res = one_frame_tool_needed(controller)
        if frame_res == "Navigate to Tool":
            return "Navigate to Tool"
        if frame_res == "Navigate to Ball":
            return "Navigate to Ball"
        for _ in range(4):
            controller.step("RotateLeft")
    return "Navigate to Tool"

# Given a frame, tell me whether we should move to left and fetch the tool or move to right and fetch the tool
# False = keep rotating
# True = OK to quit
def one_frame_locate_tool(controller):
    global left
    output = controller.step("Pass")
    loc_result = run_model(output)
    lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
    occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder", False)
    bad_tool_found, bad_tool_top_left, bad_tool_bottom_right = find_item(loc_result, "bad tool")
    good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
    # if there is no lava, then just see whether our target's center, if the target is in the right part, we move from left
    # otherwise, we move from right (sounds countertuitive, but if the target is in the right, you also move right, then you need to circle 
    # a long way around to find good tool class object)
    if not lava_found:
        if occluder_found or good_tool_found or bad_tool_found:
            if bad_tool_found:
                bad_tool_middle = (bad_tool_top_left[0] + bad_tool_bottom_right[0]) / 2
                if bad_tool_middle < w / 2:
                    left = False
            return True
        else:
            return False
    # if there is lava, there is supposed to be either occluder or tool, then we see tool (occluder) is on the right or the left, if it is
    # on the right of lava, then move right (if move left, we have to circle the long way around lava to fetch the tool)
    if occluder_found:
        if occluder_bottom_right[0][0] >= lava_bottom_right[0] - 10:
            left = False
        return True
    elif good_tool_found or bad_tool_found:
        left_border = max(bad_tool_bottom_right[0], good_tool_bottom_right[0])
        print(left_border, lava_bottom_right[0])
        if lava_found and left_border > lava_bottom_right[0]:
            left = False
        return True
    return False

# We need another function to rotate and get the frame for one_frame_locate_tool to use
def analysis_locate_tool(controller):
    locate_tool_res = False
    while not locate_tool_res:
        controller.step("RotateLeft")
        locate_tool_res = one_frame_locate_tool(controller)

# run tool_needed first, then locate_tool
def analysis(controller):
    global h, left
    mode = analysis_tool_needed(controller)
    print(mode)
    if mode == "Navigate to Ball":
        # this again checks whether we should move left or right to circumvent obstacles.
        output = controller.step("Pass")
        loc_result = run_model(output)
        lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
        occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder")
        bad_tool_found, bad_tool_top_left, bad_tool_bottom_right = find_item(loc_result, "bad tool")
        good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
        left_border = min(lava_top_left[0], occluder_top_left[0], bad_tool_top_left[0], good_tool_top_left[0])
        right_border = min(lava_bottom_right[0], occluder_bottom_right[0], bad_tool_bottom_right[0], good_tool_bottom_right[0])
        if w - right_border > left_border:
            left = False
        controller.step("LookDown")
        controller.step("LookDown")
        return mode
    analysis_locate_tool(controller)
    controller.step("LookDown")
    controller.step("LookDown")
    return mode

# This function is used in Navigate to ball and tool, more specifically used in walk around obstacles submode, you can to circumvent all obstacles to reach the goal
def make_one_move(controller, bad_left_border, bad_right_border, bad_bottom_border, good_left_border, good_right_border, good_bottom_border):
    global w, h, left, first
    print("information")
    print(bad_left_border, bad_right_border, bad_bottom_border, good_left_border, good_right_border, good_bottom_border)
    return_status = "SUCCESSFUL"
    # too close to these obstacles, move back, especially when obstacles are lava
    if bad_bottom_border + 50 >= h:
        for _ in range(5):
            return_status = controller.step("MoveBack").return_status
        if return_status == "OBSTRUCTED":
            move_obstructed(controller)
        return True
    # the target is in front of all obstacles
    if 0 <= bad_bottom_border <= good_bottom_border + 35 <= h + 35:
        print("right in front")
        first = True
        return True
    # the target is on the left (or the right) of all obstacles
    if 0 <= good_right_border < bad_left_border - 20 <= w - 20 or w + 20 >= good_left_border + 20 > bad_right_border >= 0:
        print("corner")
        first = True
        return True
    # otherwise, try to move left (right) to circumvent obstacles
    if left and bad_left_border <= w/2:
        for _ in range(5):
            return_status = controller.step("MoveLeft").return_status
        if return_status == "OBSTRUCTED":
            move_obstructed(controller, steps = 5, direction = "ahead_back")
        return True
    if not left and bad_right_border >= w/2:
        for _ in range(5):
            return_status = controller.step("MoveRight").return_status
        if return_status == "OBSTRUCTED":
            move_obstructed(controller, steps = 5, direction = "ahead_back")
        return True
    # sometimes when we just rotate 90 degrees, all obstacles disappear, when this happens, move left (right) according the left global variables
    if bad_left_border > w or bad_right_border < 0:
        if good_left_border > w or good_right_border < 0:
            move_action = "MoveLeft" if left else "MoveRight"
            for _ in range(5):
                return_status = controller.step(move_action).return_status
            if return_status == "OBSTRUCTED":
                move_obstructed(controller, steps = 5, direction = "ahead_back")
            return True
        else:
            return False
    # almost pass all obstacles (half of frames clear), turn 90 degrees
    if left:
        for _ in range(9):
            controller.step("RotateRight")
        return True
    if not left:
        for _ in range(9):
            controller.step("RotateLeft")
        return True
    return True

# walk around obstacles for navigate to tool, target is good tool, others are obstacles
def one_move_to_tool(controller):
    global left, first
    output = controller.step("Pass")
    loc_result = run_model(output)
    lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
    valid_lava_found, valid_lava_top_left, valid_lava_bottom_right = find_item(loc_result, "valid lava")
    occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder", False)
    bad_tool_found, bad_tool_top_left, bad_tool_bottom_right = find_item(loc_result, "bad tool")
    good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
    if first:
        first = False
        return not good_tool_found
    # as i mentioned, now i just use valid lava class as a complement of lava, if lava is not found, use valid lava
    if not lava_found and valid_lava_found:
        lava_bottom_right = valid_lava_bottom_right
        lava_top_left = valid_lava_top_left
        lava_found = True
    # if nothing is found, follow left global variable to move, and there is no need for make_one_move
    if not lava_found and not occluder_found and not bad_tool_found and not good_tool_found:
        if left:
            for _ in range(5):
                controller.step("MoveLeft")
        else:
            for _ in range(5):
                controller.step("MoveRight")
        return True
    # collect all borders we have to pass to make_one_move
    left_border = min(bad_tool_top_left[0], valid_lava_top_left[0])
    right_border = max(bad_tool_bottom_right[0], valid_lava_bottom_right[0])
    bottom_border = max(bad_tool_bottom_right[1], lava_bottom_right[1])
    for coordinate in occluder_bottom_right:
        right_border = max(right_border, coordinate[0])
        bottom_border = max(bottom_border, coordinate[1])
    for coordinate in occluder_top_left:
        left_border = min(left_border ,coordinate[0])
    return make_one_move(controller, left_border, right_border, bottom_border, good_tool_top_left[0], good_tool_bottom_right[0], good_tool_bottom_right[1])

# if this is good tool target, move left or right to adjust
# if this is ball target, rotate to adjust
def navigate(controller, top_left, bottom_right, cw, ch, ball = True):
    left_border = top_left[0]
    right_border = bottom_right[0]
    if cw < left_border:
        print(left_border, right_border, "1")
        if not ball:
            controller.step("MoveRight")
            controller.step("MoveRight")
            if controller.step("MoveRight").return_status != "SUCCESSFUL":
                return False
        else:
            controller.step("RotateRight")
        return True
    if cw > right_border:
        print(left_border, right_border, "2")
        if not ball:
            controller.step("MoveLeft")
            controller.step("MoveLeft")
            if controller.step("MoveLeft").return_status != "SUCCESSFUL":
                return False
        else:
            controller.step("RotateLeft")
        return True
    return False

# after good tool is able to get, we start to move to it
def move_to_tool(controller):
    global left, w, h
    output = controller.step("Pass")
    loc_result = run_model(output)
    good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
    lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
    # Again, the most important thing is still avoiding lava, if there is lava in our way
    if lava_found and lava_top_left[0] < w / 2 and lava_bottom_right[0] > w / 2 and lava_bottom_right[1] >= good_tool_bottom_right[1] >= 0:
        if left:
            for _ in range(3):
                controller.step("MoveLeft")
        else:
            for _ in range(3):
                controller.step("MoveRight")
        return True
    # adjust by moving left or right
    if good_tool_found:
        if navigate(controller, good_tool_top_left, good_tool_bottom_right, w / 2, h / 2, False):
            return True
    return_status = "SUCCESSFUL"
    # move ahead, if obstructed, then that means we have already walked into the tool
    for _ in range(5):
        return_status = controller.step("MoveAhead").return_status
    if return_status == "OBSTRUCTED":
        return False
    else:
        return True

# use tool end object and occluder obeject to adjust your direction, make sure you are facing towards the tool end
def adjust_direction_to_tool(controller):
    global w
    output = controller.step("Pass")
    loc_result = run_model(output)
    good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
    occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder")
    # I forget my logic here but I remember this is added when I ran a scene
    if occluder_found:
        if occluder_top_left[0] >= good_tool_bottom_right[0] - 20:
            controller.step("RotateRight")
            return True
        elif occluder_bottom_right[0] <= good_tool_top_left[0] + 20:
            controller.step("RotateLeft")
            return True
        else:
            return False
    # firstly try to rotate left to find tool end, then we rotate right if rotating left finds no tool end
    for _ in range(4):
        loc_result = run_model(output)
        tool_end_found, tool_end_top_left, tool_end_bottom_right = find_item(loc_result, "tool end")
        if tool_end_found:
            # if find tool_end, make last adjustment
            if (tool_end_top_left[0] + tool_end_bottom_right[0]) / 2 > w / 2:
                controller.step("RotateRight")
                controller.step("RotateRight")
            else:
                controller.step("RotateLeft")
                controller.step("RotateLeft")
            return False
        output = controller.step("RotateLeft")
        output = controller.step("RotateLeft")
    # mvoe back to the original direction
    for _ in range(4):
        output = controller.step("RotateRight")
        output = controller.step("RotateRight")
    for _ in range(4):
        loc_result = run_model(output)
        tool_end_found, tool_end_top_left, tool_end_bottom_right = find_item(loc_result, "tool end")
        if tool_end_found:
            # if find tool_end, make last adjustment
            if (tool_end_top_left[0] + tool_end_bottom_right[0]) / 2 > w / 2:
                controller.step("RotateRight")
                controller.step("RotateRight")
            else:
                controller.step("RotateLeft")
                controller.step("RotateLeft")
            return False
        output = controller.step("RotateRight")
        output = controller.step("RotateRight")
    return False

# just make sure good tool object is in the middle of the frame
def move_tool_to_middle(controller):
    global w
    output = controller.step("Pass")
    loc_result = run_model(output)
    good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
    times = 0
    while (good_tool_top_left[0] + good_tool_bottom_right[0]) / 2 < (w / 2 - 30) or (good_tool_top_left[0] + good_tool_bottom_right[0]) / 2 > (w / 2 + 30):
        times += 1
        if times >= 3:
            break
        if (good_tool_top_left[0] + good_tool_bottom_right[0]) / 2 < w / 2:
            controller.step("MoveLeft")
        else:
            controller.step("MoveRight")

def navigate_to_tool(controller):
    global w, h
    #submode #1: walk around obstacles
    while one_move_to_tool(controller):
        continue
    print("move to tool")
    # submode #2: move towards tool
    while move_to_tool(controller):
        continue
    print("adjust direction")
    # submode #3: facing towards tool end
    adjust_direction_to_tool(controller)
    move_tool_to_middle(controller)
    while controller.step("MoveAhead").return_status != "OBSTRUCTED":
        continue
    return "Push Ball"

# (use clockwise as example), rotate object clockwise, rotate right, move back, move left, move ahead
def rotate_object(controller, clockwise, x= 300, y = 300):
    global rotate, steps
    params = {"objectImageCoordsX": x, "objectImageCoordsY": y, "clockwise" : clockwise}
    status = controller.step("RotateObject", **params).return_status
    # in terms of move left (right), 2 steps are too small, 3 steps are too large.
    if status == "SUCCESSFUL":
        if steps <= 3:
            steps += 0.5
        else:
            steps = 2
    if clockwise:
        # because agent rotates 10 degrees each time, tool rotates 5 degrees instead
        if rotate:
            controller.step("RotateRight")
            rotate = False
        else:
            rotate = True
        for _ in range(2):
            controller.step("MoveBack")
        for _ in range(int(steps)):
            controller.step("MoveLeft")
        # adjust direction and move to middle is to prevent we miss the good tool in rotation and push
        adjust_direction_to_tool(controller)
        if rotate:
            if status != "SUCCESSFUL":
                controller.step("RotateLeft")
        move_tool_to_middle(controller)
    else:
        if rotate:
            controller.step("RotateLeft")
            rotate = False
        else:
            rotate = True
        for _ in range(2):
            controller.step("MoveBack")
        for _ in range(int(steps)):
            controller.step("MoveRight")
        adjust_direction_to_tool(controller)
        if rotate:
            if status != "SUCCESSFUL":
                controller.step("RotateRight")
        move_tool_to_middle(controller)
    for _ in range(4):
        controller.step("MoveAhead")

# push three times, record how many times you succeed
def push_object(controller, times, total_times, x= 300, y = 300):
    move_tool_to_middle(controller)
    params = {"objectImageCoordsX": x, "objectImageCoordsY": y, "force" : 0.85}
    times = 0
    status1 = controller.step("PushObject", **params).return_status
    if status1 == "SUCCESSFUL":
        times +=1
    status2 = controller.step("PushObject", **params).return_status
    if status2 == "SUCCESSFUL":
        times +=1
    status3 = controller.step("PushObject", **params).return_status
    if status3 == "SUCCESSFUL":
        times +=1
    print(status1)
    print(status2)
    print(status3)
    # move twice to prevent occluder obstruct agent
    while controller.step("MoveAhead").return_status != "OBSTRUCTED":
        continue
    move_obstructed(controller)
    while controller.step("MoveAhead").return_status != "OBSTRUCTED":
        continue
    return times

def ball_moved(controller):
    output = controller.step("Pass")
    loc_result = run_model(output)
    ball_found1, ball_top_left1, ball_bottom_right1 = find_item(loc_result, "ball")
    for _ in range(8):
        output = controller.step("Pass")
    loc_result = run_model(output)
    ball_found2, ball_top_left2, ball_bottom_right2 = find_item(loc_result, "ball")
    return np.abs(ball_bottom_right2[1] - ball_bottom_right1[1]) >= 5

def push_ball(controller):
    global h, left
    times = 0
    total_times = 15
    output = controller.step("Pass")
    loc_result = run_model(output)
    tool_end_found, tool_end_top_left, tool_end_bottom_right = find_item(loc_result, "tool end")
    if tool_end_top_left[1] < 55:
        total_times = 15
    else:
        total_times = 12
    # if there are occluders, we might push the tool into occluder and the tool stop moving, so we need to push several more times
    #if occluder_found:
    #    total_times = 15
    while True:
        output = controller.step("Pass")
        loc_result = run_model(output)
        good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
        tool_end_found, tool_end_top_left, tool_end_bottom_right = find_item(loc_result, "tool end")
        occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder")
        tool_center = (good_tool_top_left[0] + good_tool_bottom_right[0]) / 2
        if tool_end_top_left[1] + 10 <= occluder_top_left[0][1]:
            params = {"objectImageCoordsX": tool_center, "objectImageCoordsY": 300, "force" : 1}
            controller.step("PushObject", **params)
            controller.step("PushObject", **params)
            controller.step("PushObject", **params)
            controller.step("PushObject", **params)
            break
        lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
        ball_found, ball_top_left, ball_bottom_right = find_item(loc_result, "ball")
        ball_center = (ball_top_left[0] + ball_bottom_right[0]) / 2 
        # according to my experience, this lava warning usually doesn not work, so hard coding total times is necessary
        if lava_bottom_right[1] >= h - 80:
            break
        # periodically adjust tool's direciton to point to ball
        if ball_found:
            if tool_end_top_left[0] > ball_top_left[0]:
                rotate_object(controller, False, x = tool_center)
                continue
            elif tool_end_bottom_right[0] < ball_bottom_right[0]:
                rotate_object(controller, True, x = tool_center)
                continue
        else:
            # if the tool is 45 degrees aligned, when agent face towards tool end, there will be no ball in front of them, so agent uses lava to tell where the ball is
            if lava_found and good_tool_found:
                if lava_top_left[0] < good_tool_top_left[0]:
                    rotate_object(controller, False, x = tool_center)
                else:
                    rotate_object(controller, True, x = tool_center)
            # if the tool is 90 degrees aligned, when agent face towards tool end, there will be no ball in front of them, so agent need to search where the ball is
            else:
                for _ in range(9):
                    output = controller.step("RotateLeft")
                loc_result = run_model(output)
                ball_found, ball_top_left, ball_bottom_right = find_item(loc_result, "ball")
                for _ in range(9):
                    output = controller.step("RotateRight")
                if ball_found:
                    rotate_object(controller, False, x = tool_center)
                else:
                    rotate_object(controller, True, x = tool_center)
            continue
        if times < total_times:
            times += push_object(controller, times, total_times, x = tool_center)
        else:
            params = {"objectImageCoordsX": tool_center, "objectImageCoordsY": 300, "force" : 1}
            controller.step("PushObject", **params)
            controller.step("PushObject", **params)
            controller.step("PushObject", **params)
            controller.step("PushObject", **params)
            break
    # if there is occluder, move back outside occluder
    occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder")
    if occluder_found:
        while True:
            output = controller.step("Pass")
            loc_result = run_model(output)
            occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder", False)
            occluder_bottom = 0
            for coordinate in occluder_bottom_right:
                occluder_bottom = max(occluder_bottom, coordinate[1])
            if occluder_found and occluder_bottom >= h - 20:
                return_status = "SUCCESSFUL"
                for _ in range(5):
                    return_status = controller.step("MoveBack").return_status
                if return_status == "OBSTRUCTED":
                    move_obstructed(controller, steps = 2)
            else:
                break
    print("adjust according to memory")
    # keep moving left or right until agent completely away from lava (I hard code it because yolo is not good when detecting lava)
    for _ in range(6):
        controller.step("MoveBack")
        for _ in range(9):
            if left:
                controller.step("MoveLeft")
            else:
                controller.step("MoveRight")
    for _ in range(2):
        if not left:
            controller.step("RotateRight")
        else:
            controller.step("RotateLeft")
        for _ in range(35):
            controller.step("MoveAhead")
    for _ in range(6):
        if not left:
            controller.step("RotateLeft")
        else:
            controller.step("RotateRight")
    return "Navigate to Ball"

# same as onve_move_to_tool, but target becomes ball
def one_move_to_ball(controller):
    global left
    output = controller.step("Pass")
    loc_result = run_model(output)
    lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
    #occluder_found, occluder_top_left, occluder_bottom_right = find_item(loc_result, "occluder", False)
    bad_tool_found, bad_tool_top_left, bad_tool_bottom_right = find_item(loc_result, "bad tool")
    good_tool_found, good_tool_top_left, good_tool_bottom_right = find_item(loc_result, "good tool")
    tool_end_found, tool_end_top_left, tool_end_bottom_right = find_item(loc_result, "tool_end")
    ball_found, ball_top_left, ball_bottom_right = find_item(loc_result, "ball")
    valid_lava_found, valid_lava_top_left, valid_lava_bottom_right = find_item(loc_result, "valid lava")
    #print(occluder_top_left, occluder_bottom_right)
    if lava_found:
        left_border = min(bad_tool_top_left[0], lava_top_left[0], good_tool_top_left[0], tool_end_top_left[0])
        right_border = max(bad_tool_bottom_right[0], lava_bottom_right[0], good_tool_bottom_right[0], tool_end_bottom_right[0])
    else:
        left_border = min(bad_tool_top_left[0], valid_lava_top_left[0], good_tool_top_left[0], tool_end_top_left[0])
        right_border = max(bad_tool_bottom_right[0], valid_lava_bottom_right[0], good_tool_bottom_right[0], tool_end_bottom_right[0])
    bottom_border = max(bad_tool_bottom_right[1], lava_bottom_right[1], good_tool_bottom_right[1], tool_end_bottom_right[1])
    #for coordinate in occluder_bottom_right:
    #    right_border = max(right_border, coordinate[0])
    #    bottom_border = max(bottom_border, coordinate[1])
    #for coordinate in occluder_top_left:
    #    left_border = min(left_border ,coordinate[0])
    print(ball_found, ball_top_left, ball_bottom_right)
    return make_one_move(controller, left_border, right_border, bottom_border, ball_top_left[0], ball_bottom_right[0], ball_bottom_right[1])

# same as move_to_tool (in the future you may add valid lava as complement of lava)
def move_to_ball(controller):
    global left, w, h
    output = controller.step("Pass")
    loc_result = run_model(output)
    ball_found, ball_top_left, ball_bottom_right = find_item(loc_result, "ball")
    lava_found, lava_top_left, lava_bottom_right = find_item(loc_result, "lava")
    if not ball_found:
        for _ in range(4):
            controller.step("RotateLeft")
        return
    if lava_found and lava_top_left[0] < w / 2 and lava_bottom_right[0] > w / 2 and lava_bottom_right[1] > h / 2:
        if left:
            for _ in range(5):
                controller.step("MoveLeft")
        else:
            for _ in range(5):
                controller.step("MoveRight")
        return
    move_ahead_step = 10
    ball_x = (ball_top_left[0] + ball_bottom_right[0]) / 2
    ball_y = (ball_top_left[1] + ball_bottom_right[1]) / 2
    params = {"objectImageCoordsX": ball_x , "objectImageCoordsY": ball_y}
    if controller.step("PickupObject", **params).return_status == "SUCCESSFUL":
        print("INFO: Picked Up soccer ball. Ending scene! :)")
        controller.end_scene()
        exit(0)
    if ball_found:
        navigate(controller, ball_top_left, ball_bottom_right, w / 2, h / 2)
        if compute_area(ball_top_left, ball_bottom_right) >= 4500:
            move_ahead_step = 1
    return_status = "SUCCESSFUL"
    for _ in range(move_ahead_step):
        return_status = controller.step("MoveAhead").return_status
    if return_status == "OBSTRUCTED":
        move_obstructed(controller, steps = 2)

def navigate_to_ball(controller):
    #while one_move_to_ball(controller):
    #    print("Not yet")
    #    continue
    while True:
        move_to_ball(controller)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    parser.add_argument(
        '--unity_path',
        type=str,
        default='/home/ubuntu/unity_app/MCS-AI2-THOR-Unity-App-v0.5.7.x86_64'
    )
    args = parser.parse_args()
    controller = mcs.create_controller(config_file_or_dict='../sample_config.ini', unity_app_file_path=args.unity_path)
    fn = args.scene_path
    if os.path.exists(fn):
        scene_data = mcs.load_scene_json_file(fn)

    output = controller.start_scene(scene_data)

    # _, params = output.action_list[0]

    epoch = 0
    while epoch <= 500:
        epoch += 1
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print(mode)
        if mode == "Analysis":
            mode = analysis(controller)
        elif mode == "Navigate to Tool":
            mode = navigate_to_tool(controller)
        elif mode == "Push Ball":
            mode = push_ball(controller)
        elif mode == "Navigate to Ball":
            mode = navigate_to_ball(controller)
    
    controller.end_scene()
