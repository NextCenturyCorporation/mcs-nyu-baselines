#!/usr/bin/env python
# coding: utf-8

import import_ipynb
import argparse
import constant as const
import copy
import sys

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import machine_common_sense as mcs
import os
import glob


import enum



threshold = 0.1
neartheball = False
explore_right = 1
prev_img = 0



def check_for_ball(im, loc_result, cw):
    for idx, res in loc_result.iterrows():
        # print(res['ymin'])
        # print("int(res['ymin'] - const.TOP_BOTTOM_CUSHION): ", int(res['ymin'] - const.TOP_BOTTOM_CUSHION))
        # print("int(res['ymax'] + const.TOP_BOTTOM_CUSHION): ", int(res['ymax'] + const.TOP_BOTTOM_CUSHION))
        # print("int(res['xmin'] - const.LEFT_RIGHT_CUSHION): ", int(res['xmin'] - const.LEFT_RIGHT_CUSHION))
        # print("int(res['xmax'] + const.LEFT_RIGHT_CUSHION): ", int(res['xmax'] + const.LEFT_RIGHT_CUSHION))
        occ_img = im[
                  max(int(res['ymin'] - const.TOP_BOTTOM_CUSHION), 0):
                  max(int(res['ymax'] + const.TOP_BOTTOM_CUSHION), 0),
                  max(int(res['xmin'] - const.LEFT_RIGHT_CUSHION), 0):
                  max(int(res['xmax'] + const.LEFT_RIGHT_CUSHION), 0)
                  ]
        occ_img = cv.cvtColor(occ_img, cv.COLOR_BGR2GRAY)
        cv.imwrite('ROI_{}.png'.format(idx), occ_img)
        rows, cols = occ_img.shape
        number_of_white_pix = sum(180 <= occ_img[i][j] <= 255 for i in range(rows) for j in range(cols))
        print("number_of_white_pix for idx ", idx, ": ", number_of_white_pix)
        if number_of_white_pix > 0:
            const.SCENE_HAS_SOCCER_BALL = True
            if res['xmin'] < cw:
                return const.INITIAL_MOVE_LEFT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))]
            else:
                return const.INITIAL_MOVE_RIGHT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ))]
    return None, None


def create_bounding_box(img, loc_result, pred_class):
    color = (0, 128, 0) if pred_class == const.SPORTS_BALL else (255, 0, 0)
    for idx, res in loc_result.iterrows():
        if (res['name'] == pred_class) and (res['confidence'] > const.threshold):
            cv.rectangle(img, (int(res['xmin']), int(res['ymax'])), (int(res['xmax']), int(res['ymin'])), color, 2)


def find_bigger_occluder(loc_result, cw):
    max_width = - sys.maxsize
    bigger_occ_idx = -1
    for idx, res in loc_result.iterrows():
        occ_height = abs(res['ymax'] - res['ymin'])
        occ_width = abs(res['xmax'] - res['xmin'])
        if occ_width > max_width and occ_height > 25:
            max_width = occ_width
            bigger_occ_idx = idx
    # print('loc_result: ', type(loc_result))
    if loc_result['xmin'][bigger_occ_idx] < cw:
        return const.INITIAL_MOVE_LEFT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))]
    else:
        return const.INITIAL_MOVE_RIGHT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ))]


def find_ball(loc_result, cw):
    for idx, res in loc_result.iterrows():
        if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):
            if res['xmin'] > cw:
                return const.INITIAL_MOVE_RIGHT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ))]
            else:
                return const.INITIAL_MOVE_LEFT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))]


def navigate(loc_result, cw, ch, actions, parameters, pred_class, output):
    global look, epoch
    for idx, res in loc_result.iterrows():
        if pred_class == const.TOOL and const.MOVE_AHEAD_OBSTRUCTED==False and epoch<300:
            const.NAVIGATE_TOOL = True
            const.PREVIOUS = loc_result

            left_border = res['xmin'] - max(const.LEFT_RIGHT_CUSHION, 0)
            right_border = res['xmax'] + max(const.LEFT_RIGHT_CUSHION, 0)
            top_border = res['ymin'] - max(const.TOP_BOTTOM_CUSHION, 0)
            bottom_border = res['ymax'] + max(const.TOP_BOTTOM_CUSHION, 0)
            if left_border <= cw <= right_border:
                actions.extend(const.STICKY_MOVE_AHEAD)
                parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                const.LEFT_RIGHT_CUSHION -= 10
                const.TOP_BOTTOM_CUSHION -= 10
            if cw < left_border:
                actions.extend(const.ROTATE_RIGHT_AHEAD)
                parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_AHEAD))])
            if cw > right_border:
                actions.extend(const.ROTATE_LEFT_AHEAD)
                parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_AHEAD))])

            return actions, parameters
            print(actions, 'coming from navigate')
            print(left_border, cw, right_border)
            actions.extend(const.STICKY_MOVE_AHEAD)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
            

        elif pred_class == const.TOOL and const.MOVE_AHEAD_OBSTRUCTED:
            const.NAVIGATE_TOOL = False
            const.SCAN_BALL = True
            const.TOOL_REACHED = True
            const.ROAM = False
            const.ROTATE_COUNT = 1
            print('tool reached, now look for ball')
            return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
        
        if pred_class == const.SPORTS_BALL and const.NAVIGATE_BALL and const.MOVE_RIGHT_OBSTRUCTED and const.FINAL_COUNT and const.COUNT>8:
            const.SCAN_BALL=True
            const.SCENE_HAS_LAVA=False
            return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
        elif pred_class == const.SPORTS_BALL and const.NAVIGATE_BALL and const.MOVE_AHEAD_OBSTRUCTED==False:
            lava(output, 1)
            const.FINAL_COUNT = True
            if const.LAVA:
                actions.extend(const.MOVE_RIGHT_SEQ_10)
                parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_10))])
               
                actions.extend(const.ACTION_MOVE_BACK)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
                actions.extend(const.ACTION_MOVE_BACK)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
                actions.extend(const.ACTION_MOVE_BACK)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
                actions.extend(const.ACTION_MOVE_BACK)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
                actions.extend(const.ACTION_MOVE_BACK)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
                actions.extend(const.ACTION_MOVE_BACK)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
                
                actions.extend(const.ACTION_ROTATE_LEFT)
                parameters.extend([{} for _ in range(len(const.ACTION_ROTATE_LEFT))])

                return actions, parameters
            else:
                return const.STICKY_MOVE_AHEAD, [{} for _ in range(len(const.STICKY_MOVE_AHEAD))]
        elif pred_class == const.SPORTS_BALL and const.NAVIGATE_BALL and const.MOVE_AHEAD_OBSTRUCTED:
            const.MOVE_AHEAD_OBSTRUCTED=False
            const.FINAL_COUNT = True
            actions.extend(const.MOVE_RIGHT_SEQ_10)
            parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_10))])
            
            actions.extend(const.ACTION_MOVE_BACK)
            parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
            actions.extend(const.ACTION_MOVE_BACK)
            parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
            actions.extend(const.ACTION_MOVE_BACK)
            parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
            actions.extend(const.ACTION_MOVE_BACK)
            parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
            actions.extend(const.ACTION_MOVE_BACK)
            parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])
            actions.extend(const.ACTION_MOVE_BACK)
            parameters.extend([{} for _ in range(len(const.ACTION_MOVE_BACK))])

            actions.extend(const.ACTION_ROTATE_LEFT)
            parameters.extend([{} for _ in range(len(const.ACTION_ROTATE_LEFT))])
            actions.extend(const.ACTION_ROTATE_LEFT)
            parameters.extend([{} for _ in range(len(const.ACTION_ROTATE_LEFT))])
            return actions, parameters
        
        
        
        if pred_class == const.SPORTS_BALL and const.TOOL_REACHED and epoch<=300:
            if const.FIRST:
                const.FIRST=False
                const.PREVIOUS = loc_result
                const.TEST = True
                const.TOOL_OBSTRUCTED = True
                return const.ACTION_PASS, [{}]
                
#                 for idx, res in loc_result.iterrows():
#                     if (res['name'] == const.) and (res['confidence'] > const.threshold):
#                         actions.extend(const.PUSH_OBJ_SEQUENCE)
#                         for act in const.PUSH_OBJ_SEQUENCE:
#                             parameters.extend([{'objectId': 'tool'} if act == 'PushObject' else {}])
#                         const.TEST = True
#                         return actions, parameters
            
            else:
                if const.LAVA and const.NAVIGATE_BALL==False and const.MOVE_TOOL_RIGHT==False and const.MOVE_TOOL_LEFT==False and const.TEST_TOOL_1==False and const.TEST_TOOL_2==False:
                    const.TEST_TOOL_1=True
                    if look<0:
                        actions.extend(const.ACTION_LOOK_UP)
                        parameters.extend([{} for _ in range(len(const.ACTION_LOOK_UP))])
                    actions.extend(const.TEST_OBSTRUCT_1)
                    parameters.extend([{} for _ in range(len(const.TEST_OBSTRUCT_1))])
                    print('test 1')
                    return actions, parameters
                
                elif const.LAVA and const.NAVIGATE_BALL==False and const.TEST_TOOL_1 and const.MOVE_AHEAD_OBSTRUCTED:
                    print('move right until no lava to push')
                    const.MOVE_TOOL_RIGHT = True
                    actions.extend(const.MOVE_RIGHT_SEQ_5)
                    parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_5))])
                    actions.extend(const.STICKY_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                    print('move right')
                    return actions, parameters
                
                elif const.LAVA and const.NAVIGATE_BALL==False and const.TEST_TOOL_1 and const.MOVE_AHEAD_OBSTRUCTED==False:
                    const.TEST_TOOL_1=False
                    const.TEST_TOOL_2=True
                    print('test 2')
                    return const.TEST_OBSTRUCT_2, [{} for _ in range(len(const.TEST_OBSTRUCT_2))]
                
                elif const.LAVA and const.NAVIGATE_BALL==False and const.TEST_TOOL_2 and const.MOVE_AHEAD_OBSTRUCTED:
                    const.MOVE_TOOL_LEFT = True
                    actions.extend(const.MOVE_LEFT_SEQ_5)
                    parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
                    print('move left')
                    return actions, parameters
                
                elif const.LAVA and const.NAVIGATE_BALL==False and const.TEST_TOOL_2 and const.MOVE_AHEAD_OBSTRUCTED==False:
                    const.TEST_TOOL_2=False
                    const.NAVIGATE_BALL=True
#                     const.SCAN_BALL=True
                    print('navigate to ball now')
                    return const.ACTION_LOOK_UP, [{}]
                    
                
                if const.MOVE_TOOL_RIGHT and const.LAVA:
                    actions.extend(const.MOVE_RIGHT_SEQ_5)
                    parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_5))])
                    actions.extend(const.STICKY_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                    print('move tool right')
                    return actions, parameters
                
                elif const.MOVE_TOOL_RIGHT and const.LAVA and (const.MOVE_AHEAD_OBSTRUCTED or const.MOVE_RIGHT_OBSTRUCTED):
                    const.MOVE_AHEAD_OBSTRUCTED=False
                    const.MOVE_RIGHT_OBSTRUCTED = False
                    actions.extend(const.ACTION_MOVE_BACK)
                    parameters.extend([{}])
                    actions.extend(const.ACTION_MOVE_BACK)
                    parameters.extend([{}])
                    print('move tool right, cant move, move back')
                    return actions, parameters
                
                elif const.LAVA==False and const.MOVE_TOOL_RIGHT:
                    const.TOOL_OBSTRUCTED=False 
                    const.TOOL_AHEAD=True 
                    const.PUSH_OR_PULL=True
                    const.MOVE_TOOL_RIGHT=False
                    print('should start pushing again')
                    
                if const.MOVE_TOOL_LEFT and const.LAVA:
                    actions.extend(const.MOVE_LEFT_SEQ_5)
                    parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
                    print('move tool left')
                    return actions, parameters
                
                elif const.MOVE_TOOL_LEFT and const.LAVA and (const.MOVE_AHEAD_OBSTRUCTED or const.MOVE_LEFT_OBSTRUCTED):
                    const.MOVE_LEFT_OBSTRUCTED=False
                    const.MOVE_AHEAD_OBSTRUCTED=False
                    actions.extend(const.ACTION_MOVE_BACK)
                    parameters.extend([{}])
                    actions.extend(const.ACTION_MOVE_BACK)
                    parameters.extend([{}])
                    print('move tool left, cant move, move back')
                    return actions, parameters
                
                elif const.LAVA==False and const.MOVE_TOOL_LEFT:
                    const.TOOL_OBSTRUCTED=False 
                    const.TOOL_AHEAD=True 
                    const.PUSH_OR_PULL=True
                    const.MOVE_TOOL_LEFT=False
                    print('should start pushing again')
                    
                
                
                if (const.TOOL_OUT_OF_REACH or const.TOOL_OBSTRUCTED) and const.TOOL_LEFT==False and const.TOOL_RIGHT ==False and const.TOOL_BEHIND==False and const.TOOL_AHEAD==False and const.TEST:
                    const.TEST = False
                    print('test_obstruct')
                    return const.TEST_OBSTRUCT, [{} for _ in range(len(const.TEST_OBSTRUCT))]
                
                if const.TOOL_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED:
                    const.TOOL_LEFT = True
                    const.TOOL_OBSTRUCTED = False
                    const.MOVE_LEFT_OBSTRUCTED = False
                    actions.extend(const.MOVE_BACK_LEFT)
                    parameters.extend([{} for _ in range(len(const.MOVE_BACK_LEFT))])
                    actions.extend(const.LOOK_DOWN_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool out of reach, move left obstructed,so move back and left')
                    return actions, parameters
                
                if const.TOOL_OBSTRUCTED and const.MOVE_RIGHT_OBSTRUCTED:
                    const.TOOL_RIGHT = True
                    const.TOOL_OBSTRUCTED = False
                    const.MOVE_RIGHT_OBSTRUCTED = False
                    actions.extend(const.MOVE_BACK_RIGHT)
                    parameters.extend([{} for _ in range(len(const.MOVE_BACK_RIGHT))])
                    actions.extend(const.LOOK_DOWN_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool out of reach, move right obstructed,so move back and right')
                    return actions, parameters
    
                
                if const.TOOL_OBSTRUCTED and const.MOVE_AHEAD_OBSTRUCTED and const.TOOL_AHEAD==False:
                    const.TOOL_AHEAD = True
                    const.TOOL_OBSTRUCTED = False
                    actions.extend(const.LOOK_DOWN_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool out of reach, push')
                    return actions, parameters
                
                if const.TOOL_OBSTRUCTED and const.MOVE_BACK_OBSTRUCTED and const.PULL_ADJ==False:
                    const.TOOL_BEHIND = True
                    const.TOOL_OBSTRUCTED = False
                    const.MOVE_BACK_OBSTRUCTED = False
                    actions.extend(const.MOVE_LEFT_BACK)
                    parameters.extend([{} for _ in range(len(const.MOVE_LEFT_BACK))])
                    actions.extend(const.LOOK_DOWN_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    const.PUSH_ADJ = True
                    print('tool obstructed, move back obstructed, pull_adj false,so move left and back, push')
                    return actions, parameters
                
                if const.TOOL_OBSTRUCTED and const.MOVE_BACK_OBSTRUCTED==False and const.TOOL_BEHIND:
                    const.TOOL_OBSTRUCTED = False
                    const.TOOL_BEHIND=False
                    const.TOOL_RIGHT = True
                    print('tool out of reach, move back not obstructed,so move back')
                    return const.MOVE_BACK, [{} for _ in range(len(const.MOVE_BACK))]
                    
                if const.TOOL_OBSTRUCTED==False and const.PUSH_OR_PULL and const.TOOL_AHEAD:
                    actions.extend(const.ACTION_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_AHEAD))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool not out of reach, ready to push ')
                    return actions, parameters
                
                if const.TOOL_OBSTRUCTED and const.MOVE_AHEAD_OBSTRUCTED:
                    const.TOOL_OBSTRUCTED=False
                    const.MOVE_AHEAD_OBSTRUCTED=False
                    if look>-5:
                        actions.extend(const.ACTION_LOOK_DOWN)
                        parameters.extend([{} for _ in range(len(const.ACTION_LOOK_DOWN))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool not out of reach, ready to push ')
                    return actions, parameters
                
                if const.TOOL_LEFT and const.MOVE_LEFT_OBSTRUCTED==False:
                    const.TOOL_AHEAD = True
                    const.TOOL_LEFT = False
                    actions.extend(const.ACTION_MOVE_LEFT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_LEFT))])
                    actions.extend(const.ACTION_MOVE_LEFT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_LEFT))])
                    actions.extend(const.ACTION_MOVE_LEFT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_LEFT))])
                    actions.extend(const.ACTION_MOVE_LEFT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_LEFT))])
                    actions.extend(const.ACTION_MOVE_LEFT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_LEFT))])
                    actions.extend(const.ACTION_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_AHEAD))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool not out of reach, ready to push ')
                    return actions, parameters
  

                if const.TOOL_RIGHT and const.MOVE_RIGHT_OBSTRUCTED:
                    const.MOVE_RIGHT_OBSTRUCTED=False
                    print('tool on the right side, but move right obstructed ,so move back')
                    return const.MOVE_BACK, [{} for _ in range(len(const.MOVE_BACK))]
                    
                if const.TOOL_RIGHT and const.MOVE_RIGHT_OBSTRUCTED==False and const.PUSH_OR_PULL:
                    const.TOOL_AHEAD = True
                    const.TOOL_RIGHT = False
                    actions.extend(const.ACTION_MOVE_RIGHT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_RIGHT))])
                    actions.extend(const.ACTION_MOVE_RIGHT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_RIGHT))])
                    actions.extend(const.ACTION_MOVE_RIGHT)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_RIGHT))])
                    actions.extend(const.ACTION_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_AHEAD))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                  
                    print('tool not out of reach, ready to push ')
                    return actions, parameters
                
                if const.TOOL_OBSTRUCTED==False and const.TOOL_AHEAD and const.PUSH_OR_PULL:
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('tool not out of reach, ready to push ')
                    return actions, parameters
                
                if const.TOOL_OBSTRUCTED and const.TOOL_AHEAD and look<=-5 and const.PUSH_OR_PULL:
                    look=0
                    const.TOOL_OBSTRUCTED = False
                    actions.extend(const.LOOK_UP_SEQ_2)
                    parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ_2))])
                    actions.extend(const.STICKY_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('look up, ready to push ')
                    return actions, parameters
                
                elif const.TOOL_OBSTRUCTED and const.TOOL_AHEAD and look>3 and const.PUSH_OR_PULL:
                    look=0
                    const.TOOL_OBSTRUCTED = False
                    actions.extend(const.LOOK_UP_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ_3))])
                    actions.extend(const.STICKY_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('look down, ready to push ')
                    return actions, parameters
                
                else:
                    actions.extend(const.ACTION_MOVE_AHEAD)
                    parameters.extend([{} for _ in range(len(const.ACTION_MOVE_AHEAD))])
                    actions.extend(const.PUSH_OBJ_SEQUENCE)
                    for act in const.PUSH_OBJ_SEQUENCE:
                        parameters.extend([{"objectImageCoordsX": 300, "objectImageCoordsY":100} if act == 'PushObject' else {}])
                    print('look down, ready to push ')
                    return actions, parameters
        
        if pred_class == const.SPORTS_BALL and const.MOVE_AHEAD_OBSTRUCTED==False:
            left_border = res['xmin'] - max(const.LEFT_RIGHT_CUSHION, 0)
            right_border = res['xmax'] + max(const.LEFT_RIGHT_CUSHION, 0)
            top_border = res['ymin'] - max(const.TOP_BOTTOM_CUSHION, 0)
            bottom_border = res['ymax'] + max(const.TOP_BOTTOM_CUSHION, 0)
            if left_border <= cw <= right_border:
                actions.extend(const.STICKY_MOVE_AHEAD)
                parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                const.LEFT_RIGHT_CUSHION -= 10
                const.TOP_BOTTOM_CUSHION -= 10
            if cw < left_border:
                actions.extend(const.ROTATE_RIGHT_AHEAD)
                parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_AHEAD))])
            if cw > right_border:
                actions.extend(const.ROTATE_LEFT_AHEAD)
                parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_AHEAD))])
            if res['ymax']>350:
                actions.extend(const.ACTION_LOOK_DOWN)
                parameters.extend([{}])
            if epoch > 6:
                actions.extend(const.PICK_UP_SEQUENCE)
                for act in const.PICK_UP_SEQUENCE:
                    parameters.extend(
                        [{
                             "objectImageCoordsX": res['xmin']+(res['xmax']-res['xmin'])//2, "objectImageCoordsY":res['ymin']+(res['ymax']-res['ymin'])//2
                         } if act == 'PickupObject' else {}]
                    )
            print(actions, 'coming from navigate')
            print(left_border, cw, right_border)
            return actions, parameters
            


#         if pred_class == const.SPORTS_BALL and const.MOVE_AHEAD_OBSTRUCTED:
#             const.MOVE_AHEAD_OBSTRUCTED=False
#             actions.extend(const.MOVE_RIGHT_SEQ_5)
#             parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_5))])
#             actions.extend(const.STICKY_MOVE_AHEAD)
#             parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
#             print('move right of tool')
#             return actions, parameters
                
#         elif pred_class == const.SPORTS_BALL and const.MOVE_AHEAD_OBSTRUCTED==False:
#             actions.extend(const.STICKY_MOVE_AHEAD_3)
#             parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_3))])
#             const.SCAN_BALL=True
#             print('scan ball')
#             return actions, parameters

    return actions, parameters

def get_xyz_list_from_img(img):
    full_dastaset = []
    for im in img: #600
        xyz_list=[]
        for i in im: #400
            [x,y,z] = i
            xyz_list.append([x,y,z])
        full_dastaset.append(xyz_list)
    return full_dastaset



def find_lip(img, i):
    w, h = int(img.shape[1]), int(img.shape[0])
    cw, ch = int(w / 2), int(h / 2)
    if i == 0:
        right_crop = img[ch + 30:ch + 80, cw + 70:]  # top_left[0]:bottom_right[0]]
        left_crop = img[ch + 30:ch + 80, :cw - 70]  # top_left[0]:bottom_right[0]]
    elif i == 1:
        right_crop = img[ch + 80:ch + 100, cw + 70:]  # top_left[0]:bottom_right[0]]
        left_crop = img[ch + 80:ch + 100, :cw - 70]  # top_left[0]:bottom_right[0]]
    # cv.line(img,(cw-70,0),(cw-70,img.shape[0]-1),(0,255,0),2)
    # cv.line(img,(cw-180,0),(cw-150,img.shape[0]-1),(0,255,0),2)
    # cv.line(img,(cw+70,0),(cw+70,img.shape[0]-1),(255,0,0),2)
    # cv.line(img,(cw+150,0),(cw+150,img.shape[0]-1),(255,0,0),2)

    img_gray_r = cv.cvtColor(right_crop, cv.COLOR_BGR2GRAY)
    img_blur_r = cv.GaussianBlur(img_gray_r, (3, 3), 0)
    edges_r = cv.Canny(image=img_blur_r, threshold1=20, threshold2=40)
    sobelx_r = cv.Sobel(edges_r, cv.CV_64F, 1, 0, ksize=5)
    cv.imwrite('RCrop.png', right_crop)
    cv.imwrite('RCanny.png', edges_r)

    img_gray_l = cv.cvtColor(left_crop, cv.COLOR_BGR2GRAY)
    img_blur_l = cv.GaussianBlur(img_gray_l, (3, 3), 0)
    edges_l = cv.Canny(image=img_blur_l, threshold1=20, threshold2=40)
    cv.imwrite('LCrop.png', left_crop)
    cv.imwrite('LCanny.png', edges_l)

    indices_l = np.where(edges_l != [0])
    
    # coordinates_l=zip(indices_l[0],indices_l[1])
    print('max_l', max(indices_l[1]))
    
    indices_r = np.where(edges_r != [0])
    print('max_r', max(indices_r[1]))
    print("Shape:", img_blur_l.shape)

    if ((len(indices_l[0]) == 0) and (len(indices_r[0]) == 0)):
        const.SCAN_LAVA = False
        actions, parameters = const.LOOK_UP_SEQ_3,[{} for _ in range(len(const.LOOK_UP_SEQ_3))]
        actions.extend(const.STICKY_MOVE_AHEAD)
        parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
        return actions, parameters
    elif (len(indices_l[0]) != 0):
        const.SCAN_LAVA = False
        if (len(indices_r[0]) != 0) and ((min(indices_l[0]) < img_blur_l.shape[0]/2) or (max(indices_r[0]) < img_blur_r.shape[0]/20)):
            const.SCENE_HAS_LAVA = True
            const.SCAN_TOOL = True
            return const.LOOK_UP_SEQ_3,[{} for _ in range(len(const.LOOK_UP_SEQ_3))]
        else:
            return const.LOOK_UP_SEQ_3,[{} for _ in range(len(const.LOOK_UP_SEQ_3))]

def lava(output, i):
    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size) 
    img_pil.putdata(pixels)
    img_array = np.array(img_pil)
    occ_img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
    new_data_set = []
    
    w, h = int(occ_img.shape[1]), int(occ_img.shape[0])
    cw, ch = int(w / 2), int(h / 2)
    if i==0: 
        occ_img = occ_img[:, :ch]
    if i == 1:
        occ_img = occ_img[cw -100:cw +100, :]  # top_left[0]:bottom_right[0]]
    if i==2:
        occ_img = occ_img[cw -100:cw +100, :ch]
    
    data = get_xyz_list_from_img(occ_img)
    for iy,y in enumerate(data):
        for ix,x in enumerate(y):
            if 10<x[0]<35 and 100<x[1]<255 and 20<x[2]<255:
                new_data_set.append(x)

    if len(new_data_set)>0:
        const.LAVA=True
    else:
        const.LAVA=False


        
def select_action(output, model):
    global first_action, epoch
    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size)
    img_pil.putdata(pixels)
    img_array = np.array(img_pil)
    img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
    if not first_action:
        img = img[:, 50:]
    display_image = copy.deepcopy(img)
    predictions = model(img, size=640)
    cw = int(display_image.shape[1] / 2)
    ch = int(display_image.shape[0] / 2)
    cv.rectangle(display_image, (cw - 5, ch - 5), (cw + 5, ch + 5), 255, 2)
    actions, parameters, predicted_classes = [], [], []
    loc_result = predictions.pandas().xyxy[0]
    for idx, res in loc_result.iterrows():
        if res['confidence'] >= const.threshold:
            predicted_classes.append(res['name'])
    print("Loc Result:", loc_result)
    print("predicted_classes:", predicted_classes)
    
 
    epoch = epoch + 1
    
    if epoch>300 and const.BALL==False and const.SCAN_BALL==False:
        const.TOOL_REACHED=False
        actions.extend(const.LOOK_UP_SEQ_3)
        parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ_3))])
        const.SCAN_BALL=True
        actions.extend(const.ROTATE_RIGHT_SEQ_3)
        parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))])
        print('epoch>300, scan ball false, now scan')
        return actions, parameters
    
    elif epoch>300 and const.SCAN_BALL and const.SPORTS_BALL not in predicted_classes:
        const.ROTATE_COUNT +=1
        actions.extend(const.ROTATE_RIGHT_SEQ_3)
        parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))])
        print('epoch>300, scanning ball now')
        if const.ROTATE_COUNT == 12:
            print('full circle, no tool detected, roam mode on')
            const.ROTATE_COUNT = 0
            const.SCAN_TOOL=True
            actions.extend(const.ACTION_LOOK_UP)
            parameters.extend([{}])
        return actions, parameters
    
    elif epoch>300 and const.SCAN_BALL and const.SPORTS_BALL in predicted_classes and const.MOVE_AHEAD_OBSTRUCTED==False and const.BALL==False:
        print('navigating to ball')
        const.BALL=True
        const.SCAN_BALL=False
        return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
    
    elif epoch>300 and const.SPORTS_BALL in predicted_classes and const.MOVE_AHEAD_OBSTRUCTED==False and const.MOVE_LEFT_OBSTRUCTED:
        print('final navigate')
        return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
    
    elif epoch>300 and const.SCAN_BALL and const.SPORTS_BALL in predicted_classes and const.MOVE_AHEAD_OBSTRUCTED and const.BALL==False:
        print('navigating to ball')
        const.MOVE_AHEAD_OBSTRUCTED=False
        actions.extend(const.MOVE_BACK_LEFT)
        parameters.extend([{} for _ in range(len(const.MOVE_BACK_LEFT))])
        return actions, parameters
    
    elif epoch>300 and const.SPORTS_BALL in predicted_classes and const.MOVE_AHEAD_OBSTRUCTED and const.BALL:
        print('move back left')
        const.MOVE_AHEAD_OBSTRUCTED=False
        actions.extend(const.MOVE_BACK_LEFT)
        parameters.extend([{} for _ in range(len(const.MOVE_BACK_LEFT))])
        return actions, parameters
        
    elif epoch>300 and const.MOVE_LEFT_OBSTRUCTED and const.BALL:
        const.MOVE_LEFT_OBSTRUCTED =False
        actions.extend(const.MOVE_BACK_LEFT)
        parameters.extend([{} for _ in range(len(const.MOVE_BACK_LEFT))])
        print('move back left again')
        return actions, parameters
    
    elif epoch>300 and const.MOVE_BACK_OBSTRUCTED and const.BALL:
        const.MOVE_BACK_OBSTRUCTED =False
        actions.extend(const.ACTION_MOVE_RIGHT)
        parameters.extend([{}])
        actions.extend(const.MOVE_BACK_LEFT)
        parameters.extend([{} for _ in range(len(const.MOVE_BACK_LEFT))])
        print('move right, move back left')
        return actions, parameters
    
    elif epoch>300 and const.MOVE_BACK_OBSTRUCTED ==False and const.MOVE_LEFT_OBSTRUCTED==False and const.BALL:
        actions.extend(const.INITIAL_MOVE_LEFT_SEQ)
        parameters.extend([{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))])
        const.SCAN_BALL=True
        const.BALL=False
        print('move left')
        return actions, parameters
        
    
    
    if epoch<=300:

        if first_action and const.SPORTS_BALL in predicted_classes:
            first_action = False
            const.SCENE_HAS_SOCCER_BALL=True
            lava(output,0)
            if const.LAVA==False:
                return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
            else:
                const.SCENE_HAS_LAVA=True
                const.SCAN_TOOL=True

        elif first_action and const.SPORTS_BALL not in predicted_classes:
            print('ball not detected now spinning to scan ')
            first_action = False
            const.SCAN_BALL = True
            const.ROTATE_COUNT = 1
            return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]

        if first_action==False and const.SPORTS_BALL not in predicted_classes and const.SCAN_BALL and const.SCENE_HAS_LAVA==False:
            print('not first action, but still spin to try to detect ball')
            const.ROTATE_COUNT +=1
            if const.ROTATE_COUNT == 12: 
                const.SCAN_BALL = False
                const.ROTATE_COUNT = 0
            return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
        
        elif first_action==False and const.SPORTS_BALL not in predicted_classes and const.SCAN_BALL and const.NAVIGATE_BALL:
            print('not first action, but still spin to try to detect ball')
            const.ROTATE_COUNT +=1
            if const.ROTATE_COUNT == 12: 
                const.SCAN_BALL = False
                const.ROTATE_COUNT = 0
            return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
        elif first_action==False and const.SPORTS_BALL in predicted_classes and const.SCAN_BALL and const.NAVIGATE_BALL:
            #if we find ball while scanning
            print('not first action, scanning for ball and found ball')
            const.ROTATE_COUNT = 0
            const.SCAN_BALL = False
            lava(output,1)
            if const.LAVA==False:
                print('no lava, navigate to ball')
                actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
                return actions, parameters
        
        elif first_action==False and const.SPORTS_BALL in predicted_classes and const.SCAN_BALL and const.SCENE_HAS_LAVA==False:
            #if we find ball while scanning
            print('not first action, scanning for ball and found ball')
            const.ROTATE_COUNT = 0
            const.SCAN_BALL = False
            lava(output,0)
            if const.LAVA==False:
                print('no lava, navigate to ball')
                actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
                return actions, parameters
            else:
                const.SCENE_HAS_LAVA=True
                const.SCAN_TOOL=True
                const.ROTATE_COUNT = 1
                print('lava, scan tool now')
                return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
        elif first_action==False and const.SPORTS_BALL in predicted_classes and const.SCAN_BALL and const.NAVIGATE_BALL:
            #if we find ball while scanning
            print('lava, scanning for ball and found ball')
            const.ROTATE_COUNT = 0
            const.SCAN_BALL = False
            lava(output,0)
            if const.LAVA==False:
                print('no lava, navigate to ball')
                actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
                return actions, parameters
            else:
                const.SCENE_HAS_LAVA=True
                const.SCAN_TOOL=True
                const.ROTATE_COUNT = 1
                print('lava, scan tool now')
                return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]


        elif first_action==False and const.SPORTS_BALL in predicted_classes and const.SCENE_HAS_LAVA==False:
            #if we find ball while scanning
            print('not first action, found ball ')
            lava(output,0)
            if const.LAVA==False:
                actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
                return actions, parameters
            else:
                const.SCENE_HAS_LAVA=True
                const.SCAN_TOOL=True
                const.ROTATE_COUNT = 1
                print('lava, scan tool now')
                return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]





        if const.TOOL not in predicted_classes and const.SCAN_TOOL and const.SCENE_HAS_LAVA:
            const.ROTATE_COUNT +=1
            print(const.ROTATE_COUNT)
            print('no tool seen, scan tool, scene has lava')
            if const.ROTATE_COUNT == 12: 
                print('full circle, no tool detected, roam mode on')
                const.ROTATE_COUNT = 0
                const.ROAM = True 
                const.SCAN_TOOL=False
            return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
        elif const.SCENE_HAS_LAVA and const.TOOL in predicted_classes and const.SCAN_TOOL:
            #if we find tool while scanning
            print('not first action, scanning for tool and found tool')
            const.ROTATE_COUNT = 0
            const.SCAN_TOOL = False
            const.SCENE_HAS_TOOL = True
            return navigate(loc_result, cw, ch, actions, parameters, const.TOOL, output)
            #change navigate function so that it goes straight towards tool until obstructed?
            #set a variable: tool_reached
        elif const.SCENE_HAS_LAVA and const.NAVIGATE_TOOL and const.TOOL in predicted_classes:
            const.SCENE_HAS_TOOL = True
            print('tool in predicted class, navigate to tool')
            return navigate(loc_result, cw, ch, actions, parameters, const.TOOL, output)
        elif const.SCENE_HAS_LAVA and const.NAVIGATE_TOOL and const.TOOL not in predicted_classes and const.SCENE_HAS_TOOL:
            print('tool not in predicted class, navigate to tool based on previous location')
            return navigate(const.PREVIOUS, cw, ch, actions, parameters, const.TOOL, output)



        if const.ROAM and const.SCAN_BALL == False and const.SCAN_TOOL ==False:
            const.SCAN_BALL = True
            print('roam on, scan_ball')
            return const.ACTION_PASS, [{}]
        elif const.ROAM and const.SCAN_BALL and const.SPORTS_BALL not in predicted_classes:
            const.ROTATE_COUNT +=1
            actions, parameters = const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
            print('scan ball +1')
            if const.ROTATE_COUNT ==12:
                print('full circle, no ball detected, what now?')
                const.ROTATE_COUNT = 0
                return const.ACTION_PASS, [{}]
            return actions, parameters
        elif const.ROAM and const.SCAN_BALL and const.SPORTS_BALL in predicted_classes:
            print('stop scanning for ball, scan for tool after rotate 90 away from ball')
            const.SCAN_BALL = False
            const.SCAN_TOOL = True
            const.ROAM_COUNT +=1
            actions, parameters = const.ROTATE_RIGHT_SEQ_9, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_9))]
            actions.extend(const.STICKY_MOVE_AHEAD_3)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_3))])
            if const.ROAM_COUNT ==4:
                const.ROAM = False
                print('end of roaming')
            return actions, parameters

        if const.ROAM and const.SCAN_TOOL:
            const.ROTATE_COUNT +=1
            print(const.ROTATE_COUNT)
            if const.ROTATE_COUNT == 12:
                print('full circle, no tool detected, roam mode on')
                const.ROTATE_COUNT = 0
                const.ROAM = True
                const.SCAN_TOOL=False
            return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]

        if const.TOOL_REACHED and const.SPORTS_BALL not in predicted_classes and const.SCAN_BALL:
            const.ROTATE_COUNT +=1
            actions, parameters = const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
            print('tool reached, now scanning for ball')
            print(const.ROTATE_COUNT)
            if const.ROTATE_COUNT ==12:
                const.ROTATE_COUNT = 0
                const.SCAN_BALL=False
                print('full circle, no ball detected, now what?')
            return actions, parameters
        elif const.TOOL_REACHED and const.SPORTS_BALL in predicted_classes and const.SCAN_BALL:
            for idx, res in loc_result.iterrows():
                if res['xmin'] > 400:
                    const.ROTATE_COUNT = 0
                    actions, parameters = const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
                    const.SCAN_BALL=False
                    const.TOOL_REACH_BALL_FOUND=True
                    print('tool reach, ball found, now navigate to push tool to ball')
                    return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
                else:
                    const.SCAN_BALL=False
                    const.TOOL_REACH_BALL_FOUND=True
                    print('tool reach, ball found, now navigate to push tool to ball')
                    return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
        elif const.TOOL_REACHED and const.TOOL_REACH_BALL_FOUND:
            const.PUSH_OR_PULL=True
            lava(output,2)
            print('tool reach, ball found but cant see ball rn, push/pull true')
            return navigate(const.PREVIOUS, cw, ch, actions, parameters, const.SPORTS_BALL, output)

        if first_action==False and const.SCAN_BALL==False and const.SCAN_LAVA == True and const.SCENE_HAS_LAVA==False:
            return find_lip(img, 0)



        elif const.SPORTS_BALL in predicted_classes:
            actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL, output)
            for idx, res in loc_result.iterrows():
                if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):

                    if epoch > 6:
                        actions.extend(const.PICK_UP_SEQUENCE)
                        for act in const.PICK_UP_SEQUENCE:
                            parameters.extend(
                                [{
                                     "objectImageCoordsX": res['xmin']+(res['xmax']-res['xmin'])//2, "objectImageCoordsY":res['ymin']+(res['ymax']-res['ymin'])//2
                                     # 'objectImageCoordsX': (res['xmax'] - res['xmin']) // 2,
                                     # 'objectImageCoordsY': (res['ymax'] - res['ymin']) // 2
                                 } if act == 'PickupObject' else {}]
                            )
                    return actions, parameters

    if actions is None or len(actions) == 0:
        actions = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'LookDown', 'PickupObject']
        parameters = [{}, {}, {}, {}, {}, {"objectImageCoordsX": 300, "objectImageCoordsY":100}]

    return actions, parameters


if __name__ == '__main__':
    first_action = True
    look = 0 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    parser.add_argument(
        '--unity_path',
        type=str,
        default='/home/ubuntu/unity_app/MCS-AI2-THOR-Unity-App-v0.5.7.x86_64'
    )
    args = parser.parse_args()
    fn = args.scene_path
    if os.path.exists(fn):
        scene_data = mcs.load_scene_json_file(fn)

    controller = mcs.create_controller(config_file_or_dict='../sample_config.ini', unity_app_file_path=args.unity_path)
    output = controller.start_scene(scene_data)

    # _, params = output.action_list[0]
    action = const.ACTION_PASS
    params = [{}]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/best_v9.pt')

    epoch = 0
    while action != '':
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print("Actions to execute: ", action)
        const.LAVA = False
#         const.MOVE_AHEAD_OBSTRUCTED = False
#         const.MOVE_LEFT_OBSTRUCTED = False
#         const.MOVE_RIGHT_OBSTRUCTED = False
#         const.MOVE_BACK_OBSTRUCTED = False
#         const.TOOL_OUT_OF_REACH = False
        for idx in range(len(action)):
            output = controller.step(action[idx], **params[idx])
            if output is None:
                controller.end_scene()
                exit()
            if action[idx] == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move ahead obstructed by occluder.")
                const.MOVE_AHEAD_OBSTRUCTED = True
                
            if action[idx] == const.ACTION_MOVE_LEFT[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move left obstructed by occluder.")
                const.MOVE_LEFT_OBSTRUCTED = True
                
            if action[idx] == const.ACTION_MOVE_RIGHT[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move right obstructed by occluder.")
                const.MOVE_RIGHT_OBSTRUCTED = True
                
            if action[idx] == const.ACTION_MOVE_BACK[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move back obstructed by occluder.")
                const.MOVE_BACK_OBSTRUCTED = True
                
            if action[idx] == const.PUSH_OBJ_SEQUENCE[0] and output.return_status == "NOT_MOVEABLE":
                print("INFO : push, not moveable.")
                print(output.return_status)
                const.TOOL_OBSTRUCTED = True

            if action[idx] == const.PUSH_OBJ_SEQUENCE[0] and output.return_status == 'OUT_OF_REACH':
                print("INFO : push out of reach.")
                print(output.return_status)
                const.TOOL_OUT_OF_REACH = True
            
            if action[idx] == const.ACTION_LOOK_DOWN[0]:
                print("INFO : look down")
                
                look -=1
                print(look)
            
            if action[idx] == const.ACTION_LOOK_UP[0]:
                print("INFO : look up")
                
                look +=1
                print(look)

            if action[idx] == const.ACTION_PICK_UP_OBJ[0] and output.return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)
                
            if const.FINAL_COUNT:
                if action[idx] == const.ACTION_ROTATE_LEFT:
                    const.COUNT +=1
            
        action, params = select_action(output, model)
        
        if const.SCAN_BALL:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/best_v11.pt')
        else:
            continue
     
        


    controller.end_scene()




