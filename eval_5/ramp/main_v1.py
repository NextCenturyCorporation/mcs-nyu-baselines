#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import constant as const
import copy
import sys
import argparse

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import glob
import os

# import spatial_scene.constants as const
import machine_common_sense as mcs

threshold = 0.1
neartheball = False
explore_right = 1
prev_img = 0
import enum

l_lim = 60
r_lim = 60


class Region(enum.Enum):
    center = 1
    left = 2
    right = 3
    unknown = 4
    
ball_region = Region.unknown
lip_region = Region.unknown

def saveimage(img, stepn):
    if stepn < 10:
        cv.imwrite("output_images/lava" + str(stepn) + ".png", img)
    elif stepn < 100:
        cv.imwrite("output_images/lavb" + str(stepn) + ".png", img)
    else:
        cv.imwrite("output_images/lavc" + str(stepn) + ".png", img)
        


def check_for_ball(im, loc_result, cw):
    #if soccer ball is in the vicinity of detected bounding boxes, 
    #then return move left or move right sequences, else return none
    
    for idx, res in loc_result.iterrows():
        # print(res['ymin'])
        # print("int(res['ymin'] - const.TOP_BOTTOM_CUSHION): ", int(res['ymin'] - const.TOP_BOTTOM_CUSHION))
        # print("int(res['ymax'] + const.TOP_BOTTOM_CUSHION): ", int(res['ymax'] + const.TOP_BOTTOM_CUSHION))
        # print("int(res['xmin'] - const.LEFT_RIGHT_CUSHION): ", int(res['xmin'] - const.LEFT_RIGHT_CUSHION))
        # print("int(res['xmax'] + const.LEFT_RIGHT_CUSHION): ", int(res['xmax'] + const.LEFT_RIGHT_CUSHION))
        
        #occ_img creates a bigger bounding box? 
        occ_img = im[
                  max(int(res['ymin'] - const.TOP_BOTTOM_CUSHION), 0): # TOP_BOTTOM_CUSHION=20
                  max(int(res['ymax'] + const.TOP_BOTTOM_CUSHION), 0),
                  max(int(res['xmin'] - const.LEFT_RIGHT_CUSHION), 0): # LEFT_RIGHT_CUSHION=20
                  max(int(res['xmax'] + const.LEFT_RIGHT_CUSHION), 0)
                  ]
        occ_img = cv.cvtColor(occ_img, cv.COLOR_BGR2GRAY)
        #cv.imwrite('ROI_{}.png'.format(idx), occ_img)
        rows, cols = occ_img.shape
        number_of_white_pix = sum(180 <= occ_img[i][j] <= 255 for i in range(rows) for j in range(cols)) 
        # using white pixels to identify soccor ball in the enlarged bounding box?
#         print("number_of_white_pix for idx ", idx, ": ", number_of_white_pix)
        if number_of_white_pix > 0:
            const.SCENE_HAS_SOCCER = True
            if res['xmin'] < cw:
                return const.INITIAL_MOVE_LEFT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))]
            else:
                return const.INITIAL_MOVE_RIGHT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ))]
    return None, None


def create_bounding_box(img, loc_result, pred_class):  
    ##img: image?
    ##pred_class:str
    ##loc_result: 
    # results.pandas().xyxy[0]  # im predictions (pandas)
    # #      xmin    ymin    xmax   ymax  confidence  class    name
    # # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    
    #returns image with a bounding box rectangle drawn on it 
    #only allows two colors for bounding boxes
    
    color = (0, 128, 0) if pred_class == const.SPORTS_BALL else (255, 0, 0)
    ## this only allows two colors for bounding boxes, if more than two labels then alter this line 
    for idx, res in loc_result.iterrows():
        if (res['name'] == pred_class) and (res['confidence'] > const.threshold):
            cv.rectangle(img, (int(res['xmin']), int(res['ymax'])), (int(res['xmax']), int(res['ymin'])), color, 2)

            

def navigate(loc_result, cw, ch, actions, parameters, pred_class,img, output):
    global off_ramp_mode, first_off_ramp
    ## pred_class: e.g. const.OCCLUDER

    ##if obj in box predicted = pred_class, expand bounding box. 
    ##if expanded box xmin <=cw<=expanded box xmax, then move ahead
    ##if cw<expanded box xmin, move right 
    ##if cw>expanded box xmax, move left 
    ##if ch<expanded box ymin, look down ?? why 
    ##if ch>expanded box ymax, look up 
    ##apend all these actions as u loop through all loc_result and then return all actions.
    
    if off_ramp_mode:
        record=True
        img_ = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img1 = np.float32(img_)
        
        if const.TEST and const.MOVE_AHEAD_OBSTRUCTED==False:
            off_ramp_mode = False
            const.SCAN_BALL = True
            const.LEVEL_BALL_SCANNED = False
            const.LEVEL_RAMP_SCANNED = False
            record=False
            actions.extend(const.MOVE_LEFT_AHEAD)
            parameters.extend([{} for _ in range(len(const.MOVE_LEFT_AHEAD))])
#             actions.extend(const.LOOK_UP_SEQ)
#             parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ))])
            print('passed test, got off ramp, now scan for ball')

            return actions, parameters
        elif const.TEST and const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED==False:
#             const.TEST = False
            off_ramp_mode = True
            actions.extend(const.MOVE_LEFT_SEQ_5)
            parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
            print('still on ramp, continue')
            return actions, parameters
        
        
        if first_off_ramp:
            first_off_ramp = False
            actions.extend(const.LOOK_DOWN_SEQ)
            parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ))])
            return actions, parameters
        
        elif const.MOVE_AHEAD_OBSTRUCTED==False and const.MOVE_LEFT_OBSTRUCTED==False and first_off_ramp ==False:
            sym1 = horizontal_detector(output,200)[1]
            print('sym1', sym1)
            sym2 = vertical_detector(output,200,200)[1]
            print('sym2', sym2)
            
            if 1<=sym1<=2:
                const.HORIZONTAL_EDGE = True
            elif sym1>10:
                const.RAMP = True
                
            if const.HORIZONTAL_EDGE==True:
                print('found horizontal edge, now getting closer')
                const.TEST = True
                const.SCAN_EDGE = False
#                 const.HORIZONTAL_EDGE = False
                actions.extend(const.MOVE_LEFT_SEQ_5)
                parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
                
#                 sym2 = vertical_detector(output,200)[1]
            
#                 if sym2>3:
#                     print('number of symmetric points', sym2)
#                     const.RAMP = True
#                     print('coming from navigate, const.ramp now true')
                return actions, parameters
            elif const.RAMP == True:
                print('found off_ramp, now getting off')
                off_ramp_mode= False
                const.SCAN_EDGE = False
            
                const.HORIZONTAL_EDGE = False
#                 const.RAMP = False
                record=False
                const.LEVEL_BALL_SCANNED = False
                const.LEVEL_RAMP_SCANNED = False
                const.REVERSE = True
                actions.extend(const.ACTION_ROTATE_LEFT)
                parameters.extend([{} for _ in range(len(const.ACTION_ROTATE_LEFT))])
                actions.extend(const.STICKY_MOVE_AHEAD_2)
                parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_2))])
                return actions, parameters
            else:
                print('scan edge true, move ahead')
                const.SCAN_EDGE = True
                actions.extend(const.ACTION_MOVE_AHEAD)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_AHEAD))])
                return actions, parameters
        
        
        if first_off_ramp==False and const.SCAN_EDGE:
            const.ROTATE_COUNT +=1
            if const.ROTATE_COUNT == 40: 
                print('rotated 40 times, no edge')
                const.SCAN_EDGE = False
                const.SCAN_RAMP = True
                const.ROTATE_COUNT = 0
                return const.STICKY_MOVE_AHEAD_2, [{} for _ in range(len(const.STICKY_MOVE_AHEAD_2))]
            return const.ACTION_ROTATE_LEFT, [{}]
        elif first_off_ramp==False and const.HORIZONTAL_EDGE ==False and const.RAMP ==False and const.MOVE_AHEAD_OBSTRUCTED and const.SCAN_EDGE == False:
            print('not first off ramp, havent found edge yet, rotate once to left')
            const.SCAN_EDGE=True
            const.ROTATE_COUNT =1
            return const.ACTION_ROTATE_LEFT, [{}]   
        
                          
        elif first_off_ramp==False and const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED==False and const.HORIZONTAL_EDGE:
            actions.extend(const.MOVE_LEFT_SEQ_5)
            parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
            print('not first_off_ramp, obstructed front, move left')
            return actions, parameters
        
        elif first_off_ramp==False and const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED and const.HORIZONTAL_EDGE ==True:
            const.MOVE_LEFT_OBSTRUCTED = False
            print('found corner, now rotate left')
            return const.ROTATE_LEFT_AHEAD_LEFT, [{} for _ in range(len(const.ROTATE_LEFT_AHEAD_LEFT))]
        
        elif first_off_ramp==False and const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED==False and const.HORIZONTAL_EDGE==False and const.SCAN_EDGE==False:
            actions.extend(const.INITIAL_MOVE_LEFT_SEQ)
            parameters.extend([{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))])
            const.TEST = True
            actions.extend(const.STICKY_MOVE_AHEAD_2)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_2))])
            print('not first_off_ramp, obstructed front, move left and forward as a test ')
            return actions, parameters
        
        else:
            off_ramp_mode= False
            const.SCAN_EDGE = False
            const.RAMP = False
            record=False
            const.LEVEL_BALL_SCANNED = False
            const.LEVEL_RAMP_SCANNED = False
#             const.REVERSE = True
            print('cant detect ramp, try to get off anyways ')
            print(recorded_action)
            return const.MOVE_LEFT_AHEAD_2, [{} for _ in range(len(const.MOVE_LEFT_AHEAD_2))]
        
            
    else:
        for idx, res in loc_result.iterrows():
            if (res['name'] == pred_class) and (res['confidence'] > const.threshold):
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
                    if pred_class==const.RAMPS:
#                         #becasue yolo detects right corner of ramp so move left and ahead
#                         actions.extend(const.MOVE_LEFT_SEQ_10)
#                         parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_10))])
                        actions.extend(const.STICKY_MOVE_AHEAD_2)
                        parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_2))])
                        const.SCAN_RAMP = True
                        const.ROTATE_COUNT = 0
                    actions.extend(const.ROTATE_RIGHT_AHEAD)
                    parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_AHEAD))])
                if cw > right_border:
                    actions.extend(const.ROTATE_LEFT_AHEAD)
                    parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_AHEAD))])
#                 print(actions, 'coming from navigate')
#                 print(left_border, cw, right_border)

    return actions, parameters

def find_ball(img, model):
    global threshold
    predictions = model(img, size=640)
    loc_result = predictions.pandas().xyxy[0]
    # print("Loc Result:",loc_result)
    found = False
    top_left = (0, 0)
    bottom_right = (0, 0)
    for idx, res in loc_result.iterrows():
        # print("FOUND:",res['name'],":",res['confidence'])
        if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > threshold) and (found == False):
            found = True
            top_left = (int(res['xmin']), int(res['ymin']))
            bottom_right = (int(res['xmax']), int(res['ymax']))
    return found, top_left, bottom_right
    
def ramp_detect(output):
    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size)
    img_pil.putdata(pixels)
    img_array = np.array(img_pil)
    img = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
    img = np.float32(img)

    corners = cv.goodFeaturesToTrack(img, 100, 0.01, 10)
    corners = np.int0(corners)
    return len(corners)

def negative_loc(new_list):
    count = 0
    loc = []
    for number in new_list:
        count += 1      #moved it outside of the if
        if number < 0:
            loc.append(count)
    return loc

def horizontal_detector(output,num):
    #change max_list to a more inclusive range when detecting 
    #since ramp can be at an angle 
    # Detect horizontal lines
    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size) 
    img_pil.putdata(pixels)
    img_array = np.array(img_pil)
    img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    edges = cv.Canny(image=img_blur, threshold1=20, threshold2=40)

    w, h = int(img.shape[1]), int(img.shape[0])
    cw, ch = int(w / 2), int(h / 2)

    thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40,1))
    detect_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    
    for c in cnts:
        cv.drawContours(img, [c], -1, (36,255,12), 2)
        
    #cv.imwrite('detector_h.png', img)

    y_x =dict()
    symmetric = 0

    if len(get_xy_list_from_contour(cnts))>1:
        for i in range(len(get_xy_list_from_contour(cnts))):
            z = get_xy_list_from_contour(cnts)[i]
            for ix in range(len(z)):
                if z[ix][1] in y_x.keys():
                    y_x[z[ix][1]].append(z[ix][0])
                else:
                    y_x.update({z[ix][1]:[z[ix][0]]})
    elif len(get_xy_list_from_contour(cnts))==1:
        z = get_xy_list_from_contour(cnts)[0]
        for ix in range(len(z)):
                if z[ix][1] in y_x.keys():
                    y_x[z[ix][1]].append(z[ix][0])
                else:
                    y_x.update({z[ix][1]:[z[ix][0]]})
    
    new_dict = dict()

#     print(y_x)
    for ix, x in enumerate(y_x.keys()):
        if 4>len(y_x[x])>1:
            if max(y_x[x]) - min(y_x[x]) >num:
                if 350>x >50:
                    new_dict.update({x:y_x[x]})
        
                    #y coordinate has to be bigger than 10 (from top of screen)
                    symmetric += 1
                    
    return new_dict, symmetric

    

def vertical_detector(output,num, num2):
    #detect vertical lines
    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size) 
    img_pil.putdata(pixels)
    img_array = np.array(img_pil)
    img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    edges = cv.Canny(image=img_blur, threshold1=20, threshold2=40)

    w, h = int(img.shape[1]), int(img.shape[0])
    cw, ch = int(w / 2), int(h / 2)

    thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,10))
    detect_vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv.findContours(detect_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        cv.drawContours(img, [c], -1, (36,255,12), 2)
        
    #cv.imwrite('detector_v.png', img)
    
    y_x =dict()
    symmetric = 0

    if len(get_xy_list_from_contour(cnts))>2:
        for i in range(len(get_xy_list_from_contour(cnts))):
            z = get_xy_list_from_contour(cnts)[i]
            for ix in range(len(z)):
                if z[ix][0] in y_x.keys():
                    y_x[z[ix][0]].append(z[ix][1])
                else:
                    y_x.update({z[ix][0]:[z[ix][1]]})
    elif len(get_xy_list_from_contour(cnts))==1:
        z = get_xy_list_from_contour(cnts)[0]
        for ix in range(len(z)):
                if z[ix][0] in y_x.keys():
                    y_x[z[ix][0]].append(z[ix][1])
                else:
                    y_x.update({z[ix][0]:[z[ix][1]]})

    new_dict = dict()
    for ix, x in enumerate(y_x.keys()):
        if 5>len(y_x[x])>1:
            if max(y_x[x]) - min(y_x[x]) >num:
                if 550>x >num2:
                    new_dict.update({x:y_x[x]})
            
                    
                    #y coordinate has to be bigger than 10 (from top of screen)
                    symmetric += 1
    return new_dict, symmetric

def get_xy_list_from_contour(contours):
    full_dastaset = []
    for contour in contours:
        xy_list=[]
        for position in contour:
            [[x,y]] = position
            xy_list.append([x,y])
        full_dastaset.append(xy_list)
    return full_dastaset

def find_ramp(output, num):
    new_list = []
    y_x_ = horizontal_detector(output,num)[0]
    y_x = dict()
    if isinstance(y_x_, dict):
        for x in y_x_.keys():
            if (10<x<390) & ((max(y_x_[x]) - min(y_x_[x])) >num):
                y_x.update({x:y_x_[x]})
        if len(y_x.keys())>0:
            
            return max(y_x.keys())
        else:
            return 0
    else:
        return 0
    
        
    

def select_action(output, model):
    #output: output = controller.step(action[idx], **params[idx])
    global first_action, epoch, on_ramp, off_ramp_mode, max_list, min_list

    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size) 
    img_pil.putdata(pixels)
    #putdata():
    #Copies pixel data from a flattened sequence object into the image. 
    #The values should start at the upper left corner (0, 0), continue 
    #to the end of the line, followed directly by the first value of the 
    #second line, and so on. Data will be read until either the image or 
    #the sequence ends. The scale and offset values are used to adjust the 
    #sequence values: pixel = value*scale + offset.
    img_array = np.array(img_pil)
    img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
    #cv2.cvtColor() method is used to convert an image from one color space to another.
    if not first_action:
        img = img[:, 50:]
        
    display_image = copy.deepcopy(img)
    predictions = model(img, size=640)
    cw = int(display_image.shape[1] / 2)
    ch = int(display_image.shape[0] / 2) #is this getting the center coordinate?
    cv.rectangle(display_image, (cw - 5, ch - 5), (cw + 5, ch + 5), 255, 2)
    #cv2.rectangle(image, start_point, end_point, color, thickness)
    actions, parameters, predicted_classes = [], [], []
    loc_result = predictions.pandas().xyxy[0]
    print(loc_result)
    # results.xyxy[0]  # im predictions (tensor)
    # results.pandas().xyxy[0]  # im predictions (pandas)
    # #      xmin    ymin    xmax   ymax  confidence  class    name
    # # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    for idx, res in loc_result.iterrows():
        #loc_result= predictions.pandas().xyxy[0]
        if res['confidence'] >= const.threshold:
            predicted_classes.append(res['name'])
    print("Loc Result:", loc_result)
    print("predicted_classes:", predicted_classes)

    if const.SPORTS_BALL in predicted_classes:
        const.SCENE_HAS_SOCCER_BALL = True
        create_bounding_box(display_image, loc_result, const.SPORTS_BALL)

    #cv.imwrite("moving_ball_scene" + str(epoch) + ".png", display_image)
    epoch = epoch + 1
    if first_action and const.SPORTS_BALL in predicted_classes:
        first_action = False
        return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output)
    elif first_action and const.SPORTS_BALL not in predicted_classes:
        print('ball not detected now spinning to scan ')
        first_action = False
        const.SCAN_BALL = True
        const.ROTATE_COUNT = 1
        return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
    
    if first_action==False and const.SPORTS_BALL not in predicted_classes and const.SCAN_BALL and const.TEST==False:
        print('not first action, but still spin to try to detect ball')
        const.ROTATE_COUNT +=1
        if const.ROTATE_COUNT == 12: 
            const.SCAN_BALL = False
            const.SCAN_RAMP = True
            const.ROTATE_COUNT = 0
            const.LEVEL_BALL_SCANNED = True
        return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
    elif first_action==False and const.SPORTS_BALL not in predicted_classes and const.SCAN_BALL and const.TEST:
        const.TEST = False
        print('just got off ramp now reverse and scan')
        const.FRESH_OFF = True
        return const.REVERSE_ACTION, [{} for _ in range(len(const.REVERSE_ACTION))]
    
    elif first_action==False and const.SPORTS_BALL not in predicted_classes and const.RAMP and const.REVERSE:
        const.RAMP = False
        const.REVERSE=False
        const.SCAN_BALL = True
        print('just got off ramp now reverse and scan')
        const.FRESH_OFF = True
        return const.REVERSE_ACTION, [{} for _ in range(len(const.REVERSE_ACTION))]
    
    elif first_action==False and const.SPORTS_BALL in predicted_classes and const.SCAN_BALL:
        #if we find ball while scanning
        print('not first action, scanning for ball and found ball')
        const.ROTATE_COUNT = 0
        const.SCAN_BALL = False
        const.SCAN_RAMP = False
        const.LEVEL_BALL_SCANNED = True
        actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output)
        for idx, res in loc_result.iterrows():
            if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):
                if res['ymax']>350:
                    actions.extend(const.LOOK_DOWN_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
        return actions, parameters
        
#add if ball detected, if > ch on the same level, on floor or anotehr ramp


    if first_action==False and const.SPORTS_BALL in predicted_classes and const.MOVE_AHEAD_OBSTRUCTED==False:
    #if ball still detected but obstructed (means we're on a ramp & ball is either on the floor
    # of on another ramp), then record ball location & record all actions from now on
        actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output)
        for idx, res in loc_result.iterrows():
            if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):
                if res['ymax']>350:
                    actions.extend(const.LOOK_DOWN_SEQ_3)
                    parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
        
        
        if epoch > 6:
            actions.extend(const.PICK_UP_SEQUENCE)
            for act in const.PICK_UP_SEQUENCE:
                parameters.extend(
                    [{
                         "objectImageCoordsX": res['xmin']+(res['xmax']-res['xmin'])//2, "objectImageCoordsY":res['ymin']+(res['ymax']-res['ymin'])//2
                     } if act == 'PickupObject' else {}]
                )
        return actions, parameters
    elif first_action==False and const.SPORTS_BALL in predicted_classes and const.MOVE_AHEAD_OBSTRUCTED==True:
        for idx, res in loc_result.iterrows():
            if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):
                
                if res['ymax']<200:
                    print('ball on another ramp')
                    off_ramp_mode = True
                    return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output) 
                elif res['ymax']>200:
                    print('ball on the level below')
                    off_ramp_mode = True
                    return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output) 

    if const.BRUTE:
        print('brute force, move ahead until obstructed')
        if const.MOVE_AHEAD_OBSTRUCTED==False and const.MOVE_LEFT_OBSTRUCTED==False:
            const.RAMP = True
            on_ramp =True
            print('found ramp, go up ')
            return const.ROTATE_RIGHT_AHEAD_LEFT, [{} for _ in range(len(const.ROTATE_RIGHT_AHEAD_LEFT))]

        elif const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED==False:
            print('found corner, now rotate left')
            return const.ROTATE_LEFT_AHEAD_RIGHT, [{} for _ in range(len(const.ROTATE_LEFT_AHEAD_RIGHT))]
        elif const.MOVE_AHEAD_OBSTRUCTED==False and const.MOVE_RIGHT_OBSTRUCTED ==False:
            const.RAMP = True
            on_ramp =True
            const.BRUTE=False
            print('found ramp, go up ')
            return const.ROTATE_RIGHT_AHEAD_LEFT, [{} for _ in range(len(const.ROTATE_RIGHT_AHEAD_LEFT))]

    
    print('const.RAMP', const.RAMP)
    print('const.SCAN_RAMP', const.SCAN_RAMP)
    print('const.MOVE ahead', const.MOVE_AHEAD_OBSTRUCTED)
    
    if first_action==False and const.RAMP==False and const.SCAN_RAMP and const.RAMPS not in predicted_classes and off_ramp_mode==False:
        print('no ramp detected in scan, spin to try to detect up ramp')
        
        const.ROTATE_COUNT += 1
        if const.FRESH_OFF:
            if const.ROTATE_COUNT <4:
                min_list.append(ramp_detect(output))
                print('min_list', min_list)
                max_list.append(find_ramp(output, 50))
                print('max_list', max_list)
                print('min_list', min_list)
        else:
            min_list.append(ramp_detect(output))
            max_list.append(find_ramp(output, 50))
            print('max_list', max_list)
            print('min_list', min_list)
        if const.ROTATE_COUNT == 12: 
            const.BRUTE_COUNT += 1
            const.FRESH_OFF=False
            const.ROTATE_COUNT = 0
            
            for i in range(0,max_list.index(max(max_list))+1):
                actions.extend(const.ROTATE_RIGHT_SEQ_3)
                parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))])
            actions.extend(const.STICKY_MOVE_AHEAD_2)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_2))])
            const.LEVEL_RAMP_SCANNED = True
            const.FRESH_OFF=False
                
            print('rotate right x3 # of times', max_list.index(min(max_list)))
            if len(actions)>0:
                const.RAMP=True
            print('finished spinning for up ramp detection, now analyze corners to identify ramp')
            print('BRUTE COUNT', const.BRUTE_COUNT)
            max_list = []
            min_list= []
            return actions, parameters

        return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]
    
    elif first_action == False and const.BRUTE_COUNT>2:
        const.BRUTE =True
        const.ROTATE_COUNT+=1
        max_list.append(find_ramp(output, 50))
        print('max_list', max_list)
        if const.ROTATE_COUNT == 12: #add condition "and ball_on_ramp = True (ch)"
            const.ROTATE_COUNT = 0
            
            for i in range(0,max_list.index(max(max_list))+1):
                actions.extend(const.ROTATE_RIGHT_SEQ_3)
                parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))])
            actions.extend(const.STICKY_MOVE_AHEAD_2)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_2))])
            const.LEVEL_RAMP_SCANNED = True
        return actions, parameters

    elif first_action==False and const.RAMP==False and const.SCAN_RAMP and const.RAMPS in predicted_classes and off_ramp_mode==False:
        return navigate(loc_result, cw, ch, actions, parameters, const.RAMPS,img,output)

    elif first_action==False and const.RAMP and const.SCAN_RAMP:
        print('ramp detected in scan, now navigate towards it')
        const.SCAN_RAMP = False
        on_ramp = True
        const.SCAN_RAMP_EDGE = True
        actions.extend(const.LOOK_DOWN_SEQ_3)
        parameters.extend([{} for _ in range(len(const.LOOK_DOWN_SEQ_3))])
        actions.extend(const.ROTATE_LEFT_SEQ_10)
        parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_SEQ_10))])
        actions.extend(const.ROTATE_LEFT_SEQ_10)
        parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_SEQ_10))])
        const.ROTATE_COUNT=0
        const.ROTATE_RIGHT_COUNT = 0 
        return actions, parameters
    
        
    elif first_action==False and on_ramp and const.SCAN_RAMP_EDGE:
        const.ROTATE_COUNT += 1
        const.ROTATE_RIGHT_COUNT +=1
        
        if const.ROTATE_COUNT == 30: #add condition "and ball_on_ramp = True (ch)"
            print('rotated 10 times, didnt find edge, scan ramp again')
            const.ROTATE_COUNT = 0
            const.ROTATE_RIGHT_COUNT =0
            const.SCAN_RAMP_EDGE = False
            const.SCAN_RAMP = True
            const.RAMP =False
            const.FRESH_OFF = True
            actions.extend(const.ROTATE_LEFT_SEQ_10)
            parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_SEQ_10))])
            actions.extend(const.LOOK_UP_SEQ_3)
            parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ_3))])
            return actions, parameters  
        
        
        y_x_cor, sym1 = horizontal_detector(output,200)
        print('sym1', sym1)
        
            

        if 1<=sym1<=2 and const.ROTATE_RIGHT_COUNT>5:
            for z in y_x_cor.keys():
                left = min(y_x_cor[z])
                right = max(y_x_cor[z])
                print('left coordinate', left)
                
            if left>200:
                actions.extend(const.MOVE_RIGHT_SEQ_5)
                parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_5))])
                
            elif right<400:
                actions.extend(const.MOVE_LEFT_SEQ_5)
                parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
                
            
            print('found horizontal edge, now going up ramp')
            actions.extend(const.LOOK_UP_SEQ_3)
            parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ_3))])
            actions.extend(const.STICKY_MOVE_AHEAD)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
            const.SCAN_RAMP_EDGE=False
            return actions, parameters
        
        elif 1<=sym1<=2 and const.ROTATE_RIGHT_COUNT < 5:
            print('already on ramp')
            actions.extend(const.LOOK_UP_SEQ_3)
            parameters.extend([{} for _ in range(len(const.LOOK_UP_SEQ_3))])
            actions.extend(const.STICKY_MOVE_AHEAD)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
            
            for iz in range(0,(20-const.ROTATE_RIGHT_COUNT+1)):
                actions.extend(const.ACTION_ROTATE_RIGHT)
                parameters.extend([{}])
            
            
            const.SCAN_RAMP_EDGE=False
            return actions, parameters
        
        else:
            print('rotate left to find horizontal ramp line')
            return const.ACTION_ROTATE_RIGHT, [{}]


        

    elif first_action==False and on_ramp and const.MOVE_AHEAD_OBSTRUCTED==False and const.SCAN_RAMP==False:
        print('on ramp, move ahead not obstructed, keep moving till obstructed')
        const.ROTATE_COUNT = 0
        const.LEVEL_RAMP_SCANNED = False
        const.LEVEL_BALL_SCANNED = False
        
        x_y_cor, sym2 = vertical_detector(output,200,50)
        print('sym2', sym2)
        
        if 1<=sym2<=2:
         
            left_most = min(x_y_cor.keys())
            right_most = max(x_y_cor.keys())
            print('left most line detected', left_most)
            
            if left_most > 100:
                actions.extend(const.MOVE_RIGHT_SEQ_5)
                parameters.extend([{} for _ in range(len(const.MOVE_RIGHT_SEQ_5))])
                
            elif right_most <500:
                actions.extend(const.MOVE_LEFT_SEQ_5)
                parameters.extend([{} for _ in range(len(const.MOVE_LEFT_SEQ_5))])
                
            
            print('found vertical edge, now moving left/right')
            actions.extend(const.STICKY_MOVE_AHEAD)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
            const.SCAN_RAMP_EDGE=False
            return actions, parameters
        else:
            actions.extend(const.STICKY_MOVE_AHEAD)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
            const.SCAN_RAMP_EDGE=False
            return actions, parameters   
        
    
    elif first_action==False and on_ramp and const.MOVE_AHEAD_OBSTRUCTED:
        print('on top of ramp now, scan for ball again')
        on_ramp=False
        const.SCAN_BALL=True
        const.RAMP = False
        return const.ROTATE_RIGHT_SEQ_3, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))]

    


        
    if first_action==False and const.RAMP and off_ramp_mode == True:
        off_ramp_mode= False
        record=False
        const.LEVEL_BALL_SCANNED = False
        const.LEVEL_RAMP_SCANNED = False
        const.REVERSE = True
        #add condition if ball > ch then reverse true
        print('ramp in sight, off_ramp =true, going off ramp now')
        return const.STICKY_MOVE_AHEAD_3, [{} for _ in range(len(const.STICKY_MOVE_AHEAD_3))]
    
    elif first_action==False and const.RAMP==False and off_ramp_mode ==True:
        print('no ramp in sight, off_ramp =true, trying to find a way to get off ramp')
        return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output)
    
    
    

    if first_action==False and const.RAMP==False and const.SCAN_EDGE and off_ramp_mode:
        print('no edge detected in scan, spin to try to detect edge')
        return navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,img,output)
    
    print('const.LEVEL_RAMP_SCANNED', const.LEVEL_RAMP_SCANNED)
    if const.LEVEL_BALL_SCANNED and const.LEVEL_RAMP_SCANNED and const.RAMP == False:
        print('const.LEVEL_RAMP_SCANNED', const.LEVEL_RAMP_SCANNED)
        print('this level scanned ball & ramp')
        #on ramp need to get down
        off_ramp_mode = True
        if const.RAMP==False:
            print('no ramp in sight')
            return navigate(loc_result, cw, ch, actions, parameters, const.RAMP, img,output)
        else:
            off_ramp_mode=False
            print('ramp in sight')
            return navigate(loc_result, cw, ch, actions, parameters, const.RAMP, img,output)
        
    if const.BRUTE_COUNT>2:
        const.BRUTE =True
        const.ROTATE_COUNT+=1
        max_list.append(find_ramp(output, 50))
        actions.extend(const.ROTATE_RIGHT_SEQ_3)
        parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))])
        print('max_list', max_list)
        if const.ROTATE_COUNT == 12: #add condition "and ball_on_ramp = True (ch)"
            const.ROTATE_COUNT = 0
            
            for i in range(0,max_list.index(max(max_list))+1):
                actions.extend(const.ROTATE_RIGHT_SEQ_3)
                parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_SEQ_3))])
            actions.extend(const.STICKY_MOVE_AHEAD_3)
            parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD_3))])
            const.LEVEL_RAMP_SCANNED = True
        print('brute started')
        return actions, parameters
    elif const.BRUTE:
        print('brute force, move ahead until obstructed')
        if const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED==False:
            print('found corner, now rotate left')
            return const.MOVE_LEFT_SEQ_5, [{} for _ in range(len(const.MOVE_LEFT_SEQ_5))]
        elif const.MOVE_AHEAD_OBSTRUCTED and const.MOVE_LEFT_OBSTRUCTED and const.MOVE_RIGHT_OBSTRUCTED==False:
            return const.ROTATE_LEFT_AHEAD_RIGHT, [{} for _ in range(len(const.ROTATE_LEFT_AHEAD_RIGHT))]
        elif const.MOVE_AHEAD_OBSTRUCTED==False and const.MOVE_RIGHT_OBSTRUCTED ==False and const.MOVE_RIGHT_OBSTRUCTED==False:
            const.RAMP = True
            const.BRUTE = False
            on_ramp =True
            print('found ramp, go up ')
            return const.ROTATE_RIGHT_AHEAD_LEFT, [{} for _ in range(len(const.ROTATE_RIGHT_AHEAD_LEFT))]
        elif const.MOVE_AHEAD_OBSTRUCTED==False:
            return const.STICKY_MOVE_AHEAD, [{} for _ in range(len(const.STICKY_MOVE_AHEAD))]

            

    if actions is None or len(actions) == 0:
        actions = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'PickupObject']
        parameters = [{}, {}, {}, {}, {"objectImageCoordsX": 300, "objectImageCoordsY":100}]
    return actions, parameters

if __name__ == '__main__':
    first_action = True
    first_off_ramp = True
    record = False
    recorded_action = [const.ACTION_PASS]
    off_ramp_mode=False
    on_ramp = False
    max_list = []
    min_list = []

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
    action = const.ACTION_LOOK_DOWN
    params = [{}]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_v11.pt')

    epoch = 0
    move_ahead_count = 0

    while action != '':
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print("Actions to execute: ", action)
        const.MOVE_AHEAD_OBSTRUCTED = False
        const.MOVE_LEFT_OBSTRUCTED = False
        const.MOVE_RIGHT_OBSTRUCTED = False
        if record == True:
            print('trying to get off ramp, actions recorded')
            recorded_action.extend(action)
        for idx in range(len(action)):
            output = controller.step(action[idx], **params[idx])
            if output is None:
                controller.end_scene()
                exit()
            if action[idx] == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed ahead.")
                const.MOVE_AHEAD_OBSTRUCTED = True
                print(const.MOVE_AHEAD_OBSTRUCTED)
            if action[idx] == const.ACTION_MOVE_LEFT[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed left.")
                const.MOVE_LEFT_OBSTRUCTED = True
            if action[idx] == const.ACTION_MOVE_RIGHT[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed right.")
                const.MOVE_RIGHT_OBSTRUCTED = True
            
            if action[idx] == 'PickupObject':
                print(output.return_status)
            if action[idx] == 'PickupObject' and output.return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)

        action, params = select_action(output, model)
        if const.SCAN_RAMP:
            const.threshold = 0.20
        else:
            const.threshold = 0.40

    controller.end_scene()
    

