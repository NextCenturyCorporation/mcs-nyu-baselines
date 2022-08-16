#!/usr/bin/env python
# coding: utf-8

# In[2]:


import import_ipynb
import constant as const
import copy
import sys

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import glob
import os

# import spatial_scene.constants as const
import machine_common_sense as mcs



# def check_for_agent(im, loc_result, cw):
#     #if soccer ball is in the vicinity of detected bounding boxes, 
#     #then return move left or move right sequences, else return none
    
#     for idx, res in loc_result.iterrows():
#         # print(res['ymin'])
#         # print("int(res['ymin'] - const.TOP_BOTTOM_CUSHION): ", int(res['ymin'] - const.TOP_BOTTOM_CUSHION))
#         # print("int(res['ymax'] + const.TOP_BOTTOM_CUSHION): ", int(res['ymax'] + const.TOP_BOTTOM_CUSHION))
#         # print("int(res['xmin'] - const.LEFT_RIGHT_CUSHION): ", int(res['xmin'] - const.LEFT_RIGHT_CUSHION))
#         # print("int(res['xmax'] + const.LEFT_RIGHT_CUSHION): ", int(res['xmax'] + const.LEFT_RIGHT_CUSHION))
        
#         #occ_img creates a bigger bounding box? 
#         occ_img = im[
#                   max(int(res['ymin'] - const.TOP_BOTTOM_CUSHION), 0): # TOP_BOTTOM_CUSHION=20
#                   max(int(res['ymax'] + const.TOP_BOTTOM_CUSHION), 0),
#                   max(int(res['xmin'] - const.LEFT_RIGHT_CUSHION), 0): # LEFT_RIGHT_CUSHION=20
#                   max(int(res['xmax'] + const.LEFT_RIGHT_CUSHION), 0)
#                   ]
#         occ_img = cv.cvtColor(occ_img, cv.COLOR_BGR2GRAY)
#         cv.imwrite('ROI_{}.png'.format(idx), occ_img)
#         rows, cols = occ_img.shape
#         number_of_white_pix = sum(180 <= occ_img[i][j] <= 255 for i in range(rows) for j in range(cols)) 
#         # using white pixels to identify soccor ball in the enlarged bounding box?
#         print("number_of_white_pix for idx ", idx, ": ", number_of_white_pix)
#         if number_of_white_pix > 0:
#             const.SCENE_HAS_SOCCER = True
#             if res['xmin'] < cw:
#                 return const.INITIAL_MOVE_LEFT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))]
#             else:
#                 return const.INITIAL_MOVE_RIGHT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ))]
#     return None, None


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

# def find_bigger_occluder(loc_result, cw):
#     ##if any bounding box exceeds width of - sys.maxsize and height >25, then return move left 
#     ##sequence if x-min of that bounding box is on left side of image, return move right sequence otherwise

#     max_width = - sys.maxsize 
#     ## maximum value of Py_ssize_t depending upon the architecture
#     bigger_occ_idx = -1
#     for idx, res in loc_result.iterrows():
#         occ_height = abs(res['ymax'] - res['ymin'])
#         occ_width = abs(res['xmax'] - res['xmin'])
#         if occ_width > max_width and occ_height > 25:
#             max_width = occ_width
#             bigger_occ_idx = idx
#     # print('loc_result: ', type(loc_result))
#     if loc_result['xmin'][bigger_occ_idx] < cw:
#         return const.INITIAL_MOVE_LEFT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ))]
#     else:
#         return const.INITIAL_MOVE_RIGHT_SEQ, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ))]


def find_agent(loc_result, cw, first_action_):
    ##loc_result: 
    # results.pandas().xyxy[0]  # im predictions (pandas)
    # #      xmin    ymin    xmax   ymax  confidence  class    name
    # # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    # cw = int(display_image.shape[1] / 2) 
    
    # returns: move right sequences of actions if x-min of boundary box around ball 
    #is to the right of the center of image, left otherwise
    global first_side
    for idx, res in loc_result.iterrows():
        if (res['name'] == const.AGENT) and (res['confidence'] > const.threshold) and (first_action_==True):
            if res['xmin'] > cw: 
                first_side = 'right'
                return const.INITIAL_MOVE_RIGHT_SEQ_1, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ_1))] 
            else:
                first_side = 'left'
                return const.INITIAL_MOVE_LEFT_SEQ_1, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ_1))]
            
        elif (res['name'] == const.AGENT) and (res['confidence'] > const.threshold) and (first_action_==False):
            if res['xmin'] > cw: 
                return const.MOVE_RIGHT_SEQ_1, [{} for _ in range(len(const.MOVE_RIGHT_SEQ_1))] 
            else:
                return const.MOVE_LEFT_SEQ_1, [{} for _ in range(len(const.MOVE_LEFT_SEQ_1))]
            

def navigate(loc_result, cw, ch, actions, parameters, pred_class):
    ## pred_class: e.g. const.OCCLUDER

    ##if obj in box predicted = pred_class, expand bounding box. 
    ##if expanded box xmin <=cw<=expanded box xmax, then move ahead
    ##if cw<expanded box xmin, move right 
    ##if cw>expanded box xmax, move left 
    ##if ch<expanded box ymin, look down ?? why 
    ##if ch>expanded box ymax, look up 
    ##apend all these actions as u loop through all loc_result and then return all actions.
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
                actions.extend(const.ACTION_ROTATE_RIGHT)
                parameters.extend([{} for _ in range(len(const.ACTION_ROTATE_RIGHT))])
                actions.extend(const.STICKY_MOVE_AHEAD)
                parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
            if cw > right_border:
                actions.extend(const.ACTION_ROTATE_LEFT)
                parameters.extend([{} for _ in range(len(const.ACTION_ROTATE_LEFT))])
                actions.extend(const.STICKY_MOVE_AHEAD)
                parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
#             if ch < top_border:
#                 actions.extend(['LookDown'])
#                 parameters.extend([{}])
#             if ch > bottom_border:
#                 actions.extend(['LookUp'])
#                 parameters.extend([{}])
            print(actions, 'coming from navigate')
            print(left_border, cw, right_border)

    return actions, parameters



def select_action(output, model):
    #output: output = controller.step(action[idx], **params[idx])
    
    global first_action, epoch, first_side
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
        img = img[:, 50:] ##why is this needed? 
        
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
    
    if const.FINAL:
        for idx, res in loc_result.iterrows():
            if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):
                actions.extend(const.PICK_UP_SEQUENCE)
                for act in const.PICK_UP_SEQUENCE:
                    parameters.extend(
                        [{
                             "objectImageCoordsX": res['xmin']+(res['xmax']-res['xmin'])//2, "objectImageCoordsY":res['ymin']+(res['ymax']-res['ymin'])//2
                         } if act == 'PickupObject' else {}])
        return actions, parameters
            
    if const.NON_AGENT in predicted_classes:
        create_bounding_box(display_image, loc_result, const.NON_AGENT)

    if const.AGENT in predicted_classes:
        const.SCENE_HAS_AGENT = True
        create_bounding_box(display_image, loc_result, const.AGENT)
        
    cv.imwrite("agent_scene" + str(epoch) + ".png", display_image)
    epoch = epoch + 1
    if first_action and const.AGENT in predicted_classes:
        first_action = False 
        return find_agent(loc_result, cw, first_action_=True)
    elif first_action and const.AGENT not in predicted_classes:
        print('First step: no agent detected')
        return const.ACTION_MOVE_AHEAD, [{} for _ in range(len(const.ACTION_MOVE_AHEAD))]

#     if first_action and const.NON_AGENT in predicted_classes and const.AGENT not in predicted_classes:
#         actions, parameters = check_for_agent(img, loc_result, cw)
#         #check for ball in expanded bounding boxes

#         if actions is None:
#             actions, parameters = find_bigger_occluder(loc_result, cw) #move based on where x-min of biggest bounding box around occluder???  
#             const.OCCLUDER_IN_FRONT = True #why is this the only other option? can't the soccor just not be in sight? doesn't seem like we've rotated view yet 
#         first_action = False
#         return actions, parameters

#     if const.OCCLUDER_IN_FRONT:
#         #assuming occluder is blocking the soccer ball from view? being directly in front of it?
#         if const.OCCLUDER in predicted_classes:


    if const.AGENT in predicted_classes:
        actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.AGENT)
        if epoch > 6:#why 6?
            for idx, res in loc_result.iterrows():
                if (res['name'] == const.AGENT) and (res['confidence'] > const.threshold):
                    actions.extend(const.INTERACT)
                    for act in const.INTERACT:
                        parameters.extend(
                            [{

                                 'objectImageCoordsX': res['xmin']+ (res['xmax'] - res['xmin']) // 2,
                                 'objectImageCoordsY': res['ymin']+ (res['ymax'] - res['ymin']) // 2
                             } if act == 'InteractWithAgent' else {}]
                        )
                    return actions, parameters
    else:
        if first_side == 'right':
            print('agent not in predicted class, first side =right')
            return const.ACTION_ROTATE_RIGHT, [{} for _ in range(len(const.ACTION_ROTATE_RIGHT))]
        else:
            print('agent not in predicted class, first side =left')
            return const.ACTION_ROTATE_LEFT, [{} for _ in range(len(const.ACTION_ROTATE_LEFT))]
        

    
    if actions is None or len(actions) == 0:
        actions = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'InteractWithAgent']
        parameters = [{}, {}, {}, {}, {'objectImageCoordsX': 300,
                                 'objectImageCoordsY': 200}]

    return actions, parameters


if __name__ == '__main__':
    first_action = True
    first_side = 'left'
#     controller = mcs.create_controller(config_file_or_dict={'metadata': 'oracle'})
    fn = sys.argv[1]
    if os.path.exists(fn):
        scene_data = mcs.load_scene_json_file(fn)

#     os.chdir('/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/')

#     controller = mcs.create_controller(config_file_or_dict={'metadata': 'oracle'})
#     scene_data = mcs.load_scene_json_file(glob.glob('./MCS/scenes/baseline_agent_identification/*.json')[4])
    output = controller.start_scene(scene_data)

    # _, params = output.action_list[0]
    action = const.ACTION_PASS
    params = [{}]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/best.pt')

    epoch = 0
    while action != '':
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print("Actions to execute: ", action)
        for idx in range(len(action)):
            output = controller.step(action[idx], **params[idx])
            if action[idx] == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed by occluder.")
                const.MOVE_AHEAD_OBSTRUCTED = True
            if action[idx] == const.INTERACT[0] and output.return_status == "SUCCESSFUL":
                print("INFO: Successfully interacted with agent")
                for i in range(1,10):
                    output = controller.step('Pass')
                output = controller.step('LookDown')
                output = controller.step('LookDown')
                


                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)
#             if action[idx] == const.ACTION_PICK_UP_OBJ[0] and output.return_status == "SUCCESSFUL":
#                 print("INFO: Picked Up soccer ball. Ending scene! :)")
#                 controller.end_scene()
#                 exit(0)
        action, params = select_action(output, model)

    controller.end_scene()


# In[ ]:




