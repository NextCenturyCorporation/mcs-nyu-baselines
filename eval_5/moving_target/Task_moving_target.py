#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        cv.imwrite('ROI_{}.png'.format(idx), occ_img)
        rows, cols = occ_img.shape
        number_of_white_pix = sum(180 <= occ_img[i][j] <= 255 for i in range(rows) for j in range(cols)) 
        # using white pixels to identify soccor ball in the enlarged bounding box?
        print("number_of_white_pix for idx ", idx, ": ", number_of_white_pix)
        if number_of_white_pix > 0:
            const.SCENE_HAS_SOCCER_BALL = True
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


# def find_ball(loc_result, cw, first_action_):
#     ##loc_result: 
#     # results.pandas().xyxy[0]  # im predictions (pandas)
#     # #      xmin    ymin    xmax   ymax  confidence  class    name
#     # # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
#     # # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
#     # # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
#     # cw = int(display_image.shape[1] / 2) 
    
#     # returns: move right sequences of actions if x-min of boundary box around ball 
#     #is to the right of the center of image, left otherwise
# #     global first_side
#     for idx, res in loc_result.iterrows():
#         if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold) and (first_action_==True):
#             if res['xmin'] > cw: 
#                 first_side = 'right'
#                 return const.INITIAL_MOVE_RIGHT_SEQ_1, [{} for _ in range(len(const.INITIAL_MOVE_RIGHT_SEQ_1))] 
#             else:
#                 first_side = 'left'
#                 return const.INITIAL_MOVE_LEFT_SEQ_1, [{} for _ in range(len(const.INITIAL_MOVE_LEFT_SEQ_1))]
            
#         elif (res['name'] == const.AGENT) and (res['confidence'] > const.threshold) and (first_action_==False):
#             if res['xmin'] > cw: 
#                 return const.MOVE_RIGHT_SEQ_1, [{} for _ in range(len(const.MOVE_RIGHT_SEQ_1))] 
#             else:
#                 return const.MOVE_LEFT_SEQ_1, [{} for _ in range(len(const.MOVE_LEFT_SEQ_1))]

                
        
            

def navigate(loc_result, cw, ch, actions, parameters, pred_class, epoch):
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
                actions.extend(const.ROTATE_RIGHT_AHEAD)
                parameters.extend([{} for _ in range(len(const.ROTATE_RIGHT_AHEAD))])
            if cw > right_border:
                actions.extend(const.ROTATE_LEFT_AHEAD)
                parameters.extend([{} for _ in range(len(const.ROTATE_LEFT_AHEAD))])
            if res['ymax'] >350:
                actions.extend(const.ACTION_LOOK_DOWN)
                parameters.extend([{} for _ in range(len(const.ACTION_LOOK_DOWN))])
        
        if epoch > 6:#why 6?
            actions.extend(const.PICK_UP_SEQUENCE)
            for act in const.PICK_UP_SEQUENCE:
                parameters.extend(
                    [{
                         "objectImageCoordsX": res['xmin']+(res['xmax']-res['xmin'])//2, "objectImageCoordsY":res['ymin']+(res['ymax']-res['ymin'])//2
                     } if act == 'PickupObject' else {}]
                )
        
        print(actions, 'coming from navigate')
        print(epoch, 'epoch')
        print(left_border, cw, right_border)
        print((res['xmax']-res['xmin'])//2, (res['ymax']-res['ymin'])//2)
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
            top_left = (int(res['xmin']), int(res['ymax']))
            bottom_right = (int(res['xmax']), int(res['ymin']))
    return found, top_left, bottom_right     

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
        return const.ACTION_MOVE_AHEAD[0], [{}]
    elif (len(indices_l[0]) != 0):
        if (len(indices_r[0]) != 0) and (max(indices_l[1]) > img_blur_l.shape[1]-20) and (max(indices_r[1]) > img_blur_r.shape[1]-20):
            if const.DIRECTION =='right':
                return const.LAVA_SEQ_L, [{} for _ in range(len(const.LAVA_SEQ_L))]
            else:
                return const.LAVA_SEQ_R, [{} for _ in range(len(const.LAVA_SEQ_R))]
            
        elif (len(indices_r[0]) != 0) and (max(indices_l[1]) > img_blur_l.shape[1]-20) and (max(indices_r[1]) > img_blur_r.shape[1]/4):
            return const.LAVA_SEQ_L, [{} for _ in range(len(const.LAVA_SEQ_L))]
        
        elif (len(indices_r[0]) != 0) and (min(indices_r[1]) < 20) & (min(indices_l[1]) < img_blur_l.shape[1]/2):
            return const.LAVA_SEQ_R, [{} for _ in range(len(const.LAVA_SEQ_R))]
        
        else:
            return const.ACTION_PASS, [{}]


def select_action(output, model):
    #output: output = controller.step(action[idx], **params[idx])
    
    global first_action, epoch, first_sighting, second_sighting,second_step
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

#     if const.SPORTS_BALL in predicted_classes:
#         create_bounding_box(display_image, loc_result, const.NON_AGENT)

    if const.SPORTS_BALL in predicted_classes:
        const.SCENE_HAS_SOCCER_BALL = True
        create_bounding_box(display_image, loc_result, const.SPORTS_BALL)

    cv.imwrite("moving_ball_scene" + str(epoch) + ".png", display_image)
    epoch = epoch + 1
    if epoch == 1:
        return const.ACTION_PASS, [{}]
    if first_action and const.SPORTS_BALL in predicted_classes:
        first_action = False
        second_step=True
        const.SCENE_HAS_SOCCER_BALL = True
        
        for idx, res in loc_result.iterrows():
            if res['name']==const.SPORTS_BALL:
                first_sighting = (res['xmin']+res['xmax'])/2
                print(first_sighting, 'first_sighting')
                return const.ACTION_PASS, [{}]
            else:
                print('ball in predicted class but not found?')
                return const.ACTION_ROTATE_LEFT, [{}]
            
    elif first_action and const.SPORTS_BALL not in predicted_classes:
        print('first action, ball not in predicted class')
        return const.ROTATE_LEFT_SEQ, [{} for _ in range(len(const.ROTATE_LEFT_SEQ))]
    
    
    
    
    if second_step and const.SPORTS_BALL in predicted_classes:
        second_step=False
        for idx, res in loc_result.iterrows():
            if res['name']==const.SPORTS_BALL:
                second_sighting = (res['xmin']+res['xmax'])/2
                
                if first_sighting < second_sighting:
                    const.DIRECTION = 'right'
                    print('epoch=2, ball found, rotating based on ball roll direction')
                    print(first_sighting, 'first')
                    print(second_sighting, 'second')
                    prev_action = 'right'
                    prev_sighting = second_sighting
                    return const.ROTATE_RIGHT_SEQ_2, [{} for _ in range(len(const.ROTATE_RIGHT_SEQ_2))]
                
                elif first_sighting > second_sighting:
                    const.DIRECTION = 'left'
                    print('epoch=2, ball found, rotating based on ball roll direction')
                    print(first_sighting, 'first')
                    print(second_sighting, 'second')
                    prev_action = 'left'
                    prev_sighting = second_sighting
                    return const.ROTATE_LEFT_SEQ_2, [{} for _ in range(len(const.ROTATE_LEFT_SEQ_2))]
                
                else:
                    print('ball didnt move?')
            
            else:
                print('ball in predicted class but not found?')
                return const.ACTION_ROTATE_LEFT, [{}]

    elif second_step and const.SPORTS_BALL not in predicted_classes:
        print('epoch!=2, ball not in predicted class')
        return const.ACTION_ROTATE_LEFT, [{}]
    

        
    if first_action ==False and second_step==False and const.LAVA in predicted_classes and const.SPORTS_BALL not in predicted_classes:
        print('we wait it out near the lava')
        return find_lip(img,0)
    elif first_action ==False and second_step==False and const.LAVA in predicted_classes and const.SPORTS_BALL in predicted_classes:
        print('epoch>2, moving based on lava')
        actions, parameters = find_lip(img,0)
        for idx, res in loc_result.iterrows():
            if (res['name'] == const.SPORTS_BALL) and (res['confidence'] > const.threshold):
                if epoch > 6:#why 6?
                    actions.extend(navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL)[0],epoch)
                    parameters.extend(navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL)[1],epoch)
                    actions.extend(const.PICK_UP_SEQUENCE)
                    for act in const.PICK_UP_SEQUENCE:
                        parameters.extend(
                            [{
                                 "objectImageCoordsX": res['xmin']+(res['xmax']-res['xmin'])//2, "objectImageCoordsY":res['ymin']+(res['ymax']-res['ymin'])//2
                             } if act == 'PickupObject' else {}]
                        )
                    print((res['xmax']-res['xmin'])//2, (res['ymax']-res['ymin'])//2)
                    return actions, parameters
    elif first_action ==False and second_step==False and const.LAVA not in predicted_classes and const.SPORTS_BALL in predicted_classes:
        print('epoch>2, no lava, navigating to ball')
        actions, parameters = navigate(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL,epoch)
        return actions, parameters
    elif first_action ==False and second_step==False and const.LAVA not in predicted_classes and const.SPORTS_BALL not in predicted_classes:
        print('epoch>2, no lava, ball still not in predicted class')
        if const.DIRECTION == 'left':
            return const.ROTATE_LEFT_SEQ, [{} for _ in range(len(const.ROTATE_LEFT_SEQ))]
        elif const.DIRECTION == 'right':
            return const.ROTATE_RIGHT_SEQ, [{} for _ in range(len(const.ROTATE_LEFT_SEQ))]
        else:
            return const.ROTATE_LEFT_SEQ, [{} for _ in range(len(const.ROTATE_LEFT_SEQ))]
    elif first_action ==False and second_step==False and const.LAVA not in predicted_classes and const.SPORTS_BALL not in predicted_classes:
        print('epoch>2, no lava, ball seen but not in predicted class rn')
        return const.ACTION_PASS, [{}]
    else:
        print('sth fell through the cracks')
       

    if actions is None or len(actions) == 0:
        actions = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'PickupObject']
        parameters = [{}, {}, {}, {}, {"objectImageCoordsX": 300, "objectImageCoordsY":100}]

    return actions, parameters


if __name__ == '__main__':
    first_action = True
    second_step = False
    first_sighting = ''
    second_sighting = ''
#     controller = mcs.create_controller(config_file_or_dict={'metadata': 'oracle'})
    fn = sys.argv[1]
    if os.path.exists(fn):
        scene_data = mcs.load_scene_json_file(fn)
    
#     os.chdir('/Users/Nina/opt/anaconda3/envs/myenv/lib/python3.8/site-packages/machine_common_sense/')

#     controller = mcs.create_controller(config_file_or_dict={'metadata': 'oracle'})
#     scene_data = mcs.load_scene_json_file(glob.glob('./MCS/scenes/baseline_moving_target_prediction_eval_5/*.json')[12])
    output = controller.start_scene(scene_data)
    

    action = const.INITIAL_ROTATE_RIGHT_SEQ
    params = [{} for _ in range(len(action))]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=const.MODEL_WEIGHTS_FILE_PATH_RAMP)

    epoch = 0
    while action != '':
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print("Actions to execute: ", action)
        for idx in range(len(action)):
            output = controller.step(action[idx], **params[idx])
            if action[idx] == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed by occluder.")
                const.MOVE_AHEAD_OBSTRUCTED = True

            if action[idx] == const.ACTION_PICK_UP_OBJ[0]:
                print(output.return_status)
            if action[idx] == const.ACTION_PICK_UP_OBJ[0] and output.return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)
        action, params = select_action(output, model)

    controller.end_scene()


# In[ ]:




