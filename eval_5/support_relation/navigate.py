import copy

import cv2 as cv
import numpy as np
from PIL import Image

import functions as func
import constants as const
import sys
sys.path.insert(1, '..')
from functions import Region


left_or_right = "unknown"


def create_bounding_box(img, loc_result, pred_class):
    color = (0, 128, 0) if pred_class == const.SPORTS_BALL else (255, 0, 0)
    for idx, res in loc_result.iterrows():
        if (res['name'] == pred_class) and (res['confidence'] > const.threshold):
            cv.rectangle(img, (int(res['xmin']), int(res['ymax'])), (int(res['xmax']), int(res['ymin'])), color, 2)


def navigate_to_door(ball_region):
    actions = []
    if ball_region == func.Region.unknown:
        raise ValueError("Ball not found in scene!")
    elif ball_region == func.Region.left and const.FIRST_ACTION:
        #actions = const.INITIAL_MOVE_LEFT_SEQ
        #actions.extend(['MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft'])
        actions = ['MoveLeft'] * 22
        const.FIRST_ACTION = False
    elif ball_region == func.Region.right and const.FIRST_ACTION:
        #actions = const.INITIAL_MOVE_RIGHT_SEQ
        #actions.extend(['MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight'])
        actions = ['MoveRight'] * 22
        const.FIRST_ACTION = False

    actions.extend(const.STICKY_MOVE_AHEAD)
    params = [{} for _ in range(len(actions))]
    print(actions)
    return actions, params


def navitage_to_ball(output, model, epoch, ball_region):
    global left_or_right, obstructed
    image = output.image_list[0]
    pixels = list(image.getdata())
    img_pil = Image.new(image.mode, image.size)
    img_pil.putdata(pixels)
    img_array = np.array(img_pil)
    img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
    display_image = copy.deepcopy(img)
    predictions = model(img, size=640)
    cw = int(display_image.shape[1] / 2)
    ch = int(display_image.shape[0] / 2)
    cv.rectangle(display_image, (cw - 5, ch - 5), (cw + 5, ch + 5), 255, 2)
    actions, parameters, predicted_classes = [], [], []
    loc_result = predictions.pandas().xyxy[0]
    ball_x = 300
    ball_y = 200
    for idx, res in loc_result.iterrows():
        if res['confidence'] >= const.threshold:
            predicted_classes.append(res['name'])
            if res['name'] == 'Ball':
                ball_x = (res['xmax'] + res['xmin']) / 2
                ball_y = (res['ymax'] + res['ymin']) / 2
                const.RIGHT_WIDTH = res['xmax']
    #print("Loc Result:", loc_result)
    #print("predicted_classes:", predicted_classes)
    create_bounding_box(display_image, loc_result, 'Ball')
    #cv.imwrite("ball_scene" + str(epoch) + ".png", display_image)
    actions = []
    # Consider the scene being divided into 3 equal parts, left part, right part, middle part, the ball can either be anywhere in these parts, and more likely to be in
    # the lower part of these 3 parts
    if left_or_right == "unknown":
        # First four passes, check the middle part, currently it is checking the rectangle in front of it, the rectangle is a 2*4 region
        # Namely
        # * * * *
        # * * * *
        # you will understand if you run the scene
        if epoch <= 108 and 'Ball' not in predicted_classes:
            actions.extend(['MoveRight'])
        elif epoch <= 112 and 'Ball' not in predicted_classes:
            actions.extend(['MoveLeft'])
        elif epoch <= 114 and 'Ball' not in predicted_classes:
            actions.extend(['MoveRight'])
        elif epoch <= 117 and 'Ball' not in predicted_classes:
            actions.extend(['LookDown', 'Pass'])
        elif epoch <= 119 and 'Ball' not in predicted_classes:
            actions.extend(['MoveRight'])
        elif epoch <= 123 and 'Ball' not in predicted_classes:
            actions.extend(['MoveLeft'])
        elif epoch <= 125 and 'Ball' not in predicted_classes:
            actions.extend(['MoveRight'])
        # That means there is no ball in the middle part, then move inside and prepare to check left and right
        elif epoch <= 126 and 'Ball' not in predicted_classes:
            actions.extend(['MoveAhead']*3)
        # Now rotate to the left, and check the left part
        elif epoch <= 131 and 'Ball' not in predicted_classes:
            actions.extend(['RotateLeft', 'Pass'])
        # Left part is clear, check the right part
        elif 'Ball' not in predicted_classes:
            actions.extend(['RotateRight', 'Pass'])
        # the ball is found
        else:
            #actions.extend(['MoveAhead'] * 3)
            print(epoch, left_or_right)
            if epoch <= 125:
                left_or_right = "middle"
            elif epoch <= 131:
                left_or_right = "left"
            else:
                left_or_right = 'right'
    elif left_or_right == 'middle':
        # We found it in the middle part of the scene but for some reason we cannot pick it up
        if 'Ball' in predicted_classes:
            actions.extend(['PickupObject'])
            if ball_y >= image.size[1] / 2:
                actions.extend(['LookDown'])
            if ball_x <= image.size[0] / 3:
                actions.extend(['MoveLeft'])
            elif ball_x >= image.size[0] - image.size[0] / 3:
                actions.extend(['MoveRight'])
            else: 
                actions.extend(['MoveAhead'])
            params = [{} for _ in range(len(actions))]
            params[0] = {'objectImageCoordsX': ball_x, 'objectImageCoordsY': ball_y}
            return actions, params
        else:
            actions.extend(['MoveBack'])
    elif left_or_right == 'right':
        if 'Ball' not in predicted_classes:
            actions.extend(['RotateRight', 'Pass'])
        elif const.MOVE_AHEAD_OBSTRUCTED:
            actions.extend(['RotateRight', 'MoveAhead', 'Pass'])
        else:
            actions.extend(['MoveAhead'])
    else:
        if 'Ball' not in predicted_classes:
            actions.extend(['RotateLeft', 'Pass'])
        elif const.MOVE_AHEAD_OBSTRUCTED:
            actions.extend(['RotateLeft', 'MoveAhead', 'Pass'])
        else:
            actions.extend(['MoveAhead'])
    actions.extend(['PickupObject'])
    params = [{} for _ in range(len(actions))]
    print(ball_x, ball_y)
    params[-1] = {'objectImageCoordsX': ball_x, 'objectImageCoordsY': ball_y}
    print(actions)
    return actions, params


def select_action(output, model, ball_region, epoch):
    if const.MOVE_AHEAD_OBSTRUCTED:
        if const.SCENE_HAS_SOCCER_BALL:
            return navitage_to_ball(output, model, epoch, ball_region)
        else:
            const.SCENE_HAS_SOCCER_BALL = True
            return ['OpenObject'], [{'objectImageCoordsX': 300, 'objectImageCoordsY': 200}]
    else:
        if epoch <= 106:
            return navigate_to_door(ball_region)
        else:
            return navitage_to_ball(output, model, epoch, ball_region)
