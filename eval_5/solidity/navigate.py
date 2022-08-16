import copy

import cv2 as cv
import numpy as np
from PIL import Image

import functions as func
import constants as const
#from spatial_scene.run_yolo_detector import create_bounding_box

def create_bounding_box(img, loc_result, pred_class):
    color = (0, 128, 0) if pred_class == "Ball" else (255, 0, 0)
    for idx, res in loc_result.iterrows():
        if (res['name'] == pred_class) and (res['confidence'] > const.threshold):
            cv.rectangle(img, (int(res['xmin']), int(res['ymax'])), (int(res['xmax']), int(res['ymin'])), color, 2)

def navigate_to_door(ball_region):
    actions = []
    if ball_region == func.Region.unknown:
        raise ValueError("Ball not found in scene!")
    elif ball_region == func.Region.left and const.FIRST_ACTION:
        actions = const.INITIAL_MOVE_LEFT_SEQ
        actions.extend(['MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft', 'MoveLeft'])
        const.FIRST_ACTION = False
    elif ball_region == func.Region.right and const.FIRST_ACTION:
        actions = const.INITIAL_MOVE_RIGHT_SEQ
        actions.extend(['MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight', 'MoveRight'])
        const.FIRST_ACTION = False

    actions.extend(const.STICKY_MOVE_AHEAD)
    params = [{} for _ in range(len(actions))]

    return actions, params


def navitage_to_ball(output, model, epoch):
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
    print("Loc Result:", loc_result)
    print("predicted_classes:", predicted_classes)
    create_bounding_box(display_image, loc_result, "Ball")
    #cv.imwrite("ball_scene" + str(epoch) + ".png", display_image)
    actions = []
    if epoch <= 107:
        actions.extend(['LookDown'])
        actions.extend(['MoveAhead']*7)
        actions.extend(['Pass'])
    if 'Ball' not in predicted_classes:
        actions.extend(['RotateLeft', 'Pass'])
    elif const.MOVE_AHEAD_OBSTRUCTED:
        actions.extend(['RotateLeft', 'MoveAhead', 'Pass'])
        const.MOVE_AHEAD_OBSTRUCTED = False
    else:
        actions.extend(['MoveAhead'])
        actions.extend(['Pass'])
    #if const.SPORTS_BALL in predicted_classes:
    #    actions.extend(const.STICKY_MOVE_AHEAD)
    #else:
    #    actions.extend(const.STICKY_MOVE_AHEAD)
    #    if const.RIGHT_WIDTH < cw:
    #        actions.extend(const.ACTION_ROTATE_LEFT * 9)
    #        actions.extend(['LookDown', 'LookDown'])
    #    else:
    #        actions.extend(const.ACTION_ROTATE_RIGHT * 9)
    #        actions.extend(['LookDown', 'LookDown'])

    actions.extend(const.PICK_UP_SEQUENCE)
    params = [{} for _ in range(len(actions))]
    #params[-1] = {"objectId": "target"}
    params[-1] = {'objectImageCoordsX': ball_x, 'objectImageCoordsY': ball_y}
    return actions, params


def select_action(output, model, ball_region, epoch):
    if const.MOVE_AHEAD_OBSTRUCTED or epoch >= 106:
        if const.SCENE_HAS_SOCCER_BALL:
            return navitage_to_ball(output, model, epoch)
        else:
            const.SCENE_HAS_SOCCER_BALL = True
            return ['OpenObject'], [{'objectImageCoordsX': 300, 'objectImageCoordsY': 200}]
    else:
        return navigate_to_door(ball_region)
