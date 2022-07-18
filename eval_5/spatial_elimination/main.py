import copy
import sys
import argparse

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import constants as const
import machine_common_sense as mcs


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


def naviagte(loc_result, cw, ch, actions, parameters, pred_class):
    for idx, res in loc_result.iterrows():
        if (res['name'] == pred_class) and (res['confidence'] > const.threshold):
            left_border = res['xmin'] - max(const.LEFT_RIGHT_CUSHION, 0)
            right_border = res['xmax'] + max(const.LEFT_RIGHT_CUSHION, 0)
            top_border = res['ymin'] - max(const.TOP_BOTTOM_CUSHION, 0)
            bottom_border = res['ymax'] + max(const.TOP_BOTTOM_CUSHION, 0)
            actions = actions.copy()
            if left_border <= cw <= right_border:
                actions.extend(const.STICKY_MOVE_AHEAD)
                parameters.extend([{} for _ in range(len(const.STICKY_MOVE_AHEAD))])
                const.LEFT_RIGHT_CUSHION -= 10
                const.TOP_BOTTOM_CUSHION -= 10
            if cw < left_border:
                actions.extend(const.ACTION_MOVE_RIGHT)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_RIGHT))])
            if cw > right_border:
                actions.extend(const.ACTION_MOVE_LEFT)
                parameters.extend([{} for _ in range(len(const.ACTION_MOVE_LEFT))])
            if ch < top_border:
                actions.extend(['LookDown'])
                parameters.extend([{}])
            if ch > bottom_border:
                actions.extend(['LookUp'])
                parameters.extend([{}])
            print(actions)

    return actions, parameters


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

    if const.OCCLUDER in predicted_classes:
        create_bounding_box(display_image, loc_result, const.OCCLUDER)
    if const.SPORTS_BALL in predicted_classes:
        const.SCENE_HAS_SOCCER_BALL = True
        create_bounding_box(display_image, loc_result, const.SPORTS_BALL)
    #cv.imwrite("ball_scene" + str(epoch) + ".png", display_image)
    epoch = epoch + 1
    if first_action and const.SPORTS_BALL in predicted_classes:
        first_action = False
        return find_ball(loc_result, cw)

    if first_action and const.OCCLUDER in predicted_classes and const.SPORTS_BALL not in predicted_classes:
        actions, parameters = check_for_ball(img, loc_result, cw)
        if actions is None:
            actions, parameters = find_bigger_occluder(loc_result, cw)
            const.OCCLUDER_IN_FRONT = True
        first_action = False
        return actions, parameters

    if const.OCCLUDER_IN_FRONT:
        if const.OCCLUDER in predicted_classes:
            actions, parameters = naviagte(loc_result, cw, ch, actions, parameters, const.OCCLUDER)
        elif const.MOVE_AHEAD_OBSTRUCTED:
            actions = const.OCCLUDER_AHEAD_SEQ
            parameters = [{} for _ in range(len(const.OCCLUDER_AHEAD_SEQ))]
        else:
            actions = const.STICKY_MOVE_AHEAD
            parameters = [{} for _ in range(len(const.STICKY_MOVE_AHEAD))]
        if epoch > 6:
            actions = actions.copy()
            actions.extend(const.PICK_UP_SEQUENCE)
            for act in const.PICK_UP_SEQUENCE:
                parameters.extend([
                    {"objectImageCoordsX": 300, "objectImageCoordsY": 100}
                    if act == 'PickupObject' else {}
                ])

    elif const.SCENE_HAS_SOCCER_BALL and const.OCCLUDER in predicted_classes \
            and const.SPORTS_BALL not in predicted_classes:
        actions, parameters = check_for_ball(img, loc_result, cw)

    elif const.SPORTS_BALL in predicted_classes:
        actions, parameters = naviagte(loc_result, cw, ch, actions, parameters, const.SPORTS_BALL)
        if epoch > 6:
            actions = actions.copy()
            actions.extend(const.PICK_UP_SEQUENCE)
            for act in const.PICK_UP_SEQUENCE:
                parameters.extend(
                    [{
                        "objectImageCoordsX": 300,
                        "objectImageCoordsY": 100
                         # 'objectImageCoordsX': (res['xmax'] - res['xmin']) // 2,
                         # 'objectImageCoordsY': (res['ymax'] - res['ymin']) // 2
                     } if act == 'PickupObject' else {}]
                )

    if actions is None or len(actions) == 0:
        actions = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'LookDown', 'PickupObject']
        parameters = [{}, {}, {}, {}, {}, {"objectImageCoordsX": 300, "objectImageCoordsY": 100}]

    return actions, parameters


if __name__ == '__main__':
    first_action = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    parser.add_argument(
        '--unity_path',
        type=str,
        default='/home/ubuntu/unity_app/MCS-AI2-THOR-Unity-App-v0.5.7.x86_64'
    )
    args = parser.parse_args()
    controller = mcs.create_controller(config_file_or_dict='../sample_config.ini', unity_app_file_path=args.unity_path)
    scene_json_file_path = args.scene_path
    scene_data = mcs.load_scene_json_file(scene_json_file_path)

    output = controller.start_scene(scene_data)

    # _, params = output.action_list[0]
    action = const.ACTION_PASS
    params = [{}]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="./best.pt")

    epoch = 0
    while action != '':
        print("#" * 20, " EPOCH: ", epoch, "#" * 20)
        print("Actions to execute: ", action)
        for idx in range(len(action)):
            output = controller.step(action[idx], **params[idx])
            if output is None:
                controller.end_scene()
                exit()
            if action[idx] == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed by occluder.")
                const.MOVE_AHEAD_OBSTRUCTED = True
            if action[idx] == const.ACTION_PICK_UP_OBJ[0] and output.return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)
        action, params = select_action(output, model)

    controller.end_scene()
