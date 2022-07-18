import argparse

import torch

import functions as f
import machine_common_sense as mcs
import constants as const
from navigate import select_action

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    args = parser.parse_args()
    scene_json_file_path = args.scene_path
    threshold = 0.1
    myrotation = 0
    lookup = 0
    myposition = (0, 0)
    neartheball = False
    explore_right = 1
    prev_img = 0
    l_lim = 60
    r_lim = 60

    # ball_region = Region.unknown
    # lip_region = Region.unknown

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--right_first', default=False, action="store_true")
    #args = parser.parse_args()

    # Unity app file will be downloaded automatically
    controller = mcs.create_controller(config_file_or_dict='../sample_config.ini')
    # mcs.init_logging()
    scene_data = mcs.load_scene_json_file(scene_json_file_path)

    output = controller.start_scene(scene_data)

    # Use your machine learning algorithm to select your next action based on the scene
    # output (goal, actions, images, metadata, etc.) from your previous action.
    action, _ = output.action_list[0]
    # actions = ['LookDown']*2
    actions = ['Pass']
    params = [{} for _ in range(len(actions))]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="./best.pt")

    epoch = 0

    # Continue to select actions until your algorithm decides to stop.
    while actions:
        print("actions: ", actions)
        for idx, action in enumerate(actions):
            # print(output.step_number, action, params[idx], lookup, output.return_status, sep=':')
            output = controller.step(action, **params[idx])
            if action == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed by Door.")
                const.MOVE_AHEAD_OBSTRUCTED = True
                # break
            if action == const.ACTION_PICK_UP_OBJ[0] and output.return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)

        if epoch < 100:
            actions, params = f.select_actions(output, model)
        else:
            actions, params = select_action(output, model, f.ball_region, epoch)
        epoch += 1

    # For interaction-based goals, your series of selected actions will be scored.
    # For observation-based goals, you will pass a classification and a confidence
    # to the end_scene function here.
    controller.end_scene()
