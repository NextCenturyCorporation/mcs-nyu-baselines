import argparse

import torch

import functions as f
import machine_common_sense as mcs
import sys
#sys.path.insert(1, '..')
import constants as const
from navigate import select_action
import argparse

if __name__ == '__main__':

    #scene_json_file_path = "/Users/lliu05/eval_5_validation_interactive_debug/eval_5_validation_interactive_support_relations_0001_09_I1_debug.json"
    #scene_json_file_path = "/Users/lliu05/baseline_support_relations_eval_5/baseline_support_relations_01_i1.json"

    # ball_region = Region.unknown
    # lip_region = Region.unknown

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    parser.add_argument(
        '--unity_path',
        type=str,
        default='/home/ubuntu/unity_app/MCS-AI2-THOR-Unity-App-v0.5.7.x86_64'
    )
    args = parser.parse_args()
    scene_json_file_path = args.scene_path

    # Unity app file will be downloaded automatically
    controller = mcs.create_controller(config_file_or_dict='../sample_config.ini', unity_app_file_path=args.unity_path)
    # mcs.init_logging()
    scene_data = mcs.load_scene_json_file(scene_json_file_path)

    output = controller.start_scene(scene_data)

    # Use your machine learning algorithm to select your next action based on the scene
    # output (goal, actions, images, metadata, etc.) from your previous action.
    action, _ = output.action_list[0]
    # actions = ['LookDown']*2
    actions = ['Pass']
    params = [{} for _ in range(len(actions))]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="./best4.pt")

    epoch = 0

    # Continue to select actions until your algorithm decides to stop.
    while actions:
        #print("actions: ", actions)
        const.MOVE_AHEAD_OBSTRUCTED = False
        for idx, action in enumerate(actions):
            #print(output.step_number, action, params[idx], lookup, output.return_status, sep=':')
            #print(action, **params[idx])
            output = controller.step(action, **params[idx])
            if output is None:
                controller.end_scene()
                exit()
            if action == const.ACTION_MOVE_AHEAD[0] and output.return_status == "OBSTRUCTED":
                print("INFO : Move obstructed by Door.")
                const.MOVE_AHEAD_OBSTRUCTED = True
                # break
            if action == const.ACTION_PICK_UP_OBJ[0] and output.return_status == "SUCCESSFUL":
                print("INFO: Picked Up soccer ball. Ending scene! :)")
                controller.end_scene()
                exit(0)
        print(epoch)
        if epoch < 100:
            actions, params = f.select_actions(output, model)
        else:
            actions, params = select_action(output, model, f.ball_region, epoch)
        epoch += 1

    # For interaction-based goals, your series of selected actions will be scored.
    # For observation-based goals, you will pass a classification and a confidence
    # to the end_scene function here.
    controller.end_scene()
