import os
import sys
import machine_common_sense as mcs
from glob import glob
from l2_norm_matrix import calc_l2_norms, inference, min_diag_cyclic_perm
from PIL import Image
import numpy as np
import torch
from natsort import natsorted
from absl import app, flags
import json


class baseline_agent:
    def __init__(self, objectid=None):
        self.state = 0
        self.buffer = {}
        self.buffer['ref'] = []
        self.buffer['ball'] = []
        self.buffer['reori'] = []
        self.yaw_offset = None
        self.inventory = []
        self.objectid = objectid

    def _only_action_available(self, last_output, action_name):
        return (
            (len(last_output.action_list) == 1)
            and (last_output.action_list[0][0] == action_name)
        )

    def _align(self, ref, reori):
        assert len(ref) == 36  # hard-coding it to be 36 frames for now
        ref_hash = [inference(x) for x in ref]

        if type(reori) is list:
            assert len(reori) == 36
            reori_hash = [inference(x) for x in reori]
            l2 = calc_l2_norms(ref_hash, reori_hash)
            result = min_diag_cyclic_perm(l2)
        elif type(reori) is Image.Image:
            reori_hash = inference(reori)
            l2 = [((reori_hash - x)**2).sum().sqrt().item() for x in ref_hash]
            result = np.argmin(l2)
        else:
            raise ValueError('`reori` must be `list` or `Image`')
        return result

    def _is_blank_frame(self, frame):
        # all pixels == (0,0,0)
        return np.sum(np.array(frame)) == 0

    def _look_around(
        self,
        last_output,
        start_step,
        buffer_target,
    ):
        elapsed_steps = last_output.step_number - start_step
        if elapsed_steps == 0:
            action, params = ('Pass', {})
        elif 1 <= elapsed_steps < 37:
            self.buffer[buffer_target].append(last_output.image_list[0])
            action, params = ('RotateLeft', {})
        else:
            action, params = ('Pass', {})
        return elapsed_steps, action, params

    def step(self, last_output):
        print('='*40)
        print('Step :', last_output.step_number)
        print('State:', self.state)
        print(last_output.position)
        print(last_output.rotation)

        if self.state == 0:   # take 36 reference frames
            elapsed_steps, action, params = self._look_around(
                last_output=last_output,
                start_step=0,
                buffer_target='ref',
            )
            if elapsed_steps >= 38:
                self.state = 1
                self.start_step = last_output.step_number + 1
                action, params = ('Pass', {})
        elif self.state == 1: # pass until kidnap
            if self._only_action_available(last_output, 'EndHabituation'):
                self.state = 2
                self.start_step = last_output.step_number + 1
                action, params = last_output.action_list[0]
            else:
                action, params = ('Pass', {})
        elif self.state == 2: # look at soccer ball drop; throw away blank frame (kidnap)
            if not self._is_blank_frame(last_output.image_list[0]):
                self.buffer['ball'].append(last_output.image_list[0])
            if self._only_action_available(last_output, 'EndHabituation'):
                self.state = 3
                self.start_step = last_output.step_number + 1
                action, params = last_output.action_list[0]
            else:
                action, params = ('Pass', {})
        elif self.state == 3: # take 36 reorientation frames
            elapsed_steps, action, params = self._look_around(
                last_output=last_output,
                start_step=self.start_step,
                buffer_target='reori',
            )
            if elapsed_steps >= 38:
                self.state = 4
                self.start_step = last_output.step_number + 1
                action, params = ('Pass', {})
        elif self.state == 4: # reorient to the soccer ball corner
            if self.yaw_offset is None:
                # calculate number of times to rotate
                reori = self._align(self.buffer['ref'], self.buffer['reori'])
                ball = self._align(self.buffer['ref'], self.buffer['ball'][0])
                if (ball == 6) or (ball == 24):
                    self.correction = 'right'
                else:
                    self.correction = 'left'
                self.yaw_offset = - reori + ball  # -reori to reset to 0; +ball to look at ball
                if self.yaw_offset < -18:
                    self.yaw_offset += 36
                elif self.yaw_offset > 18:
                    self.yaw_offset -= 36
            if self.yaw_offset != 0:
                if self.yaw_offset > 0:  # rotate CCW
                    self.yaw_offset -= 1
                    action, params = ('RotateLeft', {})
                else:                    # rotate CW
                    self.yaw_offset += 1
                    action, params = ('RotateRight', {})
            else:
                self.state = 5
                self.start_step = last_output.step_number + 1
                action, params = ('Pass', {})
        elif self.state == 5: # walk towards corner
            elapsed_steps = last_output.step_number - self.start_step
            if elapsed_steps < 65:
                if elapsed_steps % 30 == 0:
                    if self.correction == 'right':
                        action, params = ('MoveRight', {})
                    else:
                        action, params = ('MoveLeft', {})
                else:
                    action, params = ('MoveAhead', {})
            else:
                self.state = 6
                self.start_step = last_output.step_number + 1
                action, params = ('Pass', {})
        elif self.state == 6: # look down
            elapsed_steps = last_output.step_number - self.start_step
            if elapsed_steps < 5:
                action, params = ('LookDown', {})
            else:
                self.state = 7
                self.start_step = last_output.step_number + 1
                action, params = ('Pass', {})
        elif self.state == 7: # pick up ball
            objs = [x for x in last_output.object_list if x.mass == 1]
            action = 'PickupObject'
#            params = {
#                "objectImageCoordsX": 224,
#                "objectImageCoordsY": 285
#            }
            params = {"objectId": self.objectid}
            #if len(objs) > 0:
            #    params = {'objectId': objs[0].uuid}
            #    self.inventory.append(objs[0].uuid)
            #else:
            #    raise ValueError("Can't find the object")

            self.state = 8
            self.start_step = last_output.step_number + 1
        elif self.state == 8:
            action, params = ('Pass', {})

        return action, params


FLAGS = flags.FLAGS
flags.DEFINE_string('scene_file', './reori_0001_01.json', 'Path to scene file')


def main(scene_data: dict, unity_app: str = None):
    controller = mcs.create_controller(
        config_file_or_dict="mcs_config.ini",
        unity_app_file_path=unity_app
    )

    objectid = [f for f in scene_data["objects"] if f["type"] =="soccer_ball"][0]["id"]

    output = controller.start_scene(scene_data)
    agent = baseline_agent(objectid=objectid)
    for i in range(1000):
        action, params = agent.step(output) 
        output = controller.step(action, **params)
        if agent.state == 8:
            break

    controller.end_scene()


if __name__=='__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python <script> <json_scene_filename>')
    scene_data, status = mcs.load_scene_json_file(sys.argv[1])
    if status is not None:
        sys.exit(status)
    main(scene_data)
