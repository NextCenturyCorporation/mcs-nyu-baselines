import glob
import os
import random
import sys
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import machine_common_sense as mcs
import torch
import torch.nn as nn
from shapely.geometry import Polygon

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.3
fontColor = (0, 0, 0)
thickness = 1

def rgb_to_torch_img(img_rgb_pil, out_size=64):
    """

    :param img_rgb_pil: image in numpy format. HxWxC where C is ordered in R,G,B
    :return: Pytorch image. BxCxHxW with C=1 and B=1
    """
    img_np = np.asarray(img_rgb_pil)
    img_np = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_NEAREST)
    img_np = np.float32(img_np) / np.float32(255.0)
    edge_map = np.zeros((out_size, out_size), dtype=np.float32)
    ddepth = cv2.CV_32F
    scale = 1
    delta = 0
    for channel in range(3):
        color = img_np[..., channel]
        # grad_x = cv2.Sobel(color, ddepth, 1, 0, ksize=5, scale=scale, delta=delta,
        #                    borderType=cv2.BORDER_DEFAULT)
        # grad_y = cv2.Sobel(color, ddepth, 0, 1, ksize=5, scale=scale, delta=delta,
        #                    borderType=cv2.BORDER_DEFAULT)
        grad_x = cv2.Scharr(color, ddepth, 1, 0, scale=scale, delta=delta,
                            borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Scharr(color, ddepth, 0, 1, scale=scale, delta=delta,
                            borderType=cv2.BORDER_DEFAULT)
        # abs_grad_x = np.abs(grad_x)
        # abs_grad_y = np.abs(grad_y)
        edge_map += np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_map /= 36  # to reduce magnitude
    im = edge_map[np.newaxis, np.newaxis, ...]  # B x C x H x W

    return torch.from_numpy(im).detach()


def torch_to_np_img(torch_img):
    frame_cv2 = torch_img.detach().cpu().numpy().copy()
    frame_cv2 = np.float32(frame_cv2[0, 0])
    frame_cv2 /= 1.2
    frame_cv2 = np.uint8(np.minimum(frame_cv2, 1.0) * 255.)
    return frame_cv2


def initialize_model(frame_predictor, posterior, prior, encoder, decoder, frames, num_past=5):
    frame_predictor.hidden = frame_predictor.init_hidden()
    prior.hidden = prior.init_hidden()
    posterior.hidden = posterior.init_hidden()

    x_out_seq = []

    h = encoder(frames[0])
    for i in range(1, len(frames)):
        h, skip = h
        h_target = encoder(frames[i])
        z_t = posterior(h_target[0])
        prior(h)
        frame_posterior = frame_predictor(torch.cat([h, z_t], 1))
        x_out_seq.append(decoder([frame_posterior, skip]).detach().cpu())
        h = h_target
    prior(h[0])

    return x_out_seq, skip


def model_predict(frame_predictor, prior, encoder, decoder, x_in, skip):
    h, _ = encoder(x_in)
    z_t_hat = prior(h)
    h = frame_predictor(torch.cat([h, z_t_hat], 1)).detach()
    x_out = decoder([h, skip])
    return x_out


def model_predict_multi(frame_predictor, prior, encoder, decoder, x_in, skip, t, pred_imgs, pred_horizon):
    with torch.no_grad():
        x_out = model_predict(frame_predictor, prior, encoder, decoder, x_in, skip)
        x_in = x_out  # predict t + 2

        assert pred_horizon % 2 == 0
        p_hidden = [(h[0].detach().clone(), h[1].detach().clone()) for h in prior.hidden]
        f_hidden = [(h[0].detach().clone(), h[1].detach().clone()) for h in frame_predictor.hidden]
        for j in range(2, pred_horizon + 1, 2):
            x_in = model_predict(frame_predictor, prior, encoder, decoder, x_in, skip)
        pred_imgs[t + pred_horizon] = x_in.detach()
        prior.hidden = p_hidden  # restore hidden states
        frame_predictor.hidden = f_hidden  # restore hidden states
    return x_out


def model_predict_posterior(frame_predictor, posterior, encoder, decoder, x_in, skip):
    h, _ = encoder(x_in)
    z_t = posterior(h)
    h = frame_predictor(torch.cat([h, z_t], 1)).detach()
    x_out = decoder([h, skip])
    return x_out


def high_pass(frame, min=10, rad=5):
    fr = frame.copy()
    fr[fr < min] = 0  # remove low brightness pixels
    if np.sum(fr) != 0:
        frame = fr

    frame = cv2.medianBlur(frame, rad)
    # frame = get_maximal_contour(frame)
    return frame


def get_polygons_from_img(img, top_n_contours=2, min_area=4*4):
    img = high_pass(img.copy(), min=8, rad=5)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:top_n_contours]
    img = cv2.drawContours(img, contours, -1, 180, 1)
    # contours = filter(lambda x: cv2.contourArea(x) > 0, contours)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) == 0:
            continue
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        poly_approx = cv2.approxPolyDP(cnt, epsilon, True)  # shape: NUM_POINTS, 1, 2 (x and y locations)
        poly_approx = np.squeeze(poly_approx, axis=1)  # compress the second dimension -> (NUM_POINTS, 2)
        try:
            poly_approx = Polygon(poly_approx)
            if poly_approx.area < min_area:
                continue
        except ValueError:
            print('too few vertices for polygon. original list: ', poly_approx)
            idx = np.random.choice(list(range(len(poly_approx))), 3, replace=True)
            poly_approx = Polygon(poly_approx[idx])
            print('resampled list: ', poly_approx)
        # poly_approx -= get_center_of_mass(poly_approx)  # center the COM at origin
        c = poly_approx.centroid.coords[0]
        cv2.circle(img, (round(c[0]), round(c[1])), 1, 150, 1)
        polys.append(poly_approx)
    return polys, img


def get_implausibility_score(poly_list1: List[Polygon], poly_list2: List[Polygon], thresh=5):
    # thresh: when max of min distance > thresh, implausibility score > 0.5
    min_dists = []
    max_min_dist = (None, -1)  # (polygon that has max min dist, value)
    for p1 in poly_list1:
        p1_centeroid = p1.centroid
        min_dist = 100000000000
        for p2 in poly_list2:
            min_dist = min(min_dist, p1_centeroid.distance(p2.centroid))
        if min_dist > max_min_dist[1]:
            max_min_dist = (p1, min_dist)
        min_dists.append(min_dist)

    poly_list1, poly_list2 = poly_list2, poly_list1
    for p1 in poly_list1:
        p1_centeroid = p1.centroid
        min_dist = 100000000000
        for p2 in poly_list2:
            min_dist = min(min_dist, p1_centeroid.distance(p2.centroid))
        if min_dist > max_min_dist[1]:
            max_min_dist = (p1, min_dist)
        min_dists.append(min_dist)
    if len(min_dists) == 0:
        return 0, None  # if both the source and prediction has nothing going on, we say it's plausible

    # plausibility = 1 / (max(3, max_min_dist) - 2)  # R+ -> [0,1]
    # return 1 - plausibility
    x = (max_min_dist[
             1] - thresh) / 2  # this means that when MMD >= 4, we have implausibility score >= 0.5 and <0.5 if not.
    implausibility = 1 / (1 + np.exp(-x))
    return implausibility, max_min_dist[0]


def make_step_prediction(choice, confidence, heatmap_img=None):
    if choice == 'plausible':
        choice = 1
    if choice == 'implausible':
        choice = 0
    if heatmap_img is not None:
        xy_list = np.argwhere(heatmap_img >= 200)
        xy_list = [{"x": int(x), "y": int(y)} for x, y in xy_list]
    else:
        xy_list = None
    return {
            "rating": choice,
            "score": confidence,
            "violations_xy_list": xy_list
    }


def run_collision_scene(scene_data, controller, models, base_path="~/logs", num_steps=200, visualize=False, pred_horizon=6):
    report = {}
    frame_predictor, posterior, prior, encoder, decoder = models

    output = controller.start_scene(scene_data)
    MOTION_THRESH = 0.0002

    images = []
    images.append(rgb_to_torch_img(output.image_list[0]).cuda())
    background = None
    motion_start_time = None
    skip = None
    pred_imgs = {}
    predicted_implausible = [False for i in range(num_steps)]
    last_pred = None
    for step in range(num_steps):
        output = controller.step("Pass")
        if output is None:
            break

        if step % 2 != 0:  # stride setting
            if last_pred:
                report[step] = make_step_prediction(choice=last_pred[0], confidence=last_pred[1], heatmap_img=last_pred[2])
            continue
        images.append(rgb_to_torch_img(output.image_list[0]).cuda())
        # if step > 113:  # manually creates "backwards video" implausibility for testing
        #     images[-1] = images[-(step - 113)]
        #     gt_plausible = False
        # else:
        #     gt_plausible = True
        if background is None:
            background = torch_to_np_img(images[-1])
        source_img = torch_to_np_img(images[-1])

        if step < 84:
            report[step] = make_step_prediction(choice='plausible', confidence=1)
            continue

        if motion_start_time is None and background is not None:
            # motion_start_time = step + 1
            x_diff = cv2.absdiff(torch_to_np_img(images[-1]), background)
            x_diff = np.mean(np.float32(x_diff))
            if visualize:
                print(x_diff)
            if x_diff > MOTION_THRESH:
                motion_start_time = step + 2  # we say the motion starts at the second frame after the first significant motion
            report[step] = make_step_prediction(choice="plausible", confidence=1)
        elif step == motion_start_time:
            # feed the 5 most recent frames, excluding this frame, into the model.
            _, skip = initialize_model(frame_predictor, posterior, prior, encoder, decoder, images[-6:-1])
        elif step > motion_start_time:
            if step - motion_start_time <= 4:
                # let the model simply observe for the first 4/2 frames
                pred_t_1_step = model_predict(frame_predictor, prior, encoder, decoder, images[-2], skip)
            else:
                # step - 2 because the stride is 2
                pred_t_1_step = model_predict_multi(frame_predictor, prior, encoder, decoder, images[-2], skip, step - 2,
                                                    pred_imgs, pred_horizon=pred_horizon)
            pred_t_1_step = torch_to_np_img(pred_t_1_step)
            if step in pred_imgs:
                pred_t_5_steps = torch_to_np_img(pred_imgs[step])
                if visualize:
                    cv2.imshow('prediction t 5 steps',
                               cv2.resize(pred_t_5_steps, (384, 384), interpolation=cv2.INTER_NEAREST))
                del pred_imgs[step]
            else:
                pred_t_5_steps = None
                if visualize:
                    cv2.imshow('prediction t 5 steps', np.zeros((384, 384)))

            pred_1_diff = cv2.absdiff(pred_t_1_step, background)
            polys_pred_1, pred_diff_marked = get_polygons_from_img(pred_1_diff)

            if visualize:
                cv2.imshow('source', cv2.resize(source_img, (384, 384), interpolation=cv2.INTER_NEAREST))
                cv2.imshow('prediction t 1 step',
                       cv2.resize(pred_t_1_step, (384, 384), interpolation=cv2.INTER_NEAREST))
            if pred_t_5_steps is not None:
                pred_5_diff = cv2.absdiff(pred_t_5_steps, background)
                polys_pred_5, pred_5_diff_marked = get_polygons_from_img(pred_5_diff)
                imp_score, polygon = get_implausibility_score(polys_pred_1, polys_pred_5)
                imp_heatmap = np.zeros((64, 64), dtype=np.uint8)
                if imp_score > 0.5:
                    pts = np.array(list(polygon.exterior.coords), dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(imp_heatmap, [pts], int(255 * imp_score))
                if imp_score > 0.5:
                    report[step] = make_step_prediction(choice="implausible", confidence=(1 - imp_score), heatmap_img=imp_heatmap.copy())
                    last_pred = ("implausible", 1 - imp_score, imp_heatmap.copy())
                    predicted_implausible[step] = True
                else:
                    report[step] = make_step_prediction(choice="plausible", confidence=(1 - imp_score), heatmap_img=imp_heatmap.copy())
                    last_pred = ("plausible", 1 - imp_score, imp_heatmap.copy())
                    predicted_implausible[step] = False
                if visualize:
                    cv2.putText(imp_heatmap, str(imp_score), (20, 20), font, fontScale, 128, thickness)
                    cv2.imshow('implausibility heatmap',
                               cv2.resize(imp_heatmap, (384, 384), interpolation=cv2.INTER_NEAREST))
            else:
                if visualize:
                    cv2.imshow('implausibility heatmap', np.zeros((384, 384)))
            if visualize:
                cv2.waitKey(0)
    detection_interval = range(80, 160, 2)
    consecutive_implausible = 0
    choice = "plausible"
    implausible_time = ""
    if motion_start_time is None:
        return
    if motion_start_time is not None:  # if no motion is detected, just say plausible
        for t in detection_interval:
            if predicted_implausible[t]:
                consecutive_implausible += 1
            else:
                consecutive_implausible = 0
            if consecutive_implausible == pred_horizon // 2 + 2:
                choice = 'implausible'
                implausible_time = " at t = " + str(t)
                # print('choosing implausible at t={}'.format(t))
                break
    print('choice: {}{}'.format(choice, implausible_time))
    choice = 1 if choice == 'plausible' else 0
    controller.end_scene(rating=choice, score=choice, report=report)


def main(scene_data: dict, unity_app: str = None):
    MCS_CONFIG_FILE_PATH = 'mcs_config.ini'  # NOTE: I ran the tests with option "size: 450". Different sizes might lead to worse results
    # raise AttributeError("Please fill out the unity app executable path and config path")
    controller = mcs.create_controller(
        config_file_or_dict=MCS_CONFIG_FILE_PATH,
        unity_app_file_path=unity_app
    )
    NUMSTEPS = 200
    BATCH_SIZE = 1
    model_path = "trained/CollisionTraining"

    models_dir = glob.glob(f'{model_path}/model_*.pth')
    latest_model = sorted(models_dir, key=lambda s: int(s[s.rfind('_e') + 2: s.rfind('.pth')]), reverse=True)[0]
    print('Loading model ', latest_model)
    saved_model = torch.load(latest_model)
    opt = saved_model['opt']
    frame_predictor = saved_model['frame_predictor'].eval().cuda()
    frame_predictor.batch_size = BATCH_SIZE
    posterior = saved_model['posterior'].eval().cuda()
    posterior.batch_size = BATCH_SIZE
    prior = saved_model['prior'].eval().cuda()
    prior.batch_size = BATCH_SIZE
    decoder = saved_model['decoder'].eval().cuda()
    encoder = saved_model['encoder'].eval().cuda()

    PREDICTION_HORIZON = 4
    run_collision_scene(scene_data, controller, (frame_predictor, posterior, prior, encoder, decoder),
                        num_steps=NUMSTEPS, visualize=False, pred_horizon=PREDICTION_HORIZON)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python <script> <json_scene_filename>')
    scene_data, status = mcs.load_scene_json_file(sys.argv[1])
    if status is not None:
        sys.exit(status)
    #scene_data["history_dir"] = f"{base_path}/log.json"
    main(scene_data)
