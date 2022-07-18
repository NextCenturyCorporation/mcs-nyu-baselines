import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import sys
sys.path.insert(1, '..')
import constants as const

hold_threshold = 0.2
ball_threshold = 0.5
neartheball = False
explore_right = 1
prev_img = 0
import enum

l_lim = 92
r_lim = 92

class Region(enum.Enum):
    center = 1
    left = 2
    right = 3
    unknown = 4

ball_region = Region.unknown
lip_region = Region.unknown
ball_backup_region = Region.unknown
hold_backup_region = Region.unknown

def saveimage(img, stepn):
    if stepn < 10:
        cv.imwrite("output_images/lava" + str(stepn) + ".png", img)
    elif stepn < 100:
        cv.imwrite("output_images/lavb" + str(stepn) + ".png", img)
    else:
        cv.imwrite("output_images/lavc" + str(stepn) + ".png", img)


def find_lip(img, i):
    w, h = int(img.shape[1]), int(img.shape[0])
    cw, ch = int(w / 2), int(h / 2)
    if i == 0:
        right_crop = img[ch + 30:ch + 80, cw + 70:cw + 200]  # top_left[0]:bottom_right[0]]
        left_crop = img[ch + 30:ch + 80, cw - 200:cw - 70]  # top_left[0]:bottom_right[0]]
    elif i == 1:
        right_crop = img[ch + 80:ch + 100, cw + 70:cw + 200]  # top_left[0]:bottom_right[0]]
        left_crop = img[ch + 80:ch + 100, cw - 200:cw - 70]  # top_left[0]:bottom_right[0]]
    #cv.line(img,(cw-70,0),(cw-70,img.shape[0]-1),(0,255,0),2)
    #cv.line(img,(cw-180,0),(cw-150,img.shape[0]-1),(0,255,0),2)
    #cv.line(img,(cw+70,0),(cw+70,img.shape[0]-1),(255,0,0),2)
    #cv.line(img,(cw+180,0),(cw+150,img.shape[0]-1),(255,0,0),2)
    #cv.line(img,(cw+103,0),(cw+103,img.shape[0]-1),(255,0,0),2)

    img_gray_r = cv.cvtColor(right_crop, cv.COLOR_BGR2GRAY)
    img_blur_r = cv.GaussianBlur(img_gray_r, (3, 3), 0)
    # thresold1 and threshold2 are set to high values so that the shadows can be filtered
    edges_r = cv.Canny(image=img_blur_r, threshold1= 60, threshold2= 90)
    cv.imwrite('RCrop.png', right_crop)
    cv.imwrite('RCanny.png', edges_r)

    img_gray_l = cv.cvtColor(left_crop, cv.COLOR_BGR2GRAY)
    img_blur_l = cv.GaussianBlur(img_gray_l, (3, 3), 0)
    edges_l = cv.Canny(image=img_blur_l, threshold1= 60, threshold2= 90)
    cv.imwrite('LCrop.png', left_crop)
    cv.imwrite('LCanny.png', edges_l)

    indices_l = np.where(edges_l != [0])
    # coordinates_l=zip(indices_l[0],indices_l[1])
    #print("Indices Left:", indices_l)

    indices_r = np.where(edges_r != [0])
    #print("Indices Right:", indices_r)
    #print("Shape:", img_blur_l.shape)
    s = 'None'
    distanc_r = 95
    distance_l = 95
    if ((len(indices_l[0]) == 0) and (len(indices_r[0]) == 0)):
        s = 'None'
    if (len(indices_l[0]) != 0):
        new_indices_l = img_blur_l.shape[1] - indices_l[1]
        distance = 70 + max(new_indices_l)
        #print("Ldistance:", distance)
        #s = 'Left'
        distance_l = distance
    if (len(indices_r[0]) != 0):
        distance = 70 + max(indices_r[1])
        #print("Rdistance:", distance)
        #s = 'Right'
        distance_r = distance

    # print("Lava:",s,":",len(indices[0]))
    #return s, distance
    return distance_l, distance_r

# find the pixels used for computing areas (it is another unstable risk, to compute the area, what I currently do is to locate a pixel 
# at (0.875 * height, 0.5* width)) and compute how many pixels that are close to it to compute the area. I tried to replace it with using a segmentation model
# but this will done unless current solutions are not satisfying enough
def detect_pixel(image):
    height = image.shape[0]
    width = image.shape[1]
    hold_pixel = image[int(height) - int(height / 8)][int(width / 2)]
    return hold_pixel

def compute_area(image):
    image = np.array(image)
    print(image.shape)
    hold_pixel = detect_pixel(image)
    area = 0
    for x in image:
        for y in x:
            # this is what I consider as pixels that are close to the hold_pixel
            if np.abs(int(y[0]) - int(hold_pixel[0])) <= 5 and np.abs(int(y[1]) - int(hold_pixel[1])) <= 5 and np.abs(int(y[2]) - int(hold_pixel[2])) <= 5:
                area += 1
    print(area)
    return area

# consider the bounding box of the ball. part of the ball will be occluded, so we should use the width of the bounding box to construct a square, use the total
# area of this bounding box's on-table area to minus bounding box's off-table area to compute the threshold
def determine_threshold(ball_top_left, ball_bottom_right, lim, left_right):
    length = ball_bottom_right[0] - ball_top_left[0]
    if left_right == 0:
        dif = - min(max(- lim + ball_bottom_right[0], 0), length) + min(max(- ball_top_left[0] + lim, 0), length)
    else:
        dif = min(max(- lim + ball_bottom_right[0], 0), length) - min(max(- ball_top_left[0] + lim, 0), length)
    return dif * length

def find_region(img_pil, img, top_left, bottom_right, ball_top_left, ball_bottom_right, ball_region):
    global l_lim, r_lim
    cw, ch = int(img.shape[1] / 2), int(img.shape[0] / 2)
    bcw, bch = int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2)
    #cv.rectangle(img, (bcw - 3, bch - 3), (bcw + 3, bch + 3), 255, 2)

    # do the segmentation, store the output image in image.png (see segmentation function definition) and Image.open it
    segmentation(img)
    img_pil = Image.open("./image.png")
    
    #The commented codes are my first thoughts, if the bounding box has more than 1/11 off the table, then we consider this object is going to fall
    #
    #if bottom_right[0] - (cw + r_lim) > 10 * (cw + r_lim - top_left[0]):
    #    ball_region = Region.right
    ## ("Rotate Right")
    #elif 10 * ( - top_left[0] + (cw - l_lim)) > (- cw + l_lim + bottom_right[0]):
    #    ball_region = Region.left
    ## print("Rotate Left")
    #elif bottom_right[0] - (cw + r_lim) <= 10 * (cw + r_lim - top_left[0]) and 10 * ( - top_left[0] + (cw - l_lim)) <= (- cw + l_lim + bottom_right[0]):
    #    ball_region = Region.center
    #else:
    #    print("Unique Condition")

    # Five situations, situation #1, #3, #5, the hold and the ball must be entirely on the left, right, center (this will be just like solidity scene)
    # situation #2 #4, the hold and the ball is on the edge and likely to fall. In this case, we compute the area of the hold (not ball) on the table
    # and off the table, and compute how big the area difference should be in order not to fall (use determine_threshold function)
    # If the on-the-table-area fails to be greater than off-the-table-area by the threshold, we consider the ball would fall down.

    # I admit that this situation is not well dealt with, one possible solution is to completely replace yolo with (panoptic) segmentaion model, 
    # I tried UNet with 4 layers but turns out that it is not working well

    # the ball is completely on the left
    if bottom_right[0] <= cw - l_lim:
        ball_region = Region.left

    # the ball is on the left edge
    elif bottom_right[0] > cw - l_lim and bottom_right[0] <= cw + r_lim and top_left[0] <= cw - l_lim:
        left_crop = img_pil.crop((top_left[0], top_left[1], cw - l_lim , bottom_right[1]))
        right_crop = img_pil.crop((cw - l_lim, top_left[1], bottom_right[0] , bottom_right[1]))
        right_area = compute_area(right_crop)
        left_area = compute_area(left_crop)
        print("threshold", determine_threshold(ball_bottom_right, ball_top_left, cw - l_lim, 0))
        if left_area >= right_area - determine_threshold(ball_bottom_right, ball_top_left, cw - l_lim, 0):
            ball_region = Region.left
        else:
            ball_region = Region.center
    
    # the ball is on the center
    elif bottom_right[0] > cw - l_lim and bottom_right[0] <= cw + r_lim and top_left[0] > cw - l_lim and top_left[0] <= cw + r_lim:
        ball_region = Region.center

    # the ball is on the right edge
    elif bottom_right[0] > cw + r_lim and top_left[0] > cw - l_lim and top_left[0] <= cw + r_lim:
        left_crop = img_pil.crop((top_left[0], top_left[1], cw + r_lim , bottom_right[1]))
        right_crop = img_pil.crop((cw + r_lim, top_left[1], bottom_right[0] , bottom_right[1]))
        right_area = compute_area(right_crop)
        left_area = compute_area(left_crop)
        print("threshold", determine_threshold(ball_bottom_right, ball_top_left, cw + r_lim, 1))
        if left_area <= right_area + determine_threshold(ball_bottom_right, ball_top_left, cw + r_lim, 1):
            ball_region = Region.right
        else:
            ball_region = Region.center

    # the ball is on the right
    elif bottom_right[0] > cw + r_lim and top_left[0] > cw + r_lim:
        ball_region = Region.right

    # Ideally this should never happen
    else:
        print("Unique Condition")
    return ball_region

# use yolo on the original image, and compute the area on the segmentated image
# to prevent different scenes move in different paces (some scenes may have balls settled in 20 steps, some may not have the balls settled in 40 steps), I
# check whether the hold stops to decided when to compute the region that the balls are supposed to be at 
# (use top_left and bottom_right returned by this function to call find_region, see find_region for more detail)
#
# However, YOLO might not be so robust to noise of different scenes, leading me to think whether I need to prepare some backup solutions
# So two backup solutions are made:
# 1. (with higher priority) as long the ball and hold are found in the scene, we consider the hold has been completely captured in the scene (no occulsion)
# We find region according to x coodinates of the hold
# 2. (with lower priority) do the same thing as solidity scene, check where the center of the ball lies
def find_support(img, model):
    global ball_threshold, hold_threshold, ball_backup_region, r_lim, l_lim
    predictions = model(img, size=640)
    loc_result = predictions.pandas().xyxy[0]
    print("Loc Result:",loc_result)
    found = False
    hold_found = False
    top_left = (0, 0)
    bottom_right = (0, 0)
    ball_top_left = (0, 0)
    ball_bottom_right = (0, 0)
    ball_found = False
    hold_confidence = 0
    ball_confidence = 0
    for idx, res in loc_result.iterrows():
        w = img.shape[1]
        h = img.shape[0]
        #print("FOUND:",res['name'],":",res['confidence'])
        # res['confidence'] > hold_confidence and abs(res['ymax'] - const.YMAX) <= 5 ====> the hold is completely put down and will not move, time to record its region
        if (res['name'] == 'Hold') and (res['confidence'] > hold_threshold) and res['confidence'] > hold_confidence and abs(res['ymax'] - const.YMAX) <= 5:
            hold_confidence = res['confidence']
            found = True
            hold_found = True
            ball_found = True
            top_left = (int(res['xmin']), int(res['ymin']))
            bottom_right = (int(res['xmax']), int(res['ymax']))
        # the hold_backup_region, record whenever both hold and ball are detected in the scene (hold_found and ball_found are both True) 
        elif (res['name'] == 'Hold') and (res['confidence'] > hold_threshold) and res['confidence'] > hold_confidence:
            hold_confidence = res['confidence']
            hold_found = True
            top_left = (int(res['xmin']), int(res['ymin']))
            bottom_right = (int(res['xmax']), int(res['ymax']))
            # keep updating const.YMAX, this is the criteria to tell whehter the hold is completely put down and will not move
            const.YMAX = res['ymax']
        # the ball_backup_region, use the center of the ball to decide the regions
        if res['name'] == 'Ball' and res['confidence'] > ball_threshold and res['confidence'] > ball_confidence:
            ball_confidence = res['confidence']
            ball_top_left = (int(res['xmin']), int(res['ymin']))
            ball_bottom_right = (int(res['xmax']), int(res['ymax']))
            # only considering ball detected if it has been put down low enough
            if (res['ymin'] + res['ymax']) / 2 >= h / 4:
                ball_found = True
            if ball_backup_region == Region.unknown:
                if ((res['xmin'] + res['xmax'])) / 2 < (w / 2 - l_lim):
                    ball_backup_region = Region.left
                elif (w / 2 - l_lim) <= ((res['xmin'] + res['xmax'])) / 2  and ((res['xmin'] + res['xmax'])) / 2 < (w / 2 + r_lim):
                    ball_backup_region = Region.center
                elif ((res['xmin'] + res['xmax'])) / 2 >= (w / 2 + r_lim):
                    ball_backup_region = Region.right
                else:
                    ball_backup_region = Region.unknown
    # (hold_found and ball_found) == True ===> time to record the hold_backup_region
    # found == True ====> hold is completely put down and stops moving, time to record its region
    return found, top_left, bottom_right, ball_bottom_right, ball_top_left, (hold_found and ball_found)

def prediction_process(predictions):
    xyxy = predictions.pandas().xyxy
    ball, hold = None
    for item in xyxy:
        if item['name'] == 'Ball':
            if ball is None:
                item.to_numpy()

# Use KMeans algorithm to do segmentation in order to achieve pixel unification, so we can compute the accurate area
def segmentation(image):
    cv.imwrite("test2.png", image)
    data = np.array(image.reshape((image.shape[0] * image.shape[1], image.shape[2])))
    kmeans = KMeans(n_clusters=15, random_state=0).fit(data)
    labels = kmeans.labels_.reshape((image.shape[0], image.shape[1]))
    a, b = labels.shape
    final_image = np.ones(image.shape)
    for i in range(a):
        for j in range(b):
            final_image[i][j] = kmeans.cluster_centers_[labels[i][j]]
    cv.imwrite("./image.png", final_image)
 
def select_actions(output, model):
    global prev_img, ball_region, l_lim, r_lim, ball_backup_region, hold_backup_region
    image = output.image_list[0]
    img_pil = Image.new(image.mode, image.size)
    img_pil.putdata(list(image.getdata()))
    img = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    cw, ch = int(img.shape[1] / 2), int(img.shape[0] / 2)
    if ball_region == Region.unknown:
        hold_found, top_left, bottom_right, ball_top_left, ball_bottom_right, backup_hold_found = find_support(img, model)
        # record the region
        if hold_found:
            ball_region = find_region(img_pil, img, top_left, bottom_right, ball_top_left, ball_bottom_right, ball_region)
        # record the hold_backup_region
        elif hold_backup_region == Region.unknown and backup_hold_found:
            hold_backup_region = find_region(img_pil, img, top_left, bottom_right, ball_top_left, ball_bottom_right, ball_region)
        # ball_backup_region has been recorded in find_region function

    if output.step_number == 95 and ball_region == Region.unknown:
        if hold_backup_region == Region.unknown:
            ball_region = ball_backup_region
        else:
            ball_region = hold_backup_region

    if output.step_number == 2:
        a, b = find_lip(img, 0)
        l_lim = a
        r_lim = b
        print(a, b)

    actions = ['Pass']
    params = [{} for _ in range(len(actions))]
    print("Final Ball region:", ball_region, ball_backup_region, hold_backup_region)
    #saveimage(img, output.step_number)
    return actions, params
