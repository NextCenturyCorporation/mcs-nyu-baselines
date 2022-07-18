import cv2 as cv
import numpy as np
from PIL import Image

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
    print("Indices Left:", indices_l)

    indices_r = np.where(edges_r != [0])
    print("Indices Right:", indices_r)
    print("Shape:", img_blur_l.shape)
    s = 'None'
    if ((len(indices_l[0]) == 0) and (len(indices_r[0]) == 0)):
        s = 'None'
    elif (len(indices_l[0]) != 0):
        new_indices_l = img_blur_l.shape[1] - indices_l[1]
        distance = 70 + max(new_indices_l)
        print("Ldistance:", distance)
        s = 'Left'
    elif (len(indices_r[0]) != 0):
        distance = 70 + max(indices_r[1])
        print("Rdistance:", distance)
        s = 'Right'

    # print("Lava:",s,":",len(indices[0]))
    return s, distance


def find_region(img, top_left, bottom_right, ball_region):
    global l_lim, r_lim
    cw, ch = int(img.shape[1] / 2), int(img.shape[0] / 2)
    bcw, bch = int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2)
    cv.rectangle(img, (cw - 5, ch - 5), (cw + 5, ch + 5), 255, 2)
    cv.rectangle(img, (bcw - 3, bch - 3), (bcw + 3, bch + 3), 255, 2)
    cv.line(img, (cw - l_lim, 0), (cw - l_lim, img.shape[0] - 1), (0, 255, 0), 2)
    cv.line(img, (cw + r_lim, 0), (cw + r_lim, img.shape[0] - 1), (0, 255, 0), 2)
    if bcw > (cw + r_lim):
        ball_region = Region.right
    # ("Rotate Right")
    elif bcw < (cw - l_lim):
        ball_region = Region.left
    # print("Rotate Left")
    elif bcw >= (cw - l_lim) and bcw <= (cw + r_lim):
        ball_region = Region.center
    else:
        print("Unique Condition")
    return ball_region


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
        if (res['name'] == 'sports ball') and (res['confidence'] > threshold) and (found == False):
            found = True
            top_left = (int(res['xmin']), int(res['ymin']))
            bottom_right = (int(res['xmax']), int(res['ymax']))
    return found, top_left, bottom_right


def select_actions(output, model):
    global ball_drop, prev_img, ball_region, l_lim, r_lim
    image = output.image_list[0]
    img_pil = Image.new(image.mode, image.size)
    img_pil.putdata(list(image.getdata()))
    img = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    cw, ch = int(img.shape[1] / 2), int(img.shape[0] / 2)
    if ball_region == Region.unknown:
        ball_found, top_left, bottom_right = find_ball(img, model)
        if ball_found:
            ball_region = find_region(img, top_left, bottom_right, ball_region)
            print("Ball region:", ball_region)

    if output.step_number == 2:
        a, b = find_lip(img, 0)
        if a == 'Left':
            l_lim = b
        elif a == 'Right':
            r_lim = b
        elif a == 'None':
            a, b = find_lip(img, 0)
            if a == 'Left':
                l_lim = b
            elif a == 'Right':
                r_lim = b

    actions = ['Pass']
    params = [{} for _ in range(len(actions))]
    print("Final Ball region:", ball_region)
    saveimage(img, output.step_number)
    return actions, params
