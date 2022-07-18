import machine_common_sense as mcs
from PIL import Image
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import argparse
threshold = 0.1
myrotation=0
lookup=0
myposition=(0,0)
neartheball=False
explore_right=1

parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str)
parser.add_argument('--right_first',default=False, action="store_true")
args = parser.parse_args()
scene_json_file_path = args.scene_path
if args.right_first:
	print("Right First")
	explore_right=1
else:
	print("Left First")
	explore_right=0

def update_pos(action):
	if action=='RotateRight':
		myrotation=myrotation+10
	elif action=='RotateLeft':
		myrotation=myrotation-10
	elif action=='MoveAhead':
		myposition=(myposition[0],myposition[1])


def find_lava(image,i):
	img_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
	img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
	#edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)
	#edges = cv.Canny(image=img_blur, threshold1=30, threshold2=70)
	edges = cv.Canny(image=img_blur, threshold1=10, threshold2=20)
	cv.imwrite('Canny'+str(i)+'.png',edges)
					
	indices=np.where(edges!=[0])
	coordinates=zip(indices[0],indices[1])
	s=True
	if (len(indices[0])==0):
		s=False
	print("Lava:",s,":",len(indices[0]))
	print("Indieces:",indices)
	return s, indices

def find_actions(img,top_left,bottom_right,i):
	global lookup, neartheball,explore_right
	cw,ch = int(img.shape[1]/2), int(img.shape[0]/2)
	bcw,bch = int((top_left[0]+bottom_right[0])/2),int((top_left[1]+bottom_right[1])/2)
	cv.rectangle(img,(cw-5,ch-5),(cw+5,ch+5),255,2)
	cv.rectangle(img,(bcw-3,bch-3),(bcw+3,bch+3),255,2)
	#cv.rectangle(img,top_left, bottom_right, 255,2)
	if bcw>(cw+80):
		actions=['RotateRight']
		#("Rotate Right")
	elif bcw<(cw-80):
		actions=['RotateLeft']
		#print("Rotate Left")
	elif bcw>=(cw-80) and bcw<=(cw+80):
		#if img.shape[0]>bottom_right[1]+70:
		if img.shape[0]>bottom_right[1]+90:
			#crop_point=max(bottom_right[1]+5,img.shape[0]-100)
			crop_point=max(bottom_right[1]+5,img.shape[0]-100)
			#crop = img[crop_point:,cw-80:cw+80]#top_left[0]:bottom_right[0]]
			crop = img[crop_point:,cw-140:cw+140]#top_left[0]:bottom_right[0]]
			cv.imwrite("Crop"+str(i)+".png",crop)
			is_lava,indices=find_lava(crop,i)
			if not is_lava:
				size_of_ball=bottom_right[0] - top_left[0]
				print("Size of ball:",bottom_right[0]-top_left[0])
				if size_of_ball<60:
					actions=['MoveAhead']
				else:
					actions=['MoveAhead']
				if size_of_ball>135:
					neartheball=True					
			else:
				if explore_right==1:
					actions=['MoveRight']
				elif explore_right==0:
					actions=['MoveLeft']
				#if indices[0][-1]>=90:
				if indices[0][-1]>=20:
					actions.append('MoveBack')
					print("Trying to move back")
				
		else:
			actions=['LookDown']
			lookup=lookup-1
			neartheball=True
#			print("Look down and try to pickup")
	else:
		#print("Unique Condition")
		actions=['RotateRight']
	return actions
	
def find_ball(img):
	global threshold
	predictions = model(img,size=640)
	loc_result = predictions.pandas().xyxy[0]
	#print("Loc Result:",loc_result)
	found = False
	top_left=(0,0)
	bottom_right=(0,0)
	for idx, res in loc_result.iterrows():
		if (res['name']=='sports ball') and (res['confidence'] > threshold) and (found==False):
			found=True
			top_left = (int(res['xmin']),int(res['ymin']))	 
			bottom_right = (int(res['xmax']),int(res['ymax']))
	return found, top_left,bottom_right
	

def select_actions(output, model):
	global lookup
	image = output.image_list[0]
	img_pil = Image.new(image.mode,image.size)
	img_pil.putdata(list(image.getdata()))
	img = cv.cvtColor(np.array(img_pil),cv.COLOR_RGB2BGR)
	actions=['RotateRight']
	ball_found, top_left, bottom_right = find_ball(img)
	if ball_found:
		actions = find_actions(img,top_left,bottom_right,output.step_number)
	elif neartheball==True:
		if lookup<-5:
			actions=['PickupObject','MoveAhead']
		else:
			actions=['LookDown','PickupObject','MoveAhead']
			lookup=lookup-1
		#'PickupObject','MoveAhead','PickupObject','MoveAhead',
		#'PickupObject','MoveAhead','PickupObject','MoveAhead',
		#'PickupObject','MoveAhead','PickupObject','MoveAhead',
		#'PickupObject','MoveAhead','PickupObject','MoveAhead']
			
	else:			
		actions=['RotateRight']*3
	#if output.step_number<100:
	#	cv.imwrite("output_images/lava"+str(output.step_number)+".png",img)
	#else:
	#	cv.imwrite("output_images/lavb"+str(output.step_number)+".png",img)
		
	return actions, params


# Unity app file will be downloaded automatically
controller = mcs.create_controller(config_file_or_dict='../sample_config.ini')
#mcs.init_logging()
scene_data = mcs.load_scene_json_file(scene_json_file_path)

output = controller.start_scene(scene_data)

# Use your machine learning algorithm to select your next action based on the scene
# output (goal, actions, images, metadata, etc.) from your previous action.
action, params = output.action_list[0]
actions = ['LookDown']*2
model = torch.hub.load('ultralytics/yolov5', 'custom', path="./best.pt")

# Continue to select actions until your algorithm decides to stop.
while actions != []:
	for action in actions:
		if action=='PickupObject':
			params={"objectId":"target"}
		print(output.step_number,action,params,lookup,output.return_status,sep=':')
		output = controller.step(action, **params)
		if output.return_status=='SUCCESSFUL':
			if action=='PickupObject':
				print("Object picked up")
				controller.end_scene()
				exit(0)
		elif output.return_status=='OBSTRUCTED' and action=='MoveLeft':
			explore_right=1
		elif output.return_status=='OBSTRUCTED' and action=='MoveRight':
			explore_right=0

			#update_pos(action)

	actions, params = select_actions(output, model)
	
# For interaction-based goals, your series of selected actions will be scored.
# For observation-based goals, you will pass a classification and a confidence
# to the end_scene function here.
controller.end_scene()
