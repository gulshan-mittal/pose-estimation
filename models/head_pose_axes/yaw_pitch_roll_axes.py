#!/usr/bin/env python


#In this head pose estimator is used to get the YAW angle.
#The angle is projected on the input images and showed on-screen as a red line.
#The images are then saved in the same folder of the script.

import numpy as np
import os
import tensorflow as tf
import cv2
from head_pose_estimation import CnnHeadPoseEstimator

#Function used to get the rotation matrix
def yaw2rotmat(yaw):
	x = 0.0
	y = 0.0
	z = yaw
	ch = np.cos(z)
	sh = np.sin(z)
	ca = np.cos(y)
	sa = np.sin(y)
	cb = np.cos(x)
	sb = np.sin(x)
	rot = np.zeros((3,3), 'float32')
	rot[0][0] = ch * ca
	rot[0][1] = sh*sb - ch*sa*cb
	rot[0][2] = ch*sa*sb + sh*cb
	rot[1][0] = sa
	rot[1][1] = ca * cb
	rot[1][2] = -ca * sb
	rot[2][0] = -sh * ca
	rot[2][1] = sh*sa*cb + ch*sb
	rot[2][2] = -sh*sa*sb + ch*cb
	return rot


sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
# Load the weights from the configuration folders
my_head_pose_estimator.load_yaw_variables(os.path.realpath("../../weights/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_roll_variables(os.path.realpath("../../weights/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("../../weights/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))

ldir = sorted(os.listdir('./data/'))
num_samples = len(ldir)
gt_labels_arr = []
if os.path.exists('./gt_labels.txt') == True:
	predict_loss_roll = 0
	predict_loss_pitch = 0
	predict_loss_yaw = 0
	gt_labels = 'gt_labels.txt'
	with open(gt_labels, 'r') as f:
		lines = f.readlines()
		for line in lines:
			striped_line = line.rstrip('\n').split(',')
			gt_roll, gt_pitch, gt_yaw = np.float(striped_line[0]), np.float(striped_line[1]),np.float(striped_line[2])
			gt_labels_arr.append([gt_roll, gt_pitch, gt_yaw])
	f.close()

for i, img in enumerate(ldir):
	file_name = './data/' + img
   
	os.makedirs('./results', exist_ok=True)
	print("Processing image ..... " + file_name)
	file_save = './results/' + os.path.splitext(img)[0] + "_axes.jpg"

	image = cv2.imread(file_name)
	cam_w = image.shape[1]
	cam_h = image.shape[0]
	c_x = cam_w / 2
	c_y = cam_h / 2
	f_x = c_x / np.tan(60/2 * np.pi / 180)
	f_y = f_x
	camera_matrix = np.float32([[f_x, 0.0, c_x],
								[0.0, f_y, c_y], 
								[0.0, 0.0, 1.0] ])
	# print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
	#Distortion coefficients
	camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
	#Defining the axes
	axis = np.float32([[0.0, 0.0, 0.0], 
					   [0.0, 0.0, 0.0], 
					   [0.0, 0.0, 0.5]])

	roll_degree = my_head_pose_estimator.return_roll(image, radians=False)  # Evaluate the roll angle using a CNN
	pitch_degree = my_head_pose_estimator.return_pitch(image, radians=False)  # Evaluate the pitch angle using a CNN
	yaw_degree = my_head_pose_estimator.return_yaw(image, radians=False)  # Evaluate the yaw angle using a CNN
	### print("Estimated [roll, pitch, yaw] (degrees) ..... [" + str(roll_degree[0,0,0]) + "," + str(pitch_degree[0,0,0]) + "," + str(yaw_degree[0,0,0])  + "]")
	roll = my_head_pose_estimator.return_roll(image, radians=True)  # Evaluate the roll angle using a CNN
	pitch = my_head_pose_estimator.return_pitch(image, radians=True)  # Evaluate the pitch angle using a CNN
	yaw = my_head_pose_estimator.return_yaw(image, radians=True)  # Evaluate the yaw angle using a CNN
	print("Estimated [roll, pitch, yaw] (radians) ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
	
	if os.path.exists('./gt_labels.txt') == True:
		predict_loss_roll += np.square(roll[0,0,0] - gt_labels_arr[i][0])
		predict_loss_pitch += np.square(pitch[0,0,0] - gt_labels_arr[i][1])
		predict_loss_yaw += np.square(yaw[0,0,0] - gt_labels_arr[i][2])
	### #Getting rotation and translation vector
	rot_matrix = yaw2rotmat(-yaw[0,0,0]) #Deepgaze use different convention for the Yaw, we have to use the minus sign

	#Attention: OpenCV uses a right-handed coordinates system:
	#Looking along optical axis of the camera, X goes right, Y goes downward and Z goes forward.
	rvec, jacobian = cv2.Rodrigues(rot_matrix)
	tvec = np.array([0.0, 0.0, 1.0], np.float) # translation vector

	imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
	p_start = (int(c_x), int(c_y))
	p_stop = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
	print("point start: " + str(p_start))
	print("point stop: " + str(p_stop))
	print("")

	cv2.line(image, p_start, p_stop, (0,0,255), 3) #RED
	cv2.circle(image, p_start, 1, (0,255,0), 3) #GREEN

	cv2.imwrite(file_save, image)

if os.path.exists('./gt_labels.txt') == True:
	print('Loss in Roll : {}'.format(predict_loss_roll/num_samples))
	print('Loss in Pitch : {}'.format(predict_loss_pitch/num_samples))
	print('Loss in Yaw : {}'.format(predict_loss_yaw/num_samples))

	print('Accuracy in Roll : {}'.format(1 - predict_loss_roll/num_samples))
	print('Accuracy in Pitch : {}'.format(1 - predict_loss_pitch/num_samples))
	print('Accuracy in Yaw : {}'.format(1 - predict_loss_yaw/num_samples))
	overall_accuracy = 1 - ((predict_loss_roll/num_samples +predict_loss_pitch/num_samples  + predict_loss_yaw/num_samples) /3 ) 
	print('Overall acuuracy: {}'.format(overall_accuracy))

