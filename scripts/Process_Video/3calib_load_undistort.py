import cv2
import numpy as np
import os
import glob
import yaml

def getallimgdir(folder, lst):
    runs = os.listdir(folder)
    for run in runs:
        imgs = os.listdir(os.path.join(folder,run))
        for im in imgs:
            fulldir = os.path.join(folder,run,im)
            lst.append(fulldir)

def undistort(img_path, k_mat, d_mat, dim):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k_mat, d_mat, np.eye(3), k_mat, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#main#
with open('../../calib_yaml/GoPro9_linear.yaml') as f:
    my_dict = yaml.safe_load(f)
    K = np.array(my_dict['K'])
    D = np.array(my_dict['D'])
    DIM = tuple(my_dict['DIM'])
folders = ["Train_frames", "Test_frames"]
img_list = []
for f in folders:
    getallimgdir(f,img_list)
images = glob.glob('./linearframes/*.png') #user-defined#
                     
for i in images:
    undistort(i,K,D,DIM)

