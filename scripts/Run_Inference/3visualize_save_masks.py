import cv2
import numpy as np
import os
from vis_mask_utils import *

def getallimgdir(dataInput_folder, lst):
    train_test_folders = os.listdir(dataInput_folder)
    for train_or_test in train_test_folders:
        runs = os.listdir(os.path.join(dataInput_folder,train_or_test))
        for run in runs:
            imgs = os.listdir(os.path.join(dataInput_folder,train_or_test,run))
            for im in imgs:
                fulldir = os.path.join(dataInput_folder,train_or_test,run,im)
                lst.append(fulldir)

mask_path_list = []
getallimgdir("dataOutput",mask_path_list)

#User-defined: Choose kernel for morph operation if any#
ksize = 5 #user-defined#
kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
kernel_x = cv2.getStructuringElement(cv2.MORPH_CROSS,(ksize,ksize))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))

#init img shape to pass into mask color function#
img_dummy = np.zeros([1080,1920,3],dtype=np.uint8)

for idx, m in enumerate(mask_path_list):
    start, end = os.path.split(m)
    new_path = start.replace("dataOutput","dataOutput_vis")
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    mask = cv2.imread(m, 0)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_rect) #user-defined: select rect kernel#
    vis_out = mask_to_color(opening,img_dummy)
    cv2.imwrite(os.path.join(new_path,end),vis_out)
    print("Done for {}/{} images".format(idx+1,len(mask_path_list)))


