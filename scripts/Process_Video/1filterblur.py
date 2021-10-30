import cv2
import os
import shutil as s
import numpy as np
from tqdm import tqdm

#Some decent image may be filtered out here but that is okay since I still have a lot of images to work with#

def getallimgdir(folder, lst):
    runs = os.listdir(folder)
    for run in runs:
        imgs = os.listdir(os.path.join(folder,run))
        for im in imgs:
            fulldir = os.path.join(folder,run,im)
            lst.append(fulldir)

def check_blur(img_lst):
    for i in tqdm(img_lst):
        run = i.split("/")[1]
        thresh = 30.0 #30.0 manually checked that this value is acceptable to detect blurred images in my data# -- #user-defined#
        img = cv2.imread(i)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img_s = cv2.filter2D(img, -1, kernel)#shapren edges if any#
        img_gray = cv2.cvtColor(img_s,cv2.COLOR_BGR2GRAY)
        img_L = cv2.Laplacian(img_gray,cv2.CV_16S,3)
        abs_dst = cv2.convertScaleAbs(img_L)
        img_area = img.shape[0]*img.shape[1]
        if (abs_dst.sum()/img_area)<thresh:
            img_name = i.split("/")[-1]
            s.move(i,"1filter_sum/"+img_name) #moves image from dataset to another folder#

folders = ["Test_frames", "Train_frames"]
img_list = []
for f in folders:
    getallimgdir(f,img_list)
img_list.sort()
check_blur(img_list)





