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

legend = cv2.imread("CityScapesLegend.png") #png generated from 0make_CS_legend.py#
legend = cv2.resize(legend,(400,810))
mask_path_list = []
img_path_list = []

#user-defined path mangement#
getallimgdir("dataOutput",mask_path_list)
getallimgdir("dataInput",img_path_list)

for idx, m in enumerate(mask_path_list):
    start, end = os.path.split(m)
    mask = cv2.imread(m, 0)
    img = cv2.imread(img_path_list[idx])
    out_before = mask_to_color(mask,img)
    img = cv2.addWeighted(out_before,0.4,img,0.6,0.0)
    ksize = 5
    type = 2
    while(1):
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
        kernel_x = cv2.getStructuringElement(cv2.MORPH_CROSS,(ksize,ksize))
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))
        k_list = [kernel_e,kernel_x,kernel_rect]
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_list[type])
        out_after = mask_to_color(opening,img)
        img2 = cv2.addWeighted(out_after,0.4,img,0.6,0.0)

        cv2.putText(out_before,"No Morph",(5,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(out_after,"Morph Open",(5,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img2,"Morph Open ",(5,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)

        top2 = cv2.hconcat([out_before,out_after])
        btm2 = cv2.hconcat([img,img2])
        output = cv2.vconcat([top2,btm2])
        output = cv2.resize(output,(1440,810))
        output = cv2.hconcat([output,legend])
        cv2.putText(output,"Ksize = " + str(ksize),(725,55),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(output,"Kernel type: Elli, X, Rect-->"+ str(type),(725,80),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
       
        cv2.imshow(m,output)
        k = cv2.waitKey(0)
        if k==43: #+
            ksize+=2

        if k==45:#-
            ksize-=2

        if k ==32: #SPACEBAR, change type
            type+=1

        if ksize<=1:
            ksize=3
        if type>2:
            type=0

        if k==27:
            cv2.destroyAllWindows()
            break
        if k == 115: #s
            cv2.imwrite("test.png",output)
