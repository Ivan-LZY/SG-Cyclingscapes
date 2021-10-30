import cv2
import numpy as np
import os

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

#Choose kernel for morph operation if any#
ksize = 5 #user-defined#
kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
kernel_x = cv2.getStructuringElement(cv2.MORPH_CROSS,(ksize,ksize))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))

counter = np.zeros(19)

for idx, m in enumerate(mask_path_list):
    mask = cv2.imread(m, 0)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_rect) #user-defined: select rect kernel#
    ids, counts = np.unique(opening, return_counts=True)
    for index, i in enumerate(ids):
        counter[i] += counts[index]

    print("Done for {}/{} images".format(idx+1,len(mask_path_list)))

print(counter)


