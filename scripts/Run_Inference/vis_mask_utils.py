import cv2
import numpy as np

CLASSES = ('road', 
'sidewalk', 
'building', 
'wall', 
'fence', 
'pole',
'traffic light', 
'traffic sign', 
'vegetation', 
'terrain', 
'sky',
'person', 
'rider', 
'car', 
'truck', 
'bus', 
'train', 
'motorcycle',
'bicycle')

PALETTE = [[128, 64, 128], 
[244, 35, 232], 
[70, 70, 70], 
[102, 102, 156],             
[190, 153, 153], 
[153, 153, 153], 
[250, 170, 30], 
[220, 220, 0],
[107, 142, 35], 
[152, 251, 152], 
[70, 130, 180], 
[220, 20, 60],
[255, 0, 0], 
[0, 0, 142], 
[0, 0, 70], 
[0, 60, 100],
[0, 80, 100], 
[0, 0, 230], 
[119, 11, 32]]


def mask_to_color(msk,img):
    out_ = np.ones(img.shape,dtype=np.uint8)*255

    for classid in range(len(CLASSES)):
        out_[:,:,0] = np.where(msk==classid,PALETTE[classid][2],out_[:,:,0])
        out_[:,:,1] = np.where(msk==classid,PALETTE[classid][1],out_[:,:,1])
        out_[:,:,2] = np.where(msk==classid,PALETTE[classid][0],out_[:,:,2])
    return out_
