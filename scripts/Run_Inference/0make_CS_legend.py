
import cv2
import numpy as np
from torch._C import Size

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

SIZE = 40
WID = 150
canvas = np.ones(((len(PALETTE))*SIZE+5,380,3),dtype=np.uint8)*20

for i in range(len(PALETTE)):
    canvas[i*SIZE+4:i*SIZE+SIZE,2:WID,:] = [PALETTE[i][2],PALETTE[i][1],PALETTE[i][0]]

for i in range(len(CLASSES)):
    cv2.putText(canvas,CLASSES[i],(WID+10,i*SIZE+SIZE-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,0)
cv2.imwrite("CityScapesLegend.png",canvas)