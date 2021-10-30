from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import numpy as np
import os
import cv2

from vis_mask_utils import *

#Note: git clone from https://github.com/open-mmlab/mmsegmentation & install mmcv to get this to work#

config_file = 'configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py'
checkpoint_file = 'configs/CHECKPOINTS/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth'


def getallimgdir(dataInput_folder, lst): #user-defined function#
    train_test_folders = os.listdir(dataInput_folder)
    for train_or_test in train_test_folders:
        runs = os.listdir(os.path.join(dataInput_folder,train_or_test))
        for run in runs:
            imgs = os.listdir(os.path.join(dataInput_folder,train_or_test,run))
            for im in imgs:
                fulldir = os.path.join(dataInput_folder,train_or_test,run,im)
                lst.append(fulldir)

img_path_lst=[]
getallimgdir("dataInput",img_path_lst)

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

for idx,img in enumerate(img_path_lst):

    #user-defined: datainput, dataoutput paths#
    start, end = os.path.split(img)
    new_path = start.replace("dataInput","dataOutput")
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    #run inference
    mask_list = inference_segmentor(model, img)

    #save mask
    mask = mask_list[0].astype(np.uint8)
    cv2.imwrite(os.path.join(new_path,end),mask)
    print("Done for {}/{} images".format(idx+1,len(img_path_lst)))

    # visualize the results in a new window
    #model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].

    #model.show_result(img, result, out_file='dataOutput/test_out.png', opacity=0.7,show=False)

    #cv2.imwrite("dataOutput/test_mask.png", opencv_result)

    # test a video and show the results

    # video = mmcv.VideoReader('video.mp4')
    # for frame in video:
    #    result = inference_segmentor(model, frame)
    #    model.show_result(frame, result, wait_time=1)