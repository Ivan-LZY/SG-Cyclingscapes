# SG-Cyclingscapes
A coarse semantic segmentation dataset for pedestrian roads in Singapore. This dataset is based on the 19 classes presented in the popular Cityscapes dataset which was recorded in European cities.
![](https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/1inference.gif)

# Background

The primary aim of this project is to have fun! Like many others, I have caught the cycling bug in this past 2 years of pandemic. Hence, coming up with a project to marry this favourite past time of mine with my curiosity for computer vision was straightforward. 

It is also interersting to find out if segmentation models trained on European cities translate well to Singapore's garden-city state.

This project came at a cost of: 

- Getting a GoPro camera.
- 7 cycling trips around Singapore which have collectively clocked around 300km.
- Time spent extracting & processing video frames.
- A bit of blood, lots of sweat and some tears of joy.

I hope that this dataset will be useful to anyone trying to develop urban autonomous systems at home. In this vein, I have provided a subset of the dataset in this repository. This subset contains 300 images with some objects of interest blurred out for privacy.

I will continue to improve on the dataset annotation quality when new state-of-the-art semantic segmentation models come out or look into ways to fine tune my image processing steps. Any technical feedback for this project is hugely appreciated!

The 7 cycling routes presented in this dataset:

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/2Routes.jpg">
  <a>Dataset train/test split was done based on the routes. Train: 2, 3, 5, 6. Test: 1, 4, 7 </a>
</p>

# Dataset Breakdown

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/3PixelBreakdown.jpg">
  <a><b>Top:</b> SG-Cyclingscapes  <b>Bottom:</b> Cityscapes equivalent breakdown, extracted from its paper: https://arxiv.org/abs/1604.01685</a>
</p>

The ranking of the classes within its grounping was similar to Cityscapes. There were also alot more trucks than cars as some of the cycling routes have ongoing building constructions nearby.<br><br>

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/4DatasetSplit.jpg">
  <a><br>Only 8,481 images were included in the final dataset.</a>
</p>

# Quick Links
[Data pipeline overview](#Data-pipeline-overview-From-raw-image-to-maskjson-annotation)<br><br>
[Data collection with the GoPro Hero9](#Data-collection-with-the-GoPro-Hero9)<br>
[Installation](#Installation)<br>
[&nbsp;&nbsp;1. Frame extraction](#1-Frame-extraction)<br>
[&nbsp;&nbsp;2. Detection and removal of blurred images](#2-Detection-and-removal-of-blurred-images)<br>
[&nbsp;&nbsp;3. Manual Filtering, Finalising of Dataset & Image Censoring](#3-manual-filtering-finalising-of-dataset--image-censoring)<br>
[&nbsp;&nbsp;4. Camera Calibration](#4-Camera-Calibration)<br>
[&nbsp;&nbsp;5. Inference with pre-trained model](#5-inference-with-pre-trained-model)<br>
[&nbsp;&nbsp;6. Morphology operation on mask](#6-morphology-operation-on-mask)<br>
[&nbsp;&nbsp;7. Conversion of masks to COCO JSON format](#7-Conversion-of-masks-to-COCO-JSON-format)<br><br>
[Dataset download](#dataset-download)<br><br>
[Future plans](#Future-plans-for-this-project)

# Data pipeline overview: From raw image to mask/json annotation
![Full data pipeline](https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/5Overview.png)

# Data collection with the GoPro Hero9
### Why a GoPro?
I have never owned one and wanted try it out! It also helps that it is a globally popular choice for action cameras. Thus, anyone can buy a GoPro to either re-create my dataset or easily deploy models trained on this dataset. This is especially important since consistent camera sensor properties and the on-board chip image processing plays a huge part in the trained model’s inference accuracy.

### Video Stabilisation
The GoPro Hero 9 has in-built electronic image stabilization (EIS) for its real time video recording options. This function is not active in its time-lapse mode.  Since its propriety EIS may involve image cropping, translation and rotation, image-to-image quality may be inconsistent. 

Hence, I chose to use its time-lapse function, which coincidentally ties in well with my objective of collecting a variety of scenes for dataset diversity.

### FOV setting 
The Linear FOV setting was chosen for its undistorted image output so that less work needs to be done for my camera calibration step. (see the section on [camera calibration](#4-Camera-Calibration) for more details)

### Resolution
The GoPro Hero9 has recording options for up to 5k resolution. For my recordings, FHD resolution of 1920x1080 was chosen primarily due to convenience. For instance, the video files for FHD are much smaller and can be quickly streamed onto my smartphone from my GoPro while out on the road. It is also easier to train on the dataset using consumer hardware, where GPU VRAM typically ranges from 4GB-11GB. So with a smaller image resolution one can easily fit in larger batch sizes when training on this dataset.

### Lighting-related settings
Due to the difference in weather conditions between each recording session, I have set these to be on auto mode, hoping to achieve some consistency in the image brightness and clarity.

# Installation

This project was developed in Ubuntu 20.04, python 3.7. Any Opencv4 and matching Numpy versions should work. To run inference, do follow mmsegmentation's installation instructions.

# Assembling a dataset
## 1. Frame extraction

The time-lapse videos were recorded at 0.5sec per frame, which corresponds to 120 frames per minute. At an average cycling speed of 16 km/h, the distance travelled between frames is roughly 2.2m. Assuming that 10m is good enough for some scene variation, then for each video, the video loop will extract one frame and skip the next four frames.

This has resulted in 11,098 extracted frames from all the runs. 

*Note: While I have cycled ~300km for this project, this number of frames only corresponds to 110.98 km. This is because I chose to stop recording on return trips so as to avoid repitition data. There were also disruptions to recording due to running out of battery and memory space.*

## 2. Detection and removal of blurred images

Since the GoPro camera was mounted onto the handlebar, there also some motion blur due to movement of it when navigating turns and bumpy roads.

To detect images with significant motion blur, the images were first filtered with an edge sharpening kernel and then put through a Laplacian operator to detect edges in the image. The sum of the edges was then computed. Images with edge sums less than a certain threshold value will be deemed as too blurry. This threshold value was determined via trial and error on few of the images that have motion blur.

As it is a common image processing approach, thresholding with the variances of the edges to detect blurring was also considered. With motion blur, the inconsistent and lack of edges in the processed image will have low edge variance. However, this also works poorly in scenes with too many edges such as scenes with a lot of grass patches and vegetation. With many edges, the variance of the edges is also low. Hence, I decided to use thresholding with edge summation to detect motion blur.

This process has filtered out 1,147 images.
<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/6blurry.jpg">
  <a>Examples of blurry images that were filtered out.</a>
</p>

## 3. Manual Filtering, Finalising of Dataset & Image Censoring

Upon taking a quick look at the remaining images, there were a few images which lack features or are repetitive. These are images of plain barriers/walls/fences or series of images captured while waiting for traffic.

Due to the above reasons, another 1,470 images were then manually removed from the dataset.

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/7traffic.jpg">
  <a>A fair bit of time was spent at the traffic lights!</a>
</p>

### Finalised Dataset

At this point, the dataset has been finalised. From a total of 11,098 extracted frames, only 8,481 made it through the filtering process. 

The dataset was then split into train and test datasets, taking into consideration of the cycling routes/runs that each image belongs to. This is to ensure that there is little to no overlap in data between the train and test datasets. 

Final dataset breakdown: 1,853 (21%) of the images was assigned to the test dataset while the rest of the 6,628 (79%) images was assigned to the train dataset.

### Censoring of images with an object detector

As the recordings were taken in a public setting, without consent from people who were unknowingly photographed, there is a need to blur out their identities. Hence, there is a need to locate them in each image and blur just enough pixels so that they are unidentifiable but still recognizable as pedestrians.

Open source face/human detectors be it ML-based or classifiers based (i.e. haar cascade) have poor accuracies and/or have full body coverage. Hence, I decided to custom train an object detector to churn out bounding boxes for the upper body.

This is done through filtering of human key points in the COCO dataset and re-annotate them as bounding box coordinates (x,y,w,h). I then use the popular object detection repository by Alexeyab: [Darknet-YOLOv4](https://github.com/AlexeyAB/darknet) to train and infer on my images:

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/8censor.jpg">
  <a>Infer with the custom trained YOLOv4 and censor the results</a>
</p>

*Note: For the creation of segmentation masks, uncensored images were used for inference. Any of my dataset shared online will be censored.*

## 4. Camera Calibration

While GoPro has an in-built Linear FOV setting that can eliminate any lens distortion, I still ran the camera calibration module provided by OpenCV [tutorial](https://learnopencv.com/camera-calibration-using-opencv/) to see if there is any improvement in image quality in terms of distortion. A 9x6 chessboard is used for camera calibration.

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/9wide_vs_narrow.jpg">
  <a>Before & After calibration results for wide and narrow FOV</a>
</p>
<br><br>
<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/10linearlens_calib.jpg">
  <a>Some loss in image quality (~2%, tested with multiple images) when a linear FOV image is calibrated</a>
</p>

After testing the calibration with various FOV settings in my GoPro, in my subjective view, there is no noticeable difference in the calibrated image captured with Linear FOV.  As for the Narrow and Wide FOV settings, camera calibration is necessary. The calibration transformation matrices for each of the FOV is provided in the [camera calibration yaml](https://github.com/Ivan-LZY/SG-Cyclingscapes/tree/main/cam_calibration_yaml) folder. A sample code applying the matrices to transform the images is also provided [here](https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/scripts/Process_Video/3calib_load_undistort.py).

## 5. Inference with pre-trained model

***For this section, all credits belong to the team at OpenMMLab for the trained weights and simple API for inference.***

Inference was done with OpenMMLab’s implementation of DeepLabV3+, pre-trained on the Cityscapes semantic segmentation dataset, with a mIoU of 80.97%.

Do refer to XXX.py for the inference code that I used to generate and save the masks and its visualizations. For the installation of the required packages, do follow the guide on OpenMMLab’s [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) repository.

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/11inference.jpg">
  <a>Inference results with DeepLabV3+ trained on Cityscapes</a>
</p>

## 6. Morphology operation on mask

As seen in the below, there may be some blobs and noise in the generated segmentation.

To remedy this, morphology operation (opening) was done to remove some of it. Some basic parameters to consider for morphology are:

1. Kernel size
2. Kernel shape

The following examples shows the effect of the different kernel size:

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/12kernel5.png">
  <a>Kernel size = 5</a>
</p>

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/13kernel19.png">
  <a>Kernel size = 19. Even less noise but human segmentation is off the ground now.</a>
</p>

From the few example that I have tried out, a kernel size of 5-9 with rectangular shape seems to yield the decent results with little compromise on segmentation quality for other classes.

As for kernel shape, between cross, elliptical and rectangular, rectangular gave the best results in removng noise present in the masks.

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/14kernel9.png">
  <a>Kernel size = 9</a>
</p>

<p align="center">
  <img src="https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/imgs/15kernel5.png">
  <a>Kernel size = 5</a>
</p>

## 7. Conversion of masks to COCO JSON format

***For this section all credits belong to the original repository owner.*** 

The conversion code was adapted from: [image-to-coco-json-converter](https://github.com/chrise96/image-to-coco-json-converter)

Refer to [gen_json_from_masks.py](https://github.com/Ivan-LZY/SG-Cyclingscapes/blob/main/scripts/Run_Inference/gen_json_from_masks.py) for my adapted code to convert Cityscapes masks to COCO-JSON format.

# Dataset download

[google drive link](https://drive.google.com/file/d/1S_wlqXRVyvowlDwwUz1jCLuGiAxHkXwu/view?usp=sharing)

# Future plans for this project

Collect data from night cycling? Cover more routes in central areas?




