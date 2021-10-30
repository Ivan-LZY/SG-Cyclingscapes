
import os
import shutil
import random
import darknet
import cv2
from tqdm import tqdm

#Note: clone from https://github.com/AlexeyAB/darknet to get darknet python inference API#

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image, error = blur_detections(detections, image_resized,image_path)
    #Do not draw boxes for saving blurred images, uncomment to view with detector results#
    #image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image,cv2.COLOR_RGB2BGR), error

def blur_detections(detections, image, path): #Added function to apply blur to detected boxes#
    error = 0
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        if left<0: #YOLO darknet may sometimes erronously feed in small negative coordinates#
            left = 1
        if top<0:
            top = 1
        cropped = image[top:bottom, left:right]
        try:
            cropped_blurred = cv2.blur(cropped,(15,15))
            image[top:bottom, left:right] = cropped_blurred
        except:
            print("skipped {},{},{},{}, {}".format(left, top, right, bottom, path))
            error = 1
    return image, error




def getallimgdir(folder, lst): #user-defined#
    runs = os.listdir(folder)
    for run in runs:
        imgs = os.listdir(os.path.join(folder,run))
        for im in imgs:
            fulldir = os.path.join(folder,run,im)
            lst.append(fulldir)



def main():
    random.seed(1)  # deterministic bbox colors
    #Set up yolo .cfg, .weights & .data filepaths here#
    network, class_names, class_colors = darknet.load_network(
        "./yolov4.cfg", #user-defined#
        "./person.data", #user-defined#
        "./yolov4.weights", #user-defined#
        batch_size=1
    )

    folders = ["Test_frames_clean", "Train_frames_clean"]
    img_list = []
    count = 0
    for f in folders:
        getallimgdir(f,img_list)   

    #feed in list of images to process#
    for i in tqdm(img_list):
        image, error = image_detection(
        i, network, class_names, class_colors, 0.2)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        #Uncomment to display results, comment out imwrite if needed#
        # cv2.imshow("Show results", image)
        # k = cv2.waitKey(0)
        # if k==27:
        #     cv2.destroyAllWindows()
        #     exit()
        if (error==0):
            cv2.imwrite(i,image)
        else:
            head, tail = os.path.split(i)
            shutil.move(i,"./unable_to_censor/" + tail) #user-defined#
            count+=1
    print("Unable to process: {} images".format(count))

if __name__ == '__main__':
    main()