import cv2
import os

#This program serves to grab & save image frames#

SKIPFRAMES = 5 #grab 1 frame once every x# of frames#  -- #user-defined#

runlist_30fps_vid = ["JurongLake","Westcoast-Queensway","SeletarAero"] #user-defined#

early_terminate_vids = [] #user-defined#

def proc_vid(vid_path,extract_path):
    vidfilename = vid_path.split("/")[2].split(".")[0]
    print("Processing: {}\n".format(vid_path))
    cap = cv2.VideoCapture(vid_path)

    i = 0
    badframe = 0
    badframe_consecutive = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    while 1:
        ret, img = cap.read()
        i +=1
        if ret:
            if (i % SKIPFRAMES == 0):
                cv2.imwrite(os.path.join(extract_path,vidfilename + "_" + str(i).zfill(6)+".png"), img) #save as png for lossless quality#
                print("Done for {}: {}/{}".format(extract_path,i,length))
                badframe_consecutive = 0
        else:
            badframe +=1
            badframe_consecutive+=1

        #detect EOF, GOPRO ocassionally has bad frames inserted into video files(esp. large ones), hence the following logic#
        if (badframe_consecutive>=100):  
            if (abs(i-length)>210):
                early_terminate_vids.append(vid_path)
                print("Early Termination detected {}".format(vid_path))
            print("Done for {}. Num_of_badframes: {}, last index {}\n ".format(vid_path,badframe, i))
            break

folders = ["Test", "Train"]

for i in folders:
    runs = os.listdir(i)
    extractfolder = i + "_frames"
    for run in runs:        
        extractfolder_path = os.path.join(extractfolder,run)
        try:
            os.mkdir(extractfolder_path)
            print("Created folder for {}".format(extractfolder_path))       
        except:
            print("Skipped folder creation for {}".format(extractfolder_path))  
        
        run_dir = os.path.join(i,run)
        vids = os.listdir(run_dir)
        for vid in vids:
            if vid[-3:] in ["MP4"]:
                vid_dir = os.path.join(run_dir,vid)
                proc_vid(vid_dir,extractfolder_path)

print(early_terminate_vids)

