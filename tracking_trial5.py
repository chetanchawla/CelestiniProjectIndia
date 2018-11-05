import time
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import sys
import os
#import left
#import right
#import ahead
import cv2
def Output(img):
    cv2.imwrite("/Users/ishani/Documents/Celestini/Phase-II/Workspace/PedestrianOnly/output.png",img)
    return 0

def pedestrian():
    vid='peopleCounter.avi'
    cap = cv2.VideoCapture(vid)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #face_cascade = cv2.CascadeClassifier('/Users/ishani/Documents/Celestini/Phase-II/Workspace/VehiclesOnly/trial2.xml')
    start_time = time.time()
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
    fno = 0
    count = 0
    hold = 0
    skip_frames = 3
    tracking = 0
    while True:
        
        ret, frames = cap.read()
        if not(tracking == 1):
            for i in range(skip_frames):
                ret, frames = cap.read()

        if ret == False:
            break

        else:
                fno = fno+1
                frameY, frameX, frameD = frames.shape
                # frames = cv2.resize(frames,(int(frameX/2), int(frameY/2)))
                frameY, frameX, frameD = frames.shape
                frames = frames[(int)(frameY*0.40):(int)(frameY*0.75), (int)(frameX*0.2):(int)(frameX*0.8)]
                frameY, frameX, frameD = frames.shape
                return_array = []

                if fno == 1:
                    # print("fx = ",frameX, "fy = ",frameY)
                    # output = "OutputVideo.mp4"
                    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
                    # out = cv2.VideoWriter(output, fourcc, 20.0, (frameX, frameY))
                    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    # (rects, weights) = hog.detectMultiScale(gray, hitThreshold = 0.3 ,winStride=(8, 8), padding=(24, 24), scale=1.05)#1.01
                    (rects, weights) = hog.detectMultiScale(gray, hitThreshold = 0.3 ,winStride=(4, 4), padding=(24, 24), scale=1.1)
                    rectss = np.array([[xC, yC, xC + wC, yC + hC] for (xC, yC, wC, hC) in rects])
                    pick = non_max_suppression(rectss, probs=None, overlapThresh=0.65)
                    for (xA, yA, xB, yB) in pick:
                        pad_w, pad_h = int(0.152*(xB-xA)), int(0.152*(yB-yA))
                        cv2.rectangle(frames, (xA+pad_w, yA+pad_h), (xB-pad_w, yB-pad_h), (0, 0, 255), 2)

                elif (not(fno == 1) and len(rects)==0) or count == 5:
                    count = 0
                    tracking = 0
                    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    (rects, weights) = hog.detectMultiScale(gray, hitThreshold = 0.3 ,winStride=(8, 8), padding=(24, 24), scale=1.05)
                    rectss = np.array([[xC, yC, xC + wC, yC + hC] for (xC, yC, wC, hC) in rects])
                    pick = non_max_suppression(rectss, probs=None, overlapThresh=0.65)
                    for (xA, yA, xB, yB) in pick:
                        pad_w, pad_h = int(0.152*(xB-xA)), int(0.152*(yB-yA))
                        cv2.rectangle(frames, (xA+pad_w, yA+pad_h), (xB-pad_w, yB-pad_h), (0, 0, 255), 2)

                elif not(len(rects)==0) and count==0:
                    count = 1
                    tracking = 1
                    tracker = []
                    init_tracker = []
                    bbox = []
                    roi_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                    roi_array = non_max_suppression(roi_array, probs=None, overlapThresh=0.65)
                    i = 0
                    # rects = roi_array
                    for (a,b,c,d) in roi_array:
                        # sss = frames[rects[i][1] : rects[i][3], rects[i][0] + rects[i][1]:rects[i][2] + rects[i][3]]
                        # rois.append(sss)
                        track = cv2.TrackerKCF_create()
                        tracker.append(track)
                        bbox.append((a,b,c-a,d-b))
                        okay  = tracker[i].init(frames, bbox[i])
                        init_tracker.append(okay)
                        i = i + 1
                    i = 0

                elif not(count==0):
                    count = count + 1
                    i = 0
##                    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
##                    (rects1, weights) = hog.detectMultiScale(gray, hitThreshold = 0.3 ,winStride=(8, 8), padding=(24, 24), scale=1.05)
##                    rects1 = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects1])
##                    rects1 = non_max_suppression(rects1, probs=None, overlapThresh=0.65)
##                    if not(len(rects1)==len(roi_array)):
##                        count = 10
##                        hold = 1
                    for i in range(len(bbox)):
##                        if hold == 1:
##                            hold = 0
##                            break
                        ok, bbox[i] = tracker[i].update(frames)
                        if ok :
                            p1 = (int(bbox[i][0]), int(bbox[i][1]))
                            p2 = (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3]))
                            pad_w, pad_h = int(0.152*bbox[i][2]), int(0.152*bbox[i][3])
                            send1 = int(( (bbox[i][0]+pad_w)+(frameX*0.2) )*2)
                            send2 = int(( (bbox[i][1]+pad_h)+(frameY*0.40))*2)
                            send3 = int(((bbox[i][2]-pad_w)+(frameX*0.2) )*2)
                            send4 = int(((bbox[i][3]-pad_h)+(frameY*0.40) )*2)
                            return_array.append((send1,send2,send3,send4))
                            cv2.rectangle(frames, (int(bbox[i][0]+pad_w), int(bbox[i][1]+pad_h)), (int(bbox[i][0] + bbox[i][2]-pad_w), int(bbox[i][1] + bbox[i][3]-pad_h)), (0,0,0), 2, 1)
                            
                            
                # out.write(frames)

                cv2.imshow("JAI MATA DI!",frames)
                if cv2.waitKey(33) == 27:
                    break
        #fileP.close()
        
        #cv2.destroyAllWindows()


pedestrian()
