#Import packages
import numpy as np
import argparse
import cv2
import os

#Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",type=str, required=True, help="path to input video")
ap.add_argument("-o","--output",type=str,required=True, help="path to output directory or cropped faces")
ap.add_argument("-d","--detector",type=str,required=True, help="path to face detector")
ap.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-s","--skip",type=int,default=16,help="Number of frames to skip b4 appplying face detection")

args = vars(ap.parse_args())

#Load face detector
print("[INFO] lOADING FACE DETECTOR...")
protoPath = os.path.sep.join([args['detector'],"deploy.prototxt"])
modelPath = os.path.sep.join([args['detector'],"res10_300x300_ssd_iter_140000.caffemodel"])

net = cv2.dnn.readNetFromCaffe(protoPath,modelPath)

vs = cv2.VideoCapture(args['input'])
read = 0
saved = 0

while True:
    #grab the frame from the file
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    
    #increment the total number of frames read
    read += 1
    
    #skip check
    if read % args['skip']!=0:
        continue
    
    #grab the frame dimensions and construct  blob from the frame
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),1.0,(300,300),(104.0,177.0,123.0))
    
    #pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    #make sure at least 1 detections was found
    if len(detections)>0:
        #assuming each image only has one face,
        #find the bounding box with the largest probability
        i = np.argmax(detections[0 , 0 ,:,2])
        confidence = detections[0,0,i,2]
        
        #ensure confidence meets minimum treshold
        if confidence>args['confidence']:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
            face = frame[startY:endY,startX:endX]
            
            #write frame to disk
            p = os.path.sep.join([args['output'],"{}.png".format(saved)])
            cv2.imwrite(p,face)
            saved+=1
            print("[INFO] SAVED {} TO DISK".format(p))
            
vs.release()
cv2.destroyAllWindows()
    
    