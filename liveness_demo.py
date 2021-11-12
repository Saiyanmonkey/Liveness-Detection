from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time

#Construct Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,required=True,help="path to trained model")
ap.add_argument("-l","--le",type=str,required=True,help="path to label encoder")
ap.add_argument("-d","--detector",type=str,required=True,help="path to face detector")
ap.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

print("[INFO] Loading face detector...")
protoPath = os.path.sep.join([args["detector"],"deploy.prototxt"])
modelPath = os.path.sep.join([args['detector'],"res10_300x300_ssd_iter_140000.caffemodel"])

net = cv2.dnn.readNetFromCaffe(protoPath,modelPath)

#load liveness detector and label encoder
print("[INFO] LOADING LIVENESS DETECTOR...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"],"rb").read())

#initialize the video stream and allow the camera sensor to warmup
print("[INFO] STARTING VIDEO STREAM...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Loop over the frames
while True:
    #grab the frame from the video
    frame = vs.read()
    #resize to have a maximum width of 600 pixels
    frame = imutils.resize(frame,width=600)
    
    #Grab the frame and convert to a blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    
    #pass the blob through the network
    #to obtain detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    #loop over the detections
    for i in range(0,detections.shape[2]):
        #extract confidence associated w the predictions
        confidence = detections[0,0,i,2]
        
        #filter weak detections
        if confidence>args["confidence"]:
            #compute the (x,y)-coordinates of the bounding boxes for the face
            #and extract face ROI
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            
            #ensure bounding box does not fall outside the dimension frame
            startX = max(0,startX)
            startY = max(0,startY)
            endX = min(w,endX)
            endY = min(h,endY)
            
            #Extract face ROI and preprocess the same way as the training data
            face = frame[startY:endY,startX:endX]
            face = cv2.resize(face,(32,32))
            face = face.astype("float")/255.0
            face = img_to_array(face)
            face = np.expand_dims(face,axis=0)
            
            #pass the face ROI through the trained liveness detector
            #model to determine if the face is real or fake
            predictions = model.predict(face)[0]
            j = np.argmax(predictions)
            #j = 0 if predictions<spoof_confidence else 1
            label = le.classes_[j]
            #label = label_name[j]
            
            #draw the label and the bounding box on the frame.
            label = "{}:{:.4f}".format(label,predictions[j])
            cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)
            
    #show the output frame and wait for keypress
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
            
    #if q is pressed, break form loop
    if key == ord("q"):
        break
            
            #cleanup
cv2.destroyAllWindows()
vs.stop()
            
    
    
