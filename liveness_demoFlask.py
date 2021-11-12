from flask import Flask, jsonify, request, make_response
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import argparse
import imutils
import pickle
import cv2
import os
import timeit
import base64

import time

app = Flask(__name__)
ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,required=True,help="path to trained model")
ap.add_argument("-l","--le",type=str,required=True,help="path to label encoder")
args = vars(ap.parse_args())

@app.route('/')
def index():
    return "<h1>Hello World!</h1>"

@app.route('/development/liveness',methods=['POST'])
def initialize():
    #print("[INFO] Loading face detector...")
    time.sleep(0.2)
    start = timeit.default_timer()
    # logger.info("Running face liveness)
    environment = request.remote_addr
    global graph
    
    tic = time.time()
    
    req = request.get_json()
    
    face = request.form['file']
    id = request.form['id']
    
    #print(face)
    
    try:
        imface = base64.b64decode(face)
        image2 = Image.open(io.BytesIO(imface))
        arrayktp = np.array(image2)
    
        protoPath = r"face_detector/deploy.prototxt"
        modelPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        
        net = cv2.dnn.readNetFromCaffe(protoPath,modelPath)
        confidence_min = 0.5
        
        le = pickle.loads(open(args["le"],"rb").read())
    
        predicted_label = ["Real","Spoof"]
        spoof_min=0.3
        frame = arrayktp
        
        
        #load liveness detector and label encoder
        print("[INFO] LOADING LIVENESS DETECTOR...")
        model = load_model(args["model"])
        #le = pickle.loads(open(args["le"],"rb").read())
        test = liveness_detection(frame,net,confidence_min,predicted_label,model,id,le)
        return test
    
    except(IndexError, ValueError, KeyError) as error:
        print(error)
        return jsonify(result=88)

def liveness_detection(frame, net, confidence_min, predicted_label ,model, id,le):
#grab the frame from the video
    
    #resize to have a maximum width of 600 pixels
    frame = imutils.resize(frame,width=600)
        
    #Grab the frame and convert to a blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
        
    #pass the blob through the network
    #to obtain detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    
    #loop over the detectionsNo 
    for i in range(0,detections.shape[2]):
       #extract confidence associated w the predictions
       confidence = detections[0,0,i,2]
            
       #filter weak detections
       if confidence>confidence_min:
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
                face = cv2.resize(face,tuple([32,32]))
                face = face.astype("float")/255.0
                face = img_to_array(face)
                face = np.expand_dims(face,axis=0)
                
                #pass the face ROI through the trained liveness detector
                #model to determine if the face is real or fake
                predictions = model.predict(face)[0]
                j = np.argmax(predictions)
                #j = 0 if  predictions < spoof_min else 1
                label = le.classes_[j]
                
                #draw the label and the bounding box on the frame.
                #label = "{}:{:.4f}".format(label,predictions[j])
                #cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                #cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)
                print("Label:",label)
                
                if label =="real":
                    #print(label)
                    print("confidence: ",confidence)
                    print("predictions: ",predictions[j])
                    return jsonify(id=id,Liveness="True",result=15,
                                   status = 200,score=float(predictions[j]))
                else:
                    #print(label)
                    print(confidence)
                    print(predictions[j])
                    return jsonify(id=id,Liveness="False",result=16,
                                   status = 200,score=float(predictions[j]))
       else:
           return jsonify(id=id,Liveness="False",result=16,
                               status = 200)
            
#show the output frame and wait for keypress

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2222, debug=True, threaded=True)
