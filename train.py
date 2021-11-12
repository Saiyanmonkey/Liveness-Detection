#set matplotlib backend so figure can be saved on the background
import matplotlib
matplotlib.use("Agg")

#import packages
import tensorflow as tf
from LiveNet.LivenessNet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pickle
import cv2
import os

#construct arguments
ap= argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",type=str,required=True,help="path to trained model")
ap.add_argument("-l","--le",type=str,required=True, help="path to label encoder")
ap.add_argument("-p","--plot",type=str,default="plot29.png",help="path to output loss/accuracy plot")
ap.add_argument("-c","--confidence matrix",type=str,default="conf.png",help="path to output confusion matrix")
ap.add_argument("-b","--batch",type=int,required=True,help="Batch size")

args = vars(ap.parse_args())


#Initialisze intial:
#Learning rate
INIT_LR = 1e-4
#Batch Size
BS = args["batch"]
#Epochs
EPOCHS = 100

#Grab the list of images in the dataset directory, then initialise
#The list of data and class images
print("[INFO] LOADING IMAGES...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#loop over all imagePaths
for imagePath in imagePaths:
    #extract the class label from the filename
    #resize into fixed 32x32 pixel
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(32,32))
    
    #update the data and labels list respectively
    data.append(image)
    labels.append(label)
    
#convert data into NumPy array, then preprocess it by scaling
#all pixel intensities to the range [0,1]
data = np.array(data,dtype="float")/255.0

#encode the labels which are strings into integers
#and one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels,2)

#partition the data into training and testing split 
(trainX,testX,trainY,testY)= train_test_split(data,labels,test_size=0.25, random_state=42)

#construct training image for data augmentation
aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

#initialize the optimizer and model...
print("[INFO] COMPILING MODEL...")
opt = Adam(lr=INIT_LR,decay = INIT_LR/EPOCHS)
model = LivenessNet.build(width=32,height=32,depth=3,classes = len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#tf.keras.utils.plot_model(model, to_file='C:/Users/alvin/OneDrive/Desktop/Alvin/Kerja/Week 3/liveness-detection-lcc/model_plot.png', show_shapes=True, show_layer_names=True)

#train the network
print("[INFO] TRAINING NETWORK FOR {} EPOCHS".format(EPOCHS))
H = model.fit(x=aug.flow(trainX,trainY,batch_size=BS),validation_data=(testX,testY),steps_per_epoch=len(trainX)//BS,epochs=EPOCHS)   



#Evaluate the network
print("[INFO] EVALUATING NETWORK")
predictions = model.predict(x = testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=le.classes_))

conf_matrix = confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1))

print(conf_matrix)

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                conf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf_matrix, annot=labels, fmt="", cmap='Blues')

#save the network to disk
print("[INFO] SERIALIZING NETWORK TO '{}'...".format(args["model"]))
model.save(args["model"],save_format="h5")

#save label encoder to disk
f = open(args["le"],"wb")
f.write(pickle.dumps(le))
f.close()

#plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS),H.history["loss"],label = "train_loss")
plt.plot(np.arange(0, EPOCHS),H.history["val_loss"],label = "val_loss")
plt.plot(np.arange(0, EPOCHS),H.history["accuracy"],label = "train_accuracy")
plt.plot(np.arange(0, EPOCHS),H.history["val_accuracy"],label = "val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


