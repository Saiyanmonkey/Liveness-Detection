import argparse
import os
import base64

ap = argparse.ArgumentParser()

ap.add_argument("-i","--input",type=str, required=True, help="path to input image")
args = vars(ap.parse_args())

imagePath = os.path.sep.join([r"C:\Users\alvin\OneDrive\Desktop\Alvin\Kerja\Week 3\liveness-detection-lcc\Val",args["input"]])

with open(imagePath, "rb") as img_file:
         my_string = base64.b64encode(img_file.read())
         
with open("Output.txt","w") as f:
    f.write(str(my_string))
