import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import os
import numpy as np
import argparse


def random_rotation(image_array: ndarray):
    #pick random degree of rotation 
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array,random_degree)

def random_noise(image_array: ndarray):
    #add random noise
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:,::-1]

transform_dict = {
    'rotate':random_rotation,
    'noise':random_noise,
    'horizontal_flip':horizontal_flip
    }


#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--input",type=str,required=True,help="path to trained model")
#ap.add_argument("-n","--num",type=int,required=True,help="path to label encoder")

#args = vars(ap.parse_args())



folder_path = r"C:\Users\alvin\OneDrive\Desktop\Alvin\Kerja\Week 3\liveness-detection-lcc\dataset\spoof"

#path = os.path.sep.join([folder_path,args["input"]])

num_files_desired = 8777
#loop all files of the folder and build the list of files paths

images = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]

num_generated_files = 0
while num_generated_files < num_files_desired:
    #random image from folder
    image_path = random.choice(images)
    #read image as 2d array
    image_to_transform = sk.io.imread(image_path)
    #apply random num of transformation 
    trans_num = random.randint(1,len(transform_dict))
    
    num_transformation = 0
    transformed_image = None
    while num_transformation < trans_num:
        #random transformation to apply for an image
        key = random.choice(list(transform_dict))
        transformed_image = transform_dict[key](image_to_transform)
        num_transformation +=1

    new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)
    
    # write image to the disk
    io.imsave(new_file_path, transformed_image)
    num_generated_files += 1
        

    
    