# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:12:28 2020

@author: Arthur Gonzaga Ribeiro
"""

import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
import sys


# Loading files
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
face_recognition = dlib.face_recognition_model_v1("resources/dlib_face_recognition_resnet_model_v1.dat")

# Input this value
classroomID = '1A' #sys.argv[1]

# Inicializing variables
names = {}
descriptors = None
index = 0

# File paths according to the classroom ID
trainFolderPath = "classrooms/{}/train/".format(classroomID)
descritorPath = "classrooms/{}/descriptor.npy".format(classroomID)
namesPath = "classrooms/{}/names.pickle".format(classroomID)


# Loop through all files in the train folder
for file in glob.glob(os.path.join(trainFolderPath, "*.jpg")):
    
    # Read the file as a picture
    image = cv2.imread(file)
    
    # Detect faces in the image (array)
    facesDetected = face_detector(image, 1)
    print(len(facesDetected))
    
    # Verify if is there just one face in the array
    if(len(facesDetected) != 1):
        print("Something is wrong with the file: {}".format(file))
        break

    # Get the face
    face = facesDetected[0]
    
    # Generate the 128D Vector from the face
    points = shape_predictor(image, face)
    
    # Generate the face descriptor from the points
    faceDescriptor = face_recognition.compute_face_descriptor(image, points)
    
    # Organize in a list
    faceDescriptorsList = [fd for fd in faceDescriptor]
    
    # Create a npArray from the list
    npArray = np.asarray(faceDescriptorsList, dtype=np.float64)
    
    # Create a new axis
    npArray = npArray[np.newaxis, :]
    
    # Verify if descriptors is already inicialized or not
    if descriptors is None:
        descriptors = npArray
    else:
        # Concatenate the descriptor from the current face in the descriptors array
        descriptors = np.concatenate((descriptors, npArray), axis=0)
        
    # Add the name of the file in the names array
    names[index] = os.path.basename(file).split(".")[0]
    print(names[index])
    
    # Increase index
    index += 1


# When the loop finish
# It save the descriptors and the names in these paths
np.save(descritorPath, descriptors)
with open(namesPath, 'wb') as file:
	cPickle.dump(names, file)