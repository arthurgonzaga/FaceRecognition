# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:12:28 2020

@author: Arthur Gonzaga Ribeiro
"""

from flask import Flask as fl
from flask import request, jsonify
from PIL import Image
import cv2
import base64
import numpy as np
import io
import dlib


# Insert your ipv4
IPV4 = '0.0.0.0'

app = fl(__name__)


@app.route('/')
def index():
    return 'Hello World'

@app.route('/image',methods=['POST'])
def analize():
    if request.method == 'POST':
        try:
            # Getting the image from the request
            base64data = request.form['image']
            image = stringToImage(base64data)
            image = toRGB(image)
            
            # Loading files
            face_detector = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
            face_recognition = dlib.face_recognition_model_v1("resources/dlib_face_recognition_resnet_model_v1.dat")
            
            # Input this value
            classroomID = '1A' #sys.argv[1]
            
            # File paths according to the classroom ID
            descriptorsPath = "classrooms/{}/descriptors.npy".format(classroomID)
            namesPath = "classrooms/{}/names.pickle".format(classroomID)
            
            # Get the students names and descriptors
            names = np.load(namesPath)
            descriptors = np.load(descriptorsPath)
            
            
            # Detect faces in the image (array)
            facesDetected = face_detector(image, 1)
            
            # Exit if doesn't detect any face in the image
            if len(facesDetected) == 0:
                print('No faces detected')
                exit(0)
                
            # If the euclidean distance is greater than 0.5
            # The two faces belong to the same person
            limiar = 0.45
            
            # Loop through the list
            for face in facesDetected:
                
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
            
                # Calculate the euclidean distance (array)
                distances = np.linalg.norm(npArray - descriptors, axis=1)
                
                # The index of the minimum distance in the array
                index = np.argmin(distances)
                
                # Get the minimum distance
                minDistance = distances[index]
                print(minDistance)
            
                # Verify if is it the same student
                if minDistance <= limiar:
                    name = names[index]
                else:
                    name = 'Unknown'
                print(name)
            
            # Returning a json with the informations
            return jsonify(facesLen=len(facesDetected))
        except Exception as err:
            return err;
        
        
# Base64 to PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# PIL Image to an RGB image (npArray)
def toRGB(image):
    print('turning rgb')
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

# Start the local server
if __name__ == '__main__':
    app.run(host=IPV4)
