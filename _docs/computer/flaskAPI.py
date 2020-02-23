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

# Insert your ipv4
IPV4 = '0.0.0.0:5000'

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
            
            # Here will happen the magic
            
            # Returning a json with the informations
            return jsonify(
            #        facesLen=len(facesDetected)
            #        studentsPresent= {}
                    )
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
