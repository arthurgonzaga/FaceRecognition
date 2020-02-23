# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:12:28 2020

@author: Arthur Gonzaga Ribeiro
"""
import cv2
import numpy as np
import sys


# Get the webcam
cap = cv2.VideoCapture(0)

# Student name (input this value)
name = 'Student_Name' #sys.argv[1]

# Student classroom (input this value)
classroomID = '1A' #sys.argv[2]

# Pictures counter
count = 0 

# Train folder path
path = "classrooms/{}/train/".format(classroomID)

# Show webcam until it has 6 pictures taken
while(count < 6):
    
    # Read webcam frame
    ret, frame = cap.read()

    # Display frame
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    # Enter pressed
    if key & 0xFF == ord('\r'):
        
        # Setting the file name according with the name and the counter value
        fileName = "{}.{}.jpg".format(name, count)
        
        # Getting the frame and saving in the train folder path
        cv2.imwrite(path + fileName, frame)
        count += 1
    elif key & 0xFF == ord('q'):
        break
        
# Stop reading frames from the webcam
cap.release()

# Close webcam window
cv2.destroyAllWindows()