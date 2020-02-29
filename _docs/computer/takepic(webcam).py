# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:12:28 2020

@author: Arthur Gonzaga Ribeiro
"""
import cv2
import numpy as np
import sys
import os
import tkinter as tk
import train
    


# Get the webcam
cap = cv2.VideoCapture(0)


root = tk.Tk()
root.geometry('330x150')
root.title('Nome do estudante')
topFrame = tk.Frame(root)
bottomFrame = tk.Frame(root)


labelName = tk.Label(topFrame, text='Nome: ')
entryName = tk.Entry(topFrame, width=35)

labelName.pack(side=tk.LEFT)
entryName.pack(side=tk.LEFT)

def openWebcam():
    # Student name (input this value)
    name = entryName.get()
    name = name.replace(' ', '_')
    print(name)
    root.destroy()
    
    cancel = False
    
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
        cv2.imshow("Webcam - Aperte 'q' para cancelar", frame)
    
        key = cv2.waitKey(1)
        # Enter pressed
        if key & 0xFF == ord('\r'):
            
            # Setting the file name according with the name and the counter value
            fileName = "{}.{}.jpg".format(name, count)
            
            # Getting the frame and saving in the train folder path
            cv2.imwrite(path + fileName, frame)
            count += 1
        elif key & 0xFF == ord('q'):
            cancel = True
            while(count != 0):
                count-= 1
                fileName = "{}.{}.jpg".format(name, count)
                os.remove(path + fileName)
            break
            
    # Stop reading frames from the webcam
    cap.release()
    
    # Close webcam window
    cv2.destroyAllWindows()
    
    if(cancel == False):
        train.start(classroomID, name)
    

    
btn = tk.Button(bottomFrame, text='       OK       ', command=openWebcam)
btn.pack()
topFrame.pack(expand=1)
bottomFrame.pack(side=tk.BOTTOM, ipady = 15)
root.mainloop()



