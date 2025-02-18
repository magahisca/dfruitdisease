#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D    # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions                              # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img                                        # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50                                                          # type: ignore
from tensorflow.keras.preprocessing import image                                                                     # type: ignore
from tensorflow.keras.models import Sequential                                                                       # type: ignore
from tensorflow.keras.models import Model, load_model                                                                            # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import cv2
import tkinter as tk
from tkinter import ttk
import PIL 
import time
from tensorflow.nn import max_pool  # Assuming this is in your model definition
import gpiozero as gpio
from gpiozero import Button
capButton = Button(26)

def my_model(inputs):
  # ... other layers ...
  x = max_pool(x, ksize=(2, 2), strides=(2, 2))  # Deprecated usage
  # ... other layers ...
  return x

# Updated code (using recommended function)
from tensorflow.nn import max_pool2d  # Assuming this is in your model definition

def my_model(inputs):
  # ... other layers ...
  x = max_pool2d(x, ksize=(2, 2), strides=(2, 2))  # Recommended usage
  # ... other layers ...
  return x



model= load_model('/home/pi/tflite1/final5.h5')




from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import os
from tkinter import PhotoImage
root = tk.Tk()
root.geometry("800x480")
root.resizable(width=False, height = False)

color1 =  '#93C572'
class ImageCaptureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.config(bg=color1)
        self.window.title(window_title)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(3, 240)
        self.vid.set(4, 180)
        self.vid.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT), bg = color1)
        self.canvas.grid(row = 0, column = 0)
        
        self.label = Label(root, text = " ", bg = color1)
        self.label.grid (row = 0, column = 1)
        
        self.image_width = 140  
        self.image_height = self.image_width  
        self.image_frame = tk.Canvas(window, width=self.image_width, height=self.image_height, bg = color1)
        self.image_frame.grid(row = 0, column = 2)
        self.image = None  
        
        photo = PhotoImage('button.png')
        self.btn_capture = tk.Button(window, text="Capture", width=10, command=self.capture)
        self.btn_capture.grid(row = 1, column = 0)
        
        self.select_button = tk.Button(root, text="Select Image", command=self.select)
        self.select_button.grid(row = 1, column = 2)
        
        Font_tuple = ("Verdana", 10, "bold")
        self.text_output = tk.Text(window, height=2, width=40)
        self.text_output.grid(row = 2, column = 0)
        self.text_output.configure(font = "Verdana")
      
        capButton.when_pressed = self.capture
        time.sleep(3)
        
        self.update()
        
        self.window.mainloop()
       
    def capture(self):
        
        ret, frame = self.vid.read()
        time.sleep(6)
        
        timestamp = int(time.time())
        if ret:
            image_path = os.path.join(r'/home/pi/tflite1/camera folder',f"image{timestamp}.png" )  
            cv2.imwrite(image_path,frame)
            dispimg = cv2.imread(image_path)  
            dispimg = cv2.GaussianBlur(dispimg, (5, 5), 0)
            dispimg = cv2.cvtColor(dispimg, cv2.COLOR_RGB2BGR) #convert lang nung captured image
            self.image = dispimg
            self.predict_image(image_path)
            self.display_image()
       #dispimg = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR) 
       #self.display_image()
       #dispimg = cv2.GaussianBlur(dispimg, (5, 5), 0)
    def predict_image(self, image_path):
        img_height, img_width = (224, 224)
        raw = cv2.imread(image_path)
        #raw = cv2.cvtColor(raw,cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(raw, (img_height, img_width))
        image_resized = image_resized / 255.0
        image = np.expand_dims(image_resized, axis=0)
        pred = model.predict(image)
        output_class = np.argmax(pred)
        class_names = ['anthracnose', 'cactus virus x', 'healthy', 'stem canker']
        predicted_label = class_names[output_class]
        self.display_prediction(predicted_label)
        
    def display_image(self):
        if self.image is not None:
            image_resized = cv2.resize(self.image, (self.image_width, self.image_height))
            self.image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image_resized))
            self.image_frame.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
        else:
            self.image_frame.delete("all") 
            
            
    def display_prediction(self, predicted_label):
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, f"Prediction:{predicted_label}\n")
        
    def select(self):
    # Open file selector and get image path
        image_path = filedialog.askopenfilename(
          initialdir="/", title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

    
        if image_path:
        
          dispimg = cv2.imread(image_path)
          self.image = dispimg
          self.predict_image(image_path)
          self.display_image()    
        
        
    def update(self):
          ret, frame = self.vid.read()
          if ret:
              self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
              self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
          self.window.after(10, self.update)

# Create a window and pass it to the ImageCaptureApp class

app = ImageCaptureApp(root, "Dragonfruit Stem Disease Classification")


# In[ ]:




