import numpy as np
import os 
import cv2
import tensorflow as tf


# from tf.keras.preprocessing import image
from cv2 import cv2

fer_model = tf.keras.models.load_model("./ck_model.h5")
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')



capture = cv2.VideoCapture(0)

while True:
    
    ret_value, image = capture.read()
    
    if not capture:
        continue

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

