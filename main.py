import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  



# Convert the image to grayscale
def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces: 
      roi_color = img[y:y + h, x:x + w] 
      return roi_color
    return None
def replace_face(img,replacement_face):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces: 
      resized_replacement_face = cv2.resize(replacement_face, (w, h))
      img[y:y + h, x:x + w] = resized_replacement_face
    return img
