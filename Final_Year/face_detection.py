import numpy as np
import cv2

import glob
import random

face_classifier1 =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture("C:\\Users\\Admin\\Desktop\\video.avi")
image = cv2.imread("C:\\Users\\Admin\\Desktop\\kaka.png")
#while(cap.isOpened()):
 #   ret, frame = cap.read()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   # faces = face_classifier1.detectMultiScale(gray, 1.2, 2)
face = face_classifier1.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),  flags=cv2.CASCADE_SCALE_IMAGE)


for (x, y, w, h) in  face:
 cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 800), 1)
 cv2.imshow('frame',image)

#cap.release()
cv2.waitKey()
cv2.destroyAllWindows()

