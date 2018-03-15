import numpy as np
from imutils.video import FPS
import imutils

import cv2
face_classifier1 =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("C:\\Users\\Desktop\\vide.mp4")
while(cap.isOpened()):
    ff,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),  flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in  face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 300), 2)
      #  Frame = imutils.resize(frame, 30, 40)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
cap.release()
cv2.destroyAllWindows()
