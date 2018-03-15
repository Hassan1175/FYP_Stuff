import numpy as np
from imutils.video import FPS
import imutils
import cv2
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream
import cv2
from imutils.video import FPS
import time

face_classifier1 =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("C:\\Users\\Admin\\Desktop\\testing.mp4")
seconds = 5

# while(cap.isOpened()):
fps = FPS().start()
streaming = cap.get(cv2.CAP_PROP_FPS)
print(streaming)
multiplier = streaming * 5
print("hello word")
mm,  frame = cap.read()
while (True):
    # print("mammmaa")
    frameid = int(round(cap.get(1)))
    # print(frameid)
    success  ,image = cap.read()
    if (frameid % multiplier == 0):
     cv2.imwrite("C:\\Users\\ADMIN\\Desktop\\Pics %d .jpg" %frameid,image)
     cv2.imshow('frame', image)
     cv2.waitKey(1)
# face = face_classifier1.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),  flags=cv2.CASCADE_SCALE_IMAGE)
# for (x, y, w, h) in  face:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 300), 2)
#         cv2.imshow('frame',frame)
#         cv2.waitKey(1)
#         fps.update()
# cap.release()
# fps.stop()
print(fps.elapsed())
print(fps.fps())
# fvs.stop()
cap.release()
print("mubarkaaaaan")
cv2.destroyAllWindows()
