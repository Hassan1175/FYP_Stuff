# from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import sys
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream
import cv2
from imutils.video import FPS
face_classifier1 =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')

#basically i want to directly pass the video file path to FileVideoStrem
fvs = FileVideoStream("C:/Users/ADMIN/Desktop/videoo.mp4").start()
time.sleep(1.0)
#print(type(fvs))
fps = FPS().start()

while fvs.more():
    print("hello word")
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # frame = imutils.resize(frame, width=450)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = np.dstack([frame, frame, frame])
    face = face_classifier1.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 300), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(50)
        fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()