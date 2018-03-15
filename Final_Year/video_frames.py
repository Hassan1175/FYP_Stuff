import cv2
import  dlib
import numpy as  np
import os
import  pickle
from imutils import face_utils
from imutils.video import FileVideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style



dlib_path = "dlibb/shape_predictor_68_face_landmarks.dat"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)
# pickle_in = open("C:\\Users\\ADMIN\\Desktop\\Frames\\LIPS_dlib.pickle","rb")
pickle_in = open("O:\\Nama_College\\FYP\\Final_Year\\full_dlib.pickle","rb")
# pickle_in = open("O:\\Nama_College\\FYP\\Final_Year\\LIPS_70_dlib.pickle","rb")
# pickle_in = open("O:\\Nama_College\\FYP\\Final_Year\\random.pickle","rb")
model = pickle.load(pickle_in)
cap = cv2.VideoCapture(0)
# cap = FileVideoStream(0)
# cap = cv2.VideoCapture("C:\\Users\\ADMIN\\Desktop\\testing.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total Number of frames ",length)
streaming = int(cap.get(cv2.CAP_PROP_FPS))
print("FPS Rate ",streaming)

B = 0
current_frame = 0


while(True):
    # plt.pause(2)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    B += 1
    if B% 5 == 0:
        print(B)
        face = detector(gray, 1)
        for (i, rect) in enumerate(face):
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            (x,y,w,h) =  face_utils.rect_to_bb(rect)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # shape =  shape[48:68]
            shape1 = shape.flatten()
            prediction = model.predict([shape1])[0]
            cv2.putText(frame,"FACE ({})".format(i+1) +" " +prediction,(x-5,y-5),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
            print(prediction)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)
        # cv2.imshow("Image",fra
                # me)
        # cv2.waitKey(0)
                cv2.imshow('hh',frame)
                k=cv2.waitKey(1)

                # FPSrate = cap.get(cv2.CAP_PROP_FPS)
                # print(FPSrate)

                if k=='q':
                    break

cap.release()
cv2.destroyAllWindows()
