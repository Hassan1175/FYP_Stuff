import cv2
import imutils
import numpy as np
import os
import dlib
from imutils import face_utils
face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
indicator = 1
path = "dlibb/shape_predictor_68_face_landmarks.dat"
t=cv2.imread("bdshbsdj.jpg")
t2=type(t)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(path)

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5)

    if faces is ():
        return None
        #print "Sorry there is no face"

      #cropping found faces
    for (x, y, w, h) in faces:
        cropped = img[y:y+h , x:x+w]
    return cropped
#
src_path =os.getcwd()+"\DATASETS\\"
items = os.listdir(src_path)
dec=face_extractor(t)
t3=type(dec)
print(items)
database=[]
for folder in items:
    folder = src_path+"\\"+folder
    print(folder)
    os.chdir(folder)
    images = os.listdir(folder)
    for  pic  in images:
        print(pic)
        photo = cv2.imread(pic)
        if t2==type(photo):
            detected_face = face_extractor(photo)
            if t3==type(detected_face):
                scale = cv2.resize(detected_face,(350,350))
                rects=detector(scale,1)
                for (i, rect) in enumerate(rects):
                  shape = predictor(scale, rect)
                  shape = face_utils.shape_to_np(shape)
                  database=database+[[shape,indicator]]
                  print(database)
    indicator =  indicator+1
print("Hello word")
l=str(database)
f = open("ff.txt","w")
#f.write(l)
f.close()