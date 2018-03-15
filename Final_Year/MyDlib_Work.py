import cv2
import dlib
import numpy as np
import os
import imutils
from imutils import face_utils

face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
dlib_path = "dlibb/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

t=cv2.imread("bdshbsdj.jpg")
database=[]


src_path = ("O:\\Nama_College\\FYP\\Final_Year\\DATASETS\\")
dst_path = ("C:\\Users\\ADMIN\\Desktop\\Pics")
global i

# That function has been commented, because i have already detected faces in the images
#and data has been resized and converted to grey scale...Thereforei directly used for training


# indicator = 1
# #face = face_classifier1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),  flags=cv2.CASCADE_SCALE_IMAGE)
#
# def face_croppped(image):
#     global crop
#     face = face_classifier.detectMultiScale(image,scaleFactor=1.1, minNeighbors=10,minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
#     if face is ():
#         return None
#         print ("Sorry there is no face")
#     for (x, y, w, h) in face:
#  #     cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 5)
#      crop = image[y:y + h, x:x + w]
#      mm= cv2.resize(crop,(300,300))
#     #cv2.imshow('frame', cropped)
#     grey = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
#
#     return grey

items = os.listdir(src_path)
#print(items)
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    for pic in images:
        photo = cv2.imread(pic)
       # cv2.imshow("sds",photo)
        #cv2.waitKey()
        face = detector(photo,1)
        for (i, rect) in enumerate(face):
            shape = predictor(photo, rect)

            shape = face_utils.shape_to_np(shape)
            database = database + [[shape, indicator]]
            # database = shape,label
            print(database)
    indicator =  indicator+1
        #cv2.imshow(face)
        #cv2.waitKey()

    #image = cv2.imread(filee)
    #s = face_croppped(image)


print("done")
cv2.destroyAllWindows()
print("Hello word")
l=str(database)
f = open("O:\\Nama_College\\FYP\\Final_Year\\dlib_landmarks.txt","w")
f.write(l)
f.close()