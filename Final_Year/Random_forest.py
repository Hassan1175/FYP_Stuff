from sklearn.ensemble import RandomForestClassifier

import cv2
import dlib
import numpy as np
import os
import imutils
import pickle
from imutils import face_utils
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
dlib_path = "dlibb/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

# t=cv2.imread("bdshbsdj.jpg")
database=[]
labels = []


src_path = ("O:\\Nama_College\\FYP\\Final_Year\\TRAINING_DATASET\\")

global i

indicator = 1
def face_croppped(image):
    global crop
    face = face_classifier.detectMultiScale(image,scaleFactor=1.1, minNeighbors=10,minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
    if face is ():
        return None
        print ("Sorry there is no face")
    for (x, y, w, h) in face:
 #     cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 5)
     crop = image[y:y + h, x:x + w]
     mm= cv2.resize(crop,(300,300))
    #cv2.imshow('frame', cropped)
    grey = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)

    return grey
items = os.listdir(src_path)
i = 0
#print(items)
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    for pic in images:
        photo = cv2.imread(pic)
        size=photo.shape

       # cv2.imshow("sds",pho1
        face = detector(photo,1)
        li=[]
        for (i, rect) in enumerate(face):
            shape = predictor(photo, rect)
            shape = face_utils.shape_to_np(shape)
            shape = shape[48:68]

            shape=shape.flatten()
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.imshow('aaa',photo)

            # moment=cv2.HuMoments(cv2.moments(hull)).flatten()
            # print(moment)

            database.append(shape)
            labels.append(img)
            print(database)
print("done")
cv2.destroyAllWindows()
# print("Hello word")
l=str(database)
Q = str(labels)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
# model = RandomForestClassifier
rf.fit(database,labels)

print("model is gonna save")

pickle_out = open("O:\\Nama_College\\FYP\\Final_Year\\random.pickle","wb")
pickle.dump(rf,pickle_out)
pickle_out.close()
print("everthing is done")



# image =  "O:\\Nama_College\\FYP\\Final_Year\\DATASETS\\BORE\\bore12.jpg"
#
# picture = cv2.imread(image)
#
# grey = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)


# photo = cv2.imread(pic)
# cv2.imshow("sds",photo)
# cv2.waitKey()

# face = detector(grey)
# for (i, rect) in enumerate(face):
#     shape = predictor(grey, rect)
#
#     shape = face_utils.shape_to_np(shape)
#     Y = shape.flatten()
#     prediction = model.predict([Y])[0]
#     cv2.putText(picture, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
#     cv2.imshow("Image", picture)
#     cv2.waitKey()
#
print("done ho gyeaa")
