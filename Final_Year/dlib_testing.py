import cv2
import dlib
import numpy as np
import os
import imutils
from imutils import face_utils
from sklearn.svm import LinearSVC
import pickle
import glob

global p

face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
dlib_path = "dlibb/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

t=cv2.imread("bdshbsdj.jpg")
t2=type(t)


src_path = ("O:\\Nama_College\\FYP\\Final_Year\\DATASETS\\")
dst_path = ("C:\\Users\\ADMIN\\Desktop\\Pics")
global i

label =1

items = os.listdir(src_path)
#i = 0
#print(items)
labels = []
database = []
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    for pic in images:
        photo = cv2.imread(pic,0)
       # cv2.imshow("sds",photo)
        #cv2.waitKey()
        face = detector(photo)
        for (i, rect) in enumerate(face):
            shape = predictor(photo, rect)
            shape = face_utils.shape_to_np(shape)
            Q = shape.flatten()

            #database = shape,label
            database.append(Q)

            S= pic.split(".")[-2]
            print(S)
            labels.append(S)
            file = open("O:\\Nama_College\\FYP\\Final_Year\\mm.txt", "w")
            l = str(database)
            file.write(l)
            file.close()
    #        print(database)

#dataset_size = len(database)

#TwoDim_dataset = database.reshape(dataset_size,-1)


print("model is gona train")
classifier = LinearSVC(C=100.0,random_state=42)
classifier.fit(database,labels)
print("model is gonna save")
pickle_out = open("O:\\Nama_College\\FYP\\Final_Year\\dlib.pickle","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()

print("testing is gonna start")

# for imgage in items:
#     folder = src_path+"\\"+imgage
#     os.chdir(folder)
#     pics = os.listdir(folder)
#     for piic in pics:
tasveer = cv2.imread("O:\\Nama_College\\FYP\\Final_Year\\DATASETS\\BORE\\bore1.jpg",0)
#gray = cv2.cvtColor(tasveer, cv2.COLOR_BGR2GRAY)
cv2.waitKey()
face = detector(tasveer)
for (i, rect) in enumerate(face):
 p = predictor(tasveer, rect)
 p = face_utils.shape_to_np(p)


#shape.flatten()
prediction = classifier.predict(p)[0]
# display the image and the prediction
cv2.putText(tasveer, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
cv2.imshow("Image", tasveer)
cv2.waitKey(0)
print("Everything is done. . . . . .")
cv2.destroyAllWindows()