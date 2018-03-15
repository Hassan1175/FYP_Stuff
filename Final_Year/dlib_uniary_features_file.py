import cv2
import dlib
import numpy as np
import os
import imutils
import pickle
import operator
from operator import add


from imutils import face_utils
from sklearn.svm import LinearSVC
face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
dlib_path = "dlibb/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

t=cv2.imread("bdshbsdj.jpg")
database=[]
labels = []


src_path = ("O:\\amer\\DATASETS\\")
dst_path = ("C:\\Users\\ADMIN\\Desktop\\Pics")
global i


xlist = []
ylist =[]


indicator = 1
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
        face = detector(photo,1)
        for (i, rect) in enumerate(face):
            shape = predictor(photo, rect)
            shape = face_utils.shape_to_np(shape)
            # k =shape.flatten()
            database.append(shape)
            labels.append(img)

m = []
print("done")
print(database)
print("gane starrrrrrt")
for i in database:
    print(i,',')
    for k in  i:
        s = sum(k)
        print(s)





# cv2.destroyAllWindows()
# print("Hello word")
# l=str(database)
# Q = str(labels)
#zz
# M = open("C:\\Users\\ADMIN\\Desktop\\Pics\\dlib_landmarks_labels.txt","w")
# f = open("C:\\Users\\ADMIN\\Desktop\\Pics\\dlib_landmarks.txt","w")
# f.write(l)
# M.write(Q)
# f.close()
# M.close()
#
# print("model trainibng has started")
# print(data)
# print(labels)

#
# model = LinearSVC(C =70.0,random_state=60)
# model.fit(database,labels)
#

#
# print("model is gonna save")
# pickle_out = open("C:\\Users\\ADMIN\\Desktop\\Pics\\dlib_model_full.pickle" ,"wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()

print("everthing is done")

#
#
# print("done ho gyeaa")
