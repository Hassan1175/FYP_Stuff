import cv2
import dlib
import numpy as np
import os
import imutils
import pickle
from imutils import face_utils
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

labels = []


src_path = ("O:\\Nama_College\\FYP\\Final_Year\\TESTING_DATASET\\")
# src_path = ("O:\\amer\\DATASETS\\")
# dst_path = ("C:\\Users\\ADMIN\\Desktop\\Pics")
global i

indicator = 1
#face = face_classifier1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),  flags=cv2.CASCADE_SCALE_IMAGE)

items = os.listdir(src_path)
#print(items)
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    for pic in images:
        photo = cv2.imread(pic)
        print(img)
        labels.append(img)

Q = str(labels)


file = open("O:\\Nama_College\\FYP\\Final_Year\\dlib_testing_labels.txt", "w" )

file.write(Q)
file.close()
print("model trainibng has started")
# print(data)
print(labels)

print("everthing is done")


