import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from MYSVM import LocalBinary
from imutils import paths
import cv2
import dlib
from imutils import face_utils
face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
dlib_path = "dlibb/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)
import argparse
import pickle
import cv2
import os

src_path = ("O:\\Nama_College\\FYP\\Final_Year\\TESTING_DATASET\\")
desc = LocalBinary(24,8)
predict = []
pickle_in = open("O:\\Nama_College\\FYP\\Final_Year\\lbp_model.pickle","rb")
model = pickle.load(pickle_in)
items = os.listdir(src_path)
for imgage in items:
    folder = src_path+"\\"+imgage
    os.chdir(folder)
    pics = os.listdir(folder)
    for piic in pics:
        tasveer = cv2.imread(piic)
        gray = cv2.cvtColor(tasveer,cv2.COLOR_BGR2GRAY)

        face = detector(tasveer,0)
        for (i, rect) in enumerate(face):
            shape = predictor(tasveer, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(tasveer, (x, y), (x + w, y + h), (0, 255, 0), 2)

        hisst = desc.describe(gray)
        prediction = model.predict([hisst])[0]
        predict.append(prediction)

        # display the image and the prediction

        cv2.putText(tasveer, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Image", tasveer)
        cv2.waitKey(0)





#The code to check the percent accuracy of the trained model
file =  open("O:\\Nama_College\\FYP\\Final_Year\\dlib_testing_labels.txt","r")
reading = file.read()
reading2 = reading.split()
new_list = []
for item in reading2:
    new_str = ""
    for entry in item:
        if entry.isalpha()==True:
            new_str = new_str+entry
    new_list.append(new_str)
print(len(new_list))
print(len(predict))
count = 0.0
correct = 0
for i in range(len(new_list)):
    count = count + 1
    if new_list[i] == predict[i]:
        correct =  correct+1

print(count)
print(correct)
M = correct/count

Accuray = M *100

print("Accuracy is ", Accuray, " percent")


print("Everything is done. . . . . .")
