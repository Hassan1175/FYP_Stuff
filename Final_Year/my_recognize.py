import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import pickle
import cv2
import os
from MYSVM import LocalBinary
src_path = ("O:\\Nama_College\\FYP\\Final_Year\\DATASETS\\")
#the method from previous file
desc = LocalBinary(24,8)
data = []
labels = []
#def training():
items = os.listdir(src_path)
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    for pic in images:
        photo = cv2.imread(pic,0)
        hist = desc.describe(photo)
       # S= pic.split(".")[-2]
        print(img)
        labels.append(img)
        indication = open("O:\\Nama_College\\FYP\\Final_Year\\indictions.txt", "w")
        data.append(hist)
        file = open("O:\\Nama_College\\FYP\\Final_Year\\hist.txt", "w")
        l = str(data)
        i = str(labels)
        file.write(l+"\n")
        indication.write(i+ "\n")
        #file.write("\n")
        file.close()
        indication.close()
print("model trainibng has started")


model = LinearSVC(C =100.0,random_state=45)
model.fit(data,labels)


print("training has done")

print("model is gonna save")


pickle_out = open("O:\\Nama_College\\FYP\\Final_Year\\dic.pickle" ,"wb")
pickle.dump(model,pickle_out)
pickle_out.close()
print("testing is gonna start")
for imgage in items:
    folder = src_path+"\\"+imgage
    os.chdir(folder)
    pics = os.listdir(folder)
    for piic in pics:
        tasveer = cv2.imread(piic)
        gray = cv2.cvtColor(tasveer, cv2.COLOR_BGR2GRAY)
        hisst = desc.describe(gray)
        prediction = model.predict([hisst])[0]
        # display the image and the prediction
        cv2.putText(tasveer, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Image", tasveer)
        cv2.waitKey(0)
print("Everything is done. . . . . .")
