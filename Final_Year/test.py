import sklearn
from sklearn import svm
from MYSVM import LocalBinary
from sklearn.svm import LinearSVC
desc = LocalBinary(24,8)
from imutils import paths
import argparse
import pickle
import cv2
import os

src_path = ("O:\\Nama_College\\FYP\\Final_Year\\TRAINING_DATASET\\")
#the method from previous file

data = []
labels = []
i = 1

#def training():
items = os.listdir(src_path)
# print(items)
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    # print(images)
    for pic in images:
        photo = cv2.imread(pic,0)
        # cv2.imshow("sd",photo)
        # cv2.waitKey()

        hist = desc.describe(photo)
        # S= pic.split(".")[-2]
        # print(img)
        labels.append(img)
        data.append(hist)
        indication = open("O:\\Nama_College\\FYP\\Final_Year\\labels.txt", "w")
        # data.append(hist)
        file = open("O:\\Nama_College\\FYP\\Final_Year\\lbp_data.txt", "w")
        l = str(data)
        i = str(labels)
        file.write(l+"\n")
        indication.write(i+ "\n")
        #file.write("\n")
        file.close()
        indication.close()
print("model trainibng has started")
print(data)
print(labels)

model = LinearSVC(C =70.0,random_state=60)
model.fit(data,labels)


print("training has done")

print("model is gonna save")
pickle_out = open("O:\\Nama_College\\FYP\\Final_Year\\lbp_model.pickle","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

print("model has saved")

#
# print("testing is gonna start")
# for imgage in items:
#     folder = src_path+"\\"+imgage
#     os.chdir(folder)
#     pics = os.listdir(folder)
#     for piic in pics:
#         tasveer = cv2.imread(piic)
#         gray = cv2.cvtColor(tasveer, cv2.COLOR_BGR2GRAY)
#         hisst = desc.describe(gray)
#         prediction = model.predict([hisst])[0]
#         # display the image and the prediction
#         cv2.putText(tasveer, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
#         cv2.imshow("Image", tasveer)
#         cv2.waitKey(0)
print("Everything is done. . . . . .")
