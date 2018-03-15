import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import pickle
import cv2
import os
import imutils
from sklearn.datasets import load_digits
from imutils import face_utils
from skimage import feature

hist_data = []
lbp_data = []
labels = []
src_path = ("O:\\Nama_College\\FYP\\Final_Year\\DATASETS\\")

def describe(image):
    eps = 1e-7
    numPoints = 25
    radius = 8
    method = "uniform"
    lbp = feature.local_binary_pattern(image, numPoints, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return lbp, hist
'''
describe(image, eps=1e-7, 8, 1,"uniform") 
'''
items = os.listdir(src_path)
for img in items:
    #print(img)
    filee = src_path + "\\" + img
    os.chdir(filee)
    images = os.listdir(filee)
    for pic in images:
        photo = cv2.imread(pic,0)
        lbp, hist = describe(photo)
        S = pic.split(".")[-2]
        print(S)

        hist_data.append(hist)
        lbp_data.append(lbp)
        labels.append(S)
        file = open("O:\\Nama_College\\FYP\\Final_Year\\hist_data.txt", "w")
        filee = open("O:\\Nama_College\\FYP\\Final_Year\\lbp_data.txt", "w")

        hd = str(hist_data)
        ld = str(lbp_data)

        file.write(ld)
        filee.write(ld)
        file.close()
        filee.close()


print ("training is gonna start for LBP")
trained_model_lbp =  LinearSVC(C=100.0,random_state=42)
trained_model_lbp.fit(lbp_data,labels)
#
# print ("training is gonna start for HISt")
# trained_model_hist =  LinearSVC(C=100.0,random_state=42)
# trained_model_hist.fit(hist_data,labels)
#

print("LBPmodel is gonna save")
pickle_out = open("O:\\Nama_College\\FYP\\Final_Year\\trained_model_lbp.pickle","wb")
pickle.dumps(trained_model_lbp,pickle_out)
pickle_out.close()


# print("HISTmodel is gonna save")
# pickle_out = open("O:\\Nama_College\\FYP\\Final_Year\\trained_model_hist.pickle","wb")
# pickle.dumps(trained_model_hist,pickle_out)
# pickle_out.close()


print("testing accuracy of LBP Model")

lbp_score = trained_model_lbp.score(lbp_data,labels)
# hist_score = trained_model_hist.score(hist_data,labels)

print("saving scores")
file = open("O:\\Nama_College\\FYP\\Final_Year\\score.txt", "w")
file.write("bbjkkjjjk","\n")
# file.write(hist_score)
file.close()
