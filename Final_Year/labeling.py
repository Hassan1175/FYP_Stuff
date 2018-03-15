import skimage
from skimage import feature
import numpy as np
import os
import dlib

src_path = ("O:\\Nama_College\\FYP\\TRUEFACE1\\DATASETS\\")
#print(src_path)
items = os.listdir(src_path)
#print(items)
for item in items:
    #print(item)
    path = src_path+ "//" +item
    os.chdir(path)
    images = os.listdir(path)
    for img in images:
        S= img.split(".")[-2]
        print(S)