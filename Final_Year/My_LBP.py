import numpy as np
from skimage import feature
import cv2
data = []
labels = []
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
def show(image):
    cv2.imshow('**', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
photo = cv2.imread('C:\\Users\\ADMIN\\Desktop\\photo.png',0)
lbp, hist = describe(photo)
print(hist)
#print(lbp)
show(photo)
print("done")
