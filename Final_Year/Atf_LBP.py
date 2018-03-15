from skimage import feature
import numpy as np
import cv2


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


def show(image):
    cv2.imshow('**', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


###################################################
img = cv2.imread("C:\\Users\\Admin\\Desktop\\kaka.png", 0)

lbp, hist = describe(img)

print(hist)

print("now these are yhje lbp point")
print(lbp)
show(img)

