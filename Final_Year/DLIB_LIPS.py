# import the necessary packages
from imutils import face_utils

import imutils
import dlib
import cv2
import numpy as np
path = "dlibb/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)
image = cv2.imread("C:\\Users\\ADMIN\\Desktop\\kaka.png")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



rects = detector(gray, 1)

for (n,rect) in enumerate(rects):

    shape = predictor(gray, rect)


    shape = face_utils.shape_to_np(shape)
    shape =  shape[48:68]
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText(image, "Face {}".format(n + 1), (x - 10, y - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

cv2.imshow("IMAGEDD", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


    # visualize all facial landmarks with a transparent overlay
    # output = face_utils.visualize_facial_landmarks(image, shape)
    # cv2.imshow("Image", output)
    # cv2.waitKey(0)

















