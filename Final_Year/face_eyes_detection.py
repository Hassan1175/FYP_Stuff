import numpy as np
import cv2

face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
eyes_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_eye.xml')

#here i am loading the image..

image = cv2.imread('C:\\Users\\ADMIN\\Desktop\\Nwaz.png')
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


#classfier will return the rectangle on the detected face as a tuple. from top left corner to bottom right corner

faces =  face_classifier.detectMultiScale(grey,1.05,6)

print('No of Faces = %d' %(len(faces)))
if faces is ():
    print("sorry ..  . .")

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),5)
    #cv2.imshow("Face detection", image)
    #cv2.waitKey(0)
    #that will crop the image of face to detetc eye
    ri_grey  =  grey[y:y+h,x:x+w]
    ri_color =  image[y:y+h,x:x+w]
    eyes = eyes_classifier.detectMultiScale(ri_grey)
    for(ex,ey,ew,eh) in eyes:
       # print  "hello word. .  ."
        cv2.rectangle(ri_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)



cv2.imshow("detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()