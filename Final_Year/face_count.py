import cv2
import numpy as np
face_classifier =  cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
# detect face and return the croppped facee
def face_extractor(img):
    grey_scale =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey_scale, 1.1, 5)
    if faces is ():
        return None
        #print "Sorry there is no face"
      #cropping found faces
    for (x, y, w, h) in faces:
        cropped = img[y:y+h , x:x+w]
    return cropped
cap = cv2.VideoCapture(0)
#just a counter
count = 0
#keeping on extracting upto 100 facess
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count +=1
        face = cv2.resize(face_extractor(frame),(400,400))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_storage = './DTST/' + str (count) + '.jpg'
        cv2.imwrite(file_storage,face)
        #put text on line image as counting
        cv2.putText(face,str(count),(60,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Cropped face ",face)
    else:
        print ("face not found. . .")
        pass
    if cv2.waitKey(1)==13 or count == 200:
        break
cap.release()
cv2.destroyAllWindows()
print ("done. . . .")
