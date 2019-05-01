import cv2 as cv
import sys

imagepath = sys.argv[1]
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml") #to read the cascade in python
#opencv is not 100% successfull in detecting the faces
img = cv.imread(imagepath) # read colour images
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
res_img = cv.resize(img, (1200,720))

faces = face_cascade.detectMultiScale(res_img,
                                      scaleFactor=1.1,
                                      minNeighbors=5)
for x, y, w, h in faces:
    img = cv.rectangle(res_img, (x,y), (x+w, y+h), (0,255,0), 2)


cv.imshow("grey", img)
cv.waitKey(0)
cv.destroyAllWindows()
