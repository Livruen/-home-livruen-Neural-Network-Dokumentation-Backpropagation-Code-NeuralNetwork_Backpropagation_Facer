import numpy as np
import cv2
from lib import ImageFunctions

THICKNESS = 2
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

imgPath = 'data/images/child.jpg'

img =[]
img = cv2.imread(imgPath)
gray = ImageFunctions.turnImageToGray(imgPath)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:

    cv2.rectangle(img, (x,y), (x+w,y+h), COLOR_BLUE, THICKNESS)
    img_grey = gray[y:y + h, x:x + w]

    img_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(img_grey)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img_color, (ex, ey), (ex + ew, ey + eh), COLOR_GREEN, THICKNESS)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()