import numpy as np
import cv2
from cv2 import cv
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')



cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    image =cv2.GaussianBlur(img,(5,5),10)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        crop_img = img[y:y+h, x:x+w] # Crop from x, y, w, h -> 100, 200, 300, 400
        height, width = img.shape[:2]
        res = cv2.resize(crop_img,(height, height), interpolation = cv2.INTER_CUBIC)
    try:
        cv2.imshow('Imagem Cortada',res)
    except NameError:
        pass
    cv2.imshow('Imagem normal',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
