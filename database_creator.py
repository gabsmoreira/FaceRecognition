import numpy as np
import cv2
from cv2 import cv

from time import time
import os
import scipy.misc
import logging
import matplotlib.pyplot as plt
import sys


number_photos = 100

def main_database():
    local = os.getcwd()
    face_cascade = cv2.CascadeClassifier('loaders/haarcascade_face.xml')

    name = raw_input("Name:")
    i=0
    os.chdir('database')
    os.mkdir('db_{0}'.format(name))
    os.chdir('db_{0}'.format(name))

    cap = cv2.VideoCapture(0)
    cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    while True:

        ret, img = cap.read()
        image =cv2.GaussianBlur(img,(5,5),10)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 2, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            crop_img = img[y:y+h, x:x+w]
            height, width = img.shape[:2]
            res = cv2.resize(crop_img,(height, height), interpolation = cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            i+=1

        try:
            if i==number_photos + 1:
                break
            cv2.imshow('Imagem Cortada',res)
            string_photo = '{0}_{1}.jpg'.format(name,i)
            scipy.misc.imsave(string_photo,rgb)
        except NameError:
            pass
        cv2.imshow('Imagem normal',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
            
    os.chdir(local)
    print(os.getcwd())
    cap.release()
    cv2.destroyAllWindows()
