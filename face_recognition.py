
import numpy as np
import cv2
from cv2 import cv
from time import time
import os
import scipy.misc
import logging
import sys
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people, load_digits
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from training import main_training, predictFace

face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
target_names = ["Borba","gabriel", "Leonardo"] #all names the program will recognize

f = open('clf.p', 'r')
clf = pickle.load(f)
f.close()
cap = cv2.VideoCapture(0)

X_test_pca, y_test, y, clf = main_training()



while True:

    ret, img = cap.read()
    image = cv2.GaussianBlur(img,(5,5),10)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    #Text that appears next to the rectangle with the person's face_cascade
    person_name = "Someone..."
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    font_color = (0,0,255)
    font_thickness = 2
    predictFace(X_test_pca, y_test, y, clf)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img, person_name, (x+w+5,y+h), fontFace, fontScale, font_color, font_thickness, cv2.CV_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        crop_img = img[y:y+h, x:x+w]
        height, width = img.shape[:2]

        res = cv2.resize(crop_img,(height, height), interpolation = cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        res_flat = res.flat
        # y_pred = clf.predict(res_flat)
        # person_name = y_pred[0]
    ##try:
        ##cv2.imshow('Imagem Cortada', res)
    ##except NameError:
        ##pass
    cv2.imshow('Imagem normal', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
