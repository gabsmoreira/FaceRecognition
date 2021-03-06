from sklearn import svm
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
from training import *
import check_list as cl


def main():
    face_cascade = cv2.CascadeClassifier('loaders/haarcascade_face.xml')
    target_names = getTargetNames()
    print(target_names)
    f = open('loaders/clf.p', 'r')
    p = open('loaders/pca.p', 'r')
    g = open('loaders/clf_ex.p', 'r')
    clf = pickle.load(f)
    pca = pickle.load(p)
    clf_ex = pickle.load(g)
    cap = cv2.VideoCapture(0)
    cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    def prepare_image(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_flat_gray = image_gray.flat
        return image_flat_gray

    while True:
        ret, img = cap.read()
        image = cv2.GaussianBlur(img,(5,5),10)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.7, 5)
        #Text that appears next to the rectangle with the person's face_cascade
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        font_color = (0,0,255)
        font_thickness = 2
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            crop_img = img[y:y+h, x:x+w]
            height, width = img.shape[:2]

            res = cv2.resize(crop_img,(height, height), interpolation = cv2.INTER_CUBIC)
            prepared_image = prepare_image(res)
            print(len(prepared_image))
            n = pca.transform(prepared_image)

            pred_contem = clf_ex.predict(n)
            pred = clf.predict(n)

            #if pred_contem < 0 :
            #    person_name = "Alguem"
            #else:
            for i in pred:
                 person_name = target_names[int(i)]

            cv2.putText(img, person_name, (x+w+5,y+h), fontFace, fontScale, font_color, font_thickness, cv2.CV_AA)

        cv2.imshow('Imagem normal', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

    cap.release()
    cv2.destroyAllWindows()
