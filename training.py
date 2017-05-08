import numpy as np
import cv2
from cv2 import cv
from time import time
import os
import scipy.misc
import matplotlib.pyplot as plt
import sys

#Scikit-learn imports (machine learning in Python)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



def getMatrix():
    name = ["Leonardo","Borba","gabriel"] #all names the program will recognize
    photo_list = []
    mat = np.zeros(((len(name))*40,480*480))
    os.chdir("database")

    #Reading all images in person's database to begin training
    for i in range(len(name)):
        os.chdir('db_{0}'.format(name[i])) #db_name will be the folder with person's photos
        for j in range (1,41):
            image = cv2.imread('{0}_{1}.jpg'.format(name[i],j))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_flat_gray = image_gray.flat
            mat[i*40+j-1] = image_flat_gray
            if j==40:
                os.chdir("../")

    return mat


def eigenfaces(mat):
    pca = PCA(n_components=100, svd_solver='randomized',whiten=True).fit(mat)

    eigenfaces = pca.components_.reshape((100, 480, 480))

    os.chdir("../")
    os.chdir("eigenfaces")
    for i in range(len(eigenfaces)):
       scipy.misc.imsave('eigenfaces_{0}.jpg'.format(i),eigenfaces[i])


def getX(mat):
    pca = PCA(n_components=100, svd_solver='randomized',whiten=True).fit(mat)
    X = []
    name = ["Leonardo","Borba","gabriel"] #all names the program will recognize
    for i in range(len(name)):
        for j in range (1,41):
            X.append(pca.transform(mat[i*40+j-1]))
    return X
