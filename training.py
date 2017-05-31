import numpy as np
import cv2
from cv2 import cv
from time import time
import os
import scipy.misc
import matplotlib.pyplot as plt
import sys
import cPickle as pickle

#Scikit-learn imports (machine learning in Python)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people, load_digits
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from database_creator import number_photos
from sklearn import svm

global target_names
def getTargetNames():
    global target_names
    target_names = [] #all names the program will recognize
    for folder in os.listdir("database"):
        if folder == "clf.p":
            pass
        else:
            target_names.append(folder[3:])
    return target_names

def getMatrix():
    mat = np.zeros(((len(target_names))*number_photos,480*480))
    os.chdir("database")

    #Reading all images in person's database to begin training
    for i in range(len(target_names)):
        os.chdir('db_{0}'.format(target_names[i])) #db_name will be the folder with person's photos
        for j in range (1,number_photos+1):
            image = cv2.imread('{0}_{1}.jpg'.format(target_names[i],j))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_flat_gray = image_gray.flat
            mat[i*number_photos+j-1] = image_flat_gray
            if j==number_photos:
                os.chdir("../")
    return mat

def eigenfaces(mat):
    pca = PCA(n_components=120, svd_solver='randomized',whiten=True).fit(mat)
    eigenfaces = pca.components_.reshape((120, 480, 480))
    os.chdir("../")
    os.chdir("eigenfaces")
    for i in range(len(eigenfaces)):
       scipy.misc.imsave('eigenfaces_{0}.jpg'.format(i),eigenfaces[i])


def getX(mat):
    pca = PCA(n_components=number_photos, svd_solver='randomized',whiten=True).fit(mat)
    X = np.zeros(((len(target_names))*number_photos,480*480))
    for i in range(len(target_names)):
        for j in range(number_photos):
            X = pca.transform(mat)
    return X

def getY():
    y = np.zeros(((len(target_names))*number_photos))
    #Setting one ID for each name
    for i in range(y.shape[0]):
        y[i] = i//number_photos
    return y

def splitTraining(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def computePCA(X_train, X_test):
    n_components = 90
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, len(X_train)))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    # eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(X_test_pca)
    print("done in %0.3fs" % (time() - t0))
    return X_train_pca, X_test_pca, pca

def trainSVM(X_train_pca, y_train):
    # Train a SVM classification model
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf.fit(X_train_pca, y_train)
    return clf

def trainSVM_Oneclass(X_train_pca, y_train):
    # Train a SVM classification model
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'nu': [0.13, 0.3, 0.6, 0.35, 0.9, 0.5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf_ex = GridSearchCV(svm.OneClassSVM(nu=0.1, kernel="rbf"), param_grid, scoring='accuracy')
    clf_ex.fit(X_train_pca, y_train)

    correct = np.mean(clf_ex.predict(X_train_pca) == 1)
    print(correct)
    return clf_ex

def main_training():
    target_names = getTargetNames()
    matrix = getMatrix()
    X = matrix
    y = getY()
    X_train, X_test, y_train, y_test = splitTraining(X, y)
    X_train_pca, X_test_pca, pca = computePCA(X_train, X_test)
    clf = trainSVM(X_train_pca, y_train)
    clf_ex = trainSVM_Oneclass(X_train_pca, y_train)
    print(clf_ex)
    os.chdir("../")

    with open('clf.p', 'w+') as f:
        pickle.dump(clf, f)

    with open('clf_ex.p', 'w+') as g:
        pickle.dump(clf_ex,g)

    with open('pca.p','w+') as p:
        pickle.dump(pca,p)
