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
from sklearn.datasets import fetch_lfw_people, load_digits
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

global target_names
target_names = ["Leonardo","Borba","gabriel"] #all names the program will recognize

def getMatrix():
    mat = np.zeros(((len(target_names))*40,480*480))
    os.chdir("database")

    #Reading all images in person's database to begin training
    for i in range(len(target_names)):
        os.chdir('db_{0}'.format(target_names[i])) #db_name will be the folder with person's photos
        for j in range (1,41):
            image = cv2.imread('{0}_{1}.jpg'.format(target_names[i],j))
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
    X = np.zeros(((len(target_names))*40,480*480))
    for i in range(len(target_names)):
        for j in range(40):
            X = pca.transform(mat)
    return X

def getY():
    y = np.zeros(((len(target_names))*40))
    #Setting one ID for each name
    for i in range(y.shape[0]):
        y[i] = i//40
    return y

def splitTraining(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def computePCA(X_train, X_test):
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 190

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, len(X_train)))
    t0 = time()
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape
    print(h, w)
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    Xlfw = lfw_people.data
    print(Xlfw.shape)
    # pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(Xlfw)
    print("done in %0.3fs" % (time() - t0))

    # eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    return X_train_pca, X_test_pca

def trainSVM(X_train_pca, y_train):
    # Train a SVM classification model
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf

def predictFace(X_test_pca, y_test, y, classifier = None):
    clf = classifier
    # Quantitative evaluation of the model quality on the test set
    n_classes = len(target_names)

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

def main():
    matrix = getMatrix()
    # X = getX(matrix)
    X = matrix
    y = getY()
    X_train, X_test, y_train, y_test = splitTraining(X, y)
    X_train_pca, X_test_pca = computePCA(X_train, X_test)
    clf = trainSVM(X_train_pca, y_train)
    predictFace(X_test_pca, y_test, y, clf)

    ## Salvar clf em disco
