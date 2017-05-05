import numpy as np
import cv2
from cv2 import cv
from time import time
import os
import scipy.misc
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

name =["Leonardo","Borba","gabriel"]
photo_list = []
mat = np.zeros(((len(name))*40,480*480))
os.chdir("database")
for i in range (len(name)):
    os.chdir('db_{0}'.format(name[i]))
    for j in range (1,41):
        image = cv2.imread('{0}_{1}.jpg'.format(name[i],j))
        #print(image.shape)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_flat_gray = image_gray.flat
        mat[i*40+j-1] = image_flat_gray
        if j==40:
            os.chdir("../")


pca = PCA(n_components=100, svd_solver='randomized',whiten=True).fit(mat)

eigenfaces = pca.components_.reshape((100, 480, 480))


os.chdir("../")
os.mkdir("eigenfaces")
os.chdir("eigenfaces")
for i in range(len(eigenfaces)):
    scipy.misc.imsave('eigenfaces_{0}.jpg'.format(i),eigenfaces[i])
