import numpy as np
import cv2
from cv2 import cv
from time import time
import os
import scipy.misc
import matplotlib.pyplot as plt
import sys

name =["Leonardo","Borba","gabriel"]
os.chdir("database")
for i in range (len(name)):
    os.chdir('db_{0}'.format(name[i]))
    for j in range 41:
        '{0}_{1}.jpg'.format(name[i],j)
