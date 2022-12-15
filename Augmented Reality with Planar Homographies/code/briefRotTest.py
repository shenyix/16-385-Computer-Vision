import numpy as np
import scipy
import cv2
from matchPics import matchPics
from skimage import data
from skimage.color import rgb2gray
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
import matplotlib.pyplot as plt

#Q3.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')

hist=[]




for i in range(36):

	temp=scipy.ndimage.rotate(cv_cover,i*10)
	matches, locs1, locs2 = matchPics(cv_cover, temp)

	rows,cols=matches.shape
	hist.append(rows)




angle = [i*10 for i in range(36)]
plt.bar(angle, height = hist, width = 4)
plt.xlabel("Degrees of rotation")
plt.ylabel("Match counts ")

plt.show()





