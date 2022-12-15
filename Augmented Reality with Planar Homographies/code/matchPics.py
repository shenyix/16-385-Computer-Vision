import numpy as np
import cv2
from skimage import data
from skimage.color import rgb2gray
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
	I1 = rgb2gray(I1)
	I2 = rgb2gray(I2)
	locs1=corner_detection(I1)
	locs2=corner_detection(I2)
	desc1, locs1 = computeBrief(I1, locs1)
	desc2, locs2 = computeBrief(I2, locs2)
	matches = briefMatch(desc1, desc2)
	
	return matches, locs1, locs2
	
	
	#Obtain descriptors for the computed feature locations
	
	
	


	#Match features using the descriptors
	

	
