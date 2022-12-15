import numpy as np
import cv2
import matplotlib.pyplot as plt
from matchPics import matchPics
import scipy
from planarH import computeH_ransac,computeH,compositeH

#Import necessary functions

def homo(vec):
	rows,cols=vec.shape 
	ones=np.ones((rows,1))
	result=np.transpose(np.hstack((vec,ones)))
	return result

cv_desk = cv2.imread('/Users/xieshenyi/Desktop/2021 fall/16385/assgn2/data/cv_desk.png')
cv_cover = cv2.imread('/Users/xieshenyi/Desktop/2021 fall/16385/assgn2/data/cv_cover.jpg')
hp_cover =cv2.imread('/Users/xieshenyi/Desktop/2021 fall/16385/assgn2/data/hp_cover.jpg')
h,w,channel1=cv_desk.shape
h1,w1,channel1=cv_cover.shape

hp_cover=cv2.resize(hp_cover,(w1,h1))


#Write script for Q3.9
matches, locs1, locs2=matchPics(cv_cover,cv_desk)
rows,cols=matches.shape
locs1match=[]
locs2match=[]
#extract matched points print(np.matul(T1Final,x1temp))
for x in range(rows):

	match1=matches[x][0]
	match2=matches[x][1]
	locs1match.append(locs1[match1]) 
	locs2match.append(locs2[match2]) 

locs1match=np.array(locs1match)
locs2match=np.array(locs2match)


bestH2to1,inliers=computeH_ransac(locs2match,locs1match)
print(bestH2to1)
output=compositeH(np.linalg.inv(bestH2to1),hp_cover,cv_desk)
cv2.imshow("HP",output)
cv2.waitKey(0)




