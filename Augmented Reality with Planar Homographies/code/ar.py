import numpy as np
import cv2
from loadVid import loadVid
from planarH import computeH_ransac,computeH,compositeH
from matchPics import matchPics
from helper import plotMatches
import matplotlib.pyplot as plt
#Import necessary functions
hp_cover =cv2.imread('../data/hp_cover.jpg')
cv_cover = cv2.imread('../data/cv_cover.jpg')
pathPanda=('../data/ar_source.mov')
pathBook=('../data/book.mov')
framesPanda=loadVid(pathPanda)
framesBook=loadVid(pathBook)
rounds=len(framesPanda)
h, w, c = cv_cover.shape

arrayFrame=[]
for x in range(rounds):
	
	frame=framesBook[x]
	frame2=framesPanda[x]
	h2,w2,c2=frame.shape
	
	h1,w1,c1=frame2.shape
	
	left=int(max(w1/2-w/2,0))
	right=int(min(w1/2+(w/2),w1))
	top=int(max(h1/2-(h/2),0))
	bottom=int(min(h1/2+h/2,h1))
	
	frame2=frame2[top+70:bottom-70,left:right]
	
	frame2=cv2.resize(frame2,(w,h))
	matches, locs1, locs2=matchPics(cv_cover,frame)
	
	if len(matches)>3:
		h1,w1,channel1=cv_cover.shape
		hp_cover=cv2.resize(hp_cover,(w1,h1))
		rows,cols=matches.shape
		locs1match=[]
		locs2match=[]
		for x in range(rows):
			match1=matches[x][0]
			match2=matches[x][1]
			locs1match.append(locs1[match1]) 
			locs2match.append(locs2[match2]) 
		locs1match=np.array(locs1match)
		locs2match=np.array(locs2match)
		bestH2to1,inliers=computeH_ransac(locs2match,locs1match)
		output=compositeH(np.linalg.inv(bestH2to1),frame2,frame)
		
	
		arrayFrame.append(output)

out=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (w2,h2))
for frame in arrayFrame:
	out.write(frame)


out.release()
cv2.destroyAllWindows()


