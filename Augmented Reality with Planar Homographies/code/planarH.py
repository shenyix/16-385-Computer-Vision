import numpy as np
import cv2
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
from random import sample
import scipy
cv_desk = cv2.imread('../data/cv_desk.png')

cv_cover = cv2.imread('../data/cv_cover.jpg')



def computeH(x11, x22):
	x1=np.copy(x11)
	x2=np.copy(x22)
	#Q3.6
	#Compute the homography between two sets of points
	rows,cols=x1.shape
	matrixA=[]

	for i in range(rows):
		x_1=x2[i][0]
		y_1=x2[i][1]
		x_2=x1[i][0]
		y_2=x1[i][1]
		temp1=[-x_1,-y_1,-1,0,0,0,x_1*x_2,y_1*x_2,x_2]
		temp2=[0,0,0,-x_1,-y_1,-1,x_1*y_2,y_1*y_2,y_2]
		matrixA.append(temp1)
		matrixA.append(temp2)
	matrixA=np.array(matrixA)
	u, s, vh=linalg.svd(matrixA)
	H =  np.reshape(vh[8], (3, 3))
	


	return H


def computeH_norm(x11, x22):
	#Q3.7
	#Compute the centroid of the points
	x1=np.copy(x11)
	x2=np.copy(x22)

	rows,cols=x1.shape
	[mean11,mean12]=np.mean(x1, axis=0)
	[mean21,mean22]=np.mean(x2, axis=0)

	ones = np.ones((rows,1))
	x1temp = np.hstack((x1,ones))
	x2temp = np.hstack((x2,ones))
	output=np.copy(x2temp)

	T1=np.array([[1,0,-mean11],
		         [0,1,-mean12],
		         [0,0,1]])


	newx1=np.transpose(np.matmul(T1,np.transpose(x1temp)))[:,0:-1]

	max1=0
	for x in range(rows):
		norm=np.linalg.norm(newx1[x])
		if norm>max1:
			max1=norm

	
	T12=np.array([[1/max1,0,0],
		         [0,1/max1,0],
		         [0,0,1]])
	

	T1Final=np.matmul(T12,T1)
	

	T2=np.array([[1,0,-mean21],
		         [0,1,-mean22],
		         [0,0,1]])


	
	
	newx2=np.transpose(np.matmul(T2,np.transpose(x2temp)))[:,0:-1]
	distance2=0
	max2=0
	for x in range(rows):
		norm=np.linalg.norm(newx2[x])
		if norm>max2:
			max2=norm
	T22=np.array([[1/max2,0,0],
		         [0,1/max2,0],
		         [0,0,1]])
	
	T2Final=np.matmul(T22,T2)
	normalized2=np.transpose(np.matmul(T2Final,np.transpose(x2temp)))[:,0:-1]
	
	normalized1=np.transpose(np.matmul(T1Final,np.transpose(x1temp)))[:,0:-1]

	H=computeH(normalized1,normalized2)
	

	result=np.matmul(np.linalg.inv(T1Final),(np.matmul(H,T2Final)))
	
	return result
def homo(vec):
	rows,cols=vec.shape 
	ones=np.ones((rows,1))
	result=np.transpose(np.hstack((vec,ones)))
	return result



	#Shift the origin of the points to the centroid

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)

	#Similarity transform 1

	#Similarity transform 2


	#Compute homography


	#Denormalization
	




#x1=Hx2
'''
def geometricDistance(locs1,locs2, h):

    p1 = np.transpose(np.matrix([locs2[0].item(0), locs2[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)
   '''






def computeH_ransac(locs11, locs22):
	locs1=np.copy(locs11)
	locs2=np.copy(locs22)
	rows,cols=locs1.shape
	threshold=2**(1/2)
	num=100
	index=[i for i in range(rows)]
	champ=0
	champI=[]
	bestH=[]
	
	while num!=0:
		acc=0
		curr=sample(index, k=4)
		temp1=locs1[curr[0]]
		temp2=locs2[curr[0]]
		for x in curr[1:]:
			temp1=np.vstack((locs1[x],temp1))
			temp2=np.vstack((locs2[x],temp2))
		
		H=computeH_norm(temp1,temp2)
		
	
		predict=np.transpose(np.matmul(H,homo(locs2)))
		
		inliers=[0]*rows
		for m in range(rows):
			if (predict[m][2]!=0):
				x=predict[m][0]/predict[m][2]
				y=predict[m][1]/predict[m][2]
				distance=np.linalg.norm(np.array([x,y])-locs1[m])
				if distance<threshold:
					acc+=1
					inliers[m]=1
				
		if acc>champ:
			champ=acc
			champI=inliers
			bestH=H
		num-=1
	inliers1=np.ones((1,2))
	inliers2=np.ones((1,2))
	
	for x in range(rows):
		if champI[x]==1:
			inliers1=np.vstack((locs1[x],inliers1))
			inliers2=np.vstack((locs2[x],inliers2))
	
	


	return computeH_norm(inliers1[0:-1,:],inliers2[0:-1,:]),np.array(champI)




			

def compositeH(H2to1, template, img):
	img=img.astype(np.uint8)
	temp=np.swapaxes(template,0,1)
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	H=np.linalg.inv(H2to1)
	#For warping the template to the image, we need to invert it.
	w,h,channel1=temp.shape
	w1,h1,channel2=img.shape

	#Create mask of same size as template
	mask = np.ones( (h , w,channel1) , dtype=np.uint8)

	mask= np.swapaxes(mask,0,1)
	#Warp mask by appropriate homography
	inter_mask= cv2.warpPerspective(mask,H,(w1,h1))
	inter_mask=np.swapaxes(inter_mask,0,1)
	warped_mask=scipy.ndimage.rotate(inter_mask,0*10)==0
	
	warped_mask.astype(np.uint8)

	

	
	#Warp template by appropriate homography
	inter_temp=cv2.warpPerspective(temp,H,(w1,h1))
	inter_temp=np.swapaxes(inter_temp,0,1)
	warped_temp=scipy.ndimage.rotate(inter_temp,0*10)
	
	#Use mask to combine the warped template and the image
	
	masked_image=warped_mask*img
	
	

	final_output=masked_image+warped_temp
	
	return final_output



