"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper
from scipy import linalg
import math
import scipy
import copy
from scipy import signal  
from scipy.linalg import inv 

import matplotlib.pyplot as plt
import math
import cv2


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""



def eight_point(pts1, pts2, M):
    #1.normalized pts
    scaler =np.array([[1/M,0,0],
              [0,1/M,0],
              [0,0,1]])
    norm_pts1=pts1/M;

    norm_pts2=pts2/M
    #2.Construct Matrix
    N,_=pts1.shape 

    matrixA=[]
    for i in range(N):
        x2=norm_pts2[i][0];
        y2=norm_pts2[i][1];
        x1=norm_pts1[i][0];
        y1=norm_pts1[i][1];
        cur_row=[x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y2,1]
        matrixA.append(cur_row)

    matrixA=np.array(matrixA)

    #3.Solve for SVD
    u1, s1, vh1=linalg.svd(matrixA)
    F =  np.reshape(vh1[8], (3, 3))


    #enforce rank2
    u2,s2,vh2=linalg.svd(F)
    s2[2]=0
    enforcedF=np.matmul(np.matmul(u2,np.diag(s2)),vh2)
  

    #unnormalized F
    refinedF=helper.refineF(enforcedF,norm_pts1,norm_pts2)
    finalF=np.matmul(np.matmul(np.transpose(scaler),refinedF),scaler)
    return finalF






"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""


# cited from hw1
'''
def gaussian_filter(sigma):
    hsize=2*math.ceil(3*sigma)+1
    gaus=scipy.signal.gaussian(hsize,sigma)
    gaus2=gaus[:,None]
    gaus.shape=(1,hsize)
    gausFinal=np.matmul(gaus2,gaus)
    gausFinal=gausFinal*(1/(np.sum(gausFinal)))

    return gausFinal
 '''




def distance(pad1, pad2):
    result=np.sum((pad1 - pad2) ** 2)
    return result

# give back the index that we used to index it
def find_candidates(pt1,F,cols):
    [x,y]=pt1
  
    homo_vector=np.array([x,y,1])
    epi_line=np.matmul(F,np.transpose(homo_vector))
    a=epi_line[0]
    b=epi_line[1]
    c=epi_line[2]
    output=[]
    for i in range(cols):
        y=round((-a*i-c)/b)
     
        output.append([y,i])
    return output

def pad(im1,im2,windowPadding):
    padArr = [(windowPadding, windowPadding), (windowPadding, windowPadding)]
    if len(im1.shape) > 2:
        padArr.append((0, 0))
    im1Pad = np.pad(im1, padArr, mode="constant", constant_values=0)
    im2Pad = np.pad(im2, padArr, mode="constant", constant_values=0)
    return im1Pad,im2Pad
def makeGaussianFiler(k_size, sigma):
    window = np.zeros( (k_size, k_size) )
    window[k_size//2, k_size//2]=1
    return gaussian_filter( window, sigma)




def find_best_candidates(candidates,pt1,im1, im2):
    window_size=5
    hSize=window_size//2
    rounds=len(candidates)
   
    img1,img2=pad(im1,im2,hSize)
    y1,x1=pt1
    x1=x1+hSize 
    y1=y1+hSize
    

    cur_pad_1=np.asarray(img1[x1-hSize:x1+hSize+1,y1-hSize:y1+hSize+1,:])
    
    #cur_pad_1=gaussian_filter(cur_pad_1,sigma=0.25)
    
    min_distance=0
    min_index=0
   
    for i in range(rounds):
        [x2,y2]=candidates[i]
      
        x2=x2+hSize 
        y2=y2+hSize
       
       
        cur_pad_2=np.asarray(img2[x2-hSize:x2+hSize+1,y2-hSize:y2+hSize+1,:])
    
        #cur_pad_2=gaussian_filter(cur_pad_2,sigma=0.25)

        dist=distance(cur_pad_1,cur_pad_2)
        

    
        if (i==0) or dist<min_distance:
            min_distance=dist
            min_index=i

    return candidates[min_index][::-1]




def epipolar_correspondences(im1, im2, F, pts1):
    row,col,channel=im2.shape
    output=[]
    
    for pt1 in pts1:
        candidates=find_candidates(pt1,F,col)
        #a two element list
        best_candidates=find_best_candidates(candidates,pt1,im1,im2)
        output.append(best_candidates)
    return np.array(output)

  

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return np.transpose(K2)@F@K1

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):

    N=len(pts2)
    output=[]
    for i in range(N):
        matrixA=[]
        y2=pts2[i][0]
        x2=pts2[i][1]
        y1=pts1[i][0]
        x1=pts1[i][1]
        matrixA.append(y1*P1[2]-P1[1])
        matrixA.append(P1[0]-x1*P1[2])
        matrixA.append(y2*P2[2]-P2[1])
        matrixA.append(P2[0]-x2*P2[2])
        matrixA=np.array(matrixA)
      
        u,s,vh=linalg.svd(matrixA)
        X=vh[-1]
        x=X[0]/X[3]
        y=X[1]/X[3]
        z=X[2]/X[3]
        output.append([x,y,z])
    return np.array(output) 


def homo(pts):
    N,x=pts.shape
    temp=np.ones((N,1))
    homo=np.hstack((pts,temp))
    return homo










"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""

def camera_center(K,R,t):
    A=-np.linalg.inv(np.matmul(K,R))
   
    B=np.matmul(K,t)
    result=np.matmul(A,B)
    return result

def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1= camera_center(K1,R1,t1)
    c1.shape=(1,3)
    c2= camera_center(K2,R2,t2)
    c2.shape=(1,3)

    r1=(c1-c2)/np.linalg.norm(c1-c2)
    r2=np.cross( r1, R1.T[:, -1]) 
    
    r3=np.cross(r2,r1)
    
    R=np.hstack((r1,r2))
    R=np.hstack((R,r3))
    R.shape=(3,3)
  
    R=np.transpose(R)
 

    t1p=-np.matmul(R,np.transpose(c1))
    t2p=-np.matmul(R,np.transpose(c2))


    M1=np.matmul(np.matmul(K2,R),inv(np.matmul(K1,R1)))
    M2=np.matmul(np.matmul(K2,R),inv(np.matmul(K2,R2)))
    return M1,M2,K2,K2,R,R,t1p,t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""


def dist(y,x,d,w,im1,im2):
    pad1=im1[y-w:y+w+1,x-w:x+w+1]
    pad2=im2[y-w:y+w+1,x-d-w:x-d+w+1]
    result=np.sum(np.square(pad1-pad2))
    return result


def get_disparity(im1, im2, max_disp, win_size):
   
    # replace pass by your implementation


    rows,cols=im1.shape
    output=None
    Dmap=np.zeros((rows,cols))
    img2pad=np.pad(im2,((0,0),(max_disp,0)),"constant")
    
    window=np.ones((win_size,win_size))
    for d in range(max_disp+1):
        img2_temp=img2pad[:,max_disp-d:cols+max_disp-d]
      
        temp=np.square(img2_temp-im1)
       
        convolved=signal.convolve2d(temp,window,mode="same",boundary='symm')
        
        if d==0:
            output=convolved
        else:
            for x in range(rows):
                for y in range(cols):
                    if(convolved[x][y]<output[x][y]):
                        Dmap[x][y]=d
                        output[x][y]=convolved[x][y]

    return Dmap


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1= camera_center(K1,R1,t1)
    c2= camera_center(K2,R2,t2)
    b=np.linalg.norm(c1-c2)
    f=K1[0][0]
    rows,cols=dispM.shape
    output=np.zeros((rows,cols))
    for y in range(rows):
        for x in range(cols):
            val=dispM[y][x]
            if val==0:
                continue
            else:
                val=b*(f/val)
                output[y][x]=val
    return output
            






"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    rows,cols=x.shape
    zeros=np.zeros((3,1))
    matrixA=[]
    for i in range(rows):
        x1=X[i][0]
        y1=X[i][1]
        z1=X[i][2]
        x_m=x[i][0]
        y_m=x[i][1]
        row1=[x1,y1,z1,1,0,0,0,0,-x_m*x1,-x_m*y1,-x_m*z1,-x_m]
        row2=[0,0,0,0,x1,y1,z1,1,-y_m*x1,-y_m*y1,-y_m*z1,-y_m]
        matrixA.append(row1)
        matrixA.append(row2)
    u,s,vh=np.linalg.svd(np.array(matrixA))
    out=vh[-1].reshape(3,4)
    return out


def performance(pts3d,P,pts2d):
    N,_=pts3d.shape
    xp=P @ np.hstack((pts3d, np.ones((N,1)))).T
    xp = xp[:2,:].T / np.vstack((xp[2,:], xp[2,:])).T
    
    length=len(pts2d)
    acc=0
    for x in range(length):
        a=np.array([pts2d[x][1],pts2d[x][0]])
        
        acc+=np.linalg.norm(a-xp[x])
    print(acc/length)

    



"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""


def estimate_params(P):
    u,s,vh=linalg.svd(P)
    c=vh[-1]
    newc=np.transpose(np.array([c[0]/c[3],c[1]/c[3],c[2]/c[3]]))

    M=P[:,0:3]
 
    K,R=linalg.rq(M)
   
    t=np.matmul(-M,newc)
 
    return K,R,t
