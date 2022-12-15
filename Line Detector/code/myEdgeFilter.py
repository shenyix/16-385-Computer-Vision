import math
import numpy as np
import scipy
import copy
from scipy import signal    # For signal.gaussian function
from myImageFilter import myImageFilter


def process(slope):

    distance=[(abs(slope-0)),(abs(slope-45)),(abs(slope-90)),(abs(slope-135)),(abs(slope-180))]

    miniIndex=distance.index(min(distance))

    return miniIndex



def myEdgeFilter(img0, sigma):

    (picHigh,picLong)=img0.shape
    hsize=2*math.ceil(3*sigma)+1

    gaus=scipy.signal.gaussian(hsize,sigma)
    gaus2=gaus[:,None]
    gaus.shape=(1,hsize)
   
    gausFinal=np.matmul(gaus2,gaus)
    gausFinal=gausFinal*(1/(np.sum(gausFinal)))
 

    imgS=myImageFilter(img0,gausFinal,False)
    horizSobel=np.array([[1,0,-1],
                         [2,0,-2],
                         [1,0,-1]])
    verSobel=np.array([[1,2,1],
                       [0,0,0],
                       [-1,-2,-1]])

    imgy=myImageFilter(imgS,verSobel,True)
    imgx=myImageFilter(imgS,horizSobel,True)
    imgg=np.sqrt(np.add(np.square(imgy),np.square(imgx)))
    for x in range(picHigh):   
        for y in range(picLong):
            if imgg[x][y]>255:
                imgg[x][y]=255
            elif imgg[x][y]<0:
                imgg[x][y]


    imgout= copy.deepcopy(imgg)
    imgg=np.pad(imgg,(1,),"edge")



    



    for x in range(picHigh):   
        for y in range(picLong):
            
            slope=np.arctan2(imgy[x,y],imgx[x,y])*180 / np.pi
            if slope<0: 
                slope+=180
         

            stepMap=[[(0,1),(0,-1)],[(-1,-1),(1,1)],[(1,0),(-1,0)],[(1,-1),(-1,1)],[(0,1),(0,-1)]]
            closeIndex=process(slope)
            step=stepMap[closeIndex]
            [(x1,y1),(x2,y2)]=step
            if((imgg[x+x1][y+y1])>imgg[x][y]) or ((imgg[x+x2][y+y2])>imgg[x][y]):
                imgout[x][y]=0

    return imgout







#issue:

#how does the scalar work acutally??


