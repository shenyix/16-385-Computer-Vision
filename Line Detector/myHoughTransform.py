import math
import numpy as np
import scipy
import copy
from scipy import signal    # For signal.gaussian function
from myImageFilter import myImageFilter

def myHoughTransform(Im, rhoRes, thetaRes):

    (picHigh,picLong)=Im.shape
    #output 
    thetaScale=np.arange(0,np.pi*2,thetaRes)
    rhoAmount=((picHigh**2+picLong**2)**(0.5))

    maxRho=int(math.ceil((rhoAmount/rhoRes)))
    maxTheta=math.ceil(math.pi*2/thetaRes)

    rhoScale=np.arange(0,rhoAmount,rhoRes)
    #Initialize the acc 
    acc=np.zeros((maxRho,maxTheta))

    for x in range(picHigh):
        for y in range(picLong):
            if Im[x][y]!=0:
                for theta in range(maxTheta):
                    thetaReal=theta*thetaRes
                    rho=x*math.sin(thetaReal)+y*math.cos(thetaReal)
                    #ignore negative value
                    if rho>=0:
                        rhoReal=rho//rhoRes

                        acc[int(rhoReal)][theta]+=1 

    return [acc, rhoScale, thetaScale]






