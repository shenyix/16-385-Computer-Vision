import numpy as np

def myImageFilter(img0, h, calc):
   
    picLong=len(img0[0])
    picHigh=len(img0)
    hSize=len(h)//2

    outImg=np.zeros(shape=(picHigh,picLong))

    #padding
    img1=np.pad(img0,(hSize,),"edge")
  
    


   

    #apply the filter
    #start from hSize because of padding
    for x in range(hSize,picHigh+hSize):
        for y in range(hSize,hSize+picLong):
        
            curPad=img1[x-hSize:x+hSize+1,y-hSize:y+hSize+1]
            
            replace=np.sum(np.multiply(curPad,h))
            if calc==False:
                if replace>255:
                    replace=255
                elif replace<0:
                    replace

            
            outImg[x-hSize][y-hSize]=replace

    return outImg







