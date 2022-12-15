import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import inv



# 1. Load the two temple images and the points from data/some_corresp.npz
img1=cv2.imread("../data/im1.png")
img2=cv2.imread("../data/im2.png")
data=np.load("../data/some_corresp.npz")
M=max(img1.shape)
# 2. Run eight_point to compute F
F=sub.eight_point(data["pts1"], data["pts2"], M)
#hlp.epipolarMatchGUI(img1, img2, F)

in_data=np.load("../data/intrinsics.npz")
K1= in_data["K1"]
K2= in_data["K2"]

E=sub.essential_matrix(F,K1,K2)

# 3. Load points in image 1 from data/temple_coords.npz
data_pts1=np.load("../data/temple_coords.npz")["pts1"]


# 4. Run epipolar_correspondences to get points in image 2
pts22=sub.epipolar_correspondences(img1, img2, F, data_pts1)

#ts.re(data_pts1,F,img1,img2)
# 5. Compute the camera projection matrix P1

ex_P1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P1=np.matmul(K1,ex_P1)
print(K1.shape)
print(E.shape)
best=None
# 6. Use camera2 to get 4 camera projection matrices P2
best=[]
Champ=0
best=None
for j in range(4):
	candidates_P2=hlp.camera2(E)[:,:,j]

	P2_temp=np.matmul(K2,candidates_P2)
	pts3d=sub.triangulate(P1, data_pts1, P2_temp, pts22)
	length=pts3d.shape[0]
	acc=0

	for i in range(length):
		if (pts3d[i][2]>=0):
			acc+=1
	
	if acc>Champ:
		Champ=acc;
		best=j


'''
mini=np.inf
for i in best:
	candidate_P2=hlp.camera2(E)[:,:,i]
	K,R,t=sub.estimate_params(candidates_P2)
	val=abs(np.linalg.det(R)-1)
	if val<mini:
		best=i
		mini=val
print(best)
'''

# 7. Run triangulate using the projection matrices

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
P2_temp=np.matmul(K2,hlp.camera2(E)[:,:,best])

pts3d=sub.triangulate(P1, data_pts1, P2_temp, pts22)
sub.performance(pts3d,P1,data_pts1)

R1=ex_P1[:,0:3]
T1=ex_P1[:,3:4]
R2=hlp.camera2(E)[:,:,best][:,0:3]
T2=hlp.camera2(E)[:,:,best][:,3:4]




outfile="../data/extrinsics.npz"
np.savez(outfile, R1=R1, R2=R2,t1=T1,t2=T2)
what=np.load("../data/extrinsics.npz")



pts3d=np.transpose(pts3d)

x=pts3d[0]
y=pts3d[1]
z=pts3d[2]

ax.scatter(x, y, z)
plt.show()

# 8. Figure out the correct P2

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
