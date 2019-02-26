# the file shows down-resampling
"""
down resampling: the pixels scale become smaller
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d as inter2
a = np.array([[2,4,3,7,5],[6,1,3,4,8],[7,2,7,5,6],[9,2,2,7,1],[4,6,9,3,4]])
x0 = np.linspace(0,4,5)
y0 = np.linspace(0,4,5)
N0x = len(x0)
N0y = len(y0)
f = inter2(x0,y0,a)
plt.imshow(a, cmap = 'binary', origin = 'lower',)
plt.show()

##########
## resampling by new "pixel" result (towards smaller pixel size)
m1 = 0.8 # coloumn scale (for row direction)
m2 = 0.8 # row scale (for coloumn direction)
N1x = np.int((N0x*(1/m1)))
N1y = np.int((N0y*(1/m2)))
M0 = np.meshgrid(x0,y0)
def down_samp(m1, m2, N1, N2, data):
    a = data
    Ny = N2
    Nx = N1
    sNy = a.shape[0]
    sNx = a.shape[1]
    x1 = np.linspace(0,Nx-1,Nx)
    y1 = np.linspace(0,Ny-1,Ny)
    M1 = np.meshgrid(x1,y1)
    sample = np.zeros((Ny, Nx), dtype = np.float)
    for k in range(sNy):
        for l in range(sNx):
            cdx = (2*l+1)/2
            cdy = (2*k+1)/2 # original pixel center
            edxl = l*1
            edxr = (l+1)*1 # edges of x direction
            edyb = k*1 
            edyu = (k+1)*1 # edges of y direction
            # calculate the covers of new pixels
            idx = (2*M1[0]+1)*m1/2-cdx
            idy = (2*M1[1]+1)*m2/2-cdy
            # select this part pixel
            ia = (idx > -(m1/2 +1/2)) & (idx < (m1/2 + 1/2))
            ib = (idy > -(m2/2 +1/2)) & (idy < (m2/2 + 1/2))
            ic = ia & ib
            iu = np.where(ic == True) # find position
            vx = np.array(iu[1])
            vy = np.array(iu[0])
            L = len(vx)
            for w in range(L):
                p = vy[w]
                s = vx[w]
                nedxl = s*m1
                nedxr = (s+1)*m1
                nedyb = p*m2
                nedyu = (p+1)*m2
                if (nedxl >= edxl) & (nedxr <= edxr) & (nedyu <= edyu) & (nedyb >= edyb):
                    sample[p,s] = np.abs(nedxr-nedxl)*np.abs(nedyu-nedyb)*a[k,l]
                elif (nedxr >= edxr) & (nedxl <= edxr) & (nedyb <= edyb) & (nedyu >= edyb):
                    if (k == 0) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l] 
                    elif (k == 0) & (l != sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-edyb)*a[k,l+1]
                    elif (k != 0) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(edxr-nedxl)*np.abs(edyb-nedyb)*a[k-1,l]
                    else:
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-edyb)*a[k,l+1]+\
                                        np.abs(nedxl-edxr)*np.abs(edyb-nedyb)*a[k-1,l]+\
                                        np.abs(nedxr-edxr)*np.abs(edyb-nedyb)*a[k-1,l+1]
                elif (nedxr <= edxr) & (nedxl >= edxl) & (nedyu >= edyb) & (nedyb <= edyb):
                    if k == 0:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]
                    else:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-nedxl)*np.abs(edyb-nedyb)*a[k-1,l]
                elif (nedxl <= edxl) & (nedxr >= edxl) & (nedyu >= edyb) & (nedyb <= edyb):
                    if (k == 0) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]
                    elif (k != 0) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxl)*np.abs(edyb-nedyb)*a[k-1,l]
                    elif (k == 0) & (l !=0 ):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(edxl-nedxl)*np.abs(nedyu-edyb)*a[k,l-1]
                    else:
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxl)*np.abs(edyb-nedyb)*a[k-1,l]+\
                                        np.abs(edxl-nedxl)*np.abs(edyb-nedyb)*a[k-1,l-1]+\
                                        np.abs(edxl-nedxl)*np.abs(nedyu-edyb)*a[k,l-1]
                elif (nedxl <= edxl) & (nedxr >= edxl) & (nedyu <= edyu) & (nedyb >= edyb):
                    if l == 0:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(nedxr-edxl)*a[k,l]
                    else:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(nedxr-edxl)*a[k,l]+\
                                        np.abs(nedyu-nedyb)*np.abs(edxl-nedxl)*a[k,l-1]
                elif (nedyu >= edyu) & (nedyb <= edyu) & (nedxl <= edxl) & (nedxr >= edxl):
                    if (k == sNy-1) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]
                    elif (k == sNy-1) & (l != 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxl-edxl)*np.abs(edyu-nedyb)*a[k,l-1]
                    elif (k != sNy-1) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedyu-edyu)*np.abs(nedxr-edxl)*a[k+1,l]
                    else:
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxl-edxl)*np.abs(edyu-nedyb)*a[k,l-1]+\
                                        np.abs(nedxl-edxl)*np.abs(nedyu-edyu)*a[k+1,l-1]+\
                                        np.abs(nedxr-edxl)*np.abs(nedyu-edyu)*a[k+1,l]
                elif (nedxl >= edxl) & (nedxr <= edxr) & (nedyu >= edyu) & (nedyb <= edyu):
                    if k == sNy-1:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]
                    else:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxr-nedxl)*np.abs(nedyu-edyu)*a[k+1,l]
                elif (nedxl <= edxr) & (nedxr >= edxr) & (nedyu >= edyu) & (nedyb <= edyu):
                    if (k == sNy-1) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]
                    elif (k == sNy-1) & (l != sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(edyu-nedyb)*a[k,l+1]
                    elif (k != sNy-1) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(edxr-nedxl)*np.abs(nedyu-edyu)*a[k+1,l]
                    else:
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(edyu-nedyb)*a[k,l+1]+\
                                        np.abs(edxr-nedxl)*np.abs(edyu-nedyu)*a[k+1,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-edyu)*a[k+1,l+1]
                elif (nedxl <= edxr) & (nedxr >= edxr) & (nedyu <= edyu) & (nedyb >= edyb):
                    if l == sNx-1:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(edxr-nedxl)*a[k,l]
                    else:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(edxr-nedxl)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-nedyb)*a[k,l+1]
                else:
                    pass
    return sample
resam = np.zeros((N1y, N1x), dtype = np.float)
resam = down_samp(m1, m2, N1x, N1y, a)
plt.imshow(resam,cmap='binary',origin='lower')
plt.show()