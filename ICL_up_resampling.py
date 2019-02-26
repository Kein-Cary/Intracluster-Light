# this file shows up-resampling
"""
up-resampling : the pixels scale become larger
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d as inter2
a = np.array([[2,4,3,7,5],[6,1,0,4,8],[7,2,7,5,6],[9,2,0,7,1],[4,6,9,3,4]])
x0 = np.linspace(0,4,5)
y0 = np.linspace(0,4,5)
N0x = len(x0)
N0y = len(y0)
f = inter2(x0,y0,a)
plt.imshow(a, cmap = 'binary', origin = 'lower',)
plt.show()

##########          
## resampling by new "pixel" result (towards bigger pixel size)
m1 = 1.2
m2 = 1.2
N2 = np.int(np.ceil(N0y*(1/m2)))
N1 = np.int(np.ceil(N0x*(1/m1)))
x0 = np.linspace(0, a.shape[1]-1, a.shape[1])
y0 = np.linspace(0, a.shape[0]-1, a.shape[0])
M = np.meshgrid(x0,y0)
resam = np.zeros((N2, N1), dtype = np.float)
def sum_samp(N1, N2, m1, m2, data):
    a = data
    Nx = N1
    Ny = N2
    sample = np.zeros((Ny, Nx), dtype = np.float)
    for p in range(Ny):
        for k in range(Nx):
            cdx = (2*k+1)*m1/2
            cdy = (2*p+1)*m2/2 # calculate the center of new pixel 
            edxl = k*m1
            edxr = (k+1)*m1 # calculate the edges in horizontal direction
            edyu = (p+1)*m2
            edyb = p*m2 # calculate the edges in virtual direction
            # check the cover pixel is normal or not
            idx = (2*M[0]+1)/2-cdx
            idy = (2*M[1]+1)/2-cdy
            # select this part pixel
            ia = (idx > -(m1/2 +1/2)) & (idx < (m1/2 + 1/2))
            ib = (idy > -(m2/2 +1/2)) & (idy < (m2/2 + 1/2))
            ic = ia & ib
            # print(ic)
            iu = np.where(ic == True) # find position
            vx = np.array(iu[1])
            vy = np.array(iu[0])
            L = len(vx)
            suma = 0
            for w in range(L):
                s = vx[w]
                l = vy[w]
                oedxl = s*1
                oedxr = (s+1)*1
                oedyu = (l+1)*1
                oedyb = l*1
                if (oedxl <= edxr) & (oedxr >= edxr) &\
                    (oedyb <= edyb) & (oedyu >= edyb):
                    suma =  suma + np.abs(edxr-oedxl)*np.abs(edyb-oedyu)*a[l,s]
                elif (oedxl >= edxl) & (oedxr <= edxr) &\
                    (oedyu >= edyb) & (oedyb <= edyb):
                    suma = suma + np.abs(oedxr-oedxl)*np.abs(oedyu-edyb)*a[l,s]
                elif (oedxl <= edxl) & (oedxr >= edxl) &\
                    (oedyu >= edyb) & (oedyb <= edyb):
                    suma = suma + np.abs(oedxr-edxl)*np.abs(oedyu-edyb)*a[l,s]
                elif (oedxl <= edxl) & (oedxr >= edxl) &\
                    (oedyu <= edyu) & (oedyb >= edyb):
                    suma = suma + np.abs(oedyu-oedyb)*np.abs(oedxr-edxl)*a[l,s]
                elif (oedxl <= edxl) & (oedxr >= edxl)&\
                    (oedyb <= edyu) & (oedyu >= edyu):
                    suma = suma + np.abs(oedxr-edxl)*np.abs(oedyb-edyu)*a[l,s]
                elif (oedxl >= edxl) & (oedxr <= edxr) &\
                    (oedyb <= edyu) & (oedyu >= edyu):
                    suma = suma + np.abs(oedxr-oedxl)*np.abs(edyu-oedyb)*a[l,s]
                elif (oedxl <= edxr) & (oedxr >= edxr) &\
                    (oedyb <= edyu) & (oedyu >= edyu):
                    suma = suma + np.abs(edxr-oedxl)*np.abs(oedyb-edyu)*a[l,s]
                elif (oedxl <= edxr) & (oedxr >= edxr) &\
                    (oedyb >= edyb) & (oedyu <= edyu):
                    suma = suma + np.abs(oedyu-oedyb)*np.abs(edxr-oedxl)*a[l,s]
                elif (oedxr <= edxr) & (oedxl >= edxl) &\
                    (oedyb >= edyb) & (oedyu <= edyu):
                    suma = suma + np.abs(oedyu-oedyb)*np.abs(oedxr-oedxl)*a[l,s]
                else:
                    pass
            sample[p,k] = suma
    return sample
# test part 
resam = sum_samp(N2, N1, m1, m2, a)
