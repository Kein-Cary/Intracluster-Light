# resample by modelu
# this file shows up-resampling
"""
sum-resampling : the pixels scale become larger
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

a = np.array([[2,4,3,7,5],[6,1,0,4,8],[7,2,7,5,6],[9,2,0,7,1],[4,6,9,3,4]])
cx0 = 2
cy0 = 2
m1 = 1.2
m2 = 1.2

##########          
## resampling by new "pixel" result (towards bigger pixel size)
def sum_samp(m1, m2, data, cx0, cy0):
    a = data
    N0x = a.shape[1]
    N0y = a.shape[0]
    x0 = np.linspace(0,N0x-1,N0x)
    y0 = np.linspace(0,N0y-1,N0y)
    M = np.meshgrid(x0,y0)
    N2 = np.int(np.ceil(N0y*(1/m2)))
    N1 = np.int(np.ceil(N0x*(1/m1)))
    Nx = N1
    Ny = N2
    # get new center pixel

    cx = np.ceil(np.ceil(cx0)/m1)
    cy = np.ceil(np.ceil(cy0)/m2)
    cpos = np.array([cx, cy])
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
            oedxl = vx*1
            oedxr = (vx+1)*1
            oedyu = (vy+1)*1
            oedyb = vy*1
            
            f1x = (oedxr > edxr)&(oedxl <= edxr)
            f1y = (oedyu > edyu)&(oedyb <= edyu)
            f1 = f1x&f1y
            w1 = np.abs(edxr-oedxl[f1])*np.abs(edyu-oedyb[f1])
            
            f2x = (oedxl > edxl)&(oedxr <= edxr)
            f2y = (oedyb <= edyu)&(oedyu > edyu)
            f2 = f2x&f2y
            w2 = np.abs(oedxr[f2]-oedxl[f2])*np.abs(edyu-oedyb[f2])
            
            f3x = (oedxl <= edxl)&(oedxr > edxl)
            f3y = (oedyb <= edyu)&(oedyu > edyu)
            f3 = f3x&f3y
            w3 = np.abs(edyu-oedyb[f3])*np.abs(oedxr[f3]-edxl)
            
            f4x = (oedxl <= edxl)&(oedxr > edxl)
            f4y = (oedyu <= edyu)&(oedyb > edyb)
            f4 = f4x&f4y
            w4 = np.abs(oedxr[f4]-edxl)*np.abs(oedyu[f4]-oedyb[f4])
            
            f5x = (oedxl <= edxl)&(oedxr > edxl)
            f5y = (oedyu > edyb)&(oedyb <= edyb)
            f5 = f5x&f5y
            w5 = np.abs(oedxr[f5]-edxl)*np.abs(oedyu[f5]-edyb)
            
            f6x = (oedxl > edxl)&(oedxr <= edxr)
            f6y = (oedyu > edyb)&(oedyb <= edyb)
            f6 = f6x&f6y
            w6 = np.abs(oedxr[f6]-oedxl[f6])*np.abs(oedyu[f6]-edyb)
            
            f7x = (oedxl <= edxr)&(oedxr > edxr)
            f7y = (oedyu > edyb)&(oedyb <= edyb)
            f7 = f7x&f7y
            w7 = np.abs(edxr-oedxl[f7])*np.abs(oedyu[f7]-edyb)
            
            f8x = (oedxr > edxr)&(oedxl <= edxr)
            f8y = (oedyu <= edyu)&(oedyb > edyb)
            f8 = f8x&f8y
            w8 = np.abs(oedyu[f8]-oedyb[f8])*np.abs(edxr-oedxl[f8])
            
            f9x = (oedxr <= edxr)&(oedxl > edxl)
            f9y = (oedyu <= edyu)&(oedyb > edyb)
            f9 = f9x*f9y
            w9 = np.abs(oedxr[f9]-oedxl[f9])*np.abs(oedyu[f9]-oedyb[f9])
            
            sample[p,k] = (np.sum(a[vy[f1],vx[f1]]*w1)+ np.sum(a[vy[f2],vx[f2]]*w2)+
                        np.sum(a[vy[f3],vx[f3]]*w3)+ np.sum(a[vy[f4],vx[f4]]*w4)+
                        np.sum(a[vy[f5],vx[f5]]*w5)+ np.sum(a[vy[f6],vx[f6]]*w6)+
                        np.sum(a[vy[f7],vx[f7]]*w7)+ np.sum(a[vy[f8],vx[f8]]*w8)+
                        np.sum(a[vy[f9],vx[f9]]*w9))           
    return sample, cpos
# test part 

resam = sum_samp(m1, m2, a, cx0, cy0)[0]
plt.imshow(resam,cmap='binary',origin='lower')
