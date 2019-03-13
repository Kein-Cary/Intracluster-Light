# resample by modelu
# this file shows up-resampling
"""
sum-resampling : the pixels scale become larger
"""
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
'''
a = np.array([[2,4,3,7,5],[6,1,0,4,8],[7,2,7,5,6],[9,2,0,7,1],[4,6,9,3,4]])
cx0 = 2
cy0 = 2
m1 = 1.2
m2 = 1.2 # for up-resamp
m3 = 0.5
m4 = 0.5 # for down-resamp
'''
##########          
def sum_samp(m1, m2, data, cx0, cy0):
    mx = m1
    my = m2
    a = data
    N0x = a.shape[1]
    N0y = a.shape[0]
    x0 = np.linspace(0,N0x-1,N0x)
    y0 = np.linspace(0,N0y-1,N0y)
    M = np.meshgrid(x0,y0)
    N2 = np.int(np.ceil(N0y*(1/my)))
    N1 = np.int(np.ceil(N0x*(1/mx)))
    Nx = N1
    Ny = N2
    # get new center pixel
    cx = np.ceil(np.ceil(cx0)/mx)
    cy = np.ceil(np.ceil(cy0)/my)
    cpos = np.array([cx, cy])
    sample = np.zeros((Ny, Nx), dtype = np.float)
    for p in range(Ny):
        W = []
        for k in range(Nx):
            cdx = (2*k+1)*mx/2
            cdy = (2*p+1)*my/2 # calculate the center of new pixel 
            edxl = k*mx
            edxr = (k+1)*mx # calculate the edges in horizontal direction
            edyu = (p+1)*my
            edyb = p*my # calculate the edges in virtual direction
            # check the cover pixel is normal or not
            idx = (2*M[0]+1)/2-cdx
            idy = (2*M[1]+1)/2-cdy
            # select this part pixel
            ia = np.abs(idx) < (mx/2+1/2)
            ib = np.abs(idy) < (my/2+1/2)
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
            f1 = f1x & f1y
            s1 = np.abs(edxr-oedxl[f1])*np.abs(edyu-oedyb[f1])
            w1 = np.sum(a[vy[f1],vx[f1]]*s1)
            W.append(w1)
            
            f2x = (oedxl > edxl)&(oedxr <= edxr)
            f2y = (oedyb <= edyu)&(oedyu > edyu)
            f2 = f2x & f2y
            s2 = np.abs(oedxr[f2]-oedxl[f2])*np.abs(edyu-oedyb[f2])
            w2 = np.sum(a[vy[f2],vx[f2]]*s2)
            W.append(w2)
            
            f3x = (oedxl <= edxl)&(oedxr > edxl)
            f3y = (oedyb <= edyu)&(oedyu > edyu)
            f3 = f3x & f3y
            s3 = np.abs(edyu-oedyb[f3])*np.abs(oedxr[f3]-edxl)
            w3 = np.sum(a[vy[f3],vx[f3]]*s3)
            W.append(w3)
            
            f4x = (oedxl <= edxl)&(oedxr > edxl)
            f4y = (oedyu <= edyu)&(oedyb > edyb)
            f4 = f4x & f4y
            s4 = np.abs(oedxr[f4]-edxl)*np.abs(oedyu[f4]-oedyb[f4])
            w4 = np.sum(a[vy[f4],vx[f4]]*s4)
            W.append(w4)
            
            f5x = (oedxl <= edxl)&(oedxr > edxl)
            f5y = (oedyu > edyb)&(oedyb <= edyb)
            f5 = f5x & f5y
            s5 = np.abs(oedxr[f5]-edxl)*np.abs(oedyu[f5]-edyb)
            w5 = np.sum(a[vy[f5],vx[f5]]*s5)
            W.append(w5)
            
            f6x = (oedxl > edxl)&(oedxr <= edxr)
            f6y = (oedyu > edyb)&(oedyb <= edyb)
            f6 = f6x & f6y
            s6 = np.abs(oedxr[f6]-oedxl[f6])*np.abs(oedyu[f6]-edyb)
            w6 = np.sum(a[vy[f6],vx[f6]]*s6)
            W.append(w6)
            
            f7x = (oedxl <= edxr)&(oedxr > edxr)
            f7y = (oedyu > edyb)&(oedyb <= edyb)
            f7 = f7x & f7y
            s7 = np.abs(edxr-oedxl[f7])*np.abs(oedyu[f7]-edyb)
            w7 = np.sum(a[vy[f7],vx[f7]]*s7)
            W.append(w7)
            
            f8x = (oedxr > edxr)&(oedxl <= edxr)
            f8y = (oedyu <= edyu)&(oedyb > edyb)
            f8 = f8x & f8y
            s8 = np.abs(oedyu[f8]-oedyb[f8])*np.abs(edxr-oedxl[f8])
            w8 = np.sum(a[vy[f8],vx[f8]]*s8)
            W.append(w8)
            
            f9x = (oedxr <= edxr)&(oedxl > edxl)
            f9y = (oedyu <= edyu)&(oedyb > edyb)
            f9 = f9x & f9y
            s9 = np.abs(oedxr[f9]-oedxl[f9])*np.abs(oedyu[f9]-oedyb[f9])
            w9 = np.sum(a[vy[f9],vx[f9]]*s9)
            W.append(w9)
            sample[p,k] = np.sum(W)
            W = []
    return sample, cpos

def down_samp(m3, m4, data, cx0, cy0):
    mx = m3
    my = m4
    a = data
    sNy = a.shape[0]
    sNx = a.shape[1]
    
    x0 = np.linspace(0,sNx-1,sNx)
    y0 = np.linspace(0,sNy-1,sNy)
    M0 = np.meshgrid(x0,y0)
    
    N1x = np.int(np.ceil((sNx*(1/mx))))
    N1y = np.int(np.ceil((sNy*(1/my))))
    Ny = N1y
    Nx = N1x
    # get the new center pixel
    cx = np.ceil(np.ceil(cx0)/mx)
    cy = np.ceil(np.ceil(cy0)/my)
    cpos = np.array([cx, cy])
    sample = np.zeros((Ny, Nx), dtype = np.float)
    for k in range(Ny):
        for l in range(Nx):
            W = 0
            s = 0
            cdx = (2*l+1)*mx/2
            cdy = (2*k+1)*my/2 # original pixel center
            edxl = l*mx
            edxr = (l+1)*mx # edges of x direction
            edyb = k*my 
            edyu = (k+1)*my # edges of y direction
            drx = (2*M0[0]+1)/2-cdx
            dry = (2*M0[1]+1)/2-cdy
            iax = np.abs(drx) < (1/2+mx/2)
            iay = np.abs(dry) < (1/2+my/2)
            ib = iax & iay
            iu = np.where(ib == True)
            vx = np.array(iu[1])
            vy = np.array(iu[0])
            oedxl = vx*1
            oedxr = (vx+1)*1
            oedyu = (vy+1)*1
            oedyb = vy*1

            f1x = (edxr >= oedxr)&(edxl <= oedxr)
            f1y = (edyu >= oedyu)&(edyb <= oedyu)
            f1 = f1x & f1y
            if (vy[f1] == a.shape[0]-1)&(vx[f1] == a.shape[1]-1):
                w1 = np.sum(f1)*(np.abs(oedxr[f1]-edxl)*np.abs(oedyu[f1]-edyb)*a[vy[f1],vx[f1]])              
            elif (vy[f1] == a.shape[0]-1)&(vx[f1] != a.shape[1]-1):
                w1 = np.sum(f1)*(np.abs(oedxr[f1]-edxl)*np.abs(oedyu[f1]-edyb)*a[vy[f1],vx[f1]] +
                        np.abs(oedyu[f1]-edyb)*np.abs(oedxr[f1]-edxr)*a[vy[f1],vx[f1]+1])
            elif (vy[f1] != a.shape[0]-1)&(vx[f1] == a.shape[1]-1):
                w1 = np.sum(f1)*(np.abs(oedxr[f1]-edxl)*np.abs(oedyu[f1]-edyb)*a[vy[f1],vx[f1]] +
                        np.abs(edyu-oedyu[f1])*np.abs(edxl-oedxr[f1])*a[vy[f1]+1,vx[f1]]) 
            else:
                w1 = np.sum(f1)*(np.abs(oedxr[f1]-edxl)*np.abs(oedyu[f1]-edyb)*a[vy[f1],vx[f1]] +
                        np.abs(edyu-oedyu[f1])*np.abs(edxl-oedxr[f1])*a[vy[f1]+1,vx[f1]] + 
                        np.abs(oedyu[f1]-edyb)*np.abs(oedxr[f1]-edxr)*a[vy[f1],vx[f1]+1] +
                        np.abs(edyu-oedyu[f1])*np.abs(edxr-oedxr[f1])*a[vy[f1]+1,vx[f1]+1])
            W = W + np.sum(w1)
            s = s + np.sum(f1)

            f2x = (edxl >= oedxl)&(edxr <= oedxr)
            f2y = (edyb <= oedyu)&(edyu >= oedyu)
            f2 = f2x & f2y
            if vy[f2] == a.shape[0]-1:
                w2 = np.sum(f2)*(np.abs(oedyu[f2]-edyb)*np.abs(edxr-edxl)*a[vy[f2],vx[f2]])              
            else:
                w2 = np.sum(f2)*(np.abs(oedyu[f2]-edyb)*np.abs(edxr-edxl)*a[vy[f2],vx[f2]] +
                        np.abs(edyu-oedyu[f2])*np.abs(edxr-edxl)*a[vy[f2]+1,vx[f2]])
            W = W + np.sum(w2)
            s = s + np.sum(f2)
            
            f3x = (edxl <= oedxl)&(edxr >= oedxl)
            f3y = (edyb <= oedyu)&(edyu >= oedyu)
            f3 = f3x & f3y
            if (vy[f3] == a.shape[0]-1) & (vx[f3] == 0):
                w3 = np.sum(f3)*(np.abs(oedxl[f3]-edxr)*np.abs(oedyu[f3]-edyb)*a[vy[f3],vx[f3]])                
            elif (vy[f3] == a.shape[0]-1) & (vx[f3] != 0):
                w3 = np.sum(f3)*(np.abs(oedxl[f3]-edxr)*np.abs(oedyu[f3]-edyb)*a[vy[f3],vx[f3]] +
                        np.abs(edyb-oedyu[f3])*np.abs(edxl-oedxl[f3])*a[vy[f3],vx[f3]-1])                
            elif (vy[f3] != a.shape[0]-1) & (vx[f3] == 0):
                w3 = np.sum(f3)*(np.abs(oedxl[f3]-edxr)*np.abs(oedyu[f3]-edyb)*a[vy[f3],vx[f3]] +
                        np.abs(edxr-oedxl[f3])*np.abs(edyu-oedyu[f3])*a[vy[f3]+1,vx[f3]])                
            else:
                w3 = np.sum(f3)*(np.abs(oedxl[f3]-edxr)*np.abs(oedyu[f3]-edyb)*a[vy[f3],vx[f3]] +
                        np.abs(edxr-oedxl[f3])*np.abs(edyu-oedyu[f3])*a[vy[f3]+1,vx[f3]] +
                        np.abs(edyu-oedyu[f3])*np.abs(edxl-oedxl[f3])*a[vy[f3]+1,vx[f3]-1] +
                        np.abs(edyb-oedyu[f3])*np.abs(edxl-oedxl[f3])*a[vy[f3],vx[f3]-1])
            W = W + np.sum(w3)
            s = s + np.sum(f3)
            
            f4x = (edxl <= oedxl)&(edxr >= oedxl)
            f4y = (edyu <= oedyu)&(edyb >= oedyb)
            f4 = f4x & f4y
            if vx[f4] == 0:
                w4 = np.sum(f4)*(np.abs(edxr-oedxl[f4])*np.abs(edyu-edyb)*a[vy[f4],vx[f4]])                
            else:
                w4 = np.sum(f4)*(np.abs(edxr-oedxl[f4])*np.abs(edyu-edyb)*a[vy[f4],vx[f4]] +
                        np.abs(edxl-oedxl[f4])*np.abs(edyu-edyb)*a[vy[f4],vx[f4]-1])
            W = W + np.sum(w4)
            s = s + np.sum(f4)
            
            f5x = (edxl <= oedxl)&(edxr >= oedxl)
            f5y = (edyu >= oedyb)&(edyb <= oedyb)
            f5 = f5x & f5y
            if (vy[f5] == 0) & (vx[f5] == 0):
                w5 = np.sum(f5)*(np.abs(edxr-oedxl[f5])*np.abs(edyu-oedyb[f5])*a[vy[f5],vx[f5]])   
            elif (vy[f5] != 0) & (vx[f5] == 0):
                w5 = np.sum(f5)*(np.abs(edxr-oedxl[f5])*np.abs(edyu-oedyb[f5])*a[vy[f5],vx[f5]] +
                        np.abs(edxr-oedxl[f5])*np.abs(edyb-oedyb[f5])*a[vy[f5]-1,vx[f5]])   
            elif (vy[f5] == 0) & (vx[f5] != 0):
                w5 = np.sum(f5)*(np.abs(edxr-oedxl[f5])*np.abs(edyu-oedyb[f5])*a[vy[f5],vx[f5]] +
                        np.abs(edxl-oedxl[f5])*np.abs(edyu-oedyb[f5])*a[vy[f5],vx[f5]-1])                
            else:
                w5 = np.sum(f5)*(np.abs(edxr-oedxl[f5])*np.abs(edyu-oedyb[f5])*a[vy[f5],vx[f5]] +
                        np.abs(edxl-oedxl[f5])*np.abs(edyu-oedyb[f5])*a[vy[f5],vx[f5]-1] +
                        np.abs(edxl-oedxl[f5])*np.abs(edyb-oedyb[f5])*a[vy[f5]-1,vx[f5]-1] +
                        np.abs(edxr-oedxl[f5])*np.abs(edyb-oedyb[f5])*a[vy[f5]-1,vx[f5]])
            W = W + np.sum(w5)
            s = s + np.sum(f5)
            
            f6x = (edxl >= oedxl)&(edxr <= oedxr)
            f6y = (edyu >= oedyb)&(edyb <= oedyb)
            f6 = f6x & f6y
            if vy[f6] == 0:
                w6 = np.sum(f6)*(np.abs(edxr-edxl)*np.abs(edyu-oedyb[f6])*a[vy[f6],vx[f6]])                
            else:
                w6 = np.sum(f6)*(np.abs(edxr-edxl)*np.abs(edyu-oedyb[f6])*a[vy[f6],vx[f6]] +
                            np.abs(edxr-edxl)*np.abs(edyb-oedyb[f6])*a[vy[f6]-1,vx[f6]])
            W = W + np.sum(w6)
            s = s + np.sum(f6)
            
            f7x = (edxl <= oedxr)&(edxr >= oedxr)
            f7y = (edyu >= oedyb)&(edyb <= oedyb)
            f7 = f7x & f7y
            if (vy[f7] == 0) & (vx[f7] == a.shape[1]-1):
                w7 = np.sum(f7)*(np.abs(edyu-oedyb[f7])*np.abs(edxl-oedxr[f7])*a[vy[f7],vx[f7]])                
            elif (vy[f7] != 0) & (vx[f7] == a.shape[1]-1):
                w7 = np.sum(f7)*(np.abs(edyu-oedyb[f7])*np.abs(edxl-oedxr[f7])*a[vy[f7],vx[f7]] +
                        np.abs(edyb-oedyb[f7])*np.abs(edxl-oedxr[f7])*a[vy[f7]-1,vx[f7]])                
            elif (vy[f7] == 0) & (vx[f7] != a.shape[1]-1):
                w7 = np.sum(f7)*(np.abs(edyu-oedyb[f7])*np.abs(edxl-oedxr[f7])*a[vy[f7],vx[f7]] +
                        np.abs(edyu-oedyb[f7])*np.abs(edxr-oedxr[f7])*a[vy[f7],vx[f7]+1])                
            else:
                w7 = np.sum(f7)*(np.abs(edyu-oedyb[f7])*np.abs(edxl-oedxr[f7])*a[vy[f7],vx[f7]] +
                        np.abs(edyu-oedyb[f7])*np.abs(edxr-oedxr[f7])*a[vy[f7],vx[f7]+1] +
                        np.abs(edyb-oedyb[f7])*np.abs(edxl-oedxr[f7])*a[vy[f7]-1,vx[f7]] +
                        np.abs(edxr-oedxr[f7])*np.abs(edyb-oedyb[f7])*a[vy[f7]-1,vx[f7]+1])
            W = W + np.sum(w7)
            s = s + np.sum(f7)

            f8x = (edxr >= oedxr)&(edxl <= oedxr)
            f8y = (edyu <= oedyu)&(edyb >= oedyb)
            f8 = f8x & f8y
            if vx[f8] == a.shape[1]-1:
                w8 = np.sum(f8)*(np.abs(edyu-edyb)*np.abs(edxl-oedxr[f8])*a[vy[f8],vx[f8]])                
            else:
                w8 = np.sum(f8)*(np.abs(edyu-edyb)*np.abs(edxl-oedxr[f8])*a[vy[f8],vx[f8]] +
                        np.abs(edxr-oedxr[f8])*np.abs(edyu-edyb)*a[vy[f8],vx[f8]+1])
            W = W + np.sum(w8)
            s = s + np.sum(f8)
            
            f9x = (edxr <= oedxr)&(edxl >= oedxl)
            f9y = (edyu <= oedyu)&(edyb >= oedyb)
            f9 = f9x & f9y
            w9 = np.sum(f9)*(np.abs(edxr-edxl)*np.abs(edyu-edyb)*a[vy[f9],vx[f9]])
            W = W + np.sum(w9)
            s = s + np.sum(f9)
            
            sample[k,l] = W/s

    return sample, cpos
'''
#resam = sum_samp(m1, m2, a, cx0, cy0)[0]
resam = down_samp(m3, m4, a, cx0, cy0)[0]
print('before=',a)
print('*'*10)
print('after=',resam)
'''