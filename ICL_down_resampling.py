# the file shows down-resampling
"""
down resampling: the pixels scale become smaller
"""
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
'''
a = np.array([[2,4,3,7,5],[6,1,0,4,8],[7,2,7,5,6],[9,2,0,7,1],[4,6,9,3,4]])
cx0 = 2
cy0 = 2
m1 = 0.5 # coloumn scale (for row direction)
m2 = 0.5 # row scale (for coloumn direction)
'''
##########
## resampling by new "pixel" result (towards smaller pixel size)
def down_samp(m1, m2, data, cx0, cy0):
    a = data
    mx = m1
    my = m2
    sNy = a.shape[0]
    sNx = a.shape[1]
    N1x = np.int(np.ceil((sNx*(1/mx))))
    N1y = np.int(np.ceil((sNy*(1/my))))
    Ny = N1y
    Nx = N1x
    x1 = np.linspace(0,Nx-1,Nx)
    y1 = np.linspace(0,Ny-1,Ny)
    M1 = np.meshgrid(x1,y1)
    # get the new center pixel
    
    cx = np.ceil(np.ceil(cx0)/m1)
    cy = np.ceil(np.ceil(cy0)/m2)
    cpos = np.array([cx, cy])
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
            ia = np.abs(idx) < (m1/2 +1/2)
            ib = np.abs(idy) < (m2/2 +1/2)
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
                    continue
                elif (nedxr >= edxr) & (nedxl <= edxr) & (nedyb <= edyb) & (nedyu >= edyb):
                    if (k == 0) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l] 
                        continue
                    elif (k == 0) & (l != sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-edyb)*a[k,l+1]
                        continue
                    elif (k != 0) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(edxr-nedxl)*np.abs(edyb-nedyb)*a[k-1,l]
                        continue
                    else:
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-edyb)*a[k,l+1]+\
                                        np.abs(nedxl-edxr)*np.abs(edyb-nedyb)*a[k-1,l]+\
                                        np.abs(nedxr-edxr)*np.abs(edyb-nedyb)*a[k-1,l+1]
                        continue
                elif (nedxr <= edxr) & (nedxl >= edxl) & (nedyu >= edyb) & (nedyb <= edyb):
                    if k == 0:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]
                        continue
                    else:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-nedxl)*np.abs(edyb-nedyb)*a[k-1,l]
                        continue
                elif (nedxl <= edxl) & (nedxr >= edxl) & (nedyu >= edyb) & (nedyb <= edyb):
                    if (k == 0) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]
                        continue
                    elif (k != 0) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxl)*np.abs(edyb-nedyb)*a[k-1,l]
                        continue
                    elif (k == 0) & (l !=0 ):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(edxl-nedxl)*np.abs(nedyu-edyb)*a[k,l-1]
                        continue
                    else:
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(nedyu-edyb)*a[k,l]+\
                                        np.abs(nedxr-edxl)*np.abs(edyb-nedyb)*a[k-1,l]+\
                                        np.abs(edxl-nedxl)*np.abs(edyb-nedyb)*a[k-1,l-1]+\
                                        np.abs(edxl-nedxl)*np.abs(nedyu-edyb)*a[k,l-1]
                        continue
                elif (nedxl <= edxl) & (nedxr >= edxl) & (nedyu <= edyu) & (nedyb >= edyb):
                    if l == 0:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(nedxr-edxl)*a[k,l]
                        continue
                    else:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(nedxr-edxl)*a[k,l]+\
                                        np.abs(nedyu-nedyb)*np.abs(edxl-nedxl)*a[k,l-1]
                        continue
                elif (nedyu >= edyu) & (nedyb <= edyu) & (nedxl <= edxl) & (nedxr >= edxl):
                    if (k == sNy-1) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]
                        continue
                    elif (k == sNy-1) & (l != 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxl-edxl)*np.abs(edyu-nedyb)*a[k,l-1]
                        continue
                    elif (k != sNy-1) & (l == 0):
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedyu-edyu)*np.abs(nedxr-edxl)*a[k+1,l]
                        continue
                    else:
                        sample[p,s] = np.abs(nedxr-edxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxl-edxl)*np.abs(edyu-nedyb)*a[k,l-1]+\
                                        np.abs(nedxl-edxl)*np.abs(nedyu-edyu)*a[k+1,l-1]+\
                                        np.abs(nedxr-edxl)*np.abs(nedyu-edyu)*a[k+1,l]
                        continue
                elif (nedxl >= edxl) & (nedxr <= edxr) & (nedyu >= edyu) & (nedyb <= edyu):
                    if k == sNy-1:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]
                        continue
                    else:
                        sample[p,s] = np.abs(nedxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxr-nedxl)*np.abs(nedyu-edyu)*a[k+1,l]
                        continue
                elif (nedxl <= edxr) & (nedxr >= edxr) & (nedyu >= edyu) & (nedyb <= edyu):
                    if (k == sNy-1) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]
                        continue
                    elif (k == sNy-1) & (l != sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(edyu-nedyb)*a[k,l+1]
                        continue
                    elif (k != sNy-1) & (l == sNx-1):
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(edxr-nedxl)*np.abs(nedyu-edyu)*a[k+1,l]
                        continue
                    else:
                        sample[p,s] = np.abs(edxr-nedxl)*np.abs(edyu-nedyb)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(edyu-nedyb)*a[k,l+1]+\
                                        np.abs(edxr-nedxl)*np.abs(edyu-nedyu)*a[k+1,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-edyu)*a[k+1,l+1]
                        continue
                elif (nedxl <= edxr) & (nedxr >= edxr) & (nedyu <= edyu) & (nedyb >= edyb):
                    if l == sNx-1:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(edxr-nedxl)*a[k,l]
                        continue
                    else:
                        sample[p,s] = np.abs(nedyu-nedyb)*np.abs(edxr-nedxl)*a[k,l]+\
                                        np.abs(nedxr-edxr)*np.abs(nedyu-nedyb)*a[k,l+1]
                        continue
                else:
                    pass
                    continue
    return sample, cpos
'''
resam = down_samp(m1, m2, a, cx0, cy0)[0]
print('before=',a)
print('*'*10)
print('after=',resam)
'''