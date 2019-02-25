# test resample
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

# assume pixel become smaller, and use interp2d to get the new sample
dx1 = 0.5
dy1 = 0.5
N1x = np.int(N0x*(1/dx1))
N1y = np.int(N0y*(1/dy1))
x1 = np.linspace(0, 4, N1x)
y1 = np.linspace(0, 4, N1y)
N1f = f(x1, y1)
plt.imshow(N1f, cmap = 'Greys', origin = 'lower') #(smaller)
plt.show()

dx2 = 1.5
dy2 = 1.5
N2x = np.int(N0x*(1/dx2))
N2y = np.int(N0y*(1/dy2))
x2 = np.linspace(0,4,N2x)
y2 = np.linspace(0,4,N2y)
N2f = f(x2, y2)
plt.imshow(N2f, cmap = 'Greys', origin = 'lower') #(bigger)
plt.show()
##########
## resampling by new "pixel" result (towards smaller pixel size)
m1 = 0.5 # coloumn scale (for row direction)
m2 = 0.5 # row scale (for coloumn direction)
resam = np.zeros((N1y, N1x), dtype = np.float)

def resam_edge_row(q, m1, m2, N, ud):
    m1 = m1*1
    m2 = m2*1
    N = N*1
    ud = ud*1
    rN = a.shape[1]
    sample = np.zeros(N, dtype = np.float)
    for k in range(N):
        for l in range(1,rN+1):
            if ((2*k+1)*m1/2-(l-1))*((2*k+1)*m1/2-l) <= 0 :
                td1 = np.abs((2*k+1)*m1/2-(l-1))
                td2 = np.abs((2*k+1)*m1/2-l)
                if (td1 >= m1/2) & (td2 >= m1/2):
                    sample[k] = m1*m2*ud[l-1]
                elif (td1 >= m1/2) & (td2 < m1/2):
                    p0 = (2*k+1)*m1/2-l
                    if p0 >= 0 :
                        sample[k] = ud[l-1]*m2*(m1/2-np.abs(p0))+ ud[l]*(m2*(m1/2+np.abs(p0)))
                    else:
                        sample[k] = ud[l-1]*m2*(m1/2+np.abs(p0))+ ud[l]*(m2*(m1/2-np.abs(p0)))
                else:
                    pass
                if l == rN :   # handle the edge pixels
                    if k == N-1 :
                        if ((2*k+1)*m1/2-rN) >= (m1/2):
                            sample[k] = 0
                        else:
                            sample[k] = ud[-1]*m2*(m1/2+(rN-(2*k+1)*m1/2))
                    else:
                        td1 = np.abs((2*k+1)*m1/2-(rN-1))
                        td2 = np.abs((2*k+1)*m1/2-rN)   
                        if (td1 >= m1/2) & (td2 >= m1/2):
                            sample[k] = m1*m2*ud[-1]
                        else:
                            pass  # end edge pixels handle 
            else:
                pass
    return sample

def resam_unedge(m1, m2, N1, N2):
    ud = a*1
    m1 = m1*1
    m2 = m2*1
    N1 = N1*1
    N2 = N2*1
    rNx = a.shape[1]
    rNy = a.shape[0]
    sample = np.zeros((N2, N1), dtype = np.float)
    for k in range(N2):
        for l in range(1, rNy+1):
            if ((2*k+1)*m2/2-(l-1))*((2*k+1)*m2/2-l) <= 0 :
                if l == rNy: # handle for the bottom line
                    td = rNy-2*(k+1)*m2/2
                    ndy = td*1
                    if td <= -1*m2/2:
                        sample[k,:] = 0
                    else:
                        for q in range(N1):
                            for s in range(1, rNx+1):
                                if ((2*q+1)*m1/2-(s-1))*((2*q+1)*m1/2-s) <= 0 :
                                    td3 = np.abs((2*q+1)*m1/2-(s-1))
                                    td4 = np.abs((2*q+1)*m1/2-s)
                                    ndx = (2*q+1)*m1/2-s
                                    if (s == rNx) & (q == N1-1) :
                                            if (2*q+1)*m1/2-rNx >= m1/2 :
                                                sample[k,q] = 0
                                            else:
                                                if td >= m2/2:
                                                    sample[k,q] = m2*(m1/2-ndx)*ud[-1,-1]
                                                else:
                                                    sample[k,q] = (m1/2-ndx)*(m2/2+ndy)*ud[-1,-1]
                                    elif (s == rNx) & (q != N1-1) :
                                        if (td > -1*m2/2)&(td <= m2/2) : 
                                            sample[k,q] = m1*(m2/2+ndy)*ud[-1,-1]
                                        else:
                                            sample[k,q] = m1*m2*ud[-1, -1]
                                    elif (s != rNx) & (q != N1-1):
                                        if (td3 >= m1/2) & (td4 >= m1/2) & (td >= m2/2) :
                                            sample[k,q] = m1*m2*ud[-1, s-1]
                                        elif (td3 >= m1/2) & (td4 < m1/2) & (td >= m2/2) :
                                            sample[k,q] = m2*(m1/2 -ndx)*ud[-1,s-1]+\
                                                            m2*(m1/2 +ndx)*ud[-1,s]
                                        elif (td3 >= m1/2) & (td4 >= m1/2) & (td < m2/2) :
                                            sample[k,q] = m1*(m2/2+ndy)*ud[-1,s-1]
                                        elif (td3 >= m1/2) & (td4 < m1/2) & (td < m2/2) :
                                            sample[k,q] = (m1/2+ndx)*(m2/2+ndy)*ud[-1,s-1]+\
                                                            (m1/2-ndx)*(m2/2+ndy)*ud[-1,s]
                                        else:
                                            pass
                                else:
                                    pass  # end handle the bottom line
                else:
                    td1 = np.abs((2*k+1)*m2/2-(l-1))
                    td2 = np.abs((2*k+1)*m2/2-l)
                    ndy = (2*k+1)*m2/2-l
                    for q in range(N1):
                        for s in range(1, rNx+1):
                            if ((2*q+1)*m1/2-(s-1))*((2*q+1)*m1/2-s) <= 0 :  
                                if (s == rNx) & (q == N1-1) :  # handle for the right-edge pixels
                                    ndx = (2*q+1)*m1/2-rNx
                                    if ndx >= m1/2:
                                        sample[k,q] = 0
                                    else:
                                        if (td1 >= m2/2) & (td2 >= m2/2):
                                            sample[k,q] = m2*(m1/2-ndx)*ud[l,-1]
                                        elif (td1 >= m2/2) & (td2 < m2/2):
                                            sample[k,q] = (m2/2-ndy)*(m1/2-ndx)*ud[l-1,-1]+\
                                                            (m2/2+ndy)*(m1/2-ndx)*ud[l,-1]
                                        else:
                                            pass
                                elif (s == rNx) & (q != N1-1):
                                    td3 = np.abs((2*q+1)*m1/2-(rNx-1))
                                    td4 = np.abs((2*q+1)*m1/2-rNx)
                                    ndy = (2*k+1)*m2/2-l
                                    if (td1 >= m2/2)&(td2 >= m2/2)&(td3 >= m1/2)&(td4 >= m1/2):
                                        sample[k,q] = m2*m1*ud[l-1,-1]
                                    elif (td1 >= m2/2)&(td2 < m2/2)&(td3 >= m1/2)&(td4 >= m1/2):
                                        sample[k,q] = m1*(m2/2-ndy)*ud[l-1,-1]+m1*(m2/2+ndy)*ud[l,-1]
                                elif (s != rNx) & (q != N1-1):
                                    td3 = np.abs((2*q+1)*m1/2-(s-1))
                                    td4 = np.abs((2*q+1)*m1/2-s)
                                    if (td1 >= m2/2)&(td2 >= m2/2)&(td3 >= m1/2)&(td4 >= m1/2):
                                        sample[k,q] = m2*m1*ud[l-1,s-1]
                                    elif (td1 >= m2/2)&(td2 < m2/2)&(td3 >= m1/2)&(td4 >= m1/2):
                                        p0 = (2*k+1)*m2/2-l
                                        if p0 >= 0:
                                            sample[k,q] = m1*(m2/2-np.abs(p0))*ud[l-1,s-1]+\
                                            m1*(m2/2+np.abs(p0))*ud[l,s-1]
                                        else:
                                            sample[k,q] = m1*(m2/2+np.abs(p0))*ud[l-1,s-1]+\
                                            m1*(m2/2-np.abs(p0))*ud[l,s-1]  
                                    elif (td1 >= m2/2)&(td2 >= m2/2)&(td3 >= m1/2)&(td4 < m1/2):
                                        p1 = (2*q+1)*m2/2-s
                                        if p1 >= 0:
                                            sample[k,q] = m2*(m1/2-np.abs(p1))*ud[l-1,s-1]+\
                                            m2*(m1/2+np.abs(p1))*ud[l-1,s]
                                        else:
                                            sample[k,q] = m2*(m1/2-np.abs(p1))*ud[l-1,s-1]+\
                                            m2*(m1/2+np.abs(p1))*ud[l-1,s]
                                    elif (td1 >= m2/2)&(td2 < m2/2)&(td3 >= m1/2)&(td4 < m1/2):
                                        p2 = (2*k+1)*m2/2-l
                                        p3 = (2*q+1)*m2/2-s
                                        if (p2 >= 0)&(p3 >= 0):
                                            sample[k,q] = (m1/2-np.abs(p3))*(m2/2-np.abs(p2))*ud[l-1,s-1]+\
                                                            (m1/2+np.abs(p3))*(m2/2-np.abs(p2))*ud[l-1,s]+\
                                                            (m1/2-np.abs(p3))*(m2/2+np.abs(p2))*ud[l,s-1]+\
                                                            (m1/2+np.abs(p3))*(m2/2+np.abs(p2))*ud[l,s]
                                        elif (p2 >= 0)&(p3 < 0):
                                            sample[k,q] = (m1/2+np.abs(p3))*(m2/2-np.abs(p2))*ud[l-1,s-1]+\
                                                            (m1/2-np.abs(p3))*(m2/2-np.abs(p2))*ud[l-1,s]+\
                                                            (m1/2+np.abs(p3))*(m2/2+np.abs(p2))*ud[l,s-1]+\
                                                            (m1/2-np.abs(p3))*(m2/2+np.abs(p2))*ud[l,s]
                                        elif (p2 < 0)&(p3 >= 0):
                                            sample[k,q] = (m1/2-np.abs(p3))*(m2/2+np.abs(p2))*ud[l-1,s-1]+\
                                                            (m1/2+np.abs(p3))*(m2/2+np.abs(p2))*ud[l-1,s]+\
                                                            (m1/2-np.abs(p3))*(m2/2-np.abs(p2))*ud[l,s-1]+\
                                                            (m1/2+np.abs(p3))*(m2/2-np.abs(p2))*ud[l,s]     
                                        else:
                                            sample[k,q] = (m1/2+np.abs(p3))*(m2/2+np.abs(p2))*ud[l-1,s-1]+\
                                                            (m1/2-np.abs(p3))*(m2/2+np.abs(p2))*ud[l-1,s]+\
                                                            (m1/2+np.abs(p3))*(m2/2-np.abs(p2))*ud[l,s-1]+\
                                                            (m1/2-np.abs(p3))*(m2/2-np.abs(p2))*ud[l,s]
                                    else:
                                        pass  # the oppsite direction pixels have no handle 
                                else:
                                    pass  # the last condition for s != rNx & q == N1-1 have no mean! 
                            else:
                                pass   # location which point handle for each time, others points have no mean
            else:
                pass  # location which line handle for each time, others lines have no mean
    
    return sample
'''
# test 
for q1 in range(N1x):
    for q2 in range(N1y):
        if q1 == 0 :
            ud = a[q1,:]
            resam[q1,:] = resam_edge_row(q1, m1, m2, N1x, ud)
        elif q1 == N1x-1 :
            ud = a[-1,:]
            resam[q1,:] = resam_edge_row(q1, m1, m2, N1x, ud)           
        elif q2 == 0 :
            ud = a[:,q2].T
            resam[:,q2] = resam_edge_row(q1, m1, m2, N1y, ud).T
        elif q2 == N1y-1:
            ud = a[:,-1].T
            resam[:,q2] = resam_edge_row(q1, m1, m2, N1y, ud).T
        else:
            pass
'''
resam = resam_unedge(m1, m2, N1x, N1y)

##########          
## resampling by new "pixel" result (towards bigger pixel size)
