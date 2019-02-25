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

# assume pixel become smaller, and use interp2d to get the new sample
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
## resampling by new "pixel" result (towards bigger pixel size)
m1 = 1.5
m2 = 1.5
resam_2 = np.zeros((N2y, N2x), dtype = np.float)

def sum_samp(N1, N2, m1, m2):
    Nx = N1
    Ny = N2
    sample = np.zeros((N2, N1), dtype = np.float)
    
    return sample
# test part 
resam = sum_samp(N2x, N2y, m1, m2)