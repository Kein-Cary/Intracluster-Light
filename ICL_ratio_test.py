# this file use to test the area ratio calculation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import scipy.stats as sts
import handy.scatter as hsc
# astronomy model
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
cir_data = aft.getdata('area_test.fits',header = True)
file = get_pkg_data_filename('area_test.fits')
hdu = aft.open(file)[0]
wcs = WCS(hdu.header)
## change the coordinate
y = np.linspace(0,1488,1489)
x = np.linspace(0,2047,2048)
vx, vy = np.meshgrid(x,y)
cx, cy = wcs.all_pix2world(vx,vy,1)
t1, t2 = wcs.all_world2pix(Ra*U.deg,Dec*U.deg,1)
## test 1: assume the cluster center is the frame center
Ra = cir_data[1]['CRVAL1']
Dec = cir_data[1]['CRVAL2']
R = np.linspace(50,1025,61)
S0 = R**2*np.pi/(0.396**2)
eta = np.zeros(len(R),dtype = np.float)
### 
for k in range(len(R)):
    dep = np.sqrt((vx-t1)**2+(vy-t2)**2)
    ad = R[k]/0.396
    ig = dep <= ad
    al = dep[ig]
    npl = len(al)
    eta[k] = npl/S0[k] 
###
plt.imshow(cir_data[0],cmap='Greys',vmin=1e-5,origin = 'lower',norm = mpl.colors.LogNorm()) 
hsc.circles(t1,t2,s = R/0.396,fc = '',ec = 'r')
plt.show()
plt.plot(R,eta,'r-')
plt.axvline(x = 745*0.396,ymin = 0,ymax = 1,c = 'b',ls = '--')
plt.axvline(x = np.sqrt(t1**2+t2**2)*0.396,ymin=0,ymax=1,c='k',ls='--')
plt.axhline(y = 1,xmin = 0,xmax = 745*0.396,c = 'g',ls = '--')
plt.xlabel(r'$R-arcsec$')
plt.ylabel(r'$\eta-[S/S0]$')
plt.title(r'$center-closed-test$')
plt.show()
## test 2: assume the cluster center is close to the angle of the frame (the left bottom corner)
t3 = 0; t4 = 0; 
R1 = np.linspace(50,1200,121)
S1 = R1**2*np.pi/(0.396**2)
eta1 = np.zeros(len(R1),dtype = np.float)
###
for k in range(len(R1)):
    dep1 = np.sqrt((vx-t3)**2+(vy-t4)**2)
    ad1 = R1[k]/0.396
    ig1 = dep1 <= ad1
    al1 = dep1[ig1]
    npl1 = len(al1)
    eta1[k] = npl1/S1[k] 
###
plt.imshow(cir_data[0],cmap='Greys',vmin=1e-5,origin = 'lower',norm = mpl.colors.LogNorm()) 
hsc.circles(t3,t4,s = R1/0.396,fc = '',ec = 'r')
plt.show()
plt.plot(R1,eta1,'r-')
plt.axvline(x = 1488*0.396, ymin = 0, ymax = 1,c = 'b',ls = '--')
plt.axvline(x = np.sqrt(1489**2+2048**2)*0.396, ymin=0, ymax= 1,c='k',ls='--')
plt.axhline(y = 0.25, xmin = 0, xmax = 1488*0.396,c = 'g',ls = '--')
plt.xlabel(r'$R-arcsec$')
plt.ylabel(r'$\eta-[S/S0]$')
plt.title(r'$corner-closed-test$')
plt.show()
## test 3: assume the cluster center is close to one edge of the frame
t5 = 1025;t6 = 0;
R2 = np.linspace(100,1300,121)
S2 = R2**2*np.pi/(0.396**2)
eta2 = np.zeros(len(R2),dtype = np.float)
###
for k in range(len(R2)):
    dep = np.sqrt((vx-t5)**2+(vy-t6)**2)
    ad = R2[k]/0.396
    ig = dep <= ad
    al = dep1[ig]
    npl = len(al)
    eta2[k] = npl/S2[k] 
### 
plt.imshow(cir_data[0],cmap='Greys',vmin=1e-5,origin = 'lower',norm = mpl.colors.LogNorm()) 
hsc.circles(t5,t6,s = R2/0.396,fc = '',ec = 'r')
plt.show()
plt.plot(R2,eta2,'r-')
plt.axvline(x = 1025*0.396, ymin = 0, ymax = 1,c = 'b',ls = '--')
plt.axvline(x = np.sqrt(1489**2+1025**2)*0.396, ymin=0, ymax= 1,c='k',ls='--')
plt.axhline(y = 0.5, xmin = 0, xmax = 1025*0.396,c = 'g',ls = '--')
plt.xlabel(r'$R-arcsec$')
plt.ylabel(r'$\eta-[S/S0]$')
plt.title(r'$edge-closed-test$')
plt.show()
