# this file used to calculation the cluster properties change with red-shift
## properties includes: luminosity, colors, angular size et al.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import scipy.stats as sts
# setlect model
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
### asummption: the L (luminosity), f (flux), As (angular size) at z~0
### asummption : all the cluster in size 1Mpc/h
z = np.linspace(1e-6,1,1000)
c0 = 1*U.Mpc.to(U.m)
alpha = 2. # for thermal radiation, the spectral index is about 2
f_v = 3.631*10**-32 # assume a constant flux : 1 nanomaggies
A_s, D_A = mark_by_plank(z,1.)
D_l = (1+z)**2*D_A
Lumi = f_v*4*np.pi*D_l**2/((1+z)**(1+alpha))
Lumi = Lumi*c0**2
V_size = A_s*U.rad.to(U.arcsec)
AS = A_s*U.rad.to(U.arcsec)
m = 20. # reference the zero point of SDSS 
SB = m + 2.5*np.log10(AS**2*np.pi)

plt.plot(z,Lumi,label = r'$L$')
plt.legend(loc = 4)
plt.yscale('log')
plt.xlabel(r'$z$')
plt.ylabel(r'$Luminosity-[W/Hz]$')
plt.title(r'$Luminosity-z$')

plt.plot(z,SB,label = r'$SB$')
plt.legend(loc = 1)
plt.xlabel(r'$z$')
plt.ylabel(r'$Surface-Brightness-[mag/arcsec^2]$')
plt.title(r'$SB-z$')

# calculate the angular size change from z1--z2
### A1*D_A1 = A2*D_A2, and then get 