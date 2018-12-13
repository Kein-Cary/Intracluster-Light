# this file used to calculation the cluster properties change with red-shift
## properties includes: luminosity, colors, angular size et al.
import matplotlib as mpl
#mpl.use('Agg')
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
### asummption : all the cluster in size 1Mpc/h
R0 = 1.
c0 = 1*U.Mpc.to(U.m)
alpha = 2. # for thermal radiation, the spectral index is about 2
f0 = 3631*10**(-23)*10**4 # SDSS zero point, in unit: (erg/s)/(m^2*Hz)
### the initial redshift and the total luminosity take the Coma cluster value
z0 = 0.023
Lumi0 = 2.6*10**44 ## x-ray luminosity of Coma cluster,in unit : (erg/s)/Hz
z = np.linspace(z0,1,1000)
# calculate the Angular size and Angular distance of cluster
Angu_s, Angu_d = mark_by_self(z,R0)
# calculate the flux in different redshift, in unit : (erg/s)/m^2
f = Lumi0*(1+z)**(1+alpha)/(4*np.pi*Angu_d**2*(1+z)**4)*(1/c0**2)
m = 22.5 - 2.5 * np.log10(f/f0) # apperant magnitude
Angu_r = Angu_s*U.rad.to(U.arcsec)
SB = m + 2.5*np.log10(Angu_r**2*np.pi) # in unit mag/arcsec^2

plt.figure()
plt.plot(z,f,'r-',label = r'$f_{physics}$')
plt.legend(loc=1)
plt.xlabel(r'$z$')
plt.ylabel(r'$f_{physics}-[erg s^{-1} m^{-2} Hz^{-1}]$')
plt.yscale('log')
plt.title(r'$flux-z$')
plt.savefig('flux_test.png',dpi=600)

plt.figure()
plt.plot(z,SB,'b-',label = r'$SB$')
plt.legend(loc=2)
plt.xlabel(r'$z$')
plt.ylabel(r'$Surface-Brightness-[mag/arcsec^2]$')
plt.title(r'$SB-z$')
plt.savefig('SB_test.png',dpi=600)
# calculate the angular size change from z1--z2
### A1*D_A1 = A2*D_A2, and then get 
"""
calculate the dimming,since f_20,ref is arbitrary, set f_20,ref = f_20
f_20 in frame calibration is f_20 = N/(10**8*(f/f0)); f,f0 is flux.
in this case, c = f_20, and c' = c * (f_20/f_20)*10**0.4*A_lambda*((1+z)/(1+z_ref))**4
"""
A_l = 2.751   # A_lambda set the r band value
z_ref = z0 # set the initial redshift as reference redshift
f_20 = 1e-3
Cunt_b = (f/f0)*10**8*f_20 # conut before rescaling
Cunt_a = Cunt_b*10**(0.4*A_l)*((1+z)/(1+z_ref))**4 # count after rescale
plt.plot(z,Cunt_a,label = r'$counts$')
plt.legend(loc = 1)
plt.xlabel(r'$z$')
plt.ylabel(r'$Counts/pixel$')
plt.yscale('log')
