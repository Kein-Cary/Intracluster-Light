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
'''
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
#f = Lumi0*(1+z)**(1+alpha)/(4*np.pi*Angu_d**2*(1+z)**4)*(1/c0**2)
f = Lumi0/(4*np.pi*Angu_d**2*(1+z)**4)*(1/c0**2)
m_e = 22.5 - 2.5 * np.log10(f/f0) # apperant magnitude
## calculate the absolute magnitude
r = Angu_d*U.Mpc.to(U.pc)
M_e = m_e + 5- 5*np.log10((z+1)**2*r)
### flux change
plt.figure()
plt.plot(z,f,'r-',label = r'$f_{cen}$')
plt.legend(loc=1)
plt.xlabel(r'$z$')
plt.ylabel(r'$f_{cen} \ [erg s^{-1} m^{-2} Hz^{-1}]$')
plt.yscale('log')
plt.title(r'$f_{cen} \ z$')
plt.savefig('flux_test.png',dpi=600)
### absolute magnitude change
plt.figure()
plt.plot(z,M_e,'b-',label = r'$SB$')
plt.legend(loc=2)
plt.xlabel(r'$z$')
plt.ylabel(r'$absolute \ magnitude \ [mag/arcsec^2]$')
plt.title(r'$absolute \ magnitude \ z$')
plt.gca().invert_yaxis()
plt.savefig('M_e_test.png',dpi=600)
'''
#####################
## calculate SB profile, assumption: m as the central surface brightness
import nstep
r_e = 20  # effective radius in unit Kpc according Zibetti's work
f_e = 1
c0 = 1*U.Mpc.to(U.m) # change Mpc to pc, mutply this parameter
f0 = 3631*10**(-23)*10**4 # SDSS zero point, in unit: (erg/s)/(m^2*Hz)
z_e = np.linspace(0.2,0.3,101)
R_clust = 1. # in unit Mpc
A_a, D_a = mark_by_self(z_e,R_clust) 
r_c = D_a*U.Mpc.to(U.pc)
# luminosity should be consistant
L0 = nstep.n_step(8)*(np.exp(7.67)/7.67**8)*np.pi*f_e*r_e**2*(U.kpc.to(U.m))**2
# assumption : I(0) = 2000Ie
f_c = np.exp(7.669)*f_e
### from fc at z == 0.2,get all the fc and fe
fc = f_c*((1+z_e[0])**4*D_a[0]**2)/((1+z_e)**4*D_a**2)
fe = fc/np.exp(7.669)
### next calculate the profile
Nbins = np.int(101)
r_clust = np.linspace(0,1000,Nbins)
r_clust_A = np.zeros((Nbins,Nbins),dtype = np.float)
I = np.zeros((Nbins,Nbins),dtype = np.float)
M_clust = np.zeros((Nbins,Nbins),dtype = np.float)
f_clust = np.zeros((Nbins,Nbins),dtype = np.float)
for p in range(len(z_e)):
    r_index = (r_clust/r_e)**(1/4)
    f_pros = fe[p]*np.exp(-7.669*(r_index-1))
    f_clust[p,:] = f_pros
    I[p,:] = 22.5-2.5*np.log10(f_pros/f0) ## apparent magnitude
    M_clust[p,:] = I[p,:]+5-5*np.log10((1+z_e[p])**2*r_c[p]) ## absolute magnituded
    r_clust_A[p,:] = (((r_clust/1000)/D_a[p])*180/np.pi)*U.deg.to(U.arcsec)
### 
plt.plot(z_e,f_clust[:,0],label = 'center flux')
plt.xlabel('z')
plt.ylabel(r'$Center \ Flux[erg s^{-1} m^{-2} Hz^{-1}]$')
plt.title(r'$CF \ z$')
plt.savefig('center flux change.png',dpi = 600)
###
plt.plot(z_e,fe)
plt.xlabel('z')
plt.ylabel(r'$Effective \ Flux[erg s^{-1} m^{-2} Hz^{-1}]$')
plt.title(r'$f_eff \ z$')
plt.savefig('effect flux change.png',dpi=600)
### sb profile change
for q in range(len(z_e)):
    if q % 20 == 0:
        plt.plot(r_clust**(1/4),I[q,:],color = mpl.cm.rainbow(q/len(z_e)),
                 label = r'$z%.3f$'%z_e[q])
plt.legend(loc = 1)
plt.gca().invert_yaxis() # change the y-axis direction
plt.xlabel(r'$R^{\frac{1}{4}} \ [kpc]$')
plt.ylabel(r'$Surface \ brightness \ [mag/arcsec^2]$')
plt.title(r'$SB \ [R] \ as \ function \ of \ z$')
plt.savefig('SB_apparent_brightness.png',dpi=600)
### absolute magnitude
for q in range(len(z_e)):
    if q % 20 == 0:
        plt.plot(r_clust**(1/4),M_clust[q,:],color = mpl.cm.rainbow(q/len(z_e)),
                 label = r'$z%.3f$'%z_e[q],alpha = 0.5)
plt.legend(loc = 1)
plt.gca().invert_yaxis() ## invers the direction of axis
plt.ylabel(r'$M_{absolute \ magnitude} \ [mag/arcsec^2]$')
plt.xlabel(r'$R^{\frac{1}{4}} \ [arcsec]$')
plt.title(r'$SB[R] \ in \ absolute \ Magnitude \ as \ z \ function$')
plt.savefig('SB_absolute_magnitude.png',dpi=600)
### test the luminosity is constant or not
################
from scipy.integrate import quad as quad
def integ_L(f,r):
    F = f
    R = r # fe, Re are effective radius and flux respectively, R in unit kpc
    L = np.zeros(Nbins, dtype = np.float)
    Integ_L = lambda r: F*np.exp(-7.669*((r/R)**(1/4)-1))*2*np.pi*r
    L = quad(Integ_L,0,np.inf)[0]*(U.kpc.to(U.m))**2
    return L
###############
L = np.zeros(len(z_e),dtype = np.float)
for k in range(len(z_e)):
    L[k] = integ_L(fe[k],r_e)
plt.plot(z_e,L/L0)
plt.xlabel('z')
plt.ylabel(r'$\eta[L/L0]$')
plt.title(r'$\eta[L/L0] \ z$')
plt.savefig('Luminosity test.png',dpi = 600)
