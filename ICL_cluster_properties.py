# this file used to calculation the cluster properties change with red-shift
## properties includes: luminosity, colors, angular size et al.
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as aft
import scipy.stats as sts
from scipy.integrate import quad as quad
from astropy.wcs import *
import astropy.wcs as awc
from ICL_angular_diameter_reshift import mark_by_self_Noh
from astropy import cosmology as apcy
##########################################
###### based on NFW profile:
import ICL_surface_mass_density as iy
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
d0 = U.rad.to(U.deg)*3600
Lsun = C.L_sun.value*10**7 # sun's luminosity in unit :erg/s
Mc = 15
Nbins = 101
c = 5
r200 = iy.sigma_m_c(Mc,Nbins)[0]
rs = r200/c
sigma = iy.sigma_m_c(Mc,Nbins)[1] # get the surface density in comoving coordinate(based on Plank 15)
rr = iy.sigma_m_c(Mc,Nbins)[2]
zn = np.linspace(0.2,0.3,Nbins)
An_a, Dn_a = mark_by_self_Noh(zn,r200/1000)
a = 1/(zn+1)
Dn_l = Dn_a*(1+zn)**2
a4 = a**4  # the fourth order of scale factor
Lc = sigma/50  # the surface luminosity in unit L_sun / kpc^-2
Ln = np.tile(a4,(Nbins,1)).T*np.tile(Lc,(Nbins,1))/(4*np.pi*d0**2) 
### 
plt.plot(rr, sigma, label = 'surface density')
plt.axvline(x=rs,c = 'r',ls = '--',label = '$r_s$')
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.yscale('log')
plt.ylabel('$\Sigma[M_\odot \ kpc^{-2}]$')
plt.savefig('surface_mass_density.png',dpi=600)
###
plt.plot(rr, Lc, label = 'surface luminosity density')
plt.axvline(x=rs,c = 'r',ls = '--',label = '$r_s$')
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.yscale('log')
plt.ylabel(r'$ \frac{ \Sigma}{50} [ L_{ \odot} kpc^{-2}]$')
plt.savefig('comv_surface_luminosity_density.png',dpi=600)
###
for k in range(len(zn)):
    if k%10 == 1:
        plt.plot(rr,Ln[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k])
plt.axvline(x=rs,c = 'r',ls = '--',label = '$r_s$')
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.yscale('log')
plt.ylabel('$\Sigma[L_\odot \ kpc^{-2}]$')
plt.title('$SB \ as \ z \ function$')
plt.savefig('surface_brightness.png',dpi=600)

### next, calculate the magnitude
#fn = np.tile(Lc,(Ln.shape[0],1))*Lsun/(4*np.pi*np.tile(Dn_l**2*c2**2,(Ln.shape[0],1)).T)
fn = Ln*Lsun/c0**2
m = 22.5 - 2.5*np.log10(fn/(10**(-9)*f0))

SB1 = m + 2.5*np.log10(1.)
M = m + 5 -5*np.log10(np.tile(Dn_l*c1,(Ln.shape[0],1)).T)

alpha = np.tile(rr*d0/1000,(Ln.shape[0],1))/np.tile(Dn_a,(Ln.shape[0],1)).T
Rs = ((rs/1000)/Dn_a)*d0

### lin with magnitude of sun (from Lc)
SB_s2 = 21.572 - 2.5*np.log10(np.tile(Lc,(Ln.shape[0],1))*10**(-6))+\
10*np.log10(np.tile(zn,(Ln.shape[0],1)).T+1)

for k in range(len(zn)):
    if k%10 == 1:
        plt.plot(rr,m[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k])
        plt.axvline(x=rs,color = mpl.cm.rainbow(k/Nbins),ls = '--',alpha = 0.5)
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.ylabel('$ SB[mag]$')
plt.gca().invert_yaxis() # change the y-axis direction
plt.title('$ SB_{app} \ as \ z \ function$')
plt.savefig('surface_brightness_apparent.png',dpi=600)
### SB in unit: mag/arcsec^2
for k in range(len(zn)):
    if k%10 == 1:
        plt.plot(rr,SB1[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k])
        plt.axvline(x=rs,color = mpl.cm.rainbow(k/Nbins),ls = '--',alpha = 0.5)
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.ylabel('$ SB[mag \ arcsec^{-2}]$')
plt.gca().invert_yaxis() # change the y-axis direction
plt.title('$ SB_{app} \ as \ z \ function$')
plt.savefig('surface_brightness_apparent_arcsec.png',dpi=600)

### absolute magnitude
for k in range(len(zn)):
    if k%10 == 1:
        plt.plot(rr,M[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k], alpha = 0.5)
plt.axvline(x=rs,c = 'r',ls = '--',label = '$r_s$')
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.ylabel('$ SB [mag]$')
plt.gca().invert_yaxis() # change the y-axis direction
plt.title('$ SB_{abs} \ as \ z \ function$')
plt.savefig('surface_luminosity_absolute.png',dpi=600)

### change x-axis as arcsec
for k in range(len(zn)):
    if k%10 == 1:
        plt.plot(alpha[k,:],SB1[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k])
        plt.axvline(x=Rs[k],color = mpl.cm.rainbow(k/Nbins),ls = '--',
                    alpha = 0.35)
plt.legend(loc = 3) 
plt.xscale('log')
plt.xlabel('$R[arcsec]$')
plt.ylabel('$ SB[mag \ arcsec^{-2}]$')
plt.gca().invert_yaxis() # change the y-axis direction
plt.title('$ SB_{app} \ as \ z \ function$')
plt.savefig('surface_brightness_apparent_arcsec2.png',dpi=600)

for k in range(len(zn)):
    if k%10 == 1:
        plt.plot(alpha[k,:],SB_s2[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k])
        plt.axvline(x=Rs[k],color = mpl.cm.rainbow(k/Nbins),ls = '--',
                    alpha = 0.35)
plt.legend(loc = 3) 
plt.xscale('log')
plt.xlabel('$R[arcsec]$')
plt.ylabel('$ SB - M_\odot[mag \ arcsec^{-2}]$')
plt.gca().invert_yaxis() # change the y-axis direction
plt.title('$ SB_{app} \ as \ z \ function$')
plt.savefig('surface_brightness_apparent_with_sun2.png',dpi=600)
