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
from scipy.integrate import quad as quad
# setlect model
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
from ICL_angular_diameter_reshift import mark_by_self_Noh
from astropy import cosmology as apcy
'''
############# from flux
## calculate SB profile, assumption: m as the central surface brightness
import nstep
r_e = 20  # effective radius in unit Kpc according Zibetti's work
f_e = 1
c0 = 1*U.Mpc.to(U.m) # change Mpc to pc, mutply this parameter
f0 = 3631*10**(-23)*10**4 # SDSS zero point, in unit: (erg/s)/(m^2*Hz)
z_e = np.linspace(0.2,0.3,101)
R_clust = 1. # in unit Mpc
A_a, D_a = mark_by_self_Noh(z_e,R_clust) 
r_c = D_a*U.Mpc.to(U.pc)
# luminosity should be consistant
L0 = nstep.n_step(8)*(np.exp(7.67)/7.67**8)*np.pi*f_e*r_e**2*(U.kpc.to(U.m))**2*(1+z_e[0])**4*D_a[0]**2
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
plt.title(r'$SB \ as \ function \ of \ z$')
plt.savefig('SB_apparent_brightness.png',dpi=600)
### absolute magnitude
for q in range(len(z_e)):
    if q % 20 == 0:
        plt.plot(r_clust**(1/4),M_clust[q,:],color = mpl.cm.rainbow(q/len(z_e)),
                 label = r'$z%.3f$'%z_e[q],alpha = 0.5)
plt.legend(loc = 1)
plt.gca().invert_yaxis() ## invers the direction of axis
plt.ylabel(r'$M_{absolute \ magnitude} \ [mag/arcsec^2]$')
plt.xlabel(r'$R^{\frac{1}{4}} \ [kpc]$')
plt.title(r'$SB \ in \ absolute \ Magnitude \ as \ z \ function$')
plt.savefig('SB_absolute_magnitude.png',dpi=600)
### test the luminosity is constant or not
################
def integ_L1(f,r,z,d):
    Z = z
    D = d
    F = f
    R = r # fe, Re are effective radius and flux respectively, R in unit kpc
    L = np.zeros(Nbins, dtype = np.float)
    Integ_L = lambda r: F*np.exp(-7.669*((r/R)**(1/4)-1))*2*np.pi*r
    L = quad(Integ_L,0,np.inf)[0]*(U.kpc.to(U.m))**2*((1+Z)**4*D**2)
    return L
###############
L = np.zeros(len(z_e),dtype = np.float)
for k in range(len(z_e)):
    L[k] = integ_L1(fe[k],r_e,z_e[k],D_a[k])
plt.plot(z_e,L/L0)
plt.xlabel('z')
plt.ylabel(r'$\frac{L}{L0}$')
plt.title(r'$\frac{L}{L0}(z)$')
plt.savefig('Luminosity test.png',dpi = 600)
'''
##########################################
###### based on NFW profile:
import ICL_surface_mass_density as iy
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
Ln = np.tile(a4,(Nbins,1)).T*np.tile(Lc,(Nbins,1))/(4*np.pi) 
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
d0 = U.rad.to(U.deg)*3600
Lsun = C.L_sun.value*10**7 # sun's luminosity in unit :erg/s
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
fn = np.tile(Lc,(Ln.shape[0],1))*Lsun/(4*np.pi*np.tile(Dn_l**2*c2**2,(Ln.shape[0],1)).T)
m = 22.5 - 2.5*np.log10(fn/f0)
angu_s = 1/(np.tile(Dn_a,(Ln.shape[0],1)).T**2)# change kpc^2 to rad^2
Angu_s = angu_s*d0**2 # change rad^2 to arcsec^2
SB1 = m + 2.5*np.log10(Angu_s)
M = m + 5 -5*np.log10(np.tile(Dn_l*c1,(Ln.shape[0],1)).T)
Test_SB = 22.5-2.5*np.log10(fn/f0)+2.5*np.log10(Angu_s) ## test the result SB1

alpha = np.tile(rr*d0/1000,(Ln.shape[0],1))/np.tile(Dn_a,(Ln.shape[0],1)).T
Rs = ((rs/1000)/Dn_a)*d0

### link with magnitude of sun (from Ln)
SB_s1 = -2.5*np.log10(Ln*10**(-6))+ 2.5*np.log10(d0**2)- 5- 2.5*np.log10(4*np.pi)
#the apparent magnitude in unit mag/arcsec^2 link with M_sun

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
        plt.plot(alpha[k,:],SB_s1[k,:],color = mpl.cm.rainbow(k/Nbins),
                 label = 'SB%.3f'%zn[k])
        plt.axvline(x=Rs[k],color = mpl.cm.rainbow(k/Nbins),ls = '--',
                    alpha = 0.35)
plt.legend(loc = 3) 
plt.xscale('log')
plt.xlabel('$R[arcsec]$')
plt.ylabel('$ SB - M_\odot[mag \ arcsec^{-2}]$')
plt.gca().invert_yaxis() # change the y-axis direction
plt.title('$ SB_{app} \ as \ z \ function$')
plt.savefig('surface_brightness_apparent_with_sun1.png',dpi=600)

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
