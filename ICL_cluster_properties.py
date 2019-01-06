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
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
from astropy import cosmology as apcy
##### from flux
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
plt.xlabel(r'$R^{\frac{1}{4}} \ [arcsec]$')
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
##########################################
###### NFW profile:
import ICL_surface_mass_density as iy
Mc = 15 # the mass of the cluster, in unit solar mass 
Ml = Mc*0.1 # the mass of ICL, here assume the same raito of ICL-mass to cluster-mass
M2l = 50 # the mass-to-light ratio of ICL
fn0 = 3631*10**(-23)*10**4 # SDSS zero point, in unit: (erg/s)/(m^2*Hz)
Lsun = C.L_sun.value*10**7 # sun's luminosity in unit :erg/s
Nbins = 101
rr = np.logspace(-5,3,Nbins)
zn = np.linspace(0.2,0.3,Nbins)
An_a, Dn_a = mark_by_self(zn,1) # the angular size and angulardiameter distance in NFW model
Dn_l = (1+zn)**2*Dn_a
fn = np.zeros((len(zn),len(zn)), dtype = np.float)
mn = np.zeros((len(zn),len(zn)), dtype = np.float)
Mn = np.zeros((len(zn),len(zn)), dtype = np.float)
sigma = iy.sigma_m(Mc, 0, rr)[2]
Lnc = sigma*Lsun/50
Ln0 = np.sum(Lnc)
for k in range(len(zn)):
    fn[k,:] = Lnc/(4*np.pi*(Dn_l[k])**2*U.Mpc.to(U.m)**2)
    mn[k,:] = 22.5 - 2.5*np.log10(fn[k,:]/fn0)
    Mn[k,:] = mn[k,:] +5 -5*np.log10(Dn_l[k]*U.Mpc.to(U.pc))
### 
fnc = fn[:,0]
plt.plot(zn,fnc,label = 'mean flux')
plt.xlabel('z')
plt.ylabel(r'$Mean \ Flux[erg s^{-1} m^{-2} Hz^{-1}]$')
plt.savefig('NFW_mean_flux.png',dpi = 600)

plt.plot(rr,sigma,label = '$\Sigma[R]$')
plt.legend(loc = 1)
plt.xlabel('$R[kpc]$')
plt.xscale('log')
plt.ylabel('$\Sigma[M_\odot kpc^{-2}]$')
plt.title('$surface \ mass \ density$')
plt.yscale('log')
plt.savefig('NFW_surface_mass_density.png',dpi = 600)
###
for q in range(len(zn)):
    if q % 20 == 0:
        plt.plot(rr**(1/4),mn[q,:],color = mpl.cm.rainbow(q/len(zn)),
                 label = r'$z%.3f$'%zn[q])
plt.legend(loc = 1)
plt.gca().invert_yaxis() # change the y-axis direction
plt.xlabel(r'$R^{\frac{1}{4}} \ [kpc]$')
plt.ylabel(r'$Surface \ brightness \ [mag/arcsec^2]$')
plt.title(r'$SB(R) \ as \ function \ of \ z$')
plt.savefig('NFW_SB_apparent_brightness.png',dpi=600)
### absolute magnitude
for q in range(len(zn)):
    if q % 20 == 0:
        plt.plot(rr**(1/4),Mn[q,:],color = mpl.cm.rainbow(q/len(zn)),
                 label = r'$z%.3f$'%zn[q],alpha = 0.5)
plt.legend(loc = 1)
plt.gca().invert_yaxis() ## invers the direction of axis
plt.ylabel(r'$M_{absolute \ magnitude} \ [mag/arcsec^2]$')
plt.xlabel(r'$R^{\frac{1}{4}} \ [arcsec]$')
plt.title(r'$SB(R) \ in \ absolute \ Magnitude \ as \ z \ function$')
plt.savefig('NFW_SB_absolute_magnitude.png',dpi=600)
####################
def f_r(m,z,r):
    Mc = m
    Z = z
    R = r
    sigma = iy.sigma_m(Mc, Z, R)[2]
    Lnc = sigma*Lsun/50
    Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
    Da = Test_model.angular_diameter_distance(Z).value
    Dl = (1+z)**2*Da
    flux = Lnc/(4*np.pi*(Dl)**2*(U.Mpc.to(U.m))**2)
    return flux
def integ_L3(m,z,r):
    M = m
    Z = z
    r = r
    flux = lambda x: f_r(M,Z,r)[2]*2*np.pi*x
    integ_L = quad(flux,r[0],np.inf)[0]*(U.kpc.to(U.m))**2
    return integ_L
###############
L = np.zeros(len(zn),dtype = np.float)
for k in range(len(zn)):
    L[k] = integ_L3(Mc,zn[k],rr)
plt.plot(zn,L/Ln0)
plt.xlabel('z')
plt.ylabel(r'$\frac{L}{L0}$')
plt.title(r'$\frac{L}{L0}(z)$')
plt.savefig('NFW_Luminosity_test.png',dpi = 600)
