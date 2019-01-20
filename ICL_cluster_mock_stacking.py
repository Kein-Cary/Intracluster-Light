## calculate the profile form mag/arcsec^2 (in observation) to L_sun/kpc^2
"""
first, produce a series of cluster at different redshift,with different surface brightness
(but the profile is the same)
"""
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import ICL_surface_mass_density as iy
cos_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311) # Plank18
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
c3 = U.rad.to(U.deg)*3600
c4 = U.L_sun.to(U.erg/U.s)
Lsun = C.L_sun.value*10**7 # sun's luminosity in unit :erg/s
c = 5 
M200 = np.linspace(14,15,6)
zc = np.linspace(0.2,0.3,6)
Nbins = 101
r200 = np.zeros(len(M200),dtype=np.float)
rs = np.zeros(len(M200),dtype=np.float)
sigma = np.zeros((len(M200),Nbins),dtype=np.float) 
r = np.zeros((len(M200),Nbins),dtype=np.float) 
for k in range(len(M200)):
    r200[k] = iy.sigma_m_c(M200[k],Nbins)[0]
    sigma[k,:] = iy.sigma_m_c(M200[k],Nbins)[1]
    r[k,:] = iy.sigma_m_c(M200[k],Nbins)[2]
rs = r200/c
Lc = sigma/50
SB = 21.572-2.5*np.log10(Lc*10**(-6))+10*np.log10(np.tile(zc,(Lc.shape[1],1)).T+1)
obSB = SB + 4.83
Da_c = cos_model.angular_diameter_distance(zc).value
alpha = (r/1000)*c3/np.tile(Da_c,(Nbins,1)).T
Rs = (rs/1000)*c3/Da_c
### get the intrinsic SB light profile
for q in range(len(M200)):
    plt.plot(r[q,:],Lc[q,:],color = mpl.cm.rainbow(q/len(M200)),
             label = 'M%.1f_z%.2f'%(M200[q],zc[q]))
    plt.axvline(x=rs[q],color = mpl.cm.rainbow(q/len(M200)),ls = '--',alpha = 0.5)
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.yscale('log')
plt.ylabel('$SB [L_ \odot/kpc^{2}]$')
plt.savefig('mock_cluste.png',dpi=600)
### get the observation SB light profile
for q in range(len(M200)):
    plt.plot(alpha[q,:],obSB[q,:],color = mpl.cm.rainbow(q/len(M200)),
             label = 'M%.1f_z%.2f'%(M200[q],zc[q]))
    plt.axvline(x=Rs[q],color = mpl.cm.rainbow(q/len(M200)),ls = '--',alpha = 0.5)
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[arcsec]$')
plt.gca().invert_yaxis()
plt.ylabel('$SB[mag/arcsec^{2}]$')
plt.savefig('mock_observation.png',dpi=600)
### take SB+M_sun(absolute magnitude of sun) as observation, and see how to get the intrinsic profile
m0 = 22.19246 
# use m0=22.5 and the SB in mag/arcsec^2 will larger than obSB (use the abs magnitude of sun) about 0.30754
intri_SB = np.tile((1+zc)**4,(Nbins,1)).T*4*np.pi*(f0*c3**2*c2**2*c4**(-1))*10**((m0-obSB)/2.5)
for q in range(len(M200)):
    plt.plot(r[q,:],intri_SB[q,:],color = mpl.cm.rainbow(q/len(M200)),
             label = 'M%.1f_z%.2f'%(M200[q],zc[q]))
    plt.plot(r[q,:],Lc[q,:],'k--')
    plt.axvline(x=rs[q],color = mpl.cm.rainbow(q/len(M200)),ls = '--',alpha = 0.5)
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[kpc]$')
plt.yscale('log')
plt.ylabel('$SB[L_\odot/kpc^2]$')
plt.savefig('mock_observation_2_intrinsic.png',dpi=600)
####################
# stacking analysis
zref = 0.25
obSB_ref = obSB+10*np.log10((1+0.25)/(np.tile(zc,(Lc.shape[1],1)).T+1))
Da_ref = cos_model.angular_diameter_distance(0.25).value
scale_ref = (1*c3)/Da_ref # also the number o pixels at z=0.25 for R=1Mpc,(set 1pixel=1arcsec)
angu_ref = ((1/c)*c3)/Da_ref 
Rs_ref = (rs/1000)*c3/Da_ref
alpha_ref = (r/1000)*c3/np.tile(Da_ref,(Nbins,len(M200))).T
for q in range(len(M200)):
    plt.plot(alpha_ref[q,:],obSB_ref[q,:],color = mpl.cm.rainbow(q/len(M200)),
             label = 'M%.1f_z%.2f'%(M200[q],zc[q]))
    plt.axvline(x=Rs_ref[q],color = mpl.cm.rainbow(q/len(M200)),ls = '--',lw=0.5)
plt.axvline(x=angu_ref,c='k',ls = '-',lw=0.5)
plt.axvline(x=scale_ref,c='k',ls = '--',lw=0.5)
plt.legend(loc = 3)
plt.xscale('log')
plt.xlabel('$R[arcsec]$')
plt.gca().invert_yaxis()
plt.ylabel('$SB_{z0.25}[mag/arcsec^{2}]$')
plt.savefig('ref_observation.png',dpi=600)
### set 300arcsec size as the stacking frame
