# new file use to record calculation
from astropy.io.ascii import BaseReader
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import handy.scatter as hsc
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
import astropy.constants as C
import astropy.units as U
import numpy as np
######## test the luminosity
file = get_pkg_data_filename('area_test.fits') # k =1
hdu = fits.open(file)[0]
wcs = WCS(hdu.header)
import mice_symbol 
######## read the SEXtractor data
#################
cir_data = fits.getdata('area_test.fits',header = True)
wcs = WCS(cir_data[1])
x0,y0 = wcs.all_world2pix(Ra[1]*U.deg, Dec[1]*U.deg, 1)
r0 = R_A[1].value/0.396
f = plt.figure(figsize = (50,50))
f.suptitle(r'$S/S0$') # control the total name of this figure
spc = grid.GridSpec(ncols = 5, nrows = 5,figure = f)
ax1 = f.add_subplot(spc[0,0],projection = wcs)
im1 = ax1.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax1.set_title('$Ra_{237.186} \ Dec_{45.769}$ \n $\lambda_{125.678} \ z_{0.225}$',fontsize = 20)
# this line shows change line of title
ax1.set_ylabel(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$', labelpad = 0.005)
ax = ax1.axes.coords[0]
ay = ax1.axes.coords[1]
ax.set_ticks(spacing = 0.05*U.deg)
ax.set_ticklabel(color = 'red')
ax.grid(color = 'red',alpha = 0.5)
ay.set_ticks(spacing = 0.05*U.deg) 
ay.set_ticklabel(color = 'green')
ay.grid(color = 'green',alpha = 0.5)
ax1.set_xticklabels(labels = [],fontsize = 6.5)
ax1.set_yticklabels(labels = [],fontsize = 6.5)
hsc.circles(1025,745,s=200,fc = '',ec = 'b')
ax1.axes.set_aspect('equal')
co1 = plt.colorbar(im1,label = 'flux',fraction = 0.035,pad = 0.001)
co1.set_ticks(np.logspace(-5,2,6))

ax12 = f.add_subplot(spc[0,1],projection = wcs)
im2 = ax12.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax12.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax12.set_ylabel(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$', labelpad = 0.005)
ax1 = ax12.axes.coords[0]
ay1 = ax12.axes.coords[1]
ax1.set_ticks(spacing = 0.05*U.deg)
ax1.set_ticklabel(color = 'red')
ax1.grid(color = 'red',alpha = 0.5)
ay1.set_ticks(spacing = 0.05*U.deg) 
ay1.set_ticklabel(color = 'green')
ay1.grid(color = 'green',alpha = 0.5)
ax12.set_xticklabels(labels = [], fontsize = 6.5)
ax12.set_yticklabels(labels = [], fontsize = 6.5)
hsc.circles(1025,745,s=200,fc = '',ec = 'b')
ax12.axes.set_aspect('equal')
#ax12.grid()
co2 = plt.colorbar(im2,label = 'flux',fraction = 0.035,pad = 0.001)
co2.set_ticks(np.logspace(-5,2,6))

ax13 = f.add_subplot(spc[0,2],projection = wcs)
im3 = ax13.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax13.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax13.set_ylabel(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$', labelpad = 0.005)
ax13.set_xticklabels(labels = [], fontsize = 6.5)
ax13.set_yticklabels(labels = [], fontsize = 6.5)
hsc.circles(1025,745,s=200,fc = '',ec = 'b')
ax13.axes.set_aspect('equal')
ax13.grid()
co3 = plt.colorbar(im3,label = 'flux',fraction = 0.035,pad = 0.001)
co3.set_ticks(np.logspace(-5,2,6))

ax14 = f.add_subplot(spc[0,3],projection = wcs)
im4 = ax14.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax14.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax14.set_ylabel(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$', labelpad = 0.005)
ax14.set_xticklabels(labels = [], fontsize = 6.5)
ax14.set_yticklabels(labels = [], fontsize = 6.5)
ax14.axes.set_aspect('equal')
ax14.grid()
co4 = plt.colorbar(im4,label = 'flux',fraction = 0.035,pad = 0.001)
co4.set_ticks(np.logspace(-5,2,6))

ax15 = f.add_subplot(spc[0,4],projection = wcs)
im5 = ax15.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax15.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax15.set_ylabel(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$', labelpad = 0.005)
ax15.set_xticklabels(labels = [], fontsize = 6.5)
ax15.set_yticklabels(labels = [], fontsize = 6.5)
ax15.axes.set_aspect('equal')
ax15.grid()
co5 = plt.colorbar(im5,label = 'flux',fraction = 0.035,pad = 0.001)
co5.set_ticks(np.logspace(-5,2,6))

ax2 = f.add_subplot(spc[1,0],projection = wcs)
ax2.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax2.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax2.set_xticklabels(labels = [], fontsize = 6.5)
ax2.set_yticklabels(labels = [], fontsize = 6.5)
ax2.grid()

ax3 = f.add_subplot(spc[2,0],projection = wcs)
ax3.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax3.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax3.set_xticklabels(labels = [], fontsize = 6.5)
ax3.set_yticklabels(labels = [], fontsize = 6.5)
ax3.grid()

ax4 = f.add_subplot(spc[3,0],projection = wcs)
ax4.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax4.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax4.set_xticklabels(labels = [], fontsize = 6.5)
ax4.set_yticklabels(labels = [], fontsize = 6.5)
ax4.grid()

ax5 = f.add_subplot(spc[4,0],projection = wcs)
ax5.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ax5.set_title(r'$Ra_{237.186} \ Dec_{45.769} \ \lambda_{125.678} \ z_{0.225}$',fontsize = 20)
ax5.set_xticklabels(labels = [], fontsize = 6.5)
ax5.set_yticklabels(labels = [], fontsize = 6.5)
ax5.grid()
#f.tight_layout(pad = 1.08, h_pad = 1.08, w_pad =1.08)
#plt.savefig('test_pdf.pdf',dpi=600)
##################################################### surface brightness calculate
from ICL_angular_diameter_reshift import mark_by_self
import ICL_surface_mass_density as iy
Mc = 15 # the mass of the cluster, in unit solar mass 
Ml = Mc*0.1 # the mass of ICL, here assume the same raito of ICL-mass to cluster-mass
M2l = 50 # the mass-to-light ratio of ICL
fn0 = 3631*10**(-26) # SDSS zero point, in unit: (erg/s)/(m^2*Hz)
Lsun = C.L_sun.value*10**7 # sun's luminosity in unit :erg/s
Nbins = 101
sigma = iy.sigma_m_c(Mc,101)[1]
rr = iy.sigma_m_c(Mc,101)[2]
zn = np.linspace(0.2,0.3,Nbins)
An_a, Dn_a = mark_by_self(zn,1) # the angular size and angulardiameter distance in NFW model
Dn_l = (1+zn)**2*Dn_a
fn = np.zeros((len(zn),len(zn)), dtype = np.float)
mn = np.zeros((len(zn),len(zn)), dtype = np.float)
Mn = np.zeros((len(zn),len(zn)), dtype = np.float)
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
###
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
        plt.plot(rr,mn[q,:],color = mpl.cm.rainbow(q/len(zn)),
                 label = r'$z%.3f$'%zn[q])
plt.legend(loc = 1)
plt.gca().invert_yaxis() # change the y-axis direction
plt.xlabel(r'$R^{\frac{1}{4}} \ [kpc]$')
plt.ylabel(r'$Surface \ brightness \ [mag/arcsec^2]$')
plt.xscale('log')
plt.title(r'$SB(R) \ as \ function \ of \ z$')
plt.savefig('NFW_SB_apparent_brightness.png',dpi=600)
### absolute magnitude
for q in range(len(zn)):
    if q % 20 == 0:
        plt.plot(rr,Mn[q,:],color = mpl.cm.rainbow(q/len(zn)),
                 label = r'$z%.3f$'%zn[q],alpha = 0.5)
plt.legend(loc = 1)
plt.gca().invert_yaxis() ## invers the direction of axis
plt.ylabel(r'$M_{absolute \ magnitude} \ [mag/arcsec^2]$')
plt.xlabel(r'$R^{\frac{1}{4}} \ [kpc]$')
plt.xscale('log')
plt.title(r'$SB(R) \ in \ absolute \ Magnitude \ as \ z \ function$')
plt.savefig('NFW_SB_absolute_magnitude.png',dpi=600)
