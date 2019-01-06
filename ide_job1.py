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
'''
#### from the Intensity
import nstep
r_e = 20  # effective radius in unit Kpc according Zibetti's work
I_e = 1   # in unit  (erg/s)/(m^2*Hz)/arcsec^2
c0 = 1*U.Mpc.to(U.m) # change Mpc to pc, mutply this parameter
c1 = 1*U.kpc.to(U.m) 
f0 = 3631*10**(-23)*10**4 # SDSS zero point, in unit: (erg/s)/(m^2*Hz)
z_e = np.linspace(0.2,0.3,101)
R_clust = 1. # in unit Mpc
A_a, D_a = mark_by_self(z_e,R_clust) 
r_c = D_a*U.Mpc.to(U.pc)
# luminosity should be consistant
L0 = nstep.n_step(8)*(np.exp(7.67)/7.67**8)*np.pi*I_e*r_e**2*(U.kpc.to(U.m))**2
# assumption : I(0) = 2000Ie
I_c = np.exp(7.669)*I_e
Ic = I_c*((1+z_e[0])**4)/((1+z_e)**4)
Ie = Ic/np.exp(7.669)
### next calculate the profile
Nbins = np.int(101)
r_clust = np.linspace(0,1000,Nbins)
I_clust = np.zeros((Nbins,Nbins),dtype = np.float)
m_clust = np.zeros((Nbins,Nbins),dtype = np.float)
M_clust = np.zeros((Nbins,Nbins),dtype = np.float)
for p in range(len(z_e)):
    r_index = (r_clust/r_e)**(1/4)
    I_pros = Ie[p]*np.exp(-7.669*(r_index-1))
    I_clust[p,:] = I_pros
    m_clust[p,:] = 22.5-2.5*np.log10(I_pros/f0) ## apparent magnitude 
    M_clust[p,:] = m_clust[p,:]+5-5*np.log10((1+z_e[p])**2*r_c[p]) ## absolute magnituded  
### 
plt.plot(z_e,I_clust[:,0],label = 'center flux')
plt.xlabel('z')
plt.ylabel(r'$Center \ Flux[erg s^{-1} m^{-2} Hz^{-1} / arcsec^2]$')
plt.savefig('center flux intensity.png',dpi = 600)
###
plt.plot(z_e,Ie)
plt.xlabel('z')
plt.ylabel(r'$Effective \ Flux[erg s^{-1} m^{-2} Hz^{-1} / arcsec^]$')
plt.savefig('effect flux intensity.png',dpi=600)
### sb profile change
for q in range(len(z_e)):
    if q % 20 == 0:
        plt.plot(r_clust**(1/4),m_clust[q,:],color = mpl.cm.rainbow(q/len(z_e)),
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
def integ_L2(I,r,z,z0):
    Z = z
    Z0 = z0
    F = I
    R = r # fe, Re are effective radius and flux respectively, R in unit kpc
    Integ_L = lambda r: F*np.exp(-7.669*((r/R)**(1/4)-1))*2*np.pi*r
    L = quad(Integ_L,0,np.inf)[0]*(U.kpc.to(U.m))**2*((1+Z)**4)
    return L
###############
L = np.zeros(len(z_e),dtype = np.float)
for k in range(len(z_e)):
    L[k] = integ_L2(Ie[k],r_e,z_e[k],z_e[0])
plt.plot(z_e,L/L[0])
plt.xlabel('z')
plt.ylabel(r'$\eta[L/L0]$')
plt.title(r'$\eta[L/L0] \ z$')
plt.savefig('Luminosity test.png',dpi = 600)
'''