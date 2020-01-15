import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Ellipse
from light_measure import light_measure, flux_recal
from scipy import ndimage

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)

Lsun = C.L_sun.value*10**7
G = C.G.value
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2
M_dot = 4.83 # the absolute magnitude of SUN

R0 = 1 # in unit Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

## stacking image
dfile_i = '/home/xkchen/jupyter/random_correct_cut_A_1283_imgs_i_band.h5'
dfile_r = '/home/xkchen/jupyter/random_correct_cut_A_1291_imgs_r_band.h5'
dfile_g = '/home/xkchen/jupyter/random_correct_cut_A_1286_imgs_g_band.h5'
dfile_test = '/home/xkchen/jupyter/stack_1000_img.h5'

with h5py.File(dfile_i, 'r') as f:
	img_i = np.array(f['a'])

with h5py.File(dfile_r, 'r') as f:
	img_r = np.array(f['a'])

with h5py.File(dfile_g, 'r') as f:
	img_g = np.array(f['a'])

with h5py.File(dfile_test, 'r') as f:
	img_test = np.array(f['a'])

x0, y0 = 2427, 1765
Nx = np.linspace(0, 4854, 4855)
Ny = np.linspace(0, 3530, 3531)

R_cut, bins = 1260, 80
R_smal, R_max = 1, 1.7e3 # kpc

SB_lel = np.arange(26, 31, 1)
SB2flux = 10**( (22.5 - SB_lel + 2.5 * np.log10(pixel**2)) / 2.5)

cen_img = img_i[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
#cen_img = img_r[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
#cen_img = img_g[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
#cen_img = img_test[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]

dnoise = 20
kernl_img = ndimage.gaussian_filter(cen_img, sigma = dnoise,  mode = 'nearest')
SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)

plt.figure()
ax = plt.subplot(111)
ax.set_title('i band stack img [1283 imgs]')
#ax.set_title('r band stack img [1291 imgs]')
#ax.set_title('g band stack img [1286 imgs]')
#ax.set_title('stack r band mock img [1000 imgs]')

# i band case
Clus0 = Circle(xy = (R_cut, R_cut), radius = Rpp, fill = False, ec = 'k', ls = '-', label = '1Mpc')
#Clus1 = Circle(xy = (R_cut, R_cut), radius = 0.15 * Rpp, fill = False, ec = 'k', ls = (0, (3, 5, 1, 5, 1, 5)),)
#Clus2 = Circle(xy = (R_cut, R_cut), radius = 0.2 * Rpp, fill = False, ec = 'k', ls = ':',)
#Clus3 = Circle(xy = (R_cut, R_cut), radius = 0.4 * Rpp, fill = False, ec = 'k', ls = '-.',)
Clus4 = Circle(xy = (R_cut, R_cut), radius = 0.6 * Rpp, fill = False, ec = 'k', ls = '--', label = '0.6Mpc')
'''
Clus0 = Circle(xy = (R_cut, R_cut), radius = Rpp, fill = False, ec = 'k', ls = '-', label = '1Mpc')
#Clus1 = Circle(xy = (R_cut, R_cut), radius = 0.3 * Rpp, fill = False, ec = 'k', ls = ':',)
Clus2 = Circle(xy = (R_cut, R_cut), radius = 0.5 * Rpp, fill = False, ec = 'k', ls = '-.', label = '0.5Mpc')
#Clus3 = Circle(xy = (R_cut, R_cut), radius = 0.15 * Rpp, fill = False, ec = 'k', ls = '--',)
'''
tf = ax.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
plt.clabel(tf, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

# i band case
ax.add_patch(Clus0)
#ax.add_patch(Clus1)
#ax.add_patch(Clus2)
#ax.add_patch(Clus3)
ax.add_patch(Clus4)
'''
ax.add_patch(Clus0)
#ax.add_patch(Clus1)
ax.add_patch(Clus2)
#ax.add_patch(Clus3)
'''
ax.imshow(cen_img, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm(),)
ax.axis('equal')
ax.set_xlim(R_cut - np.ceil(Rpp), R_cut + np.ceil(Rpp))
ax.set_ylim(R_cut - np.ceil(Rpp), R_cut + np.ceil(Rpp))
ax.legend(loc = 2)

plt.savefig('i_band_stack_img_contour.png', dpi = 300)
#plt.savefig('r_band_stack_img_contour.png', dpi = 300)
#plt.savefig('g_band_stack_img_contour.png', dpi = 300)
#plt.savefig('r_band_stack_mock_img_contour.png', dpi = 300)
plt.show()
