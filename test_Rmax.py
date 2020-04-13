import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import h5py
import random
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy.optimize import curve_fit, minimize
from light_measure import light_measure, light_measure_rn

## constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
Lsun2erg = U.L_sun.to(U.erg/U.s)
rad2asec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Angu_ref = rad2asec / Da_ref
Rpp = Angu_ref / pixel

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def get_pro(img_array, R_bins, R_min, R_max, cx, cy, pix_scale, z):

	Intns, Intns_r, Intns_err, Npix = light_measure(img_array, R_bins, R_min, R_max, cx, cy, pixel, z)
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB - SB0
	err1 = SB1 - SB
	id_nan = np.isnan(SB)
	SB, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
	pR, err0, err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.

	return Intns_r, Intns, Intns_err, pR, SB, err0, err1

bins = 95
R_smal, R_max = 1, 3e3# kpc
r_a0, r_a1 = 1e3, 1.1e3

## read the image
x0, y0 = 2427, 1765
for kk in range(3):

	for nn in range(20):

		with h5py.File(load + 'rich_sample/test_img/%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
			img = np.array(f['a'])

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%s band %d sub-stack img' % (band[kk], nn) )
		clust0 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5, label = '1 Mpc')
		clust1 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'g', ls = '-', alpha = 0.5, label = '2 Mpc')
		clust2 = Circle(xy = (x0, y0), radius = 3 * Rpp, fill = False, ec = 'b', ls = '-', alpha = 0.5, label = '3 Mpc')

		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e1, norm = mpl.colors.LogNorm())
		ax.add_patch(clust0)
		ax.add_patch(clust1)
		ax.add_patch(clust2)
		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])
		ax.legend(loc = 1, frameon = False)
		plt.savefig(home + '%s_band_%d_sub-stack_img.png' % (band[kk], nn), dpi = 300)
		plt.close()
raise
