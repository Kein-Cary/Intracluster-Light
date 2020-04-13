import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pds
import numpy as np
import h5py
import time

import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from matplotlib.patches import Circle
from astropy import cosmology as apcy
from scipy.ndimage import map_coordinates as mapcd
from astropy.io import fits as fits
from scipy import interpolate as interp
from light_measure import light_measure_Z0, light_measure
from scipy.stats import binned_statistic as binned

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Rpp = (rad2asec / Da_ref) / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

band = ['r', 'g', 'i', 'u', 'z']
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'

## differ richness bin Mag hist
for kk in range(3):
	dat = pds.read_csv(load + 'selection/%s_band_sky_catalog.csv' % band[kk])
	set_ra, set_dec, set_z, set_rich = np.array(dat.ra), np.array(dat.dec), np.array(dat.z), np.array(dat.rich)
	com_Mag = np.array(dat.r_Mag)

	idx0 = (set_rich >= 20) & (set_rich <= 30)
	z0 = set_z[idx0]
	cc_Mag = com_Mag[idx0]
	Dl_0 = Test_model.luminosity_distance(z0).value
	Mag_0 = cc_Mag + 5 - 5 * np.log10(Dl_0 * 1e6)

	idx1 = (set_rich >= 30) & (set_rich <= 50)
	z1 = set_z[idx1]
	cc_Mag = com_Mag[idx1]
	Dl_1 = Test_model.luminosity_distance(z1).value
	Mag_1 = cc_Mag + 5 - 5 * np.log10(Dl_1 * 1e6)

	idx2 = set_rich >= 50
	z2 = set_z[idx2]
	cc_Mag = com_Mag[idx2]
	Dl_2 = Test_model.luminosity_distance(z2).value
	Mag_2 = cc_Mag + 5 - 5 * np.log10(Dl_2 * 1e6)

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%s band BCG Magnitude' % band[kk])
	ax.hist(Mag_0, bins = 21, density = True, histtype = 'step', color = 'b', label = '$ 20 \\leqslant \\lambda \\leqslant 30 $', alpha = 0.5)
	ax.axvline(x = np.mean(Mag_0), ls = '--', color = 'b', label = 'Mean', alpha = 0.5)
	ax.axvline(x = np.median(Mag_0), ls = '-.', color = 'b', label = 'Median', alpha = 0.5)

	ax.hist(Mag_1, bins = 21, density = True, histtype = 'step', color = 'g', label = '$ 30 \\leqslant \\lambda \\leqslant 50 $', alpha = 0.5)
	ax.axvline(x = np.mean(Mag_1), ls = '--', color = 'g', alpha = 0.5)
	ax.axvline(x = np.median(Mag_1), ls = '-.', color = 'g', alpha = 0.5)

	ax.hist(Mag_2, bins = 21, density = True, histtype = 'step', color = 'r', label = '$ 50 \\leqslant \\lambda $', alpha = 0.5)
	ax.axvline(x = np.mean(Mag_2), ls = '--', color = 'r', alpha = 0.5)
	ax.axvline(x = np.median(Mag_2), ls = '-.', color = 'r', alpha = 0.5)

	ax.set_xlabel(' Absolute Magnitude ')
	ax.set_ylabel(' PDF ')
	ax.legend(loc = 2)
	ax.invert_xaxis()

	plt.savefig(load + 'rich_sample/test_img/%s_band_rich_bin_Mag.png' % band[kk], dpi = 300)
	plt.close()

for lamda_k in range(3):

	plt.figure()
	ax = plt.subplot(111)
	if lamda_k == 0:
		ax.set_title('$ BCG \; Mag [20 \\leqslant \\lambda \\leqslant 30]$')
	elif lamda_k == 1:
		ax.set_title('$ BCG \; Mag [30 \\leqslant \\lambda \\leqslant 50]$')
	else:
		ax.set_title('$ BCG \; Mag [50 \\leqslant \\lambda ]$')

	for kk in range(3):
		dat = pds.read_csv(load + 'selection/%s_band_sky_catalog.csv' % band[kk])
		set_ra, set_dec, set_z, set_rich = np.array(dat.ra), np.array(dat.dec), np.array(dat.z), np.array(dat.rich)

		com_Mag = np.array(dat.r_Mag)

		if lamda_k == 0:
			idx = (set_rich >= 20) & (set_rich <= 30)
		elif lamda_k == 1:
			idx = (set_rich >= 30) & (set_rich <= 50)
		else:
			idx = set_rich >= 50
		z0 = set_z[idx]
		cc_Mag = com_Mag[idx]

		Dl_0 = Test_model.luminosity_distance(z0).value
		Mag_25 = cc_Mag + 5 - 5 * np.log10(Dl_0 * 1e6) ## absolute magnitude

		ax.hist(Mag_25, bins = 21, density = True, histtype = 'step', color = mpl.cm.rainbow(kk / 3), 
			label = '%s band' % band[kk], alpha = 0.5)
		if kk == 0:
			ax.axvline(x = np.mean(Mag_25), ls = '--', color = mpl.cm.rainbow(kk / 3), label = 'Mean', alpha = 0.5)
			ax.axvline(x = np.median(Mag_25), ls = '-.', color = mpl.cm.rainbow(kk / 3), label = 'Median', alpha = 0.5)
		else:
			ax.axvline(x = np.mean(Mag_25), ls = '--', color = mpl.cm.rainbow(kk / 3), alpha = 0.5)
			ax.axvline(x = np.median(Mag_25), ls = '-.', color = mpl.cm.rainbow(kk / 3), alpha = 0.5)
	ax.set_xlabel(' Absolute Magnitude ')
	ax.set_ylabel(' PDF ')
	ax.legend(loc = 1)
	ax.invert_xaxis()
	if lamda_k == 0:
		plt.savefig(load + 'rich_sample/test_img/low_rich_Mag_hist.png', dpi = 300)
	elif lamda_k == 1:
		plt.savefig(load + 'rich_sample/test_img/media_rich_Mag_hist.png', dpi = 300)
	else:
		plt.savefig(load + 'rich_sample/test_img/high_rich_Mag_hist.png', dpi = 300)
	plt.close()

raise

## residual sky flux hist

#N_bin = np.array([ [1857, 1069, 342], [1860, 1071, 339], [1853, 1068, 337] ])
N_bin = np.array([ [1857, 1069, 342], [1860, 1071, 340], [1853, 1068, 337] ])

plt.figure()
ax = plt.subplot(111)
ax.set_title('difference img flux pdf')

for kk in range(3):

	for lamda_k in range(2, 3):
		# median difference
		with h5py.File(load + 'rich_sample/stack_sky_median_%d_imgs_%s_band_%drich.h5' % 
			(N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
			BCG_sky = np.array(f['a'])

		with h5py.File(load + 'rich_sample/M_sky_rndm_median_%d_imgs_%s_band_%drich.h5' % 
			(N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
			rand_sky = np.array(f['a'])
		differ_img = BCG_sky - rand_sky

		idx = np.isnan(differ_img)
		pdf_img = differ_img[idx == False]

		Nbin = np.linspace(-0.002, 0.002, 26)
		if kk == 0:
			ax.hist(pdf_img, bins = Nbin, histtype = 'step', color = 'r', density = True, alpha = 0.5, label = 'r band')
			ax.axvline(x = np.mean(pdf_img), ls = '--', color = 'r', label = 'Mean', alpha = 0.5)
			ax.axvline(x = np.median(pdf_img), ls = ':', color = 'r', label = 'Median', alpha = 0.5)
		elif kk == 1:
			ax.hist(pdf_img, bins = Nbin, histtype = 'step', color = 'g', density = True, alpha = 0.5, label = 'g band')
			ax.axvline(x = np.mean(pdf_img), ls = '--', color = 'g', alpha = 0.5)
			ax.axvline(x = np.median(pdf_img), ls = ':', color = 'g', alpha = 0.5)
		else:
			ax.hist(pdf_img, bins = Nbin, histtype = 'step', color = 'b', density = True, alpha = 0.5, label = 'i band')
			ax.axvline(x = np.mean(pdf_img), ls = '--', color = 'b', alpha = 0.5)
			ax.axvline(x = np.median(pdf_img), ls = ':', color = 'b', alpha = 0.5)
ax.set_xlabel('flux[nmaggy]')
ax.set_ylabel('PDF')
ax.legend(loc = 1)
plt.savefig(home + '2D_flux_check.png', dpi = 300)
plt.close()
