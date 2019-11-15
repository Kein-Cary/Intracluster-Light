import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse

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
from light_measure import light_measure, flux_recal
from resample_modelu import down_samp, sum_samp

# constant
vc = C.c.to(U.km/U.s).value
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg_s = U.L_sun.to(U.erg/U.s)
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
DH = vc/H0

pixel = 0.396 # the pixel size in unit arcsec
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])
with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

goal_data = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
redshift = np.array(goal_data.Z_SPEC)
Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
Mag_err = np.array(goal_data.MODEL_MAGERR_R)
com_Mag = Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag_err = Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

tmp = '/home/xkchen/mywork/ICL/data/test_data/tmp/'

def sers_pro(r, mu_e, r_e):
	n = 4.
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def sngl_pro():
	NMGY = 5e-3 ## change mock img to flux
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	bins, zN = 65, 20
	sub_R = np.logspace(1, 3.01, bins)
	## select sample based on mag
	idx0 = com_Mag <= 17
	idx1 = (com_Mag >= 17) & (com_Mag <= 18)
	idx2 = com_Mag >= 18

	sub_z0 = z[idx0][:zN]
	sub_ra0 = ra[idx0][:zN]
	sub_dec0 = dec[idx0][:zN]
	sub_SB0 = np.zeros((zN, bins), dtype = np.float)

	sub_z1 = z[idx1][:zN]
	sub_ra1 = ra[idx1][:zN]
	sub_dec1 = dec[idx1][:zN]
	sub_SB1 = np.zeros((zN, bins), dtype = np.float)

	sub_z2 = z[idx2][:zN]
	sub_ra2 = ra[idx2][:zN]
	sub_dec2 = dec[idx2][:zN]
	sub_SB2 = np.zeros((zN, bins), dtype = np.float)

	for kk in range(3 * zN):
		if kk // zN == 0:
			z_g = sub_z0[kk % zN]
			ra_g = sub_ra0[kk % zN]
			dec_g = sub_dec0[kk % zN]
			Da_g = Test_model.angular_diameter_distance(z_g).value
			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
			data = fits.open(load + file)
			img = data[0].data
			wcs = awc.WCS(data[0].header)
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)


		elif kk // zN == 1:
			z_g = sub_z1[kk % zN]
			ra_g = sub_ra1[kk % zN]
			dec_g = sub_dec1[kk % zN]
			Da_g = Test_model.angular_diameter_distance(z_g).value

		elif kk // zN == 2:
			z_g = sub_z1[kk % zN]
			ra_g = sub_ra1[kk % zN]
			dec_g = sub_dec1[kk % zN]
			Da_g = Test_model.angular_diameter_distance(z_g).value

		else:
			continue

	raise

def pro_compare():
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	idu = (com_Mag >= 17) & (com_Mag <= 18)

	sub_ra = ra[idu]
	sub_dec = dec[idu]
	sub_z = z[idu]
	zN = 20
	fig = plt.figure(figsize = (20,20))
	fig.suptitle('BCG pros')
	gs = gridspec.GridSpec(zN // 5, 5)

	cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
						4.63,  7.43,  11.42,  18.20,  28.20, 
						44.21, 69.00, 107.81, 168.20, 263.00])
	for q in range(zN):
		ra_g = sub_ra[q]
		dec_g = sub_dec[q]
		z_g = sub_z[q]
		Da_g = Test_model.angular_diameter_distance(z_g).value	
		file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('g', ra_g, dec_g, z_g) ## data
		data = fits.open(load + file)
		img = data[0].data
		wcs = awc.WCS(data[0].header)
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
		r_bcg = (0.1 * rad2asec / Da_g) / pixel

		## measure the BCG magnitude
		R_smal, R_max, bins = 1, 200, 55
		Intns, Intns_r, Intns_err, Npix = light_measure(img, bins, R_smal, R_max, cx, cy, pixel, z_g)
		id_nan = np.isnan(Intns)
		flux = Intns[id_nan == False]
		flux_err = Intns_err[id_nan == False]
		bin_R = Intns_r[id_nan == False]
		flux0 = flux + flux_err
		flux1 = flux - flux_err

		SB_img = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = SB_img - SB0
		err1 = SB1 - SB_img
		id_nan = np.isnan(SB1)
		err1[id_nan] = 100.
		img_err0 = err0 * 1
		img_err1 = err1 * 1

		## cat profile of SDSS
		# the band info. 0, 1, 2, 3, 4 --> u, g, r, i, z
		cat_pro = pds.read_csv(tmp + 'BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g), skiprows = 1)
		dat_band = np.array(cat_pro.band)
		dat_bins = np.array(cat_pro.bin)
		dat_pro = np.array(cat_pro.profMean) # in unit of nmaggy / arcsec^2
		dat_pro_err = np.array(cat_pro.profErr)

		idx = dat_band == 1
		tt_pro = dat_pro[idx]
		tt_bin = dat_bins[idx]
		tt_proErr = dat_pro_err[idx]
		tt_r = (cat_Rii[tt_bin] * Da_g / rad2asec) * 1e3 # arcsec --> kpc

		cat_SB = 22.5 - 2.5 * np.log10(tt_pro)
		SB0 = 22.5 - 2.5 * np.log10(tt_pro + tt_proErr)
		SB1 = 22.5 - 2.5 * np.log10(tt_pro - tt_proErr)
		err0 = cat_SB - SB0
		err1 = SB1 - cat_SB
		id_nan = np.isnan(cat_SB)
		cc_SB, SB0, SB1 = cat_SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False]
		cc_r, err0, err1 = tt_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
		id_nan = np.isnan(SB1)
		err1[id_nan] = 100.
		cc_err0 = err0 * 1
		cc_err1 = err1 * 1

		ax = plt.subplot(gs[q // 5, q % 5])
		ax.errorbar(bin_R, SB_img, yerr = [img_err0, img_err1], xerr = None, ls = '-', fmt = 'g.', label = 'Me pipe', alpha = 0.5)
		ax.errorbar(cc_r, cc_SB, yerr = [cc_err0, cc_err1], xerr = None, ls = '--', fmt = 'r.', label = 'SDSS cat.', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('k[kpc]')
		ax.set_ylabel('SB[mag/arcsec^2]')
		ax.set_ylim(20, 28)
		ax.set_xlim(1, 200)
		ax.legend(loc = 1)
		ax.invert_yaxis()

	plt.tight_layout()
	plt.savefig('BCG_pros.png', dpi = 300)
	plt.close()

	raise

def rand_pro():
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	bins, zN0, zN1 = 65, 0, 5
	sub_ra, sub_dec, sub_z = ra[zN0:zN1], dec[zN0:zN1], z[zN0:zN1]
	dx = np.linspace(0, 35, 8)
	dy = np.linspace(0, 35, 8)
	dN = np.int(zN1 - zN0)

	for kk in range(dN):
		ra_g = sub_ra[kk]
		dec_g = sub_dec[kk]
		z_g = sub_z[kk]

		file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
		data_f = fits.open(load+file)
		img = data_f[0].data
		Da_g = Test_model.angular_diameter_distance(z_g).value
		R_p = (rad2asec / Da_g) / pixel

		head_inf = data_f[0].header
		wcs = awc.WCS(head_inf)
		cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		fig = plt.figure()
		fig.suptitle('rand-position SB profile ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g) )
		ax = plt.subplot(111)

		R_smal, R_max = 10, 10**3.02
		for jj in range(len(dx)):
			cx = cx_BCG - dx[jj]
			cy = cy_BCG - dy[jj]
			Intns, Intns_r, Intns_err, Npix = light_measure(img, bins, R_smal, R_max, cx, cy, pixel, z_g)
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

			ax.plot(pR, SB, linestyle = '-', color = mpl.cm.rainbow(jj / len(dx)), label = 'dx = %.0f' % dx[jj], alpha = 0.5)
			ax.set_xscale('log')
			ax.set_xlabel('R[kpc]')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.legend(loc = 1)
			ax.set_xlim(9, 1010)
			ax.set_ylim(21, 27)
			ax.invert_yaxis()
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.savefig('rand_SB_%d.png' % kk, dpi = 300)
		plt.close()
	return

def main():
	sngl_pro()
	#pro_compare()
	#rand_pro()

if __name__ == "__main__":
	main()
