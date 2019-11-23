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

from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from extinction_redden import A_wave
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy.optimize import curve_fit, minimize
from light_measure import light_measure, flux_recal, flux_scale
from resample_modelu import down_samp, sum_samp

import sfdmap
Rv = 3.1
sfd = SFDQuery()
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
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

goal_data = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
redshift = np.array(goal_data.Z_SPEC)
Mag_bcgs = np.array(goal_data.MODEL_MAG_R) ## mag for r band
Mag_err = np.array(goal_data.MODEL_MAGERR_R)

g_Mag_bcgs = np.array(goal_data.MODEL_MAG_G)
i_Mag_bcgs = np.array(goal_data.MODEL_MAG_I)

com_Mag = Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag_err = Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

com_Mag = Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_g_Mag = g_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_i_Mag = i_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
tmp = '/home/xkchen/mywork/ICL/data/test_data/'

def sers_pro(r, mu_e, r_e):
	n = 4.
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def Z05_compare():
	## SB Mine
	x0, y0, R_cut = 2427, 1765, 900
	R_smal, R_max, bins = 10, 10**3.02, 65
	with h5py.File('/home/xkchen/jupyter/stack_Amask_3363_in_i_band.h5', 'r') as f:
		i_pro = np.array(f['a'])
	sub_img = i_pro[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
	Intns, Intns_r, Intns_err, Npix = light_measure(sub_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
	id_nan = np.isnan(Intns)
	flux = Intns[id_nan == False]
	flux_err = Intns_err[id_nan == False]
	ipro_R = Intns_r[id_nan == False]
	flux0 = flux + flux_err
	flux1 = flux - flux_err

	SB_i_pro = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB_i_pro - SB0
	err1 = SB1 - SB_i_pro
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.
	ipro_err0 = err0 * 1
	ipro_err1 = err1 * 1

	with h5py.File('/home/xkchen/jupyter/stack_Amask_3377_in_g_band.h5', 'r') as f:
		g_pro = np.array(f['a'])
	sub_img = g_pro[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
	Intns, Intns_r, Intns_err, Npix = light_measure(sub_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
	id_nan = np.isnan(Intns)
	flux = Intns[id_nan == False]
	flux_err = Intns_err[id_nan == False]
	gpro_R = Intns_r[id_nan == False]
	flux0 = flux + flux_err
	flux1 = flux - flux_err

	SB_g_pro = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB_g_pro - SB0
	err1 = SB1 - SB_g_pro
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.
	gpro_err0 = err0 * 1
	gpro_err1 = err1 * 1

	with h5py.File('/home/xkchen/jupyter/stack_Amask_3378_in_r_band.h5', 'r') as f:
		r_pro = np.array(f['a'])
	sub_img = r_pro[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
	Intns, Intns_r, Intns_err, Npix = light_measure(sub_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
	id_nan = np.isnan(Intns)
	flux = Intns[id_nan == False]
	flux_err = Intns_err[id_nan == False]
	rpro_R = Intns_r[id_nan == False]
	flux0 = flux + flux_err
	flux1 = flux - flux_err

	SB_r_pro = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB_r_pro - SB0
	err1 = SB1 - SB_r_pro
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.
	rpro_err0 = err0 * 1
	rpro_err1 = err1 * 1

	## check the query SB profile
	cat0 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_BCG_ICL.csv')
	R0, SB0 = cat0['(1000R)^(1/4)'], cat0['mag/arcsec^2']
	R0 = R0**4

	cat1 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/g_band_BCG_ICL.csv')
	R1, SB1 = cat1['(1000R)^(1/4)'], cat1['mag/arcsec^2']
	R1 = R1**4

	cat2 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/i_band_BCG_ICL.csv')
	R2, SB2 = cat2['(1000R)^(1/4)'], cat2['mag/arcsec^2']
	R2 = R2**4

	devi_g = SB_g_pro + 2.5 * np.log10(2.40)
	devi_i = SB_i_pro - 2.5 * np.log10(1.60)
	devi_r = SB_r_pro + 2.5 * np.log10(1.11)
	'''
	plt.figure(figsize = (16, 9))
	ax = plt.subplot(111)
	ax.set_title('SB profile comparison')
	ax.plot(R0, SB0, 'r-', label = 'r band [Z05]', alpha = 0.5)
	ax.errorbar(rpro_R, SB_r_pro, yerr = [rpro_err0, rpro_err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'r', label = 'r band [stacking]', elinewidth = 1, alpha = 0.5)
	ax.plot(rpro_R, devi_r, 'r:', label = 'shift r band [stacking]', alpha = 0.5)

	ax.plot(R1, SB1, 'g-', label = 'g band [Z05]', alpha = 0.5)
	ax.errorbar(gpro_R, SB_g_pro, yerr = [gpro_err0, gpro_err1], xerr = None, color = 'g', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'g', label = 'g band [stacking]', elinewidth = 1, alpha = 0.5)
	ax.plot(gpro_R, devi_g, 'g:', label = 'shift g band [stacking]', alpha = 0.5)

	ax.plot(R2, SB2, 'b-', label = 'i band [Z05]', alpha = 0.5)
	ax.errorbar(ipro_R, SB_i_pro, yerr = [ipro_err0, ipro_err1], xerr = None, color = 'b', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'b', label = 'i band [stacking]', elinewidth = 1, alpha = 0.5)
	ax.plot(ipro_R, devi_i, 'b:', label = 'shift i band [stacking]', alpha = 0.5)

	ax.legend(loc = 1)
	ax.grid(which = 'both', axis = 'both')
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylabel('$ SB[mag/arcsec^2] $')
	ax.set_xlim(10, 1e3)
	ax.set_ylim(22, 32)
	ax.invert_yaxis()
	#plt.savefig('Zibetti_pro.png', dpi = 300)
	plt.savefig('pro_compare.png', dpi = 300)
	plt.show()
	'''
	return rpro_R, SB_r_pro, gpro_R, SB_g_pro, ipro_R, SB_i_pro

def sngl_pro():
	"""
	this part main including resampling process
	"""
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	idu = (com_Mag >= 17) & (com_Mag <= 18)
	sub_ra, sub_dec, sub_z = ra[idu], dec[idu], z[idu]
	sub_Mag, sub_g_Mag, sub_i_Mag = com_Mag[idu], com_g_Mag[idu], com_i_Mag[idu]
	zN = 20
	cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
						4.63,  7.43,  11.42,  18.20,  28.20, 
						44.21, 69.00, 107.81, 168.20, 263.00])

	## hist the magnitude of sample
	DL0 = Test_model.luminosity_distance(sub_z).value
	samp_r_mag = sub_Mag[:zN] + 5 - 5 * np.log10(DL0[:zN] * 1e6)
	samp_g_mag = sub_g_Mag[:zN] + 5 - 5 * np.log10(DL0[:zN] * 1e6)
	samp_i_mag = sub_i_Mag[:zN] + 5 - 5 * np.log10(DL0[:zN] * 1e6)
	# sample in sgi server
	N_tt = len(z)
	serv_r_mag = np.zeros(N_tt, dtype = np.float)
	for ll in range(N_tt):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]
		with h5py.File('/home/xkchen/mywork/ICL/data/redmapper/Except_r_sample.h5', 'r') as f:
			except_cat = np.array(f['a'])
		except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
		except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
		if  identi == True: 
			continue
		else:
			dl = Test_model.luminosity_distance(z_g).value
			serv_r_mag[ll] = com_Mag[ll] + 5 - 5 * np.log10(dl * 1e6)
	id_zeros = serv_r_mag == 0.
	serv_r_mag[id_zeros] = np.nan

	serv_g_mag = np.zeros(N_tt, dtype = np.float)
	for ll in range(N_tt):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]
		with h5py.File('/home/xkchen/mywork/ICL/data/redmapper/Except_g_sample.h5', 'r') as f:
			except_cat = np.array(f['a'])
		except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
		except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
		if  identi == True: 
			continue
		else:
			dl = Test_model.luminosity_distance(z_g).value
			serv_g_mag[ll] = com_g_Mag[ll] + 5 - 5 * np.log10(dl * 1e6)
	id_zeros = serv_g_mag == 0.
	serv_g_mag[id_zeros] = np.nan

	serv_i_mag = np.zeros(N_tt, dtype = np.float)
	for ll in range(N_tt):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]
		with h5py.File('/home/xkchen/mywork/ICL/data/redmapper/Except_i_sample.h5', 'r') as f:
			except_cat = np.array(f['a'])
		except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
		except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
		if  identi == True: 
			continue
		else:
			dl = Test_model.luminosity_distance(z_g).value
			serv_i_mag[ll] = com_i_Mag[ll] + 5 - 5 * np.log10(dl * 1e6)
	id_zeros = serv_i_mag == 0.
	serv_i_mag[id_zeros] = np.nan

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('PDF of r band Magnitude')
	ax.hist(samp_r_mag, color = 'b', alpha = 0.5, density = True, label = 'sample_20')
	ax.hist(serv_r_mag, color = 'r', alpha = 0.5, density = True, label = 'sample_3000+')
	ax.axvline(x = np.nanmean(samp_r_mag), linestyle = '--', alpha = 0.5, color = 'b')
	ax.axvline(x = np.nanmean(serv_r_mag), linestyle = '--', alpha = 0.5, color = 'r')
	ax.legend(loc = 1)
	ax.set_xlabel('$M_{r} [mag]$')
	ax.set_ylabel('PDF')
	ax.grid(which = 'both', axis = 'both')
	ax.invert_xaxis()
	plt.savefig('sample_r_mag.png', dpi = 300)
	plt.close()

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('PDF of g band Magnitude')
	ax.hist(samp_g_mag, color = 'b', alpha = 0.5, density = True, label = 'sample_20')
	ax.hist(serv_g_mag, color = 'r', alpha = 0.5, density = True, label = 'sample_3000+')
	ax.axvline(x = np.nanmean(samp_g_mag), linestyle = '--', alpha = 0.5, color = 'b')
	ax.axvline(x = np.nanmean(serv_g_mag), linestyle = '--', alpha = 0.5, color = 'r')
	ax.legend(loc = 1)
	ax.set_xlabel('$M_{g} [mag]$')
	ax.set_ylabel('PDF')
	ax.grid(which = 'both', axis = 'both')
	ax.invert_xaxis()
	plt.savefig('sample_g_mag.png', dpi = 300)
	plt.close()

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('PDF of i band Magnitude')
	ax.hist(samp_i_mag, color = 'b', alpha = 0.5, density = True, label = 'sample_20')
	ax.hist(serv_i_mag, color = 'r', alpha = 0.5, density = True, label = 'sample_3000+')
	ax.axvline(x = np.nanmean(samp_i_mag), linestyle = '--', alpha = 0.5, color = 'b')
	ax.axvline(x = np.nanmean(serv_i_mag), linestyle = '--', alpha = 0.5, color = 'r')
	ax.legend(loc = 1)
	ax.set_xlabel('$M_{i} [mag]$')
	ax.set_ylabel('PDF')
	ax.grid(which = 'both', axis = 'both')
	ax.invert_xaxis()
	plt.savefig('sample_i_mag.png', dpi = 300)
	plt.close()
	raise

def pro_compare():
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	idu = (com_Mag >= 17) & (com_Mag <= 18)
	sub_ra, sub_dec, sub_z = ra[idu], dec[idu], z[idu]
	sub_Mag, sub_g_Mag, sub_i_Mag = com_Mag[idu], com_g_Mag[idu], com_i_Mag[idu]
	zN = 20
	raise
	cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
						4.63,  7.43,  11.42,  18.20,  28.20, 
						44.21, 69.00, 107.81, 168.20, 263.00])

	## absolute magnitude
	DL0 = Test_model.luminosity_distance(sub_z).value
	samp_r_mag = sub_Mag[:zN] + 5 - 5 * np.log10(DL0[:zN] * 1e6)
	samp_g_mag = sub_g_Mag[:zN] + 5 - 5 * np.log10(DL0[:zN] * 1e6)
	samp_i_mag = sub_i_Mag[:zN] + 5 - 5 * np.log10(DL0[:zN] * 1e6)
	# sample in sgi server
	N_tt = len(z)
	serv_r_mag = np.zeros(N_tt, dtype = np.float)
	for ll in range(N_tt):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]
		with h5py.File('/home/xkchen/mywork/ICL/data/redmapper/Except_r_sample.h5', 'r') as f:
			except_cat = np.array(f['a'])
		except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
		except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
		if  identi == True: 
			continue
		else:
			dl = Test_model.luminosity_distance(z_g).value
			serv_r_mag[ll] = com_Mag[ll] + 5 - 5 * np.log10(dl * 1e6)
	id_zeros = serv_r_mag == 0.
	serv_r_mag[id_zeros] = np.nan

	serv_g_mag = np.zeros(N_tt, dtype = np.float)
	for ll in range(N_tt):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]
		with h5py.File('/home/xkchen/mywork/ICL/data/redmapper/Except_g_sample.h5', 'r') as f:
			except_cat = np.array(f['a'])
		except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
		except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
		if  identi == True: 
			continue
		else:
			dl = Test_model.luminosity_distance(z_g).value
			serv_g_mag[ll] = com_g_Mag[ll] + 5 - 5 * np.log10(dl * 1e6)
	id_zeros = serv_g_mag == 0.
	serv_g_mag[id_zeros] = np.nan

	serv_i_mag = np.zeros(N_tt, dtype = np.float)
	for ll in range(N_tt):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]
		with h5py.File('/home/xkchen/mywork/ICL/data/redmapper/Except_i_sample.h5', 'r') as f:
			except_cat = np.array(f['a'])
		except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
		except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
		if  identi == True: 
			continue
		else:
			dl = Test_model.luminosity_distance(z_g).value
			serv_i_mag[ll] = com_i_Mag[ll] + 5 - 5 * np.log10(dl * 1e6)
	id_zeros = serv_i_mag == 0.
	serv_i_mag[id_zeros] = np.nan

	## mean of catalogue
	ref_pro = []
	ref_Rii = []
	for q in range(zN):
		ra_g = sub_ra[q]
		dec_g = sub_dec[q]
		z_g = sub_z[q]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		## cat profile of SDSS
		# the band info. 0, 1, 2, 3, 4 --> u, g, r, i, z
		cat_pro = pds.read_csv(
		'/home/xkchen/mywork/ICL/data/BCG_pros/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g), skiprows = 1)
		dat_band = np.array(cat_pro.band)
		dat_bins = np.array(cat_pro.bin)
		dat_pro = np.array(cat_pro.profMean) # in unit of nmaggy / arcsec^2
		dat_pro_err = np.array(cat_pro.profErr)

		#idx = dat_band == 1
		#idx = dat_band == 2
		idx = dat_band == 3
		tt_pro = dat_pro[idx] * ((1 + z_g) / (1 + z_ref))**4
		tt_bin = dat_bins[idx]
		tt_proErr = dat_pro_err[idx] * ((1 + z_g) / (1 + z_ref))**4
		tt_r = (cat_Rii[tt_bin] * Da_g / rad2asec) * 1e3 # arcsec --> kpc

		ref_pro.append(tt_pro)
		ref_Rii.append(tt_r)

	#ref_pro = np.array(ref_pro)
	#ref_Rii = np.array(ref_Rii)
	mm_R = (cat_Rii * Da_ref / rad2asec) * 1e3 # set the radius in z = 0.25
	mm_F = np.zeros((zN, len(cat_Rii)), dtype = np.float)
	for q in range(zN):
		put_pro = ref_pro[q]
		put_r = ref_Rii[q]
		Len_r = len(put_r)

		for pp in range(Len_r):
			dr = np.abs(mm_R - put_r[pp])
			idx = dr == np.nanmin(dr)
			mm_F[q,:][idx] = put_pro[pp] * 1

	id_zero = mm_F == 0.
	mm_F[id_zero == True] = np.nan
	mean_flux = np.nanmean(mm_F, axis = 0)
	std_flux = np.nanstd(mm_F, axis = 0)
	mm_Ns = np.zeros(len(mean_flux), dtype = np.float)
	for q in range(len(cat_Rii)):
		id_nan = np.isnan(mm_F[:, q])
		mm_Ns[q] = len(mm_F[id_nan == False])
	std_flux = std_flux / np.sqrt(mm_Ns)

	ref_SB = 22.5 - 2.5 * np.log10(mean_flux)
	SB0 = 22.5 - 2.5 * np.log10(mean_flux + std_flux)
	SB1 = 22.5 - 2.5 * np.log10(mean_flux - std_flux)
	err0 = ref_SB - SB0
	err1 = SB1 - ref_SB
	id_nan = np.isnan(ref_SB)
	ref_SB, SB0, SB1 = ref_SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False]
	ref_R, err0, err1 = mm_R[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.
	ref_err0, ref_err1 = err0 * 1, err1 * 1

	## mean of pipe. measured
	bins = 55
	sub_rr = np.logspace(0, np.log10(200), bins)
	sub_pro = np.zeros((zN, bins), dtype = np.float)
	sub_Rii = np.zeros((zN, bins), dtype = np.float)
	for q in range(zN):
		ra_g = sub_ra[q]
		dec_g = sub_dec[q]
		z_g = sub_z[q]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		'''
		#file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('g', ra_g, dec_g, z_g) ## data
		#data = fits.open(load + file)

		file = 'mask/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % ('g', ra_g, dec_g, z_g)
		data = fits.open(tmp + file)
		img = data[0].data
		wcs = awc.WCS(data[0].header)
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		R_smal, R_max = 1, 200
		Intns, Intns_r, Intns_err, Npix = light_measure(img, bins, R_smal, R_max, cx, cy, pixel, z_g)

		tt_array = np.array([Intns, Intns_r, Intns_err])
		with h5py.File(tmp + 'tmp/SB_pro_%s_ra%.3f_dec%.3f_z%.3f.h5' % ('g', ra_g, dec_g, z_g), 'w') as f:
			f['a'] = np.array(tt_array)
		with h5py.File(tmp + 'tmp/SB_pro_%s_ra%.3f_dec%.3f_z%.3f.h5' % ('g', ra_g, dec_g, z_g) ) as f:
			for tt in range(len(tt_array)):
				f['a'][tt,:] = tt_array[tt,:]
		'''
		#with h5py.File(tmp + 'tmp/SB_pro_%s_ra%.3f_dec%.3f_z%.3f.h5' % ('g', ra_g, dec_g, z_g), 'r') as f:
		#with h5py.File(tmp + 'tmp/SB_pro_%s_ra%.3f_dec%.3f_z%.3f.h5' % ('r', ra_g, dec_g, z_g), 'r') as f:
		with h5py.File(tmp + 'tmp/SB_pro_%s_ra%.3f_dec%.3f_z%.3f.h5' % ('i', ra_g, dec_g, z_g), 'r') as f:
			tt_array = np.array(f['a'])
		Intns, Intns_r, Intns_err = tt_array[0,:], tt_array[1,:], tt_array[2,:]

		Len_r = len(Intns_r)
		for kk in range(Len_r):
			dr = np.abs(sub_rr - Intns_r[kk])
			idr = dr == np.nanmin(dr)
			sub_pro[q,:][idr] = (Intns[kk] / pixel**2) * ((1 + z_g) / (1 + z_ref))**4
			sub_Rii[q,:][idr] = Intns_r[kk]

	id_zeros = sub_pro == 0.
	sub_pro[id_zeros] = np.nan
	sub_Rii[id_zeros] = np.nan

	mm_pro = np.nanmean(sub_pro, axis = 0)
	sub_std = np.nanstd(sub_pro, axis = 0)
	mm_Rii = np.nanmean(sub_Rii, axis = 0)

	mm_Nt = np.zeros(bins, dtype = np.float)
	for q in range(bins):
		id_nan = np.isnan(sub_pro[:, q])
		mm_Nt = len(sub_pro[id_nan == False])
	sub_std = sub_std / np.sqrt(mm_Nt)

	pipe_SB = 22.5 - 2.5 * np.log10(mm_pro)
	SB0 = 22.5 - 2.5 * np.log10(mm_pro + sub_std)
	SB1 = 22.5 - 2.5 * np.log10(mm_pro - sub_std)
	err0 = pipe_SB - SB0
	err1 = SB1 - pipe_SB
	id_nan = np.isnan(pipe_SB)
	Min_SB, SB0, SB1 = pipe_SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False]
	Min_R, err0, err1 = mm_Rii[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.
	Min_err0, Min_err1 = err0 * 1, err1 * 1

	## for mask image, measure the profile from image
	#with h5py.File('/home/xkchen/mywork/ICL/data/test_data/Mean_img_Amask_g_band.h5', 'r') as f:
	#with h5py.File('/home/xkchen/mywork/ICL/data/test_data/Mean_img_Amask_r_band.h5', 'r') as f:
	with h5py.File('/home/xkchen/mywork/ICL/data/test_data/Mean_img_Amask_i_band.h5', 'r') as f:
		A_stack = np.array(f['a'])

	R_cut = 900
	x0, y0 = 2427, 1765
	R_smal, R_max = 1, 200 # kpc
	ss_img = A_stack[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
	Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)

	SB_obs = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(Intns + Intns_err) + 2.5 *np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(Intns - Intns_err) + 2.5 *np.log10(pixel**2)
	id_nan = np.isnan(SB_obs)
	SB_obs, R_obs = SB_obs[id_nan == False], Intns_r[id_nan == False]
	SB0, SB1 = SB0[id_nan == False], SB1[id_nan == False]
	err0 = SB_obs - SB0
	err1 = SB1 - SB_obs
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100.
	obs_err0, obs_err1 = err0 * 1, err1 * 1

	## Zibetti 2005 result
	#SB_Z05 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/g_band_BCG_ICL.csv')
	#SB_Z05 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_BCG_ICL.csv')
	SB_Z05 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/i_band_BCG_ICL.csv')
	R_tt, SB_tt = SB_Z05['(1000R)^(1/4)'], SB_Z05['mag/arcsec^2']
	R_tt = R_tt**4

	## stack 3000+ sample
	rpro_R, SB_r_pro, gpro_R, SB_g_pro, ipro_R, SB_i_pro = Z05_compare()

	plt.figure()
	ax = plt.subplot(111)
	#ax.set_title('Mean SB profile in g band')
	#ax.set_title('Mean SB profile in r band')
	ax.set_title('Mean SB profile in i band')

	ax.errorbar(ref_R, ref_SB, yerr = [ref_err0, ref_err1], xerr = None, ls = '--', fmt = 'k.', label = 'Mean of SDSS cat_pro', alpha = 0.5)
	ax.errorbar(Min_R, Min_SB, yerr = [Min_err0, Min_err1], xerr = None, ls = '--', fmt = 'b.', label = 'Mean of pipe_pro', alpha = 0.5)
	ax.errorbar(R_obs, SB_obs, yerr = [obs_err0, obs_err1], xerr = None, ls = '--', fmt = 'r.', label = 'stack img [20]', alpha = 0.5)
	ax.plot(R_tt, SB_tt, 'g:', label = 'Profile of Z05', alpha = 0.5)

	#ax.plot(gpro_R, SB_g_pro, 'm--', label = 'stack img [3000]', alpha = 0.5)
	#ax.plot(rpro_R, SB_r_pro, 'm--', label = 'stack img [3000]', alpha = 0.5)
	ax.plot(ipro_R, SB_i_pro, 'm--', label = 'stack img [3000]', alpha = 0.5)
	
	#ax.text(2, 27, s = '$ \overline{M} [20] = %.3f $' % np.nanmean(samp_g_mag), color = 'r')
	#ax.text(2, 26, s = '$ \overline{M} [3000] = %.3f $' % np.nanmean(serv_g_mag), color = 'm')

	#ax.text(2, 27, s = '$ \overline{M} [20] = %.3f$' % np.nanmean(samp_r_mag), color = 'r')
	#ax.text(2, 26, s = '$ \overline{M} [3000] = %.3f$' % np.nanmean(serv_r_mag), color = 'm')

	ax.text(2, 27, s = '$ \overline{M} [20] = %.3f$' % np.nanmean(samp_i_mag), color = 'r')
	ax.text(2, 26, s = '$ \overline{M} [3000] = %.3f$' % np.nanmean(serv_i_mag), color = 'm')

	ax.set_xscale('log')
	ax.set_xlabel('k[kpc]')
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.set_ylim(19, 28)
	ax.set_xlim(1, 200)
	ax.legend(loc = 1)
	ax.invert_yaxis()

	#plt.savefig('Mean_BCG_pros_g_band.png', dpi = 300)
	#plt.savefig('Mean_BCG_pros_r_band.png', dpi = 300)
	plt.savefig('Mean_BCG_pros_i_band.png', dpi = 300)
	plt.close()

	'''
	fig = plt.figure(figsize = (20,20))
	fig.suptitle('BCG pros')
	gs = gridspec.GridSpec(zN // 5, 5)
	## single cluster compare
	for q in range(zN):
		ra_g = sub_ra[q]
		dec_g = sub_dec[q]
		z_g = sub_z[q]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		#file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('g', ra_g, dec_g, z_g) ## data
		#data = fits.open(load + file)

		file = 'mask/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % ('g', ra_g, dec_g, z_g)
		data = fits.open(tmp + file)

		img = data[0].data
		wcs = awc.WCS(data[0].header)
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		## measure the BCG magnitude
		#R_smal, R_max, bins = 1, 200, 55
		#Intns, Intns_r, Intns_err, Npix = light_measure(img, bins, R_smal, R_max, cx, cy, pixel, z_g)

		with h5py.File(tmp + 'tmp/SB_pro_%s_ra%.3f_dec%.3f_z%.3f.h5' % ('g', ra_g, dec_g, z_g), 'r') as f:
			tt_array = np.array(f['a'])
		Intns, Intns_r, Intns_err = tt_array[0,:], tt_array[1,:], tt_array[2,:]

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
		cat_pro = pds.read_csv(
		'/home/xkchen/mywork/ICL/data/BCG_pros/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g), skiprows = 1)
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
		ax.set_title('z = %.3f' % z_g)
		ax.errorbar(bin_R, SB_img, yerr = [img_err0, img_err1], xerr = None, ls = '-', fmt = 'g.', label = 'Me pipe', alpha = 0.5)
		ax.errorbar(cc_r, cc_SB, yerr = [cc_err0, cc_err1], xerr = None, ls = '-', fmt = 'r.', label = 'SDSS cat.', alpha = 0.5)
		ax.errorbar(ref_R, ref_SB, yerr = [ref_err0, ref_err1], xerr = None, ls = '--', fmt = 'k.', label = 'Stack SDSS cat_pro[z=0.25]', alpha = 0.5)
		ax.errorbar(Min_R, Min_SB, yerr = [Min_err0, Min_err1], xerr = None, ls = '--', fmt = 'b.', label = 'Stack Me pipe_pro[z=0.25]', alpha = 0.5)

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
	'''
	raise

def extin_pro():
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	idu = (com_Mag >= 17) & (com_Mag <= 18)

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	sub_ra = ra[idu]
	sub_dec = dec[idu]
	sub_z = z[idu]
	zN = 20
	cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
						4.63,  7.43,  11.42,  18.20,  28.20, 
						44.21, 69.00, 107.81, 168.20, 263.00])

	R_smal, R_max, bins = 1, 200, 55

	plt.figure(figsize = (20,20))
	gs = gridspec.GridSpec(zN // 5, 5)

	for kk in range(zN):
		ra_g = sub_ra[kk]
		dec_g = sub_dec[kk]
		z_g = sub_z[kk]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		## original image
		file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('g', ra_g, dec_g, z_g) ## data
		data = fits.open(load + file)
		img = data[0].data
		wcs = awc.WCS(data[0].header)
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		Intns, Intns_r, Intns_err, Npix = light_measure(img, bins, R_smal, R_max, cx, cy, pixel, z_g)
		id_nan = np.isnan(Intns)
		flux = Intns[id_nan == False]
		flux_err = Intns_err[id_nan == False]
		sub_R0 = Intns_r[id_nan == False]
		flux0 = flux + flux_err
		flux1 = flux - flux_err

		sub_SB0 = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = sub_SB0 - SB0
		err1 = SB1 - sub_SB0
		id_nan = np.isnan(SB1)
		err1[id_nan] = 100.
		sub0_err0 = err0 * 1
		sub0_err1 = err1 * 1

		## SFD 1998
		ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		BEV_98 = sfd(pos)
		Av_98 = Rv * BEV_98
		Al = A_wave(l_wave[1], Rv) * Av_98
		c0_img = img * 10**(Al / 2.5)

		Intns, Intns_r, Intns_err, Npix = light_measure(c0_img, bins, R_smal, R_max, cx, cy, pixel, z_g)
		id_nan = np.isnan(Intns)
		flux = Intns[id_nan == False]
		flux_err = Intns_err[id_nan == False]
		sub_R1 = Intns_r[id_nan == False]
		flux0 = flux + flux_err
		flux1 = flux - flux_err

		sub_SB1 = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = sub_SB1 - SB0
		err1 = SB1 - sub_SB1
		id_nan = np.isnan(SB1)
		err1[id_nan] = 100.
		sub1_err0 = err0 * 1
		sub1_err1 = err1 * 1

		## SFD 2011
		BEV_11 = sfd(pos) * 0.86
		Av_11 = Rv * BEV_11
		Al = A_wave(l_wave[1], Rv) * Av_11
		c1_img = img * 10**(Al / 2.5)

		Intns, Intns_r, Intns_err, Npix = light_measure(c1_img, bins, R_smal, R_max, cx, cy, pixel, z_g)
		id_nan = np.isnan(Intns)
		flux = Intns[id_nan == False]
		flux_err = Intns_err[id_nan == False]
		sub_R2 = Intns_r[id_nan == False]
		flux0 = flux + flux_err
		flux1 = flux - flux_err

		sub_SB2 = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = sub_SB2 - SB0
		err1 = SB1 - sub_SB2
		id_nan = np.isnan(SB1)
		err1[id_nan] = 100.
		sub2_err0 = err0 * 1
		sub2_err1 = err1 * 1

		## SDSS catalogue
		cat_pro = pds.read_csv(
		'/home/xkchen/mywork/ICL/data/BCG_pros/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g), skiprows = 1)
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

		ax = plt.subplot(gs[kk // 5, kk % 5])
		ax.errorbar(sub_R0, sub_SB0, yerr = [sub0_err0, sub0_err1], xerr = None, ls = '-', fmt = 'r.', label = 'No correction', alpha = 0.5)
		ax.errorbar(sub_R1, sub_SB1, yerr = [sub1_err0, sub1_err1], xerr = None, ls = '-', fmt = 'g.', label = 'SFD 1998', alpha = 0.5)
		ax.errorbar(sub_R2, sub_SB2, yerr = [sub2_err0, sub2_err1], xerr = None, ls = '-', fmt = 'b.', label = 'SFD 2011', alpha = 0.5)
		ax.errorbar(cc_r, cc_SB, yerr = [cc_err0, cc_err1], xerr = None, ls = '--', fmt = 'k.', label = 'SDSS cat.', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('k[kpc]')
		ax.set_ylabel('SB[mag/arcsec^2]')
		ax.set_ylim(20, 28)
		ax.set_xlim(1, 200)
		ax.legend(loc = 1)
		ax.invert_yaxis()

	plt.tight_layout()
	plt.savefig('Extinct_BCG_pros_g_band.png', dpi = 300)
	plt.close()

	raise

def main():
	#Z05_compare()
	#sngl_pro()
	pro_compare()
	#extin_pro()

if __name__ == "__main__":
	main()
