import astropy.units as U
import astropy.constants as C

import h5py
import time
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits
import statistics as sts

from scipy import interpolate as interp
from astropy import cosmology as apcy

## constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Lsun = C.L_sun.value*10**7
G = C.G.value

## cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

## profile catalogue [in unit of 'arcsec']
cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00]) # in unit 'arcsec'
## the band info. of SDSS BCG pro. : 0, 1, 2, 3, 4 --> u, g, r, i, z

def cumula_flux(angl_r, bin_fdens,):
	N_bin = len( angl_r )
	flux_arr = np.zeros( N_bin, dtype = np.float32)

	for kk in range( N_bin ):

		if kk == 0:
			cfi = np.pi * angl_r[kk]**2 * bin_fdens[kk]
			flux_arr[kk] = cfi + flux_arr[kk]

		else:
			tps = kk + 0
			cfi = 0
			while tps > 0:
				cfi = cfi + np.pi * (angl_r[tps]**2 - angl_r[tps-1]**2) * bin_fdens[tps]
				tps = tps - 1

			cfi = cfi + np.pi * angl_r[0]**2 * bin_fdens[0]

			flux_arr[kk] = cfi + flux_arr[kk]

	return flux_arr

def fdens_deriv(r_angle, obs_r, obs_fmean, ):

	cumu_f = cumula_flux(obs_r, obs_fmean)

	asinh_x = np.log(obs_r + np.sqrt(obs_r**2 + 1) )
	asinh_f = np.log(cumu_f + np.sqrt(cumu_f**2 + 1) )

	fit_func = interp.splrep(asinh_x, asinh_f, s = 0)

	new_x = np.log(r_angle + np.sqrt(r_angle**2 + 1) )
	new_f = interp.splev(new_x, fit_func, der = 0)

	deri_f = interp.splev(new_x, fit_func, der = 1)

	dy_f = ( np.exp(new_f) + np.exp(-1 * new_f) ) / 2
	dx_f = 1 / np.sqrt( r_angle**2 + 1)

	sb_f = deri_f * dy_f * dx_f / (2 * np.pi * r_angle)
	'''
	fit_func = interp.splrep(obs_r, cumu_f, s = 0)
	new_x = r_angle
	new_f = interp.splev(new_x, fit_func, der = 1)

	sb_f = new_f / (2 * np.pi * r_angle)
	'''
	return sb_f

def BCG_SB_pros_func(band_id, set_z, set_ra, set_dec, pros_file, z_ref, out_file, r_bins,):

	if band_id == 0:
		pro_id = 2
	if band_id == 1:
		pro_id = 1
	if band_id == 2:
		pro_id = 3

	zn = len(set_z)
	fdens_arr = np.zeros((zn, len(r_bins) ), dtype = np.float32) + np.nan

	for tt in range( zn ):

		z_g, ra_g, dec_g = set_z[tt], set_ra[tt], set_dec[tt]

		Da_g = Test_model.angular_diameter_distance(z_g).value
		r_angl = (r_bins * 1e-3) / Da_g * rad2asec

		cat_pro = pds.read_csv( pros_file % (z_g, ra_g, dec_g), skiprows = 1)

		dat_band = np.array(cat_pro.band)
		dat_bins = np.array(cat_pro.bin)
		dat_pro = np.array(cat_pro.profMean) # in unit of nmaggy / arcsec^2
		dat_pro_err = np.array(cat_pro.profErr)

		idx = dat_band == pro_id
		tt_pro = dat_pro[idx]
		tt_proErr = dat_pro_err[idx]
		tt_bin = dat_bins[idx]
		tt_r = cat_Rii[tt_bin]

		id_lim = r_angl <= tt_r.max()
		use_angl_r = r_angl[ id_lim ]
		fdens = fdens_deriv( use_angl_r, tt_r, tt_pro,)

		fdens = fdens * ( (1 + z_g) / (1 + z_ref) )**4
		fdens_arr[tt][id_lim] = fdens

	m_fdens = np.nanmean( fdens_arr, axis = 0 )
	std_fdens = np.nanstd( fdens_arr, axis = 0 )

	SB_pro = 22.5 - 2.5 * np.log10(m_fdens) + mag_add[ band_id ]

	SB0 = 22.5 - 2.5 * np.log10(m_fdens + std_fdens) + mag_add[band_id]
	SB1 = 22.5 - 2.5 * np.log10(m_fdens - std_fdens) + mag_add[band_id]
	err0 = SB_pro - SB0
	err1 = SB1 - SB_pro

	keys = ['R_ref', 'SB_mag', 'SB_mag_err0', 'SB_mag_err1', 'SB_fdens', 'SB_fdens_err']
	values = [r_bins, SB_pro, err0, err1, m_fdens, std_fdens]

	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

