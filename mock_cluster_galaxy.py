import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import skimage
import numpy as np
import pandas as pds
from scipy import interpolate as interp

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy.optimize import curve_fit, minimize
from ICL_surface_mass_density import sigma_m_c
from light_measure_tmp import light_measure, flux_recal
from resample_modelu import down_samp, sum_samp

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
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*Jy # zero point in unit (erg/s)/cm^-2
Lstar = 2e10 # in unit L_sun ("copy from Galaxies in the Universe", use for Luminosity function calculation)

# observation catalogue
with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
load = '/home/xkchen/mywork/ICL/data/mock_frame/'
# divide by z == 0.25
set_z0, set_z1 = z[ z > 0.25], z[ z < 0.25]
ra_z0, ra_z1 = ra[ z > 0.25], ra[ z < 0.25]
dec_z0, dec_z1 = dec[ z > 0.25], dec[ z < 0.25]

set_z = np.r_[ set_z0[:10], set_z1[:10] ]
set_ra = np.r_[ ra_z0[:10], ra_z1[:10] ]
set_dec = np.r_[ dec_z0[:10], dec_z1[:10] ]

def lumi_fun(M_arr, Mag_c, alpha_c, phi_c):
	M_c = Mag_c 
	alpha = alpha_c
	phi = phi_c

	M_arr = M_arr
	X = 10**(-0.4 * (M_arr - M_c))
	lumi_f = phi * X**(alpha + 1) * np.exp(-1 * X)
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(M_arr, np.log10(lumi_f), 'r-')
	ax.set_title('galaxy luminosity function')
	ax.set_xlabel('$M_r - 5 log h$')
	ax.set_ylabel('$log(\phi) \, h^2 Mpc^{-2}$')
	ax.text(-17, 0, s = '$Mobasher \, et \, al.2003$' + '\n' + 'for Coma cluster')
	ax.invert_xaxis()
	plt.savefig('/home/xkchen/mywork/ICL/code/mock_lumifun.png', dpi = 300)
	plt.close()
	'''
	dM = M_arr[1:] - M_arr[:-1]
	mid_M = 0.5 * (M_arr[1:] + M_arr[:-1])
	Acum_l = np.zeros(len(dM), dtype = np.float)
	for kk in range(1, len(dM) + 1):
		sub_x = mid_M[:kk] * dM[:kk]
		Acum_l[kk - 1] = np.sum(sub_x)
	Acum_l = Acum_l / Acum_l[-1]
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(mid_M, np.log10(Acum_l), 'r-')
	ax.set_title('integration of luminosity function')
	ax.set_xlabel('$M_r - 5 log h$')
	ax.set_ylabel('$log(\phi \, dM) \, h^2 Mpc^{-2}$')
	ax.invert_xaxis()
	plt.savefig('/home/xkchen/mywork/ICL/code/acum_lumifun.png', dpi = 300)
	plt.close()
	'''
	L_set = interp.interp1d(Acum_l, mid_M)
	new_P = np.linspace(np.min(Acum_l), np.max(Acum_l), len(M_arr))
	new_M = L_set(new_P)
	gal_L = 10**((M_c - new_M) / 2.5) * Lstar #mock galaxy luminosity, in unit of L_sun

	return gal_L

def galaxy():
	ng = 500
	m_set = np.linspace(-23, -16, ng)
	Mag_c = -21.79
	alpha_c = -1.18
	phi_c = 9.5 # in unit "h^2 Mpc^-2"
	ly = lumi_fun(m_set, Mag_c, alpha_c, phi_c) # in unit "L_sun"
	raise
	return

def SB_fit(r, mu_e0, mu_e1, mu_e2, r_e0, r_e1, r_e2, ndex0, ndex1, ndex2):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	mock_SB0 = mu_e0 + 1.086 * (2 * ndex0 - 0.324) * ( (r / r_e0)**(1 / ndex0) - 1) # in unit mag/arcsec^2
	mock_SB1 = mu_e1 + 1.086 * (2 * ndex1 - 0.324) * ( (r / r_e1)**(1 / ndex1) - 1)
	mock_SB2 = mu_e2 + 1.086 * (2 * ndex2 - 0.324) * ( (r / r_e2)**(1 / ndex2) - 1)
	f_SB0 = 10**( (22.5 - mock_SB0 + 2.5 * np.log10(pixel**2)) / 2.5 ) # in unit nmaggy
	f_SB1 = 10**( (22.5 - mock_SB1 + 2.5 * np.log10(pixel**2)) / 2.5 )
	f_SB2 = 10**( (22.5 - mock_SB2 + 2.5 * np.log10(pixel**2)) / 2.5 )
	mock_SB = 22.5 - 2.5 * np.log10(f_SB0 + f_SB1 + f_SB2) + 2.5 * np.log10(pixel**2)
	return mock_SB

def SB_dit(r, mu_e0, mu_e1, mu_e2, r_e0, r_e1, r_e2, ndex0, ndex1, ndex2):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	mock_SB0 = mu_e0 + 1.086 * (2 * ndex0 - 0.324) * ( (r / r_e0)**(1 / ndex0) - 1) # in unit mag/arcsec^2
	mock_SB1 = mu_e1 + 1.086 * (2 * ndex1 - 0.324) * ( (r / r_e1)**(1 / ndex1) - 1)
	mock_SB2 = mu_e2 + 1.086 * (2 * ndex2 - 0.324) * ( (r / r_e2)**(1 / ndex2) - 1)
	f_SB0 = 10**( (22.5 - mock_SB0 + 2.5 * np.log10(pixel**2)) / 2.5 ) # in unit nmaggy
	f_SB1 = 10**( (22.5 - mock_SB1 + 2.5 * np.log10(pixel**2)) / 2.5 )
	f_SB2 = 10**( (22.5 - mock_SB2 + 2.5 * np.log10(pixel**2)) / 2.5 )
	mock_SB = 22.5 - 2.5 * np.log10(f_SB0 + f_SB1 + f_SB2) + 2.5 * np.log10(pixel**2)
	return mock_SB0, mock_SB1, mock_SB2, mock_SB

def SB_pro():
	SB0 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_1.csv', skiprows = 1)
	R_r = SB0['Mpc']
	SB_r0 = SB0['mag/arcsec^2'] # Intrinsic SB

	SB1 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_2.csv', skiprows = 1)
	SB_r1 = SB1['mag/arcsec^2'] # Intrinsic SB + residual sky

	SB2 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_3.csv', skiprows = 1)
	SB_r2 = SB2['mag/arcsec^2'] # total component [BCG + ICL]

	## fit the profile
	mu_e0, mu_e1, mu_e2 = 23.87, 30, 20 # mag/arcsec^2
	Re_0, Re_1, Re_2 = 19.29, 120, 10 # kpc
	ndex0, ndex1, ndex2 = 4., 4., 4.

	r_fit = R_r * 10**3
	po = np.array([mu_e0, mu_e1, mu_e2, Re_0, Re_1, Re_2, ndex0, ndex1, ndex2])
	popt, pcov = curve_fit(SB_fit, r_fit, SB_r0, p0 = po, 
			bounds = ([21, 27, 18, 18, 100, 9, 1., 1., 1.], [24, 32, 21, 22, 500, 18, 6., 12., 4.]), method = 'trf')
	mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2 = popt
	mock_SB0, mock_SB1, mock_SB2, mock_SB = SB_dit(r_fit, mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2)

	'''
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Mock SB')
	ax.plot(r_fit, mock_SB0, 'r--', alpha = 0.5)
	ax.plot(r_fit, mock_SB1, 'g--', alpha = 0.5)
	ax.plot(r_fit, mock_SB2, 'm--', alpha = 0.5)
	ax.plot(r_fit, mock_SB, 'b-', label = 'Mock', alpha = 0.5)
	ax.plot(r_fit, SB_r0, 'r^', label = 'Z 2005', alpha = 0.5)
	ax.legend(loc = 1)
	ax.set_xscale('log')
	ax.set_ylim(20, 34)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	ax.text(500, 24, s = '$ \mu_{e} = %.2f$' % mu_fit0 + '\n' + '$R_{e} = %.2f$' % re_fit0 + '\n' + '$n = %.2f$' % ndex_fit0, color = 'r')
	ax.text(500, 26, s = '$ \mu_{e} = %.2f$' % mu_fit1 + '\n' + '$R_{e} = %.2f$' % re_fit1 + '\n' + '$n = %.2f$' % ndex_fit1, color = 'g')
	ax.text(500, 28, s = '$ \mu_{e} = %.2f$' % mu_fit2 + '\n' + '$R_{e} = %.2f$' % re_fit2 + '\n' + '$n = %.2f$' % ndex_fit2, color = 'm')

	ax.set_xlabel('$ R[kpc] $')
	ax.set_ylabel('$ SB[mag/arcsec^2] $')
	plt.savefig('mock_SB_r.png', dpi = 300)
	plt.close()
	'''
	r = np.logspace(0, 3.08, 1000)
	r_sc = r / 10**3
	r_max = np.max(r_sc)
	r_min = np.min(r_sc)

	SB_r = SB_fit(r, mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2) # profile at z = 0.25
	#SB_r = SB_fit(r_fit, mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2)

	## change the SB_r into counts / s
	NMGY = 5e-3 # mean value of the data sample
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	exp_time = 54 # exposure time, in unit second
	DN = 10**( (22.5 - SB_r + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY
	## error
	err_N = np.sqrt( DN / gain )

	'''
	## save the SB_ref, and mock SB for different redshift
	sub_SB = np.zeros( (len(set_z), len(r)), dtype = np.float)
	sub_flux = np.zeros( (len(set_z), len(r)), dtype = np.float)
	for kk in range( len(set_z) ):
		ttx = SB_r - 10 * np.log10((1 + z_ref) / (1 + set_z[kk]))
		sub_SB[kk, :] = ttx * 1
		sub_flux[kk, :] = 10**( (22.5 - ttx + 2.5*np.log10(pixel**2) ) / 2.5 )

	keys = ['%.3f' % ll for ll in set_z]
	keys.append('r')
	values = []
	[ values.append(ll) for ll in sub_flux ]
	values.append(r)
	fill = dict( zip(keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv(load + 'mock_flux_data.csv')

	keys = ['%.3f' % ll for ll in set_z]
	keys.append('r')
	values = []
	[ values.append(ll) for ll in sub_SB ]
	values.append(r)
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(load + 'mock_SB_data.csv')

	key0 = ['%.3f' % z_ref, 'r']
	value0 = [SB_r, r]
	fill0 = dict(zip(key0, value0))
	data = pds.DataFrame(fill0)
	data.to_csv(load + 'mock_intrinsic_SB.csv')
	'''
	## add sky
	sky = 21. # mag/arcsec^2 (from SDSS dr14: the image quality)
	N_sky = 10**( (22.5 - sky + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY
	err_sky = np.sqrt( N_sky / gain )
	N_tot =  DN + N_sky
	err_tot = np.sqrt( N_tot / gain + err_sky**2 )
	mock_SBt = 22.5 - 2.5 * np.log10(N_tot * NMGY) + 2.5 * np.log10(pixel**2)

	'''
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Mock SB')
	ax.plot(r_fit, mock_SBt, 'g--', label = 'Mock + sky', alpha = 0.5)
	ax.plot(r_fit, mock_SB, 'b-', label = 'Mock', alpha = 0.5)
	ax.plot(r_fit, SB_r0, 'r^', label = 'Z 2005', alpha = 0.5)
	ax.axhline(y = sky, linestyle = '--', color = 'k', label = 'sky')
	ax.legend(loc = 3)
	ax.set_xscale('log')
	ax.set_ylim(20, 34)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_xlabel('$ R[kpc] $')
	ax.set_ylabel('$ SB[mag/arcsec^2] $')
	plt.savefig('mock_SB_sky_r.png', dpi = 300)
	plt.show()

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('counts profile')
	ax.plot(r_fit, DN, 'r--', label = 'BCG + ICL', alpha = 0.5)
	ax.plot(r_fit, N_tot, 'b-', label = 'BCG + ICL + sky', alpha = 0.5)
	#ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_ylabel('DN')
	ax.set_xlabel('R[kpc]')
	ax.legend(loc = 1)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.savefig('DN_tot_profile.png', dpi = 300)
	plt.close()

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('counts error profile')
	ax.plot(r_fit, err_N, 'r--', label = 'BCG + ICL', alpha = 0.5)
	ax.plot(r_fit, err_tot, 'b-', label = 'BCG + ICL + sky', alpha = 0.5)
	#ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_ylabel('DN')
	ax.set_xlabel('R[kpc]')
	ax.legend(loc = 1)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.savefig('DNtot_err_profile.png', dpi = 300)	
	plt.close()
	'''

	f_DN = interp.interp1d(r_sc, DN, kind = 'cubic')
	## apply DN to CCD
	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	frame = np.zeros((len(y0), len(x0)), dtype = np.float)
	Nois = np.zeros((len(y0), len(x0)), dtype = np.float)
	pxl = np.meshgrid(x0, y0)

	xc, yc = 1025, 745
	dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
	dr_sc = dr / Rpp
	DN_min = np.min(DN)
	for kk in range(dr_sc.shape[0]):
		for jj in range(dr_sc.shape[1]):
			if (dr_sc[kk, jj] >= r_max ) | (dr_sc[kk, jj] <= r_min ):
				lam_x = DN_min * gain / 10
				N_e = lam_x + N_sky * gain
				rand_x = np.random.poisson( N_e )
				frame[kk, jj] += lam_x # electrons number
				Nois[kk, jj] += rand_x
			else:
				lam_x = f_DN( dr_sc[kk, jj] ) * gain
				N_e = lam_x + N_sky * gain
				rand_x = np.random.poisson( N_e )
				frame[kk, jj] += lam_x # electrons number
				Nois[kk, jj] += rand_x

	N_mock = frame / gain
	N_ele = (frame + N_sky * gain) / gain
	N_sub = Nois / gain - N_sky
	Noise = N_mock - N_sub
	# change N_sub to flux in unit 'nmaggy'
	bins = 65
	N_flux = N_sub * NMGY
	Intns, Intns_r, Intns_err, Npix = light_measure(N_flux, bins, 10, Rpp, xc, yc, pixel, z_ref)
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	cc_err0 = Intns_err * 1.
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB - SB0
	err1 = SB1 - SB
	id_nan = np.isnan(SB)
	SB, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
	pR, err0, err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	id_nan = np.isnan(SB1)
	err1[id_nan] = 100. # set a large value for show the break out errorbar

	N_mooth = N_mock * NMGY
	Intns, Intns_r, Intns_err, Npix = light_measure(N_mooth, bins, 10, Rpp, xc, yc, pixel, z_ref)
	## for smooth image, the err should be the Poisson Noise
	f_err = np.sqrt( (Intns / NMGY) / (Npix * gain) + 2 * N_sky / (gain * Npix) ) * NMGY

	cc_err1 = f_err * 1.
	flux0 = Intns + f_err
	flux1 = Intns - f_err
	SB_mooth = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0_mooth = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1_mooth = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0_mooth = SB_mooth - SB0_mooth
	err1_mooth = SB1_mooth - SB_mooth
	id_nan = np.isnan(SB_mooth)
	SB_mooth, SB0_mooth, SB1_mooth = SB_mooth[id_nan == False], SB0_mooth[id_nan == False], SB1_mooth[id_nan == False]
	pR_mooth, err0_mooth, err1_mooth = Intns_r[id_nan == False], err0_mooth[id_nan == False], err1_mooth[id_nan == False]
	id_nan = np.isnan(SB1_mooth)
	err1_mooth[id_nan] = 100.

	eta = cc_err0 / cc_err1
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(Intns_r, eta, 'b-', label = 'Z05 / Poisson', alpha = 0.5)
	ax.set_xlim(9, 1010)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylabel('$ err_{Z05} / err_{poisson} \, [in \; flux \; term] $')
	ax.legend(loc = 1)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.savefig('err_compared.png', dpi = 300)
	plt.show()
	raise
	'''
	plt.figure( figsize = (16, 9) )
	ax0 = plt.subplot(221)
	ax1 = plt.subplot(222)
	ax2 = plt.subplot(223)
	ax3 = plt.subplot(224)

	ax0.set_title('intrinsic SB')
	tf = ax0.imshow(N_mock, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'DN')

	ax1.set_title('intrinsic SB + sky')
	tf = ax1.imshow(N_ele, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'DN')

	ax2.set_title('intrinsic SB + sky + noise')
	tf = ax2.imshow(Nois / gain, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = 'DN')

	ax3.set_title('sky subtracted')
	tf = ax3.imshow(N_sub, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax3, fraction = 0.035, pad = 0.01, label = 'DN')

	plt.tight_layout()
	plt.savefig('mock_image.png', dpi = 300)
	plt.close()
	'''
	plt.figure( )
	bx = plt.subplot(111)

	bx.errorbar(pR, SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'ro', label = ' noise image [sky subtracted] ', alpha = 0.5)
	bx.errorbar(pR_mooth, SB_mooth, yerr = [err0_mooth, err1_mooth], xerr = None, ls = '', fmt = 'gs', label = ' smooth image ', alpha = 0.5)
	bx.plot(r, SB_r, 'b-', label = ' intrinsic SB ', alpha = 0.5)
	bx.set_xscale('log')
	bx.set_xlabel('R [kpc]')
	bx.set_xlim(9, 1010)
	bx.set_ylim(19, 34)
	bx.invert_yaxis()
	bx.set_ylabel('$ SB[mag / arcsec^2] $')
	bx.legend(loc = 1)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')

	bx1 = bx.twiny()
	xtik = bx.get_xticks()
	xtik = np.array(xtik)
	xR = xtik * 10**(-3) * rad2asec / Da_ref
	bx1.set_xscale('log')
	bx1.set_xticks(xtik)
	bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
	bx1.set_xlim(bx.get_xlim())
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.savefig('SB_measure_test.png', dpi = 300)
	plt.close()

	raise

def mock_ccd():
	NMGY = 5e-3 # mean value of the data sample
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	exp_time = 54 # exposure time, in unit second
	sky = 21. # mag/arcsec^2 (from SDSS dr14: the image quality)
	N_sky = 10**( (22.5 - sky + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY

	mock_flux = pds.read_csv(load + 'mock_flux_data.csv')
	mock_SB = pds.read_csv(load + 'mock_SB_data.csv')
	ins_SB = pds.read_csv(load + 'mock_intrinsic_SB.csv')
	r = ins_SB['r']
	r_sc = r / 10**3
	r_max = np.max(r_sc)
	r_min = np.min(r_sc)

	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	Nx0, Ny0 = len(x0), len(y0)
	pxl = np.meshgrid(x0, y0)

	Nz = len(set_z)
	for dd in range(Nz):
		xc = random.randint(900, 1100)
		yc = random.randint(800, 1000)
		zc = set_z[dd]

		Da0 = Test_model.angular_diameter_distance(zc).value
		Angu_r = rad2asec / Da0
		R_pixel = Angu_r / pixel

		dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
		dr_sc = dr / R_pixel

		cc_Lob = mock_flux['%.3f' % zc] / NMGY
		DN_min = np.min(cc_Lob)
		mock_DN = interp.interp1d(r_sc, cc_Lob, kind = 'cubic')

		ix = np.abs(x0 - xc)
		iy = np.abs(y0 - yc)
		ix0 = np.where(ix == np.min(ix))[0][0]
		iy0 = np.where(iy == np.min(iy))[0][0]
		frame = np.zeros((Ny0, Nx0), dtype = np.float)
		Nois = np.zeros((Ny0, Nx0), dtype = np.float)

		for kk in range(Ny0):
			for jj in range(Nx0):
				if (dr_sc[kk, jj] >= r_max ) | ( dr_sc[kk, jj] <= r_min ):

					lam_x = DN_min * gain / 10
					N_e = lam_x + N_sky * gain
					rand_x = np.random.poisson( N_e )
					frame[kk, jj] += lam_x # electrons number
					Nois[kk, jj] += rand_x

				else:

					lam_x = mock_DN( dr_sc[kk, jj] ) * gain
					N_e = lam_x + N_sky * gain
					rand_x = np.random.poisson( N_e )
					frame[kk, jj] += lam_x # electrons number
					Nois[kk, jj] += rand_x

		N_mock = frame / gain
		N_sub = Nois / gain - N_sky
		'''
		plt.figure(figsize = (12, 6))
		ax = plt.subplot(121)
		bx = plt.subplot(122)
		ax.imshow(N_mock, origin = 'lower', norm = mpl.colors.LogNorm())
		bx.imshow(N_sub, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.tight_layout()
		plt.show()
		'''
		x = frame.shape[1]
		y = frame.shape[0]
		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
				'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		values = ['T', 32, 2, x, y, np.int(ix0), np.int(iy0), ix0, iy0, ix0, iy0, set_ra[dd], set_dec[dd], zc, pixel]
		ff = dict(zip(keys, values))
		fil = fits.Header(ff)

		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/mock/mock_z%.3f_ra%.3f_dec%.3f.fits' % 
			(zc, set_ra[dd], set_dec[dd]), N_mock, header = fil, overwrite=True)
		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % 
			(zc, set_ra[dd], set_dec[dd]), N_sub, header = fil, overwrite=True)

		# add noise + mask
		N_s = 450
		xa0 = np.max( [xc - np.int(R_pixel), 0] )
		xa1 = np.min( [xc + np.int(R_pixel), 2047] )
		ya0 = np.max( [yc - np.int(R_pixel), 0] )
		ya1 = np.min( [yc + np.int(R_pixel), 1488] )
		lox = np.array([random.randint(xa0, xa1) for ll in range(N_s)])
		loy = np.array([random.randint(ya0, ya1) for ll in range(N_s)])
		Lr = np.abs(np.random.normal(0, 1.5, N_s) * 40)
		Sr = Lr * np.random.random(N_s)
		Phi = np.random.random(N_s) * 180

		mask = np.ones((1489, 2048), dtype = np.float)
		ox = np.linspace(0, frame.shape[1] - 1, frame.shape[1])
		oy = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
		basic_coord = np.array(np.meshgrid(ox,oy))
		major = Lr / 2
		minor = Sr / 2
		senior = np.sqrt(major**2 - minor**2)
		tdr = np.sqrt((lox - xc)**2 + (loy - yc)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]
		for k in range(N_s):
			posx = lox[k]
			posy = loy[k]
			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = Phi[k] * np.pi / 180

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(posx - set_r), 0])
			la1 = np.min( [np.int(posx + set_r +1), frame.shape[1] - 1] )
			lb0 = np.max( [np.int(posy - set_r), 0] ) 
			lb1 = np.min( [np.int(posy + set_r +1), frame.shape[0] - 1] )

			if k == dr00[0] :
				continue
			else:
				df1 = lr**2 - cr**2*np.cos(chi)**2
				df2 = lr**2 - cr**2*np.sin(chi)**2
				fr = ((basic_coord[0,:][lb0: lb1, la0: la1] - posx)**2*df1 + (basic_coord[1,:][lb0: lb1, la0: la1] - posy)**2*df2
					- cr**2*np.sin(2*chi)*(basic_coord[0,:][lb0: lb1, la0: la1] - posx)*(basic_coord[1,:][lb0: lb1, la0: la1] - posy))
				idr = fr / (lr**2*sr**2)
				jx = idr <= 1

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
				iv[iu] = np.nan
				mask[lb0: lb1, la0: la1] = mask[lb0: lb1, la0: la1] * iv
		frame2 = mask * N_sub
		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % 
			(zc, set_ra[dd], set_dec[dd]), frame2, header = fil, overwrite=True)

	#raise

def light_test():
	# sample review
	'''
	fig = plt.figure(figsize = (18,9))
	gs = gridspec.GridSpec(5,4)
	for k in range(20):
		data = fits.getdata(load + 'noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		img = data[0]
		ax = plt.subplot(gs[ k // 4, k % 4])
		ax.set_title('sample %d [z%.3f]' %(k, set_z[k]) )
		tf = ax.imshow(img, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[DN]$')
		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])
	plt.tight_layout()
	plt.savefig('noise_sample_view.png', dpi = 300)
	plt.close()

	fig = plt.figure(figsize = (18,9))
	gs = gridspec.GridSpec(5,4)
	for k in range(20):
		data = fits.getdata(load + 'noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		img = data[0]
		ax = plt.subplot(gs[ k // 4, k % 4])
		ax.set_title('sample %d [z%.3f]' %(k, set_z[k]) )
		tf = ax.imshow(img, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[DN]$')
		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])
	plt.tight_layout()
	plt.savefig('add_mask_sample_view.png', dpi = 300)
	plt.close()
	'''
	# SB profile measurement
	NMGY = 5e-3 # mean value of the data sample
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	exp_time = 54 # exposure time, in unit second
	bins, Nz = 65, len(set_z)
	mock_flux = pds.read_csv(load + 'mock_flux_data.csv')
	mock_SB = pds.read_csv(load + 'mock_SB_data.csv')
	ins_SB = pds.read_csv(load + 'mock_intrinsic_SB.csv')
	r = ins_SB['r']
	r_sc = r / 10**3

	R_t = np.zeros((Nz, bins), dtype = np.float)
	SB_t = np.zeros((Nz, bins), dtype = np.float)
	err_up = np.zeros((Nz, bins), dtype = np.float)
	err_botm = np.zeros((Nz, bins), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		#data = fits.getdata(load + 'noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)

		img = data[0]
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		Rp = (rad2asec / Dag) / pixel
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']

		N_flux = img * NMGY
		Intns, Intns_r, Intns_err, Npix = light_measure(N_flux, bins, 2, Rp, cenx, ceny, pixel, set_z[k])
		flux0 = Intns + Intns_err
		flux1 = Intns - Intns_err
		SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = SB - SB0
		err1 = SB1 - SB
		R_t[k, :], SB_t[k, :], err_up[k, :], err_botm[k, :] = Intns_r, SB, err0, err1

	plt.figure(figsize = (20, 24))
	gs = gridspec.GridSpec(5, 4)
	for k in range(Nz):
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		cc_SB = mock_SB['%.3f' % set_z[k]]
		f_SB = interp.interp1d(r, cc_SB, kind = 'cubic')
		id_nan = np.isnan(SB_t[k,:])
		ivx = id_nan == False
		ss_R = R_t[k, ivx]
		ss_SB = SB_t[k, ivx]
		iux = ( ss_R > np.min(r) ) & ( ss_R < np.max(r) )
		ddsb = ss_SB[iux] - f_SB( ss_R[iux] )
		ddsr = ss_R[iux]
		std = np.nanstd(ddsb)
		aver = np.nanmean(ddsb)
		err0 = err_up[k, ivx]
		err1 = err_botm[k, ivx]
		id_nan = np.isnan(err1)
		err1[id_nan] = 100. # set a large value for show the break out errorbar

		gs0 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec = gs[ k // 4, k % 4])
		ax0 = plt.subplot(gs0[:4])
		ax1 = plt.subplot(gs0[-1])

		ax0.plot(r, cc_SB, 'r-', label = '$ Intrinsic $', alpha = 0.5)
		ax0.errorbar(ss_R, ss_SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'bo', label = ' noise image [sky subtracted] ', alpha = 0.5)
		#ax0.errorbar(ss_R, ss_SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'bo', label = ' noise + mask image [sky subtracted] ', alpha = 0.5)
		ax0.set_xscale('log')
		ax0.set_ylabel('$SB[mag/arcsec^2]$')

		ax0.legend(loc = 1)
		ax0.set_xlim(9, 1010)
		ax0.set_ylim(19, 34)
		ax0.invert_yaxis()
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		bx1 = ax0.twiny()
		xtik = ax0.get_xticks()
		xtik = np.array(xtik)
		xR = xtik * 10**(-3) * rad2asec / Dag
		bx1.set_xscale('log')
		bx1.set_xticks(xtik)
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx1.set_xlim(ax0.get_xlim())
		ax0.set_xticks([])

		ax1.plot(ddsr, ddsb, 'b-', alpha = 0.5)
		ax1.errorbar(ddsr, ddsb, yerr = [err0[iux], err1[iux]], xerr = None, ls = '', fmt = 'bo', alpha = 0.5)
		ax1.axhline(y = 0, linestyle = '--', color = 'r', alpha = 0.5, label = '$ \Delta{SB} = 0 $')
		ax1.axhline(y = aver, linestyle = '--', color = 'b', alpha = 0.5)
		ax1.axhline(y = aver + std, linestyle = '--', color = 'k', alpha = 0.5)
		ax1.axhline(y = aver - std, linestyle = '--', color = 'k', alpha = 0.5)

		ax1.set_xlim(9, 1010)
		ax1.set_ylim(-1, 1)
		ax1.set_xscale('log')
		ax1.set_xlabel('$R[kpc]$')
		ax1.set_ylabel('$ SB_{M} - SB_{I} $')
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('noise_light_measure_test.pdf', dpi = 300)
	#plt.savefig('add_mask_light_measure_test.pdf', dpi = 300)
	plt.close()

	raise

def resamp_test():
	# SB profile measurement
	NMGY = 5e-3 # mean value of the data sample
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	exp_time = 54 # exposure time, in unit second
	bins, Nz = 65, len(set_z)
	mock_flux = pds.read_csv(load + 'mock_flux_data.csv')
	mock_SB = pds.read_csv(load + 'mock_SB_data.csv')
	ins_SB = pds.read_csv(load + 'mock_intrinsic_SB.csv')
	r = ins_SB['r']
	INS_SB = ins_SB['0.250']
	f_SB = interp.interp1d(r, INS_SB, kind = 'cubic')

	R_t = np.zeros((Nz, bins), dtype = np.float)
	SB_t = np.zeros((Nz, bins), dtype = np.float)
	err_up = np.zeros((Nz, bins), dtype = np.float)
	err_botm = np.zeros((Nz, bins), dtype = np.float)

	R_s = np.zeros((Nz, bins), dtype = np.float)
	SB_s = np.zeros((Nz, bins), dtype = np.float)
	err_s_up = np.zeros((Nz, bins), dtype = np.float)
	err_s_botm = np.zeros((Nz, bins), dtype = np.float)

	for k in range(Nz):
		data = fits.getdata(load + 'noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		#data = fits.getdata(load + 'noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)

		img = data[0]
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		Rp = (rad2asec / Dag) / pixel
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']
		Len_ref = Da_ref * pixel / rad2asec
		Len_z0 = Dag * pixel / rad2asec
		eta = Len_ref/Len_z0
		mu = 1 / eta

		N_flux = img * NMGY
		scale_img = flux_recal(N_flux, set_z[k], z_ref)
		if eta > 1:
			resamt, xn, yn = sum_samp(eta, eta, scale_img, cenx, ceny)
		else:
			resamt, xn, yn = down_samp(eta, eta, scale_img, cenx, ceny)

		xn = np.int(xn)
		yn = np.int(yn)
		Nx = resamt.shape[1]
		Ny = resamt.shape[0]
		## PS : the flux saved in resample file is in unit "nmaggy", not DN (DN is for previous files)
		keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'CENTER_X', 'CENTER_Y', 'ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, Nx, Ny, xn, yn, set_z[k], pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(load + 'resamp/resamp-noise-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), resamt, header = fil, overwrite=True)
		#fits.writeto(load + 'resamp/resamp-mask-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), resamt, header = fil, overwrite=True)

		Intns, Intns_r, Intns_err, Npix = light_measure(N_flux, bins, 2, Rp, cenx, ceny, pixel, set_z[k])
		flux0 = Intns + Intns_err
		flux1 = Intns - Intns_err
		SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
		SB = SB - 10 * np.log10( (1 + set_z[k]) / (1 + z_ref) )
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = SB - SB0
		err1 = SB1 - SB
		R_t[k, :], SB_t[k, :], err_up[k, :], err_botm[k, :] = Intns_r, SB, err0, err1

		Intns, Intns_r, Intns_err, Npix = light_measure(resamt, bins, 2, Rpp, xn, yn, pixel, z_ref)
		flux0 = Intns + Intns_err
		flux1 = Intns - Intns_err
		SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
		SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
		SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
		err0 = SB - SB0
		err1 = SB1 - SB
		R_s[k, :], SB_s[k, :], err_s_up[k, :], err_s_botm[k, :] = Intns_r, SB, err0, err1

	plt.figure(figsize = (20, 24))
	gs = gridspec.GridSpec(5, 4)
	for k in range(Nz):

		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		id_nan = np.isnan(SB_t[k,:])
		ivx = id_nan == False
		ss_R = R_t[k, ivx]
		ss_SB = SB_t[k, ivx]
		iux = ( ss_R > np.min(r) ) & ( ss_R < np.max(r) )
		ddsb = ss_SB[iux] - f_SB( ss_R[iux] )
		ddsr = ss_R[iux]
		std = np.nanstd(ddsb)
		aver = np.nanmean(ddsb)
		err0 = err_up[k, ivx]
		err1 = err_botm[k, ivx]
		id_nan = np.isnan(err1)
		err1[id_nan] = 100. # set a large value for show the break out errorbar

		id_nan = np.isnan(SB_s[k,:])
		ipx = id_nan == False
		tt_R = R_s[k, ipx]
		tt_SB = SB_s[k, ipx]
		iqx = ( tt_R > np.min(r) ) & ( tt_R < np.max(r) )
		ddtb = tt_SB[iqx] - f_SB( tt_R[iqx] )
		ddtr = tt_R[iqx]
		t_std = np.nanstd(ddtb)
		t_aver = np.nanmean(ddtb)
		t_err0 = err_s_up[k, ipx]
		t_err1 = err_s_botm[k, ipx]
		id_nan = np.isnan(t_err1)
		t_err1[id_nan] = 100. # set a large value for show the break out errorbar

		gs0 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec = gs[ k // 4, k % 4])
		ax0 = plt.subplot(gs0[:4])
		ax1 = plt.subplot(gs0[-1])

		ax0.plot(r, INS_SB, 'r-', label = '$ Intrinsic $', alpha = 0.5)
		ax0.errorbar(ss_R, ss_SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'b^', label = ' sky subtracted ', alpha = 0.5)
		ax0.errorbar(tt_R, tt_SB, yerr = [t_err0, t_err1], xerr = None, ls = '', fmt = 'gs', label = ' sky subtracted + resampled ', alpha = 0.5)

		ax0.set_xscale('log')
		ax0.set_ylabel('$SB[mag/arcsec^2]$')
		ax0.legend(loc = 1)
		ax0.set_xlim(9, 1010)
		ax0.set_ylim(19, 34)
		ax0.invert_yaxis()
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		bx1 = ax0.twiny()
		xtik = ax0.get_xticks()
		xtik = np.array(xtik)
		xR = xtik * 10**(-3) * rad2asec / Dag
		bx1.set_xscale('log')
		bx1.set_xticks(xtik)
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx1.set_xlim(ax0.get_xlim())
		ax0.set_xticks([])

		ax1.plot(ddsr, ddsb, 'b-', alpha = 0.5)
		ax1.axhline(y = 0, linestyle = '--', color = 'r', alpha = 0.5, label = '$ \Delta{SB} = 0 $')

		#ax1.errorbar(ddsr, ddsb, yerr = [err0[iux], err1[iux]], xerr = None, ls = '', fmt = 'b^', alpha = 0.5)
		ax1.plot(ddsr, ddsb, 'b-', alpha = 0.5)
		ax1.axhline(y = aver, linestyle = '--', color = 'b', alpha = 0.5)
		ax1.axhline(y = aver + std, linestyle = '--', color = 'b', alpha = 0.5)
		ax1.axhline(y = aver - std, linestyle = '--', color = 'b', alpha = 0.5)

		#ax1.errorbar(ddtr, ddtb, yerr = [t_err0[iqx], t_err1[iqx]], xerr = None, ls = '', fmt = 'gs', alpha = 0.5)
		ax1.plot(ddtr, ddtb, 'g-', alpha = 0.5)
		ax1.axhline(y = t_aver, linestyle = '--', color = 'g', alpha = 0.5)
		ax1.axhline(y = t_aver + t_std, linestyle = '-.', color = 'g', alpha = 0.5)
		ax1.axhline(y = t_aver - t_std, linestyle = '-.', color = 'g', alpha = 0.5)

		ax1.set_xlim(9, 1010)
		ax1.set_ylim(-1, 1)
		ax1.set_xscale('log')
		ax1.set_xlabel('$R[kpc]$')
		ax1.set_ylabel('$ SB_{M} - SB_{I} $')
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('noise_resample_SB.pdf', dpi = 300)
	#plt.savefig('mask_resample_SB.pdf', dpi = 300)
	plt.close()

	raise

def stack_test():
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	NMGY = 5e-3 # mean value of the data sample
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	exp_time = 54 # exposure time, in unit second
	bins, Nz = 65, len(set_z)
	mock_flux = pds.read_csv(load + 'mock_flux_data.csv')
	mock_SB = pds.read_csv(load + 'mock_SB_data.csv')
	ins_SB = pds.read_csv(load + 'mock_intrinsic_SB.csv')
	r = ins_SB['r']
	INS_SB = ins_SB['0.250']
	f_SB = interp.interp1d(r, INS_SB, kind = 'cubic')
	## Noise sample
	sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'resamp/resamp-noise-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), header = True)
		img = data[0]
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']

		la0 = np.int(y0 - ceny)
		la1 = np.int(y0 - ceny + img.shape[0])
		lb0 = np.int(x0 - cenx)
		lb1 = np.int(x0 - cenx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + img[idv]
		count_array_D[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array_D)
		id_fals = np.where(id_nan == False)
		p_count_D[id_fals] = p_count_D[id_fals] + 1
		count_array_D[la0: la1, lb0: lb1][idv] = np.nan

	mean_array_D = sum_array_D / p_count_D
	where_are_inf = np.isinf(mean_array_D)
	mean_array_D[where_are_inf] = np.nan
	id_zeros = np.where(p_count_D == 0)
	mean_array_D[id_zeros] = np.nan

	Intns, Intns_r, Intns_err, Npix = light_measure(mean_array_D, bins, 2, Rpp, x0, y0, pixel, z_ref)
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB - SB0
	err1 = SB1 - SB

	id_nan = np.isnan(SB)
	ivx = id_nan == False
	t1_R = Intns_r[ivx]
	t1_SB = SB[ivx]
	t1_err0 = err0[ ivx ]
	t1_err1 = err1[ ivx ]
	id_nan = np.isnan(t1_err1)
	t1_err1[id_nan] = 100. # set a large value for show the break out errorbar
	iux = (t1_R > np.min(r)) & (t1_R < np.max(r))
	ddsb0 = t1_SB[ iux ] - f_SB( t1_R[ iux ] )
	ddsr0 = t1_R[ iux ]
	std0 = np.nanstd(ddsb0)
	aver0 = np.nanmean(ddsb0)

	## add mask sample
	sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'resamp/resamp-mask-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), header = True)
		img = data[0]
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']

		la0 = np.int(y0 - ceny)
		la1 = np.int(y0 - ceny + img.shape[0])
		lb0 = np.int(x0 - cenx)
		lb1 = np.int(x0 - cenx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + img[idv]
		count_array_D[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array_D)
		id_fals = np.where(id_nan == False)
		p_count_D[id_fals] = p_count_D[id_fals] + 1
		count_array_D[la0: la1, lb0: lb1][idv] = np.nan

	mean_array_D = sum_array_D / p_count_D
	where_are_inf = np.isinf(mean_array_D)
	mean_array_D[where_are_inf] = np.nan
	id_zeros = np.where(p_count_D == 0)
	mean_array_D[id_zeros] = np.nan

	Intns, Intns_r, Intns_err, Npix = light_measure(mean_array_D, bins, 2, Rpp, x0, y0, pixel, z_ref)
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0 = SB - SB0
	err1 = SB1 - SB

	id_nan = np.isnan(SB)
	ivx = id_nan == False
	t2_R = Intns_r[ivx]
	t2_SB = SB[ivx]
	t2_err0 = err0[ ivx ]
	t2_err1 = err1[ ivx ]
	id_nan = np.isnan(t2_err1)
	t2_err1[id_nan] = 100. # set a large value for show the break out errorbar
	iux = (t2_R > np.min(r)) & (t2_R < np.max(r))
	ddsb1 = t2_SB[ iux ] - f_SB( t2_R[ iux ] )
	ddsr1 = t2_R[ iux ]
	std1 = np.nanstd(ddsb1)
	aver1 = np.nanmean(ddsb1)	

	plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])

	ax0.set_title('stack image')
	ax0.plot(r, INS_SB, 'r-', label = '$ Intrinsic $', alpha = 0.5)
	ax0.errorbar(t1_R, t1_SB, yerr = [t1_err0, t1_err1], xerr = None, ls = '', fmt = 'b^', label = ' Noise ', alpha = 0.5)
	ax0.errorbar(t2_R, t2_SB, yerr = [t2_err0, t2_err1], xerr = None, ls = '', fmt = 'gs', label = ' Noise + Mask ', alpha = 0.5)

	ax0.set_xscale('log')
	ax0.set_xlabel('$R[kpc]$')
	ax0.set_xlim(9, 1010)
	ax0.set_ylim(19, 34)
	ax0.set_ylabel('$SB[mag/arcsec^2]$')
	ax0.invert_yaxis()
	ax0.legend(loc = 1)
	ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

	bx1 = ax0.twiny()
	xtik = ax0.get_xticks()
	xtik = np.array(xtik)
	Dag = Test_model.angular_diameter_distance(z_ref).value
	xR = xtik * 10**(-3) * rad2asec / Dag
	bx1.set_xscale('log')
	bx1.set_xticks(xtik)
	bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
	bx1.set_xlim(ax0.get_xlim())
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax0.set_xticks([])

	ax1.axhline(y = 0, linestyle = '--', color = 'k', alpha = 0.5, label = '$ \Delta{SB} = 0 $')
	ax1.plot(ddsr0, ddsb0, 'b--', alpha = 0.5)
	ax1.axhline(y = aver0, linestyle = '--', color = 'b', alpha = 0.5)
	ax1.axhline(y = aver0 + std0, linestyle = '--', color = 'b', alpha = 0.5)
	ax1.axhline(y = aver0 - std0, linestyle = '--', color = 'b', alpha = 0.5)

	ax1.plot(ddsr1, ddsb1, 'g-.', alpha = 0.5)
	ax1.axhline(y = aver1, linestyle = '-.', color = 'g', alpha = 0.5)
	ax1.axhline(y = aver1 + std1, linestyle = '-.', color = 'g', alpha = 0.5)
	ax1.axhline(y = aver1 - std1, linestyle = '-.', color = 'g', alpha = 0.5)	

	ax1.set_xscale('log')
	ax1.set_xlim(9, 1010)
	ax1.set_ylim(-1, 1)
	ax1.set_xlabel('$R[kpc]$')
	ax1.set_ylabel('$ SB_{stacking} - SB_{reference} $')
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(hspace = 0)
	plt.savefig('stack_test.png', dpi = 300)
	plt.close()

	raise

def main():
	#galaxy()
	SB_pro()

	#mock_ccd()
	#light_test()
	#resamp_test()
	#stack_test()

if __name__ == "__main__":
	main()
