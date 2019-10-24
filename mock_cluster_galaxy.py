import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
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
from light_measure_tmp import light_measure

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
Lstar = 2e10 # in unit L_sun ("copy from Galaxies in the Universe")

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

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
	SB_r0 = SB0['mag/arcsec^2']

	SB1 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_2.csv', skiprows = 1)
	SB_r1 = SB1['mag/arcsec^2']

	SB2 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_3.csv', skiprows = 1)
	SB_r2 = SB2['mag/arcsec^2']
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(R_r, SB_r0, 'r^', label = 'masking incompleteness corrected')
	ax.plot(R_r, SB_r1, 'go', label = 'BCG+ICL')
	ax.plot(R_r, SB_r2, 'bs', label = 'total')
	ax.set_xscale('log')
	ax.invert_yaxis()
	ax.set_xlabel('$ R[Mpc] $')
	ax.set_ylabel('$ SB [mag/arcsec^2] $')
	ax.set_xlim(1e-2, 1.2e0)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.legend(loc = 1)

	bx1 = ax.twiny()
	xtik = ax.get_xticks()
	xtik = np.array(xtik)
	xR = xtik * rad2asec / Da_ref
	bx1.set_xscale('log')
	bx1.set_xticks(xtik)
	bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
	bx1.set_xlim(ax.get_xlim())
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.tight_layout()
	plt.savefig('ref_SB_r.png', dpi = 300)
	plt.show()
	'''
	# fitting the SB profile
	mu_e0 = 23.87 # mag/arcsec^2
	mu_e1 = 30
	mu_e2 = 20

	Re_0 = 19.29 # kpc
	Re_1 = 120
	Re_2 = 10

	ndex0 = 4.
	ndex1 = 4.
	ndex2 = 4.

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
	r_sc = r / np.max(r)
	SB_r = SB_fit(r, mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2) # profile at z = 0.25
	#SB_r = SB_fit(r_fit, mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2)

	## change the SB_r into counts / s
	NMGY = 5e-3 # mean value of the data sample
	exp_time = 54 # exposure time, in unit second
	DN = 10**( (22.5 - SB_r + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	## error
	err_N = np.sqrt( DN / gain ) 

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

	xc = 1025
	yc = 745
	dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
	dr_sc = dr / Rpp

	for kk in range(dr_sc.shape[0]):
		for jj in range(dr_sc.shape[1]):
			if (dr_sc[kk, jj] >= np.max(r_sc) ) | (dr_sc[kk, jj] <= np.min(r_sc) ):
				lam_x = np.min(DN) * gain / 10
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
	Intns, Intns_r, Ar, Intns_err = light_measure(N_flux, bins, 10, Rpp, xc, yc, pixel, z_ref)
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
	err1[id_nan] = 100. # set a large value for show the break out errorbar

	N_smooth = N_mock * NMGY
	Intns, Intns_r, Ar, Intns_err = light_measure(N_smooth, bins, 10, Rpp, xc, yc, pixel, z_ref)
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	SB_mooth = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
	SB0_mooth = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
	SB1_mooth = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
	err0_mooth = SB_mooth - SB0_mooth
	err1_mooth = SB1_mooth - SB_mooth

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

	plt.figure( figsize = (12, 6) )
	ax = plt.subplot(121)
	bx = plt.subplot(122)

	ax.hist(N_mock.flatten(), bins = 25, histtype = 'step', color = 'b', density = True, label = 'intrinsic')
	ax.hist(N_sub.flatten(), bins = 25, histtype = 'step', color = 'r', density = True, label = 'add noise')
	ax.set_title('image DN distribution')
	ax.set_yscale('log')
	ax.set_xlabel('DN')
	ax.set_ylabel('PDF')
	ax.legend(loc = 1)

	bx.errorbar(pR, SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'ro', label = ' noise image [sky subtracted] ', alpha = 0.5)
	bx.errorbar(Intns_r, SB_mooth, yerr = [err0_mooth, err1_mooth], xerr = None, ls = '', fmt = 'gs', label = ' smooth image ', alpha = 0.5)
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

	plt.tight_layout()
	plt.savefig('SB_measure_test.png', dpi = 300)
	plt.close()

	raise

def main():
	#galaxy()
	SB_pro()

if __name__ == "__main__":
	main()
