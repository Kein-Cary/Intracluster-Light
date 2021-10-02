import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i']

### ==== ### SB model for random image case ### ==== ###
def cc_inves_x2(x, x0, A, alpha, B):
	return A * (np.abs(x - x0))**(-1*alpha) + B

def cc_rand_sb_func(x, a, b, x0, A, alpha, B):
	pf0 = a * np.log10(x) + b
	pf1 = cc_inves_x2(np.log10(x), x0, A, alpha, B)
	pf = pf0 + pf1
	return pf

def err_func(p, x, y, yerr):
	a, b, x0, A, alpha, B = p[:]
	pf = cc_rand_sb_func(x, a, b, x0, A, alpha, B)
	return np.sum( (pf - y)**2 / yerr**2 )

### ==== ### introduce data below ### ==== ###
def random_SB_fit_func(r_arr, sb_arr, err_arr, po, out_file, end_point = 1, R_psf = 10,):
	"""
	R_psf : the seeing PSF size, in unit of kpc, fitting data beyond R_psf only
			10 kpc is for SDSS
	r_arr, sb_arr, err_arr : radius, surface brightness and the err
	end_point : points closed to the curve end may have large uncertainty, here choose to ignore
				how many points when fitting, end_point = 1, means with fit data points to the second last
	po : the initial value start the fitting, list type
	out_file : save the  
	"""
	idnx = r_arr >= R_psf
	fit_r = r_arr[ idnx ]
	fit_sb = sb_arr[ idnx ]
	fit_err = err_arr[ idnx ]

	p0 = po
	P_return = optimize.minimize(err_func, x0 = np.array(p0), args = ( fit_r[:-end_point], fit_sb[:-end_point], fit_err[:-end_point] ), method = 'Powell',)
	p1 = P_return.x

	e_a, e_b, e_x0, e_A, e_alpha, e_B = p1

	keys = [ 'e_a', 'e_b', 'e_x0', 'e_A', 'e_alpha', 'e_B' ]
	values = [ e_a, e_b, e_x0, e_A, e_alpha, e_B ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_file )

	return

### ==== ### SB model for cluster image case ### ==== ###
##: auume the large scale is : random profile + C + n = 2.1 sersic (Zhang et a., 2019, for large scale, n~2.1)
def sersic_func(r, Ie, re):

	ndex = 2.1

	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def err_fit_func(p, x, y, params, yerr):
	a, b, x0, A, alpha, B = params 
	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)

	d_off, I_e, R_e = p[:]
	pf1 = sersic_func(x, I_e, R_e)
	pf = pf0 + pf1 - d_off
	return np.sum( (pf - y)**2 / yerr**2 )

def clust_SB_fit_func( R_arr, sb_arr, sb_err_arr, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, out_params_file, out_pros_file, trunk_R = 2e3,):
	"""
	by default, trunk_R = 2Mpc, which meanse the model signal beyond 2Mpc will be treated as 
	background and subtracted from the observation
	params_file : description for random image stacking result (which is mean of rand stacking)
	lo_R_lim, hi_R_lim : region limits for fitting is given by lo_R_lim (use data beyond lo_R_lim)
						hi_R_lim is the point at which SB profile stop decreasing.
	p0, bounds : initial arr and bounds for fitting
	out_params_file, out_pros_file : save the fitting result and params (.csv file)
	"""
	idmx = R_arr >= R_psf # use data points beyond psf scale
	com_r = R_arr[ idmx ]
	com_sb = sb_arr[ idmx ]
	com_err = sb_err_arr[ idmx ]

	## read params of random point SB profile
	p_dat = pds.read_csv( params_file )
	( e_a, e_b, e_x0, e_A, e_alpha, e_B ) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0],
											np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0],)

	idx1 = ( com_r >= lo_R_lim ) # normal
	# idx1 = ( com_r >= lo_R_lim ) & ( com_r <= hi_R_lim ) # adjust

	idx2 = ( com_r >= lo_R_lim ) & ( com_r <= hi_R_lim )

	fx = com_r[idx1]
	fy = com_sb[idx1]
	ferr = com_err[idx1]

	params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B])

	po = list( p0 )
	bonds = bounds
	E_return = optimize.minimize( err_fit_func, x0 = np.array(po), args = (fx, fy, params, ferr), method = 'L-BFGS-B', bounds = bonds,)
	popt = E_return.x
	offD, I_e, R_e = popt

	print(E_return)
	print(popt)

	fit_rnd_sb = cc_rand_sb_func(com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)  
	sign_fit = sersic_func( com_r, I_e, R_e)
	BG_pros = fit_rnd_sb - offD
	comb_F = BG_pros + sign_fit

	# chi2 value
	# chi2 = np.sum( (comb_F[idx1] - com_sb[idx1])**2 / com_err[idx1]**2 )
	# chi_ov_nu = chi2 / ( len(fx) - len( po ) )
	# chi_ov_nu = np.ones( len(com_r) ) * chi_ov_nu

	chi2 = E_return.fun
	chi_ov_nu = chi2 / ( len(fx) - len( po ) )

	chi_inner = np.sum( (comb_F[idx2] - com_sb[idx2])**2 / com_err[idx2]**2 )
	chi_inner_m = chi_inner / ( len(com_r[idx2]) - 3 )

	## normalize with SB value at 2Mpc of the model fitting
	sb_2Mpc = sersic_func( trunk_R, I_e, R_e )
	norm_sign = sign_fit - sb_2Mpc
	norm_BG = comb_F - norm_sign

	# save the background profile and params
	full_r = R_arr

	full_r_fit = cc_rand_sb_func( full_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)
	full_BG = full_r_fit - offD + sb_2Mpc

	chi_ov_nu = np.ones( len(R_arr) ) * chi_ov_nu
	chi_inner_m = np.ones( len(R_arr) ) * chi_inner_m

	keys = [ 'R_kpc', 'BG_sb', 'chi2nu', 'chi2nu_inner']
	values = [ full_r, full_BG, chi_ov_nu, chi_inner_m ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_pros_file )

	keys = ['e_a', 'e_b', 'e_x0', 'e_A', 'e_alpha', 'e_B', 'offD', 'I_e', 'R_e']
	values = [ e_a, e_b, e_x0, e_A, e_alpha, e_B, offD, I_e, R_e ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_params_file )

	return

