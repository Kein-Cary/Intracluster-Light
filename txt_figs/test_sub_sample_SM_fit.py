import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.special as special
import astropy.units as U
import astropy.constants as C

import emcee
import corner
import time

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import signal
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set
from color_2_mass import get_c2mass_func, gi_band_c2m_func
from multiprocessing import Pool

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)

H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### sersic profile
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def log_parabolic_func(r, Im, Rm, am, bas):

	Ir = 10**Im * ( r / Rm)**( -10**am - 10**bas * np.log( r / Rm) )
	return Ir

def log_norm_func(r, Im, R_pk, L_trans):

	# f1 = 1 / ( r * L_trans * np.sqrt(2 * np.pi) )
	# f2 = np.exp( -0.5 * (np.log(r) - R_crit)**2 / L_trans**2 )

	#... scaled version
	scl_r = r / R_pk       # r / R_crit
	scl_L = L_trans / R_pk # L_trans / R_crit

	cen_p = 0.25 # R_crit / R_pk

	f1 = 1 / ( scl_r * scl_L * np.sqrt(2 * np.pi) )
	f2 = np.exp( -0.5 * (np.log( scl_r ) - cen_p )**2 / scl_L**2 )

	Ir = 10**Im * f1 * f2

	return Ir

def para_err_fit_f(p, x, y, params, yerr):

	cov_mx = params[0]

	_Ie, _Re, _ae, bas = p[:]
	_mass_cen = log_parabolic_func( x, _Ie, _Re, _ae, bas)
	_mass_2Mpc = log_parabolic_func( 2e3, _Ie, _Re, _ae, bas)

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

def lg_norm_err_fit_f(p, x, y, params, yerr):

	cov_mx = params[0]

	_Ie, _R_pk, L_trans = p[:]

	_mass_cen = log_norm_func( x, _Ie, _R_pk, L_trans )
	_mass_2Mpc = log_norm_func( 2e3, _Ie, _R_pk, L_trans )

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	chi2 = delta.T.dot( cov_inv ).dot(delta)
	# chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === ### fitting outer region
def likelihood_func(p, x, y, params, yerr):

	bf = p[0]

	cov_mx, lg_sigma = params[:]

	_mass_out = 10**lg_sigma

	_sum_mass = np.log10( _mass_out * 10**bf )

	delta = _sum_mass - y
	# cov_inv = np.linalg.pinv( cov_mx )
	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def prior_p_func( p ):

	bf = p[0]

	identi = 10**(-4.0) < 10**bf < 10**(-2.5)

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

### === ### fitting mid-region
def mid_like_func(p, x, y, params, yerr):

	cov_mx = params[0]

	_Ie, _R_pk, L_trans = p[:]
	M_x = log_norm_func( x, _Ie, _R_pk, L_trans)
	M_2Mpc = log_norm_func( 2e3, _Ie, _R_pk, L_trans)

	cros_M = np.log10( M_x - M_2Mpc )

	delta = cros_M - y

	cov_inv = np.linalg.pinv( cov_mx )
	chi2 = delta.T.dot( cov_inv ).dot( delta )

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def mid_prior_p_func( p ):

	_Ie, _R_pk, L_trans = p[:]
	identi = ( 10**5.0 <= 10**_Ie <= 10**6.5 ) & ( 10 <= _R_pk <= 90 ) & ( 10 <= L_trans <= 90 )

	if identi:
		return 0
	return -np.inf

def mid_ln_p_func(p, x, y, params, yerr):

	pre_p = mid_prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + mid_like_func(p, x, y, params, yerr)

### === load data
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'
path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/SBs/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

fit_path = '/home/xkchen/tmp_run/data_files/figs/mass_pro_fit/'

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
mark_s = ['s', 'o']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

## miscen params for high mass
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24]
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

out_path = '/home/xkchen/tmp_run/data_files/figs/mass_pro_fit/'

a_ref = 1 / (z_ref + 1)
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = ( lo_xi * 1e3 * rho_m ) / a_ref**2 * h
lo_rp = lo_rp * 1e3 / h * a_ref

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) / a_ref**2 * h
hi_rp = hi_rp * 1e3 / h * a_ref

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )

"""
## fitting outer region
for band_str in ('gi',):

	for mm in range( 1,2 ):

		# dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		# obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

		id_rx = obs_R >= 9
		obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

		id_cov = np.where( id_rx )[0][0]
		cov_arr = cov_arr[id_cov:, id_cov:]

		lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

		## .. use xi_hm for sigma estimation
		if mm == 0:
			misNFW_sigma = lo_interp_F( obs_R )
			sigma_2Mpc = lo_xi2M_2Mpc + 0.
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		if mm == 1:
			misNFW_sigma = hi_interp_F( obs_R )
			sigma_2Mpc = hi_xi2M_2Mpc + 0.
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		out_lim_R = 250 # 200, 250, 300 kpc
		idx_lim = obs_R >= out_lim_R
		id_dex = np.where( idx_lim == True )[0][0]

		fit_R = obs_R[idx_lim]
		fit_M = lg_M[idx_lim]
		fit_Merr = lg_M_err[idx_lim]
		cut_cov = cov_arr[id_dex:, id_dex:]

		put_params = [ cut_cov, lg_M_sigma[idx_lim] ]

		n_walk = 50

		# initial
		put_x4 = np.random.uniform( -4.0, -2.5, n_walk ) # lgfb

		L_chains = 1e4
		param_labels = ['$lgf_{b}$']

		pos = np.array( [put_x4] ).T
		n_dim = pos.shape[1]

		file_name = out_path + '%s_%s-band-based_xi2-sigma_beyond-%dkpc_fit_mcmc_fit.h5' % (cat_lis[mm], band_str, out_lim_R)

		backend = emcee.backends.HDFBackend( file_name )
		backend.reset( n_walk, n_dim )

		with Pool( 2 ) as pool:
			sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = ( fit_R, fit_M, put_params, fit_Merr), pool = pool, backend = backend,)
			sampler.run_mcmc(pos, L_chains, progress = True, )

		sampler = emcee.backends.HDFBackend( file_name )

		try:
			tau = sampler.get_autocorr_time()
			flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)

		## params estimate
		mc_fits = []

		for oo in range( n_dim ):

			samp_arr = flat_samples[:, oo]
			mc_fit_oo = np.median( samp_arr )
			mc_fits.append( mc_fit_oo )

		## figs
		fig = corner.corner( flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
			levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
			plot_datapoints = True, plot_density = False, fill_contours = True,)
		axes = np.array( fig.axes ).reshape( (n_dim, n_dim) )

		for jj in range( n_dim ):
			ax = axes[jj, jj]
			ax.axvline( mc_fits[jj], color = 'r', ls = '-', alpha = 0.75,)

		for yi in range( n_dim ):
			for xi in range( yi ):
				ax = axes[yi, xi]

				ax.axvline( mc_fits[xi], color = 'r', ls = '-', alpha = 0.75,)
				ax.axhline( mc_fits[yi], color = 'r', ls = '-', alpha = 0.75,)

				ax.plot( mc_fits[xi], mc_fits[yi], 'ro', alpha = 0.75,)

		ax = axes[0, n_dim - 2 ]
		ax.set_title( fig_name[mm] + ',%s band-based' % band_str )
		plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_beyond-%dkpc_fit_params.png' % (cat_lis[mm], band_str, out_lim_R), dpi = 300)
		plt.close()

		bf_fit = mc_fits[0]
		# n_dim = 1
		# bf_fit = -2.794

		_out_M = 10**lg_M_sigma * 10** bf_fit
		_sum_fit = np.log10( _out_M )

		devi_lgM = np.log10( surf_M - _out_M )

		##.. model lines
		new_R = np.logspace( 0, np.log10(2.5e3), 100)

		if mm == 0:
			fit_out_M = (lo_interp_F( new_R ) - sigma_2Mpc) * 10**bf_fit

		if mm == 1:
			fit_out_M = (hi_interp_F( new_R ) - sigma_2Mpc) * 10**bf_fit

		c_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit.csv' % (cat_lis[mm], band_str) )
		Ie_fit, Re_fit, Ne_fit = np.array( c_dat['Ie'] )[0], np.array( c_dat['Re'] )[0], np.array( c_dat['ne'] )[0] 
		_cen_M = sersic_func( new_R, 10**Ie_fit, Re_fit, Ne_fit)

		cut_inv = np.linalg.pinv( cut_cov )
		cut_delta = _sum_fit[idx_lim] - lg_M[idx_lim]
		cut_chi2 = cut_delta.T.dot( cut_inv ).dot(cut_delta)

		cut_chi2nu = cut_chi2 / ( np.sum(idx_lim) - n_dim )
		print( cut_chi2nu )

		## sersic fitting on devi_lgM
		id_nan = np.isnan( devi_lgM )
		devi_R = obs_R[ id_nan == False ]
		devi_lgM = devi_lgM[ id_nan == False ]
		devi_err = lg_M_err[ id_nan == False ]

		id_nnx = np.where( id_nan == True )[0]
		devi_cov = np.delete( cov_arr, list(id_nnx), axis = 0)
		devi_cov = np.delete( devi_cov, list(id_nnx), axis = 1)


		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[mm] + ',%s-band based' % band_str)

		ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = '.', ls = 'none', ecolor = 'r', 
			alpha = 0.75, mec = 'r', mfc = 'r', label = 'observed')

		ax.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', 
			alpha = 0.75, mec = 'b', mfc = 'none', label = 'observed - scaled $NFW_{projected}^{miscentering}$')

		ax.plot( new_R, np.log10( fit_out_M ), ls = '-', color = 'b', alpha = 0.75, 
			label = 'scaled $NFW_{projected}^{miscentering}$',)

		ax.plot( new_R, np.log10( _cen_M ), ls = '-', color = 'g', alpha = 0.75, label = 'n=4 sersic',)

		ax.text( 1e1, 3.5, s = '$\\chi^{2} / \\nu[R>=300kpc] = %.5f$' % cut_chi2nu, color = 'k',)

		ax.legend( loc = 1, )
		ax.set_ylim( 3, 8.5)
		ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

		ax.set_xlim( 1e1, 3e3)
		ax.set_xlabel( 'R [kpc]')
		ax.set_xscale( 'log' )
		plt.savefig('/home/xkchen/%s_%s-band_based_beyond-%dkpc_fit_test.png' % (cat_lis[mm], band_str, out_lim_R), dpi = 300)
		plt.close()
"""

"""
### === cross part fitting
for band_str in ('gi',):

	for mm in range( 2 ):

		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

		id_rx = obs_R >= 9
		obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

		lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

		## .. use xi_hm for sigma estimation
		if mm == 0:
			misNFW_sigma = lo_interp_F( obs_R )
			sigma_2Mpc = lo_xi2M_2Mpc + 0.
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		if mm == 1:
			misNFW_sigma = hi_interp_F( obs_R )
			sigma_2Mpc = hi_xi2M_2Mpc + 0.
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

		id_cov = np.where( id_rx )[0][0]
		cov_arr = cov_arr[id_cov:, id_cov:]

		out_lim_R = 250 # 200, 250, 300 kpc

		## ...
		pre_file = out_path + '%s_%s-band-based_xi2-sigma_beyond-%dkpc_fit_mcmc_fit.h5' % (cat_lis[0], band_str, out_lim_R)
		pre_sampler = emcee.backends.HDFBackend( pre_file )

		try:
			tau = sampler.get_autocorr_time()
			flat_samples = pre_sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = pre_sampler.get_chain( discard = 3000, thin = 300, flat = True)

		mc_fits = []
		n_dim = flat_samples.shape[1]

		for oo in range( n_dim ):

			samp_arr = flat_samples[:, oo]
			mc_fit_oo = np.median( samp_arr )
			mc_fits.append( mc_fit_oo )

		bf_fit = mc_fits[0]

		#.. use params of total sample
		# c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-300kpc_xi2M-fit.csv')
		# lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
		# lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
		# lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

		# bf_fit = lg_fb_gi

		_out_M = 10**lg_M_sigma * 10** bf_fit

		## .. centeral deV profile
		c_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit.csv' % (cat_lis[mm], band_str) )
		Ie_fit, Re_fit, Ne_fit = np.array( c_dat['Ie'] )[0], np.array( c_dat['Re'] )[0], np.array( c_dat['ne'] )[0]

		_cen_M = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit)
		_cen_M_2Mpc = sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)

		##.. mid-region
		devi_lgM = np.log10( surf_M - _out_M - ( _cen_M - _cen_M_2Mpc ) )

		id_nan = np.isnan( devi_lgM )
		id_M_lim = devi_lgM < 4.5 # mass density higher than 10**4.5 M_sun / kpc^2

		if mm == 1:
			id_rx = obs_R < 40
			id_lim = id_nan | id_M_lim | id_rx
		else:
			id_lim = id_nan | id_M_lim

		lis_x = np.where( id_lim )[0]

		mid_cov = np.delete( cov_arr, tuple(lis_x), axis = 1)
		mid_cov = np.delete( mid_cov, tuple(lis_x), axis = 0)

		devi_R = obs_R[ id_lim == False ]
		devi_lgM = devi_lgM[ id_lim == False ]
		devi_err = lg_M_err[ id_lim == False ]

		po_param = [ mid_cov ]

		#... Log-norm
		po = [ 6.5, 50, 50 ]
		bounds = [ [3.5, 9.5], [5, 500], [5, 1000] ]
		E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = ( devi_R, devi_lgM, po_param, devi_err), method = 'L-BFGS-B', bounds = bounds,)

		print(E_return)
		popt = E_return.x
		Ie_min, Rpk_min, L_tran_min = popt
		fit_cross = np.log10( log_norm_func( obs_R, Ie_min, Rpk_min, L_tran_min ) - log_norm_func( 2e3, Ie_min, Rpk_min, L_tran_min ) )


		# plt.figure()
		# ax = plt.subplot(111)

		# ax.set_title( fig_name[mm] + ',%s-band based' % band_str)
		# ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = '.', ls = 'none', ecolor = 'r', 
		# 	alpha = 0.5, mec = 'r', mfc = 'r', capsize = 3.5, label = 'signal')

		# ax.plot( obs_R, np.log10( _cen_M - _cen_M_2Mpc ), ls = '--', color = 'g', alpha = 0.5,)
		# ax.plot( obs_R, np.log10( _out_M), ls = '-', color = 'g', alpha = 0.5, label = '$ NFW_{mis} $',)
		# ax.plot( obs_R, np.log10( 10**lg_M_sigma / 620), 'c-',)

		# ax.set_xlim( 9, 1.2e3 )
		# ax.legend( loc = 1, )
		# ax.set_ylim( 3, 8.5)
		# ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

		# ax.set_xlabel( 'R [kpc]')
		# ax.set_xscale( 'log' )
		# plt.savefig('/home/xkchen/%s_pre_compare.png' % cat_lis[mm], dpi = 300)
		# plt.show()

		plt.figure()
		plt.plot( obs_R, fit_cross, ls = '--', color = line_c[mm],)

		plt.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = line_c[mm], marker = '^', ms = 4, ls = 'none', 
			ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3,)

		plt.xlim( 3e1, 4e2)
		plt.xscale( 'log' )
		plt.ylim( 4.0, 6.25)
		plt.savefig( '/home/xkchen/%s_%s-band-based_mass-profile_mid-region_fit-test.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()


		put_params = [ mid_cov ]
		n_walk = 50

		put_x0 = np.random.uniform( 5.0, 6.5, n_walk ) # lgIx
		put_x1 = np.random.uniform( 10, 90, n_walk ) # R_pk
		put_x2 = np.random.uniform( 10, 90, n_walk ) # L_trans

		param_labels = [ '$lg\\Sigma_{x}$', '$R_{pk}$', '$L_{trans}$' ]
		pos = np.array( [ put_x0, put_x1, put_x2 ] ).T

		L_chains = 2e4

		n_dim = pos.shape[1]

		mid_file_name = fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.h5' % (cat_lis[mm], band_str)
		backend = emcee.backends.HDFBackend( mid_file_name )
		backend.reset( n_walk, n_dim )

		with Pool( 2 ) as pool:

			sampler = emcee.EnsembleSampler(n_walk, n_dim, mid_ln_p_func, args = (devi_R, devi_lgM, put_params, devi_err), pool = pool, backend = backend,)
			sampler.run_mcmc(pos, L_chains, progress = True, )
		try:
			tau = sampler.get_autocorr_time()
			flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)
		## params estimate
		mc_fits = []

		for oo in range( n_dim ):
			samp_arr = flat_samples[:, oo]
			mc_fit_oo = np.median( samp_arr )
			mc_fits.append( mc_fit_oo )

		## figs
		fig = corner.corner( flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
			levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
			plot_datapoints = True, plot_density = False, fill_contours = True,)
		axes = np.array( fig.axes ).reshape( (n_dim, n_dim) )

		for jj in range( n_dim ):
			ax = axes[jj, jj]
			ax.axvline( mc_fits[jj], color = 'r', ls = '-', alpha = 0.5,)

		for yi in range( n_dim ):
			for xi in range( yi ):
				ax = axes[yi, xi]

				ax.axvline( mc_fits[xi], color = 'r', ls = '-', alpha = 0.5,)
				ax.axhline( mc_fits[yi], color = 'r', ls = '-', alpha = 0.5,)

				ax.plot( mc_fits[xi], mc_fits[yi], 'ro', alpha = 0.5,)

		ax = axes[0, n_dim - 2 ]
		ax.set_title( fig_name[mm] + ',%s band-based' % band_str )
		plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_mid-region_mcmc-fit_params.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()


		Ie_fit, Rpk_fit, Le_fit = mc_fits[:]

		keys = ['Ie', 'R_pk', 'L_trans']
		values = [ Ie_fit, Rpk_fit, Le_fit ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill, index = ['k', 'v'])
		out_data.to_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (cat_lis[mm], band_str),)
		fit_cross = np.log10( log_norm_func( obs_R, Ie_fit, Rpk_fit, Le_fit) - log_norm_func( 2e3, Ie_fit, Rpk_fit, Le_fit) )

		plt.figure()
		plt.plot( obs_R, fit_cross, ls = '--', color = line_c[mm],)
		plt.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = line_c[mm], marker = '^', ms = 4, ls = 'none', 
			ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3,)
		plt.xlim( 3e1, 4e2)
		plt.xscale( 'log' )
		plt.ylim( 4.5, 6.25)
		plt.savefig( '/home/xkchen/%s_%s-band-based_mass-profile_mid-region_fit-compare.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()


		#... total mass profile compare
		mod_sum_M = 10**fit_cross + _out_M + ( _cen_M - _cen_M_2Mpc )
		lg_mod_M = np.log10( mod_sum_M )

		delta = lg_M - lg_mod_M
		cov_inv = np.linalg.pinv( cov_arr )
		chi2 = delta.T.dot( cov_inv ).dot(delta)
		chi2nu = chi2 / ( len(obs_R) - 1)

		print( chi2nu )

		plt.figure()
		ax = plt.subplot(111)

		ax.set_title( fig_name[mm] + ',%s-band based' % band_str)
		ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = '.', ls = 'none', ecolor = 'r', 
			alpha = 0.5, mec = 'r', mfc = 'r', capsize = 3.5, label = 'signal')

		ax.plot( obs_R, np.log10( _cen_M - _cen_M_2Mpc ), ls = '-', color = 'g', alpha = 0.5,)
		ax.plot( obs_R, fit_cross, ls = '--', color = 'g', alpha = 0.5,)

		ax.plot( obs_R, np.log10( _out_M), ls = ':', color = 'g', alpha = 0.5, label = '$ NFW_{mis} $',)
		ax.plot( obs_R, lg_mod_M, ls = '-', color = 'b', alpha = 0.5,)

		ax.text( 1e1, 4.0, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k',)

		ax.set_xlim( 9, 1.2e3 )
		ax.legend( loc = 1, )
		ax.set_ylim( 3, 8.5)
		ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

		ax.set_xlabel( 'R [kpc]')
		ax.set_xscale( 'log' )
		plt.savefig('/home/xkchen/%s_%s-band_mass-pro_separate-fit_test.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()
"""


### === sample compare
for band_str in ('gi',):

	# fig = plt.figure( figsize = (15.168, 4.8) ) # log coordinate
	# ax0 = fig.add_axes([0.04, 0.12, 0.28, 0.85])
	# ax1 = fig.add_axes([0.36, 0.12, 0.28, 0.85])
	# ax2 = fig.add_axes([0.695, 0.12, 0.28, 0.85])

	fig = plt.figure( figsize = (15.40, 4.8) )
	ax0 = fig.add_axes([0.05, 0.12, 0.275, 0.85])
	ax1 = fig.add_axes([0.38, 0.12, 0.275, 0.85])
	ax2 = fig.add_axes([0.71, 0.12, 0.275, 0.85])

	tt_cros_r, tt_cros_m, tt_cros_err = [], [], []

	for mm in range( 2 ):

		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

		id_rx = obs_R >= 9
		obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

		lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

		id_cov = np.where( id_rx )[0][0]
		cov_arr = cov_arr[id_cov:, id_cov:]

		## .. use xi_hm for sigma estimation
		if mm == 0:
			misNFW_sigma = lo_interp_F( obs_R )
			sigma_2Mpc = lo_xi2M_2Mpc + 0.
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		if mm == 1:
			misNFW_sigma = hi_interp_F( obs_R )
			sigma_2Mpc = hi_xi2M_2Mpc + 0.
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )


		out_lim_R = 250 # 200, 250, 300 kpc

		## .. 
		file_name = out_path + '%s_%s-band-based_xi2-sigma_beyond-%dkpc_fit_mcmc_fit.h5' % (cat_lis[0], band_str, out_lim_R)
		sampler = emcee.backends.HDFBackend( file_name )

		try:
			tau = sampler.get_autocorr_time()
			flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)

		mc_fits = []
		n_dim = flat_samples.shape[1]

		for oo in range( n_dim ):

			samp_arr = flat_samples[:, oo]
			mc_fit_oo = np.median( samp_arr )
			mc_fits.append( mc_fit_oo )
		bf_fit = mc_fits[0]

		# c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-300kpc_xi2M-fit.csv')
		# lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
		# lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
		# lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

		# bf_fit = lg_fb_gi


		_out_M = 10**lg_M_sigma * 10** bf_fit
		_sum_fit = np.log10( _out_M )

		## .. centeral deV profile
		c_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit.csv' % (cat_lis[mm], band_str) )
		Ie_fit, Re_fit, Ne_fit = np.array( c_dat['Ie'] )[0], np.array( c_dat['Re'] )[0], np.array( c_dat['ne'] )[0]

		_cen_M = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit)
		_cen_M_2Mpc = sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)

		devi_surf_M = surf_M - _out_M - ( _cen_M - _cen_M_2Mpc )

		devi_lgM = np.log10( surf_M - _out_M - ( _cen_M - _cen_M_2Mpc ) )

		id_nan = np.isnan( devi_lgM )
		id_M_lim = devi_lgM >= 4.5 # mass density higher than 10**4.5 M_sun / kpc^2
		
		if mm == 1:
			idx_lim = obs_R >= 40
			id_lim = (id_nan == False) & id_M_lim & ( idx_lim )

		else:
			id_lim = (id_nan == False) & id_M_lim

		devi_R = obs_R[ id_lim ]
		devi_lgM = devi_lgM[ id_lim ]
		devi_err = lg_M_err[ id_lim ]

		mid_surf_M = devi_surf_M[ id_lim ]
		mid_err = surf_M_err[ id_lim ]

		tt_cros_r.append( devi_R )
		tt_cros_m.append( devi_lgM )
		tt_cros_err.append( devi_err )

		#... model part 
		new_R = np.logspace( 0, np.log10(2.5e3), 100)

		if mm == 0:
			fit_out_M = (lo_interp_F( new_R ) - sigma_2Mpc) * 10**bf_fit

		if mm == 1:
			fit_out_M = (hi_interp_F( new_R ) - sigma_2Mpc) * 10**bf_fit

		fit_cen_M = sersic_func( new_R, 10**Ie_fit, Re_fit, Ne_fit) - sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)

		#... trans part
		tr_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (cat_lis[mm], band_str),)
		# tr_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_cp-total-outer_Lognorm-mcmc-fit.csv' % (cat_lis[mm], band_str),)
		Im_fit, Rm_fit, L_tr_fit = np.array( tr_dat['Ie'] )[0], np.array( tr_dat['R_pk'] )[0], np.array( tr_dat['L_trans'] )[0]

		fit_cross = log_norm_func( obs_R, Im_fit, Rm_fit, L_tr_fit) - log_norm_func( 2e3, Im_fit, Rm_fit, L_tr_fit)

		const = 10**(-bf_fit)

		if mm == 0:
			ax = ax0
		if mm == 1:
			ax = ax1

		# ax.errorbar( obs_R / 1e3, lg_M, yerr = lg_M_err, xerr = None, color = line_c[mm], marker = mark_s[mm], ms = 8, ls = 'none', ecolor = line_c[mm], 
		# 	mec = line_c[mm], mfc = line_c[mm], capsize = 3, label = '$\\Sigma_{\\ast}^{tot}$', alpha = 0.75)

		# ax.errorbar( devi_R / 1e3, devi_lgM, yerr = devi_err, xerr = None, color = line_c[mm], marker = mark_s[mm], ms = 8, ls = 'none', 
		# 	ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3, alpha = 0.75, 
		# 	label = '$\\Sigma_{\\ast}^{tran} \, = \, \\Sigma_{\\ast}^{tot} {-} (\\Sigma_{deV} {+} \\Sigma_{dm} / \\mathrm{%.0f} )$' % const,)

		ax.plot( obs_R / 1e3, surf_M, ls = '-', color = line_c[mm], alpha = 0.80, label = '$\\Sigma_{\\ast}^{BCG{+}ICL} $',)
		ax.fill_between( obs_R / 1e3, y1 = surf_M - surf_M_err, y2 = surf_M + surf_M_err, color = line_c[mm], alpha = 0.15,)

		ax.plot( new_R / 1e3, fit_cen_M, ls = ':', color = line_c[mm], alpha = 0.75, label = '$\\Sigma_{\\ast}^{deV}$',)

		ax.plot( new_R / 1e3, fit_out_M, ls = '--', color = 'k', alpha = 0.75, label = '$\\Sigma_{tot} / \\mathrm{%.0f} $' % const)

		ax.plot( new_R / 1e3, fit_cen_M + fit_out_M, ls = '-.', color = line_c[mm], alpha = 0.75, 
			label = '$\\Sigma_{\\ast}^{deV} {+} $' + '$\\Sigma_{tot} / \\mathrm{%.0f} $' % const,)

		ax.errorbar( devi_R / 1e3, mid_surf_M, yerr = mid_err, xerr = None, color = line_c[mm], marker = mark_s[mm], ms = 8, ls = 'none',
			ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3, alpha = 0.75,
			label = '$\\Sigma_{\\ast}^{tran} \, = \, \\Sigma_{\\ast}^{BCG{+}ICL} {-} (\\Sigma_{\\ast}^{deV} {+} \\Sigma_{tot} / \\mathrm{%.0f} )$' % const,)


		# ax2.errorbar( devi_R / 1e3, devi_lgM, yerr = devi_err, xerr = None, color = line_c[mm], marker = mark_s[mm], ms = 8, ls = 'none', 
		# 	ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3, alpha = 0.75, 
		# 	label = '$\\Sigma_{\\ast}^{tran} \,$' + '(%s)' % fig_name[mm],)

		ax2.errorbar( devi_R / 1e3, mid_surf_M, yerr = mid_err, xerr = None, color = line_c[mm], marker = mark_s[mm], ms = 8, ls = 'none',
			ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3, alpha = 0.75,
			label = fig_name[mm],)

		ax2.plot( obs_R / 1e3, fit_cross, ls = '-.', color = line_c[mm], alpha = 0.75, )

		handles, labels = ax.get_legend_handles_labels()
		handl_s = [ handles[4], handles[3], handles[0], handles[2], handles[1] ]
		label_s = [ labels[4], labels[3], labels[0], labels[2], labels[1] ]
		ax.legend( handl_s, label_s, loc = 1, frameon = False, fontsize = 12.5, markerfirst = False)
		# ax.legend(loc = 3, frameon = False, fontsize = 12.5,)

	ax.set_ylim( 10**3.5, 10**8.6 )
	ax.set_yscale( 'log' )
	ax.set_ylabel( '$\\Sigma_{\\ast} \, [M_{\\odot} \, / \, kpc^2] $', fontsize = 15)

	ax.set_xlim( 9e-3, 1.1 )
	ax.set_xlabel( 'R [Mpc]', fontsize = 15,)
	ax.set_xscale( 'log' )
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax0.annotate( text = fig_name[0], xy = (0.05, 0.05), xycoords = 'axes fraction', color = 'k', fontsize = 15,)
	ax1.annotate( text = fig_name[1], xy = (0.05, 0.05), xycoords = 'axes fraction', color = 'k', fontsize = 15,)

	ax0.set_ylim( 10**3.5, 10**8.6)
	ax0.set_yscale( 'log' )
	ax0.set_ylabel( '$\\Sigma_{\\ast} \, [M_{\\odot} \, / \, kpc^2] $', fontsize = 15)

	ax0.set_xlim( 9e-3, 1.1 )
	ax0.set_xlabel( 'R [Mpc]', fontsize = 15,)
	ax0.set_xscale( 'log' )
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax2.set_ylim( 3 * 10**4, 10**6.25 )
	ax2.set_yscale( 'log' )
	ax2.set_ylabel( '$\\Sigma_{\\ast}^{tran} \, [M_{\\odot} \, / \, kpc^2] $', fontsize = 15)

	ax2.legend(loc = 3, frameon = False, fontsize = 14, markerfirst = False)

	ax2.set_xlim( 3e-2, 4e-1)
	ax2.set_xlabel( 'R [Mpc]', fontsize = 15,)
	ax2.set_xscale( 'log' )

	ax2.set_xticks( [0.03, 0.05, 0.2, 0.4], minor = True,)
	ax2.set_xticklabels( labels = ['$\\mathrm{0.03}$', '$\\mathrm{0.05}$', '$\\mathrm{0.20}$', '$\\mathrm{0.40}$'], minor = True,)
	ax2.set_xticks( [0.1] )
	ax2.set_xticklabels( labels = ['$\\mathrm{0.10}$'])

	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/mass-bin_%s-band_based_beyond-%dkpc_fit_test.pdf' % (band_str, out_lim_R), dpi = 300)
	plt.close()
