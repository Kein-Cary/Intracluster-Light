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

from surface_mass_profile_decompose import cen_ln_p_func, mid_ln_p_func
from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set
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

### === ### fitting central region
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def log_norm_pdf_func( r, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * ( np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_P = -1 * ( lg_A0 + lg_A1)
	return 10**lg_P

def log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e ) * ( np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_M = lg_SM0 - lg_A0 - lg_A1

	return 10**lg_M

def lg_norm_err_fit_f(p, x, y, params, yerr):

	cov_mx = params[0]

	_lg_SM0, _R_t, _sigm_tt = p[:]

	_mass_cen = log_norm_func( x, _lg_SM0, _R_t, _sigm_tt )
	_mass_2Mpc = log_norm_func( 2e3, _lg_SM0, _R_t, _sigm_tt )

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )

	# chi2 = delta.T.dot( cov_inv ).dot(delta) # for low-mass bin

	chi2 = np.sum( delta**2 / yerr**2 ) # for high mass bin

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === ### fitting mid-region
def mid_like_func(p, x, y, params, yerr):

	cov_mx = params[0]

	_lg_SM0, _R_t, _sigm_tt = p[:]

	_mass_cen = log_norm_func( x, _lg_SM0, _R_t, _sigm_tt )
	_mass_2Mpc = log_norm_func( 2e3, _lg_SM0, _R_t, _sigm_tt )

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )

	# chi2 = delta.T.dot( cov_inv ).dot(delta)  # for low-mass bin

	chi2 = np.sum( delta**2 / yerr**2 ) # for high mass bin

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def mid_prior_p_func( p ):

	_lg_SM0, _R_t, _sigm_tt = p[:]
	identi = ( 3.5 <= _lg_SM0 <= 9.5 ) & ( 10 <= _R_t <= 500 ) & ( 0.1 <= _sigm_tt <= 3 )

	if identi:
		return 0
	return -np.inf

def mid_ln_p_func(p, x, y, params, yerr):

	pre_p = mid_prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + mid_like_func(p, x, y, params, yerr)


### === load data
color_s = [ 'r', 'g', 'darkred' ]
line_c = ['b', 'r']
mark_s = ['s', 'o']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value


#. estimate mass profile of large scale from xi_mh
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


xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

misNFW_sigma = xi_to_Mf( lo_rp )
sigma_2Mpc = xi_to_Mf( 2e3 )
lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

tot_interp_M_F = interp.interp1d( lo_rp, misNFW_sigma, kind	= 'linear', fill_value = 'extrapolate',)


BG_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

band_str = 'gri'

# mass estimation with deredden or not
id_dered = True
dered_str = '_with-dered'

# id_dered = False
# dered_str = ''

"""
### === fitting central region
for mm in range( 2 ):

	if id_dered == False:
		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )

		_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['medi_correct_surf_M']), np.array(dat['surf_M_err'])
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

	if id_dered == True:
		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str) )

		_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['medi_correct_surf_M']), np.array(dat['surf_M_err'])
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )		

	id_rx = obs_R >= 8 # 10, 9, 8
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

	id_cov = np.where( id_rx )[0][0]
	cov_arr = cov_arr[id_cov:, id_cov:]

	#. mass in lg_Mstar
	lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

	## inner part
	idx_lim = obs_R <= 20

	id_dex = np.where( idx_lim == True)[0][-1]
	cut_cov = cov_arr[:id_dex+1, :id_dex+1]

	fit_R, fit_M, fit_err = obs_R[idx_lim], lg_M[idx_lim], lg_M_err[idx_lim]

	## .. fit test
	put_params = [ cut_cov, 4 ]

	# initial
	n_walk = 50

	put_x0 = np.random.uniform( 5.5, 9.5, n_walk ) # lgI
	put_x1 = np.random.uniform( 5, 30, n_walk ) # Re

	L_chains = 1e4
	param_labels = ['$lg\\Sigma_{e}$', '$R_{e}$' ]

	pos = np.array( [ put_x0, put_x1 ] ).T
	n_dim = pos.shape[1]

	file_name = fit_path + '%s_%s-band-based_mass-profile_cen-deV_mcmc_fit%s.h5' % (cat_lis[mm], band_str, dered_str)

	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 2 ) as pool:

		sampler = emcee.EnsembleSampler(n_walk, n_dim, cen_ln_p_func, args = (fit_R, fit_M, put_params, fit_err), pool = pool, backend = backend,)
		sampler.run_mcmc(pos, L_chains, progress = True, )
	try:
		tau = sampler.get_autocorr_time()
		flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain( discard = 1000, thin = 200, flat = True)
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

	plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_cen-deV_fit_params%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()

	#. fitting compare
	Ie_fit, Re_fit = mc_fits
	Ne_fit = 4

	#. save parameters
	keys = [ 'Ie', 'Re', 'ne' ]
	values = [ Ie_fit, Re_fit, Ne_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[mm], band_str, dered_str),)

	_cen_M = sersic_func( _cp_R, 10**Ie_fit, Re_fit, Ne_fit) - sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)


	plt.figure()
	ax = plt.subplot(111)

	ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', 
		alpha = 0.75, mec = 'r', mfc = 'none', label = 'signal', ms = 5,)

	ax.plot( _cp_R, np.log10( _cp_SM ), 'g:')
	ax.plot( obs_R, lg_M, ls = '-', color = 'r', alpha = 0.5,)

	ax.plot( _cp_R, np.log10( _cen_M ), ls = '--', color = 'b', alpha = 0.75, label = 'BCG',)
	ax.text( 1e1, 4.5, s = '$ lg\\Sigma_{e} = %.2f$' % Ie_fit + '\n' + '$R_{e} = %.2f$' % Re_fit + 
		'\n' + '$ n = %.2f $' % Ne_fit, color = 'b',)
	ax.legend( loc = 1, )
	ax.set_ylim( 4, 9.5 )

	ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )
	ax.set_xlabel( 'R [kpc]')
	ax.set_xscale( 'log' )

	plt.savefig('/home/xkchen/%s_%s-band_based_mass-pro_compare%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()

raise
"""

### === cross part fitting
def view_on_lognorm():
	lg_xM = 6.5
	Rx = np.logspace(0, 3, 100)

	Rt = 100
	sigm_t = np.arange( 1, 5, 0.5)
	pre_N = len( sigm_t )
	test_lis = 'sigma_t'

	# Rt = np.arange( 50, 550, 50)
	# sigm_t = 1
	# pre_N = len( Rt )
	# test_lis = 'Rt'


	plt.figure()

	for kk in range( pre_N ):

		_tt_M = log_norm_func( Rx, lg_xM, Rt, sigm_t[kk] )
		plt.plot( Rx, _tt_M, ls = '-', color = mpl.cm.rainbow(kk / pre_N ), alpha = 0.5, label = '$\\sigma_{t} = %.1f$' % sigm_t[kk])

		# _tt_M = log_norm_func( Rx, lg_xM, Rt[kk], sigm_t )
		# plt.plot( Rx, _tt_M, ls = '-', color = mpl.cm.rainbow(kk / pre_N ), alpha = 0.5, label = 'Rt = %d' % Rt[kk])

	plt.legend( loc = 4,)
	plt.xscale( 'log' )
	plt.xlabel( 'Mpc' )
	plt.yscale( 'log' )
	plt.ylabel( '$\\Sigma_{\\ast}$' )
	plt.savefig('/home/xkchen/%s_lg_normal_SM_pro_test.png' % test_lis, dpi = 300)
	plt.close()

# view_on_lognorm()


out_lim_R = 350 # 400

for mm in range( 1,2 ):

	dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )

	_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['medi_correct_surf_M']), np.array(dat['surf_M_err'])
	obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	id_rx = obs_R >= 10.
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

	##.. cov_arr
	with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
		cov_arr = np.array( f['cov_MX'] )
		cor_arr = np.array( f['cor_MX'] )

	id_cov = np.where( id_rx )[0][0]
	cov_arr = cov_arr[id_cov:, id_cov:]

	#. mass in lg_Mstar
	lg_M, lg_M_err = np.log10( surf_M ), np.sqrt( np.diag(cov_arr) ) # surf_M_err / ( np.log(10) * surf_M )


	#.. use params of total sample for large scale
	if id_dered == False:
		c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R )
	if id_dered == True:
		c_dat = pds.read_csv( fit_path + 'with-dered_total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R )

	lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
	lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
	lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

	# _out_M = ( tot_interp_M_F( obs_R ) - sigma_2Mpc ) * 10**lg_fb_gi
	if mm == 0:
		_out_M = ( lo_interp_F( obs_R ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi

	if mm == 1:
		_out_M = ( hi_interp_F( obs_R ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi


	## .. centeral deV profile
	c_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[mm], band_str, dered_str),)
	Ie_fit, Re_fit, Ne_fit = np.array( c_dat['Ie'] )[0], np.array( c_dat['Re'] )[0], np.array( c_dat['ne'] )[0]

	_cen_M = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit)
	_cen_M_2Mpc = sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)


	##.. mid-region
	diffi_lgM = np.log10( surf_M - _out_M - ( _cen_M - _cen_M_2Mpc ) )

	id_nan = np.isnan( diffi_lgM )

	devi_R = obs_R[ id_nan == False ]
	devi_lgM = diffi_lgM[ id_nan == False ]
	devi_err = lg_M_err[ id_nan == False ]

	#. use to fitting
	if mm == 0:
		# id_M_lim = diffi_lgM < 4.55 # 4.6 # mass density limit
		# id_rx = obs_R < 45

		#. adjust
		# id_rx0 = obs_R < 30
		# id_rx1 = ( 45 <= obs_R ) & ( obs_R < 50 )
		# id_rx2 = ( 90 <= obs_R ) & ( obs_R < 110 )
		# id_rx = id_rx0 | id_rx1 | id_rx2

		id_M_lim = diffi_lgM < 4.0

		id_rx0 = obs_R < 40
		id_rx1 = obs_R >= 300
		id_rx2 = ( 45 <= obs_R ) & ( obs_R < 50 )
		id_rx = id_rx0 | id_rx1 | id_rx2

	if mm == 1:

		# id_M_lim = diffi_lgM < 4.
		# id_rx = obs_R < 40

		id_M_lim = diffi_lgM < 4. # 4.

		id_rx0 = obs_R < 30 # 35, 45
		id_rx1 = obs_R > 300 # 270
		id_rx = id_rx0 | id_rx1

	id_lim = ( id_nan | id_M_lim ) | id_rx

	lis_x = np.where( id_lim )[0]


	mid_cov = np.delete( cov_arr, tuple(lis_x), axis = 1)
	mid_cov = np.delete( mid_cov, tuple(lis_x), axis = 0)

	mid_cor = np.delete( cor_arr, tuple(lis_x), axis = 1)
	mid_cor = np.delete( mid_cor, tuple(lis_x), axis = 0)

	fit_R = obs_R[ id_lim == False ]
	fit_lgM = diffi_lgM[ id_lim == False ]
	fit_err = lg_M_err[ id_lim == False ]

	po_param = [ mid_cov ]


	#... Log-norm
	po = [ 6, 100, 1 ]
	bounds = [ [3.5, 9.5], [10, 500], [0.1, 3] ]
	E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = ( fit_R, fit_lgM, po_param, fit_err), method = 'L-BFGS-B', bounds = bounds,)

	print(E_return)
	popt = E_return.x

	lg_SM0_fit, Rt_fit, sigm_tt_fit = popt
	fit_cross = np.log10( log_norm_func( obs_R, lg_SM0_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM0_fit, Rt_fit, sigm_tt_fit ) )

	plt.figure()
	plt.title( fig_name[mm] )
	plt.imshow( mid_cor, origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1,)
	plt.savefig('/home/xkchen/%s_%s-band-based_mid-cor%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()

	plt.figure()

	plt.plot( obs_R, fit_cross, ls = '--', color = 'b',)
	plt.plot( fit_R, fit_lgM, 'r*',)
	plt.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = 'b', marker = '^', ms = 4, ls = 'none', 
		ecolor = 'b', mec = 'b', mfc = 'none', capsize = 3,)

	# plt.axvline( x = 280, ls = ':', color = 'k')
	# plt.axhline( y = 5, ls = '-', color = 'k')
	plt.xlim( 9e0, 6e2)
	plt.xscale( 'log' )
	plt.xlabel('R [kpc]')
	plt.ylim( 3.5, 6.25)
	plt.ylabel('$\\lg M_{\\ast}$')
	plt.savefig( '/home/xkchen/%s_%s-band-based_mass-profile_mid-region_fit-test%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()



	put_params = [ mid_cov ]
	n_walk = 50

	put_x0 = np.random.uniform( 3.5, 9.5, n_walk ) # lgIx
	put_x1 = np.random.uniform( 10, 500, n_walk ) # R_trans
	put_x2 = np.random.uniform( 0.1, 3, n_walk ) # L_trans

	param_labels = [ '$lg\\Sigma_{x}$', '$R_{t}$', '$\\sigma_{t}$' ]
	pos = np.array( [ put_x0, put_x1, put_x2 ] ).T

	L_chains = 1e4
	n_dim = pos.shape[1]

	mid_file_name = fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.h5' % (cat_lis[mm], band_str, dered_str)
	backend = emcee.backends.HDFBackend( mid_file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 2 ) as pool:

		sampler = emcee.EnsembleSampler(n_walk, n_dim, mid_ln_p_func, args = ( fit_R, fit_lgM, put_params, fit_err), pool = pool, backend = backend,)
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
	plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_mid-region_mcmc-fit_params%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()


	lg_SM_fit, Rt_fit, sigm_tt_fit = mc_fits[:]

	keys = ['lg_M0', 'R_t', 'sigma_t']
	values = [ lg_SM_fit, Rt_fit, sigm_tt_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[mm], band_str, dered_str),)


	#... total mass profile compare
	fit_cross = np.log10( log_norm_func( obs_R, lg_SM_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit ) )
	mod_sum_M = 10**fit_cross + _out_M + ( _cen_M - _cen_M_2Mpc )
	lg_mod_M = np.log10( mod_sum_M )

	delta = lg_M - lg_mod_M
	cov_inv = np.linalg.pinv( cov_arr )
	chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2nu = chi2 / ( len(obs_R) - 1)


	plt.figure()
	plt.plot( obs_R, fit_cross, ls = '--', color = line_c[mm],)
	plt.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = line_c[mm], marker = '^', ms = 4, ls = 'none', 
		ecolor = line_c[mm], mec = line_c[mm], mfc = 'none', capsize = 3,)
	plt.xlim( 3e1, 4e2)
	plt.xscale( 'log' )
	plt.ylim( 4.5, 6.25)
	plt.savefig( '/home/xkchen/%s_%s-band-based_mass-profile_mid-region_fit-compare%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()


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
	plt.savefig('/home/xkchen/%s_%s-band_mass-pro_separate-fit_test%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()

raise
