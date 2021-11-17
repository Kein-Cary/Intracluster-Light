import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.special as special
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy import optimize
from scipy import signal
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_profile_decompose import cen_ln_p_func, mid_ln_p_func
from surface_mass_density import input_cosm_model, cosmos_param, rhom_set

import corner
import emcee
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

def sersic_err_fit_f(p, x, y, params, yerr):

	cov_mx, _ne = params[:]

	_Ie, _Re = p[:]
	_mass_cen = sersic_func( x, 10**_Ie, _Re, _ne)
	_mass_2Mpc = sersic_func( 2e3, 10**_Ie, _Re, _ne)

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	chi2 = delta.T.dot( cov_inv ).dot(delta)

	# chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === ### logNormal function
def log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * (np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
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

	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === 
def mid_like_func(p, x, y, params, yerr):

	cov_mx = params[0]

	_lg_SM0, _R_t, _sigm_tt = p[:]

	_mass_cen = log_norm_func( x, _lg_SM0, _R_t, _sigm_tt )
	_mass_2Mpc = log_norm_func( 2e3, _lg_SM0, _R_t, _sigm_tt )

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )

	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

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


### === ### load obs data
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

#. flux scaling correction
BG_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

band_str = 'gri'

#. mass estimation with deredden or not
id_dered = True
dered_str = 'with-dered_'

# id_dered = False
# dered_str = ''


### === fitting central part
def central_fit():
	if id_dered == False:

		dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
		_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

		##.. cov_arr
		with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str, 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

	if id_dered == True:

		dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )
		_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

		##.. cov_arr
		with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5', 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

	id_rx = obs_R >= 10 # 10, 9, 8
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

	id_cov = np.where( id_rx )[0][0]
	cov_arr = cov_arr[id_cov:, id_cov:]

	lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

	## inner part
	idx_lim = obs_R <= 20 # 20

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

	file_name = fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_mcmc_fit.h5' % (dered_str, band_str)

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

	plt.savefig('/home/xkchen/%stotal-sample_%s-band-based_mass-profile_cen-deV_fit_params.png' % (dered_str, band_str), dpi = 300)
	plt.close()


	Ie_fit, Re_fit = mc_fits
	Ne_fit = 4


	## .. save
	keys = [ 'Ie', 'Re', 'ne' ]
	values = [ Ie_fit, Re_fit, Ne_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % (dered_str, band_str),)

	_cen_M = sersic_func( _cp_R, 10**Ie_fit, Re_fit, Ne_fit) - sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)


	plt.figure()
	ax = plt.subplot(111)

	# ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', 
	# 	alpha = 0.75, mec = 'r', mfc = 'none', label = 'signal', ms = 5,)

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

	plt.savefig('/home/xkchen/%stotal-sample_%s-band_based_mass-pro_compare.png' % (dered_str, band_str), dpi = 300)
	plt.close()

# central_fit()


### === fitting middle part

## ... DM mass profile
lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = ( lo_xi * 1e3 * rho_m ) * h / a_ref**2
lo_rp = lo_rp * 1e3 * a_ref / h

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) * h / a_ref**2
hi_rp = hi_rp * 1e3 * a_ref / h

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )

#...
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

sigma_2Mpc = xi_to_Mf( 2e3 )


# SM(r)
if id_dered == False:

	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
	_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
	obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

	##.. cov_arr
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str, 'r') as f:
		cov_arr = np.array( f['cov_MX'] )
		cor_arr = np.array( f['cor_MX'] )

if id_dered == True:

	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )
	_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
	obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	##.. cov_arr
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5', 'r') as f:
		cov_arr = np.array( f['cov_MX'] )
		cor_arr = np.array( f['cor_MX'] )

id_rx = obs_R >= 9 # 9, 10
obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]
lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

id_cov = np.where( id_rx )[0][0]
cov_arr = cov_arr[id_cov:, id_cov:]

# central part
p_dat = pds.read_csv( fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % (dered_str,band_str),)
c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

# parameters of scaled relation
out_lim_R = 350 # 400

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % (dered_str,out_lim_R),)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]


#...trans part
cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
fit_cen_M = sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc

devi_M = surf_M - ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi - ( sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc )

devi_lgM = np.log10( devi_M )
devi_err = lg_M_err
devi_R = obs_R

# np.savetxt('/home/xkchen/total_mid-SM_data_log.txt', np.array([ devi_R, devi_lgM, devi_err ]).T, )
# np.savetxt('/home/xkchen/total_mid-SM_data.txt', np.array([ devi_R, devi_M, surf_M_err ]).T, )


id_nan = np.isnan( devi_lgM )

id_M_lim = devi_lgM < 4.0
# id_M_lim = devi_lgM < 4.5 # 4.6

id_R_x0 = obs_R < 10
id_R_x1 = obs_R >= 300
id_R_lim = id_R_x0 | id_R_x1

id_lim = (id_nan | id_M_lim) | id_R_lim
lis_x = np.where( id_lim )[0]

mid_cov = np.delete( cov_arr, tuple(lis_x), axis = 1)
mid_cov = np.delete( mid_cov, tuple(lis_x), axis = 0)

fit_R = obs_R[ id_lim == False ]
fit_lgM = devi_lgM[ id_lim == False ]
fit_err = lg_M_err[ id_lim == False ]


#. pre-fitting test
po_param = [ mid_cov ]
po = [ 7, 100, 0.8 ]
bounds = [ [3.5, 9.5], [10, 500], [0.1, 3] ]
E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = ( fit_R, fit_lgM, po_param, fit_err), method = 'L-BFGS-B', bounds = bounds,)

popt = E_return.x
lg_SM0_fit, Rt_fit, sigm_tt_fit = popt
fit_cross = np.log10( log_norm_func( obs_R, lg_SM0_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM0_fit, Rt_fit, sigm_tt_fit ) )


plt.figure()

plt.plot( devi_R, fit_cross, ls = '--', color = 'k',)
plt.plot( fit_R, fit_lgM, 'b*',)
plt.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = 'r', marker = '^', ms = 4, ls = 'none', 
	ecolor = 'r', mec = 'r', mfc = 'none', capsize = 3,)

plt.xlim( 1e1, 5e2)
plt.xscale( 'log' )
plt.ylim( 4.0, 6.3 )
plt.savefig( '/home/xkchen/%stotal-sample_mid-region_fit-test.png' % dered_str, dpi = 300)
plt.close()

raise

# #. mcmc fitting
# put_params = [ mid_cov ]
# n_walk = 50

# put_x0 = np.random.uniform( 3.5, 9.5, n_walk ) # lgIx
# put_x1 = np.random.uniform( 10, 500, n_walk ) # R_trans
# put_x2 = np.random.uniform( 0.1, 3, n_walk ) # L_trans

# param_labels = [ '$lg\\Sigma_{x}$', '$R_{t}$', '$\\sigma_{t}$' ]
# pos = np.array( [ put_x0, put_x1, put_x2 ] ).T

# L_chains = 1e4
# n_dim = pos.shape[1]

# mid_file_name = fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.h5' % (dered_str, band_str)
# backend = emcee.backends.HDFBackend( mid_file_name )
# backend.reset( n_walk, n_dim )

# with Pool( 2 ) as pool:

# 	sampler = emcee.EnsembleSampler(n_walk, n_dim, mid_ln_p_func, args = (fit_R, fit_lgM, put_params, fit_err), pool = pool, backend = backend,)
# 	sampler.run_mcmc(pos, L_chains, progress = True, )
# try:
# 	tau = sampler.get_autocorr_time()
# 	flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
# except:
# 	flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)
# ## params estimate
# mc_fits = []

# for oo in range( n_dim ):
# 	samp_arr = flat_samples[:, oo]
# 	mc_fit_oo = np.median( samp_arr )
# 	mc_fits.append( mc_fit_oo )

# ## figs
# fig = corner.corner( flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
# 	levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
# 	plot_datapoints = True, plot_density = False, fill_contours = True,)
# axes = np.array( fig.axes ).reshape( (n_dim, n_dim) )

# for jj in range( n_dim ):
# 	ax = axes[jj, jj]
# 	ax.axvline( mc_fits[jj], color = 'r', ls = '-', alpha = 0.5,)

# for yi in range( n_dim ):
# 	for xi in range( yi ):
# 		ax = axes[yi, xi]

# 		ax.axvline( mc_fits[xi], color = 'r', ls = '-', alpha = 0.5,)
# 		ax.axhline( mc_fits[yi], color = 'r', ls = '-', alpha = 0.5,)

# 		ax.plot( mc_fits[xi], mc_fits[yi], 'ro', alpha = 0.5,)

# ax = axes[2, 2]
# ax.set_title( 'All clusters,%s band-based' % band_str )
# plt.savefig('/home/xkchen/%stotal_%s-band-based_mass-profile_mid-region_mcmc-fit_params.png' % (dered_str, band_str), dpi = 300)
# plt.close()


# lg_SM_fit, Rt_fit, sigm_tt_fit = mc_fits[:]

# keys = ['lg_M0', 'R_t', 'sigma_t']
# values = [ lg_SM_fit, Rt_fit, sigm_tt_fit ]
# fill = dict( zip( keys, values) )
# out_data = pds.DataFrame( fill, index = ['k', 'v'])
# out_data.to_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (dered_str, band_str),)


# #... total mass profile compare
# fit_cross = np.log10( log_norm_func( obs_R, lg_SM_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit ) )

# _out_M = ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi
# mod_sum_M = 10**fit_cross + _out_M + fit_cen_M

# lg_mod_M = np.log10( mod_sum_M )

# delta = lg_M - lg_mod_M
# cov_inv = np.linalg.pinv( cov_arr )
# chi2 = delta.T.dot( cov_inv ).dot(delta)
# chi2nu = chi2 / ( len(obs_R) - 1)


# plt.figure()
# plt.plot( obs_R, fit_cross, ls = '--', color = 'g',)
# plt.errorbar( devi_R, devi_lgM, yerr = devi_err, xerr = None, color = 'g', marker = '^', ms = 4, ls = 'none', 
# 	ecolor = 'b', mec = 'b', mfc = 'none', capsize = 3,)
# plt.xlim( 3e1, 4e2)
# plt.xscale( 'log' )
# plt.ylim( 4.5, 6.25)
# plt.savefig( '/home/xkchen/%stotal_%s-band-based_mass-profile_mid-region_fit-compare.png' % (dered_str, band_str), dpi = 300)
# plt.close()


# plt.figure()
# ax = plt.subplot(111)
# ax.set_title('All clusters,%s-band based' % band_str)
# ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = '.', ls = 'none', ecolor = 'r', 
# 	alpha = 0.5, mec = 'r', mfc = 'r', capsize = 3.5, label = 'signal')
# ax.plot( obs_R, np.log10( fit_cen_M ), ls = '-', color = 'g', alpha = 0.5,)
# ax.plot( obs_R, fit_cross, ls = '--', color = 'g', alpha = 0.5,)

# ax.plot( obs_R, np.log10( _out_M), ls = ':', color = 'g', alpha = 0.5, label = '$ NFW_{mis} $',)
# ax.plot( obs_R, lg_mod_M, ls = '-', color = 'b', alpha = 0.5,)

# ax.text( 1e1, 4.0, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k',)

# ax.set_xlim( 9, 1.2e3 )
# ax.legend( loc = 1, )
# ax.set_ylim( 3, 8.5)
# ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

# ax.set_xlabel( 'R [kpc]')
# ax.set_xscale( 'log' )
# plt.savefig('/home/xkchen/%stotal_%s-band_mass-pro_separate-fit_test.png' % (dered_str, band_str), dpi = 300)
# plt.close()

