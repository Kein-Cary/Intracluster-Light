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
import scipy.interpolate as interp

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
from scipy import optimize
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param
from color_2_mass import get_c2mass_func
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

### === ### miscentering nfw profile (Zu et al. 2020, section 3.)
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""

	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

	theta = np.linspace( 0, 2 * np.pi, 100)
	d_theta = np.diff( theta )
	N_theta = len( theta )

	try:
		NR = len( rp )
	except:
		rp = np.array( [rp] )
		NR = len( rp )

	r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
	off_pdf = mis_p_func( r_off, sigma_off )
	dr_off = np.diff( r_off )

	NR_off = len( r_off )

	surf_dens_off = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		surf_dens_arr = np.zeros( (NR_off, N_theta), dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )
			surf_dens_arr[jj,:] = sigmam( r_cir, lgM, z, c_mass,)

		## integration on theta
		medi_surf_dens = ( surf_dens_arr[:,1:] + surf_dens_arr[:,:-1] ) / 2
		sum_theta_fdens = np.sum( medi_surf_dens * d_theta, axis = 1) / ( 2 * np.pi )

		## integration on r_off
		integ_f = sum_theta_fdens * off_pdf

		medi_integ_f = ( integ_f[1:] + integ_f[:-1] ) / 2

		surf_dens_ii = np.sum( medi_integ_f * dr_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	if NR == 1:
		return off_sigma[0]
	return off_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

	off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	norm_sigma = sigmam( rp, lgM, z, c_mass)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

### === ###
def likelihood_func(p, x, y, params, yerr):

	_Ie, _Re, Ie_x, Re_x, lg_bf = p[:]

	cov_mx, lg_sigma, _ne, ne_x = params[:]

	_mass_out = 10**lg_sigma

	_mass_cen = sersic_func( x, 10**_Ie, _Re, _ne) + sersic_func( x, 10**Ie_x, Re_x, ne_x)
	_mass_2Mpc = sersic_func( 2e3, 10**_Ie, _Re, _ne) + sersic_func( 2e3, 10**Ie_x, Re_x, ne_x)

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc + _mass_out * 10**lg_bf )

	delta = _sum_mass - y

	cov_inv = np.linalg.pinv( cov_mx )

	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

"""
## initial
def prior_p_func( p ):

	_Ie, _Re, Ix_, Rx_, lg_bf = p[:]

	identi = ( 10**6 <= 10**_Ie <= 10**9 ) & ( 5 < _Re < 35 ) & ( 10**3 <= 10**Ix_ <= 10**7 ) & ( 5e1 < Rx_ < 6e2 ) & ( 10**(-6) <= 10**lg_bf <= 10**(-0.823) )

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)
"""
def prior_p_func( p, samp_id):

	_Ie, _Re, Ix_, Rx_, lg_bf = p[:]

	# younger-bin
	if samp_id == 0:
		identi = ( 10**7.7 <= 10**_Ie <= 10**8.7) & (5 < _Re < 35) & (10**4.8 <= 10**Ix_ <= 10**6.5) & (5e1 < Rx_ < 3e2) & (10**(-3.6) <= 10**lg_bf <= 10**(-2.4) )

	# older-bin
	if samp_id == 1:
		identi = ( 10**7.8 <= 10**_Ie <= 10**8.8) & (5 < _Re < 35) & (10**4.8 <= 10**Ix_ <= 10**6.8) & (5e1 < Rx_ < 3e2) & (10**(-3.6) <= 10**lg_bf <= 10**(-2.6) )

	# # high-rich
	# if samp_id == 1:
	# 	identi = ( 10**8.0 <= 10**_Ie <= 10**9.0) & (5 < _Re < 35) & (10**4.8 <= 10**Ix_ <= 10**6.8) & (5e1 < Rx_ < 3e2) & (10**(-3.4) <= 10**lg_bf <= 10**(-2.4) )

	# # low-rich
	# if samp_id == 0:
	# 	# identi = ( 10**7.6 <= 10**_Ie <= 10**8.5) & (5 < _Re < 25) & (10**5.5 <= 10**Ix_ <= 10**8.0) & (5e1 < Rx_ < 1.5e2) & (10**(-3.6) <= 10**lg_bf <= 10**(-3.0) )
	# 	# identi = ( 10**7.6 <= 10**_Ie <= 10**8.5) & (5 < _Re < 25) & (10**5.5 <= 10**Ix_ <= 10**8.0) & (5e1 < Rx_ < 1.2e2) & (10**(-3.6) <= 10**lg_bf <= 10**(-3.0) )
	# 	identi = ( 10**7.6 <= 10**_Ie <= 10**8.5) & (5 < _Re < 25) & (10**5.0 <= 10**Ix_ <= 10**8.0) & (5e1 < Rx_ < 1.2e2) & (10**(-3.6) <= 10**lg_bf <= 10**(-3.0) )

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr, samp_id,):

	pre_p = prior_p_func( p, samp_id)
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

v_m = 200 # rho_mean = 200 * rho_c * omega_m

## miscen params for high mass
c_mass = [5.5, 6.5]
Mh0 = [14.24, 14.24]
off_set = [200, 200] # in unit kpc / h
f_off = [0.20, 0.20]

# band_str = ['gi', 'ri', 'gr']

rich_id = True

path = '/home/xkchen/mywork/ICL/code/BCG_M_based_cat/SB_pros/'
BG_path = '/home/xkchen/mywork/ICL/code/BCG_M_based_cat/BG_pros/'

if rich_id == True:
	cat_lis = [ 'low-rich', 'hi-rich' ]
	fig_name = [ 'low $\\lambda$', 'high $\\lambda$' ]

if rich_id == False:
	cat_lis = [ 'low-age', 'hi-age' ]
	fig_name = [ 'younger', 'older' ]

fit_path = '/home/xkchen/figs/tmp_mass_scaled_fit/'

"""
# for band_str in ('gi', 'ri'):
for band_str in ('gi', ):

	for mm in range( 2 ):

		dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

		## .. change mass to lg_Mass
		lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

		## .. ICL part
		## mis-NFW sigma
		misNFW_sigma = obs_sigma_func( obs_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m )
		misNFW_sigma = misNFW_sigma * h # in unit of M_sun / kpc^2
		sigma_2Mpc = obs_sigma_func( 2e3 * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h

		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		## .. fit test
		## mcmc
		put_params = [ cov_arr, lg_M_sigma, 4, 1]

		n_walk = 50

		# initial
		# put_x0 = np.random.uniform( 6, 9, n_walk ) # lgI
		# put_x1 = np.random.uniform( 5, 35, n_walk ) # Re
		# put_x2 = np.random.uniform( 3, 7, n_walk ) # lgIx
		# put_x3 = np.random.uniform( 5e1, 6e2, n_walk ) # Rx
		# put_x4 = np.random.uniform( -6, -0.823, n_walk ) # lgfb

		## younger sample
		if mm == 0:
			put_x0 = np.random.uniform( 7.7, 8.7, n_walk ) # lgI
			put_x1 = np.random.uniform( 5, 35, n_walk ) # Re
			put_x2 = np.random.uniform( 4.8, 6.5, n_walk ) # lgIx
			put_x3 = np.random.uniform( 5e1, 3e2, n_walk ) # Rx
			put_x4 = np.random.uniform( -3.6, -2.4, n_walk ) # lgfb

		## older sample
		if mm == 1:
			put_x0 = np.random.uniform( 7.8, 8.8, n_walk ) # lgI
			put_x1 = np.random.uniform( 5, 35, n_walk ) # Re
			put_x2 = np.random.uniform( 4.8, 6.8, n_walk ) # lgIx
			put_x3 = np.random.uniform( 5e1, 3e2, n_walk ) # Rx
			put_x4 = np.random.uniform( -3.6, -2.6, n_walk ) # lgfb

		'''
		## high rich sample
		if mm == 1:
			put_x0 = np.random.uniform( 8.0, 9.0, n_walk ) # lgI
			put_x1 = np.random.uniform( 5, 35, n_walk ) # Re
			put_x2 = np.random.uniform( 4.8, 6.8, n_walk ) # lgIx
			put_x3 = np.random.uniform( 5e1, 3e2, n_walk ) # Rx
			put_x4 = np.random.uniform( -3.4, -2.4, n_walk ) # lgfb

		## low rich sample
		if mm == 0:

			# put_x0 = np.random.uniform( 7.6, 8.5, n_walk ) # lgI
			# put_x1 = np.random.uniform( 5, 25, n_walk ) # Re
			# put_x2 = np.random.uniform( 5.5, 8.0, n_walk ) # lgIx
			# put_x3 = np.random.uniform( 5e1, 1.5e2, n_walk ) # Rx
			# put_x4 = np.random.uniform( -3.6, -3.0, n_walk ) # lgfb

			# put_x0 = np.random.uniform( 7.6, 8.5, n_walk ) # lgI
			# put_x1 = np.random.uniform( 5, 25, n_walk ) # Re
			# put_x2 = np.random.uniform( 5.5, 8.0, n_walk ) # lgIx
			# put_x3 = np.random.uniform( 5e1, 1.2e2, n_walk ) # Rx
			# put_x4 = np.random.uniform( -3.6, -3.0, n_walk ) # lgfb

			put_x0 = np.random.uniform( 7.6, 8.5, n_walk ) # lgI
			put_x1 = np.random.uniform( 5, 25, n_walk ) # Re
			put_x2 = np.random.uniform( 5.0, 8.0, n_walk ) # lgIx
			put_x3 = np.random.uniform( 5e1, 1.2e2, n_walk ) # Rx
			put_x4 = np.random.uniform( -3.6, -3.0, n_walk ) # lgfb
		'''

		L_chains = 2e4
		param_labels = ['$lg\\Sigma_{e}$', '$R_{e}$', '$lg\\Sigma_{x}$', '$R_{x}$', '$lgf_{b}$']

		pos = np.array( [put_x0, put_x1, put_x2, put_x3, put_x4] ).T
		n_dim = pos.shape[1]

		file_name = fit_path + 'fixed-BCG-M_%s_%s-band-based_mass-profile_2sersic-misNFW_mcmc_fit.h5' % (cat_lis[mm], band_str)

		backend = emcee.backends.HDFBackend( file_name )
		backend.reset( n_walk, n_dim )

		with Pool( 2 ) as pool:

			# sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = (obs_R, lg_M, put_params, lg_M_err), pool = pool, backend = backend,)
			sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = (obs_R, lg_M, put_params, lg_M_err, mm), pool = pool, backend = backend,)

			sampler.run_mcmc(pos, L_chains, progress = True, )

		# sampler = emcee.backends.HDFBackend( file_name )
		try:
			tau = sampler.get_autocorr_time()
			flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)

		# ## the params sets
		# bf_samp = flat_samples[:,-1]
		# idx_lim = bf_samp > -3.0

		## params estimate
		mc_fits = []
		_flat_samples = []

		for oo in range( n_dim ):

			samp_arr = flat_samples[:, oo]#[ idx_lim ]

			mc_fit_oo = np.median( samp_arr )
			mc_fits.append( mc_fit_oo )

			# _flat_samples.append( samp_arr )

		# _flat_samples = np.array( _flat_samples ).T

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
		plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_2sersic-misNFW_mcmc_params.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()

		Ie_fit, Re_fit, Ix_fit, Rx_fit, bf_fit = mc_fits[:]
		Ne_fit, nx_fit = 4, 1

		## .. save params
		keys = ['Ie', 'Re', 'Ix', 'Rx', 'lgbf']
		values = [ Ie_fit, Re_fit, Ix_fit, Rx_fit, bf_fit ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill, index = ['k', 'v'])
		out_data.to_csv( fit_path + 'fixed-BCG-M_%s_%s-band-based_mass-profile_2sersic-misNFW_mcmc.csv' % (cat_lis[mm], band_str),)


		_cen_M = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit) + sersic_func( obs_R, 10**Ix_fit, Rx_fit, nx_fit)
		_cen_M_2Mpc = sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit) + sersic_func( 2e3, 10**Ix_fit, Rx_fit, nx_fit)

		_out_M = 10**lg_M_sigma * 10** bf_fit

		_sum_fit = np.log10( _cen_M - _cen_M_2Mpc + _out_M )

		fit_cen_0 = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit)
		fit_cen_1 = sersic_func( obs_R, 10**Ix_fit, Rx_fit, nx_fit)

		delta = _sum_fit - lg_M
		cov_inv = np.linalg.pinv( cov_arr )

		chi2 = delta.T.dot( cov_inv ).dot(delta)
		chi2nu = chi2 / ( len(obs_R) - n_dim )

		## chi2 for r >= 200 kpc
		idx_lim = obs_R >= 200
		id_dex = np.where( idx_lim == True )[0][0]

		cut_cov = cov_arr[id_dex:, id_dex:]
		cut_inv = np.linalg.pinv( cut_cov )

		cut_delta = _sum_fit[idx_lim] - lg_M[idx_lim]
		cut_chi2 = cut_delta.T.dot( cut_inv ).dot(cut_delta)

		cut_chi2nu = cut_chi2 / ( np.sum(idx_lim) - n_dim )
		print( cut_chi2nu )


		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[mm] + ',%s-band based' % band_str)

		ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', 
			alpha = 0.5, mec = 'r', mfc = 'r', label = 'signal')

		ax.plot( obs_R, np.log10( fit_cen_0 ), ls = '-', color = 'g', alpha = 0.5,)
		ax.plot( obs_R, np.log10( fit_cen_1 ), ls = '--', color = 'g', alpha = 0.5,)

		ax.plot( obs_R, np.log10( _out_M), ls = ':', color = 'g', alpha = 0.5, label = '$ NFW_{mis} $',)

		ax.plot( obs_R, _sum_fit, ls = '-', color = 'b', alpha = 0.5,)

		# ax.plot( obs_R, np.log10( 10**lg_M_sigma * 10**(-3.1) ), ls = '-', color = 'g', alpha = 0.5,)

		ax.text( 4e1, 7.5, s = '$ lg\\Sigma_{e} = %.2f$' % Ie_fit + '\n' + '$R_{e} = %.2f$' % Re_fit + 
			'\n' + '$ n = %.2f $' % Ne_fit, color = 'b',)
		ax.text( 1e2, 7.5, s = '$ lg\\Sigma_{e} = %.2f$' % Ix_fit + '\n' + '$R_{e} = %.2f$' % Rx_fit + 
			'\n' + '$ n = %.2f $' % nx_fit, color = 'b',)

		ax.text( 1e1, 4.0, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k',)
		ax.text( 1e1, 3.5, s = '$\\chi^{2} / \\nu[R>=200] = %.5f$' % cut_chi2nu, color = 'k',)

		ax.legend( loc = 1, )
		ax.set_ylim( 3, 8.5)
		ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

		ax.set_xlabel( 'R [kpc]')
		ax.set_xscale( 'log' )
		plt.savefig('/home/xkchen/%s_%s-band_based_2sersic-misNFW_fit_test.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()
"""

### === ### cross compare
for band_str in ('gi', ):

	fig = plt.figure()
	ax0 = fig.add_axes([0.15, 0.15, 0.75, 0.75])

	for mm in range( 2 ):

		dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

		## .. change mass to lg_Mass
		lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

		## .. ICL part (mis_NFW)
		misNFW_sigma = obs_sigma_func( obs_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m )
		misNFW_sigma = misNFW_sigma * h # in unit of M_sun / kpc^2

		sigma_2Mpc = obs_sigma_func( 2e3 * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h

		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		## fit params
		p_dat = pds.read_csv( fit_path + 'fixed-BCG-M_%s_%s-band-based_mass-profile_2sersic-misNFW_mcmc.csv' % (cat_lis[mm], band_str) )

		Ie_fit, Re_fit = np.array(p_dat['Ie'])[0], np.array(p_dat['Re'])[0]
		Ix_fit, Rx_fit, bf_fit = np.array(p_dat['Ix'])[0], np.array(p_dat['Rx'])[0], np.array(p_dat['lgbf'])[0]

		Ne_fit, nx_fit = 4, 1

		n_dim = 5

		_cen_M = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit) + sersic_func( obs_R, 10**Ix_fit, Rx_fit, nx_fit)
		_cen_M_2Mpc = sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit) + sersic_func( 2e3, 10**Ix_fit, Rx_fit, nx_fit)
		_out_M = 10**lg_M_sigma * 10** bf_fit
		_sum_fit = np.log10( _cen_M - _cen_M_2Mpc + _out_M )

		## model lines
		fit_cen_0 = sersic_func( obs_R, 10**Ie_fit, Re_fit, Ne_fit) - sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)
		fit_cen_1 = sersic_func( obs_R, 10**Ix_fit, Rx_fit, nx_fit) - sersic_func( 2e3, 10**Ix_fit, Rx_fit, nx_fit)

		delta = _sum_fit - lg_M
		cov_inv = np.linalg.pinv( cov_arr )

		chi2 = delta.T.dot( cov_inv ).dot(delta)
		chi2nu = chi2 / ( len(obs_R) - n_dim )

		## chi2 for r >= 200 kpc
		idx_lim = obs_R >= 200
		id_dex = np.where( idx_lim == True )[0][0]

		cut_cov = cov_arr[id_dex:, id_dex:]
		cut_inv = np.linalg.pinv( cut_cov )

		cut_delta = _sum_fit[idx_lim] - lg_M[idx_lim]
		cut_chi2 = cut_delta.T.dot( cut_inv ).dot(cut_delta)

		cut_chi2nu = cut_chi2 / ( np.sum(idx_lim) - n_dim )
		print( cut_chi2nu )


		## mcmc fit results
		file_name = fit_path + 'fixed-BCG-M_%s_%s-band-based_mass-profile_2sersic-misNFW_mcmc_fit.h5' % (cat_lis[mm], band_str)
		sampler = emcee.backends.HDFBackend( file_name )

		try:
			tau = sampler.get_autocorr_time()
			flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)

		mc_fits = []

		for oo in range( n_dim ):

			samp_arr = flat_samples[:, oo]

			mc_fit_0 = np.percentile( samp_arr, 16 )
			mc_fit_1 = np.percentile( samp_arr, 84 )
			mc_fits.append( [ mc_fit_0, mc_fit_1 ] )	

		devi_Ie = mc_fits[0] - Ie_fit
		devi_Re = mc_fits[1] - Re_fit
		devi_Ix = mc_fits[2] - Ix_fit
		devi_Rx = mc_fits[3] - Rx_fit
		devi_bf = mc_fits[4] - bf_fit

		ax0.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = line_c[mm], marker = '.', ls = 'none', ecolor = line_c[mm], 
			alpha = 0.5, mec = line_c[mm], mfc = line_c[mm], capsize = 3, label = fig_name[mm] + ' at fixed $M_{\\ast}^{BCG}$',)

		ax0.plot( obs_R, np.log10( fit_cen_0 ), ls = ':', color = line_c[mm], alpha = 0.5, )
		ax0.plot( obs_R, np.log10( fit_cen_1 ), ls = '-', color = line_c[mm], alpha = 0.5, )

		ax0.plot( obs_R, np.log10( _out_M), ls = '-.', color = line_c[mm], alpha = 0.5, )
		ax0.plot( obs_R, _sum_fit, ls = '--', color = line_c[mm], alpha = 0.5, linewidth = 1.25,)

	legend_2 = plt.legend( ['n=4 sersic', 'n=1 sersic', 'scaled $NFW_{projected}^{miscentering}$', 'Best fitting'], loc = 3, frameon = False, fontsize = 13,)
	legend_20 = ax0.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.gca().add_artist( legend_2 )

	ax0.legend( loc = 1, fontsize = 13.5, frameon = False,)
	ax0.set_ylim( 3, 8.5)
	ax0.set_xlim( 1e1, 1e3)
	ax0.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)

	ax0.set_xlabel( 'R [kpc]', fontsize = 15,)
	ax0.set_xscale( 'log' )
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	if rich_id == True:
		plt.savefig('/home/xkchen/rich-bin_mass_pro_fit_compare.png', dpi = 300)
	if rich_id == False:
		plt.savefig('/home/xkchen/age-bin_mass_pro_fit_compare.png', dpi = 300)
	plt.close()
