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

from surface_mass_profile_decompose import cen_ln_p_func
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


### === ### load obs data
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

#.
BG_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'


band_str = 'gri'

#. mass estimation with deredden or not
id_dered = True
dered_str = 'with-dered_'


### === fitting central part
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

