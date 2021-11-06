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

from surface_mass_density import sigmam, sigmac
from surface_mass_density import input_cosm_model, cosmos_param, rhom_set
from color_2_mass import get_c2mass_func, gi_band_c2m_func

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

### ... minimize func.
def sersic_func(r, Ie, re, ndex):

	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def sersic_err_fit_f(p, x, y, params, yerr):

	cov_mx = params[0]

	_Ie, _Re, _ne = p[:]
	_mass_cen = sersic_func( x, 10**_Ie, _Re, _ne)
	_mass_2Mpc = sersic_func( 2e3, 10**_Ie, _Re, _ne)

	_sum_mass = np.log10( _mass_cen - _mass_2Mpc )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )

	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### ...mcmc fit
### === ### fitting center region
def cen_like_func(p, x, y, params, yerr):

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
		return -0.5 * chi2
	return -np.inf

def cen_prior_p_func( p ):

	_Ie, _Re = p[:]

	identi = (10**3.5 <= 10**_Ie <= 10**9.5) & ( 5 < _Re < 30 )

	if identi:
		return 0
	return -np.inf

def cen_ln_p_func(p, x, y, params, yerr):

	pre_p = cen_prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + cen_like_func(p, x, y, params, yerr)

### === ### fitting outer region
def likelihood_func(p, x, y, params, yerr):

	bf = p[0]

	cov_mx, lg_sigma = params[:]
	_mass_out = 10**lg_sigma
	_sum_mass = np.log10( _mass_out * 10**bf )
	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def prior_p_func( p, icl_mode):

	bf = p[0]

	if icl_mode == 'xi2M':
		identi = 10**(-4.0) < 10**bf < 10**(-2.5)

	if icl_mode == 'SG_N':
		identi = 9.5 <= bf <= 11.5

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr, icl_mode):
	"""
	icl_mode = 'xi2M' or 'SG_N', 
				scaling the total mass profile or satellite galaxy number density
	"""
	pre_p = prior_p_func( p, icl_mode)
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

def outer_err_fit_func(p, x, y, params, yerr):

	bf = p[0]
	cov_mx, lg_sigma = params[:]

	_mass_out = 10**lg_sigma
	_sum_mass = np.log10( _mass_out * 10**bf )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === ### fitting mid-region
#...logNormal function
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
	chi2 = delta.T.dot( cov_inv ).dot(delta)

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
	chi2 = delta.T.dot( cov_inv ).dot(delta)

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

