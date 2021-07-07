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

### ...mcmc fit
### === ### fitting center region
def cen_like_func(p, x, y, params, yerr):

	_Ie, _Re = p[:]

	cov_mx, _ne = params[:]

	_mass_cen = sersic_func( x, 10**_Ie, _Re, _ne)

	_sum_mass = np.log10( _mass_cen )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def cen_prior_p_func( p ):

	_Ie, _Re = p[:]

	identi = (10**5.5 <= 10**_Ie <= 10**9.5) & ( 5 < _Re < 30 )

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

