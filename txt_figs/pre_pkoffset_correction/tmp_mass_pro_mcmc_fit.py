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

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize

from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param
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

### === ### halo profile model
from colossus.cosmology import cosmology
from colossus.halo import profile_einasto
from colossus.halo import profile_nfw
from colossus.halo import profile_hernquist

cosmos_param = {'flat': True, 'H0': H0, 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8' : 0.811, 'ns': 0.965}
cosmology.addCosmology('myCosmo', cosmos_param )
cosmo = cosmology.setCosmology( 'myCosmo' )

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

def misHern_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m, query_func):

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

			surf_dens_arr[jj,:] = query_func( r_cir ) # unit M_sun h / kpc^2

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

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m, profile_str = 'NFW'):

	if profile_str == 'NFW':
		off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
		norm_sigma = sigmam( rp, lgM, z, c_mass)

	if profile_str == 'Hern':
		rho_Hern = profile_hernquist.HernquistProfile( M = 10**(lgM), c = c_mass, z = z, mdef = '200m')
		off_sigma = misHern_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
		norm_sigma = rho_Hern.surfaceDensity( rp )

	if profile_str == 'Ein':
		rho_einasto = profile_einasto.EinastoProfile( M = 10**(lgM), c = c_mass, z = z, mdef = '200m')
		off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
		norm_sigma = rho_einasto.surfaceDensity( rp )

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

### === ### fitting function
def likelihood_func(p, x, y, params, yerr):

	_Ie, _Re, _ne, bf = p[:]

	cov_mx, lg_sigma = params[:]

	_mass_cen = sersic_func( x, 10**_Ie, _Re, _ne)

	_mass_out = 10**lg_sigma

	_sum_mass = np.log10( _mass_cen + _mass_out * 10**bf )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )

	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def prior_p_func( p ):

	_Ie, _Re, _ne, bf = p[:]

	identi = ( 1e7 <= 10**_Ie <= 1e11 ) & ( 5 < _Re < 50 ) & ( 1 <= _ne <= 10 ) & (1e-4 < 10**bf < 1e-2)

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)


"""
## adjust for younger (misHern)
def prior_p_func( p, samp_str):

	_Ie, _Re, _ne, bf = p[:]

	if samp_str == 'ri':
		identi = ( 10**7.75 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 15 ) & ( 3 <= _ne <= 6.5 ) & ( 10**(-3.3) < 10**bf < 10**(-2.7) )

	if samp_str == 'gi':
		identi = ( 10**8.0 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 15 ) & ( 4 <= _ne <= 10 ) & ( 10**(-3.6) < 10**bf < 10**(-2.7) )

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr, samp_str):

	pre_p = prior_p_func( p, samp_str )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)
"""

"""
## adjust for mass bin (misHern)
def prior_p_func( p, samp_id, samp_str):

	_Ie, _Re, _ne, bf = p[:]

	if (samp_id == 0) & (samp_str == 'ri'):
		identi = ( 10**7.5 <= 10**_Ie <= 10**8.8 ) & ( 2 < _Re < 15 ) & ( 2 <= _ne <= 10 ) & ( 10**(-3.5) < 10**bf < 10**(-2.7) )

	if (samp_id == 0) & (samp_str == 'gi'):
		identi = ( 10**8.0 <= 10**_Ie <= 10**9.0 ) & ( 2 < _Re < 15 ) & ( 2 <= _ne <= 9 ) & ( 10**(-3.5) < 10**bf < 10**(-2.8) )

	if (samp_id == 1) & (samp_str == 'ri'):
		identi = ( 10**8.0 <= 10**_Ie <= 10**8.7 ) & ( 4.5 < _Re < 15 ) & ( 6 <= _ne <= 10 ) & ( 10**(-3.5) < 10**bf < 10**(-2.8) )

	if (samp_id == 1) & (samp_str == 'gi'):
		identi = ( 10**7.8 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 20 ) & ( 1 <= _ne <= 6 ) & ( 10**(-3.3) < 10**bf < 10**(-2.7) )

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr, samp_id, samp_str):

	pre_p = prior_p_func( p, samp_id, samp_str)
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)
"""

"""
## adjust (misNFW)
def prior_p_func( p, samp_id, samp_str):

	_Ie, _Re, _ne, bf = p[:]

	## age-bin
	# if (samp_id == 0) & (samp_str == 'ri'):
	# 	identi = ( 10**7.5 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 15 ) & ( 6 <= _ne <= 10 ) & ( 10**(-2.8) < 10**bf < 10**(-2.0) )

	# if (samp_id == 0) & (samp_str == 'gi'):
	# 	identi = ( 10**8.0 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 15 ) & ( 6 <= _ne <= 10 ) & ( 10**(-3.4) < 10**bf < 10**(-2.4) )

	# if (samp_id == 1) & (samp_str == 'ri'):
	# 	identi = ( 10**7.0 <= 10**_Ie <= 10**8.0 ) & ( 5 < _Re < 25 ) & ( 6 <= _ne <= 10 ) & ( 10**(-3.2) < 10**bf < 10**(-2.4) )

	# if (samp_id == 1) & (samp_str == 'gi'):
	# 	identi = ( 10**7.7 <= 10**_Ie <= 10**8.4 ) & ( 5 < _Re < 20 ) & ( 6 <= _ne <= 10 ) & ( 10**(-3.5) < 10**bf < 10**(-2.5) )

	## mass-bin
	if (samp_id == 0) & (samp_str == 'ri'):
		identi = ( 10**7.75 <= 10**_Ie <= 10**8.6 ) & ( 3 < _Re < 15 ) & ( 4 <= _ne <= 10 ) & ( 10**(-3.0) < 10**bf < 10**(-2.4) )

	if (samp_id == 0) & (samp_str == 'gi'):
		identi = ( 10**8.4 <= 10**_Ie <= 10**9.2 ) & ( 1 < _Re < 15 ) & ( 5.5 <= _ne <= 10 ) & ( 10**(-3.2) < 10**bf < 10**(-2.4) )

	if (samp_id == 1) & (samp_str == 'ri'):
		identi = ( 10**7.8 <= 10**_Ie <= 10**8.5 ) & ( 1 < _Re < 25 ) & ( 6 <= _ne <= 10 ) & ( 10**(-3.0) < 10**bf < 10**(-2.4) )

	if (samp_id == 1) & (samp_str == 'gi'):
		identi = ( 10**7.8 <= 10**_Ie <= 10**8.4 ) & ( 5 < _Re < 20 ) & ( 1 <= _ne <= 6 ) & ( 10**(-3.1) < 10**bf < 10**(-2.3) )

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr, samp_id, samp_str):

	pre_p = prior_p_func( p, samp_id, samp_str)
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)
"""

"""
## up-limit lgfb fitting
def prior_p_func( p, samp_id, samp_str):

	_Ie, _Re, _ne, bf = p[:]

	## age-bin
	if (samp_id == 0) & (samp_str == 'gi'):
		identi = ( 10**7.8 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 15 ) & ( 4 <= _ne <= 10 ) & ( 10**(-4.0) < 10**bf <= 10**(-3.2) )

	## mass-bin
	# if (samp_id == 1) & (samp_str == 'gi'):
	# 	identi = ( 10**7.5 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 25 ) & ( 4 <= _ne <= 10 ) & ( 10**(-4.0) < 10**bf <= 10**(-3.2) )

	if (samp_id == 1) & (samp_str == 'gi'):
		identi = ( 10**7.5 <= 10**_Ie <= 10**8.5 ) & ( 5 < _Re < 25 ) & ( 2 <= _ne <= 8 ) & ( 10**(-4.0) < 10**bf <= 10**(-3.2) )

	if identi:
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr, samp_id, samp_str):

	pre_p = prior_p_func( p, samp_id, samp_str)
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)
"""


### === ### load data
path = '/home/xkchen/project/tmp_mass_pro_fit/'
BG_path = '/home/xkchen/project/tmp_mass_pro_fit/'

# out_path = '/home/xkchen/project/tmp_mass_pro_fit/'
out_path = '/home/xkchen/figs/'

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

v_m = 200 # rho_mean = 200 * rho_c * omega_m

## miscen params for high mass
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24]
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

NFW_on = True

# mass_id = False

for mass_id in (True, False):

	if mass_id == True:

		cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
		fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

	if mass_id == False:

		cat_lis = ['younger', 'older']
		fig_name = ['younger', 'older']

	# for band_str in ('gi', 'gr', 'ri'):
	for band_str in ('gi', ): ## up-limit in lgfb fitting

		for mm in range( 2 ):

			dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
			obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

			## .. change mass to lg_Mass
			lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

			with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
				cov_arr = np.array( f['cov_MX'] )

			## .. ICL part
			if NFW_on == True:
				## mis-NFW sigma
				misNFW_sigma = obs_sigma_func( obs_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m )
				misNFW_sigma = misNFW_sigma * h # in unit of M_sun / kpc^2

				lg_M_sigma = np.log10( misNFW_sigma )

			if NFW_on == False:
				## mis-Hern sigma (Hernquist)
				theta = np.linspace( 0, 2 * np.pi, 100)
				r_off = np.arange( 0, 15 * off_set[mm], 0.02 * off_set[mm] ) ## in unit kpc / h

				R_min = np.inf
				R_max = -np.inf

				for ii in range( len(theta) ):

					for jj in range( len(r_off) ):

						r_cir = np.sqrt( (obs_R * h)**2 + 2 * ( obs_R * h ) * r_off[jj] * np.cos( theta[ii] ) + r_off[jj]**2 )

						R_min = np.min( [ R_min, r_cir.min() ] )
						R_max = np.max( [ R_max, r_cir.max() ] )

				full_R = np.logspace( np.log10(R_min), np.log10(R_max), 10000)

				rho_Hern = profile_hernquist.HernquistProfile( M = 10**( Mh0[mm] ), c = c_mass[mm], z = z_ref, mdef = '200m')
				Hern_sigma = rho_Hern.surfaceDensity( full_R )
				intep_Hern_f = interp.interp1d( full_R, Hern_sigma, kind = 'slinear', fill_value = 'extrapolate')

				off_Hern_sigm = misHern_sigma_func( obs_R * h, off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m, intep_Hern_f )
				norm_Hern_ = rho_Hern.surfaceDensity( obs_R * h )
				obs_Hern_sigm = f_off[mm] * off_Hern_sigm + ( 1 - f_off[mm] ) * norm_Hern_

				lg_M_sigma = np.log10( obs_Hern_sigm )

			## mcmc
			put_params = [ cov_arr, lg_M_sigma ]

			n_walk = 70

			## .. initial
			put_x0 = np.random.uniform( 7, 11, n_walk ) # lgI
			put_x1 = np.random.uniform( 5, 5e1, n_walk ) # Re
			put_x2 = np.random.uniform( 1, 10, n_walk ) # n
			put_x3 = np.random.uniform( -6, -0.823, n_walk ) # lgbf

			## Hernquist
			# if mass_id == False:

			# 	if (mm == 0) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.75, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 3, 6.5, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.3, -2.7, n_walk ) # lgbf

			# 	if (mm == 0) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 8.0, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 4, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.6, -2.7, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.0, 8.0, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.3, -2.8, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.8, 8.8, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 3, 9, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.5, -2.5, n_walk ) # lgbf

			# if mass_id == True:

			# 	if (mm == 0) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.5, 8.8, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 2, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 2, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.5, -2.7, n_walk ) # lgbf

			# 	if (mm == 0) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 8.0, 9.0, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 2, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 2, 9, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.5, -2.8, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 8.0, 8.7, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 4.5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.5, -2.8, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.8, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 20, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 1, 6, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.3, -2.7, n_walk ) # lgbf

			## NFW
			# if mass_id == False:

			# 	if (mm == 0) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.8, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -2.8, -2.0, n_walk ) # lgbf

			# 	if (mm == 0) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 8.0, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.4, -2.4, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.0, 8.0, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 25, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.2, -2.4, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.7, 8.4, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 20, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.5, -2.5, n_walk ) # lgbf

			# if mass_id == True:

			# 	if (mm == 0) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.75, 8.6, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 3, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 4, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.0, -2.4, n_walk ) # lgbf

			# 	if (mm == 0) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 8.4, 9.2, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 1, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 5.5, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.2, -2.4, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'ri'):
			# 		put_x0 = np.random.uniform( 7.8, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 1, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 6, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.0, -2.4, n_walk ) # lgbf

			# 	if (mm == 1) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.8, 8.4, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 20, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 1, 6, n_walk ) # n
			# 		put_x3 = np.random.uniform( -3.1, -2.3, n_walk ) # lgbf

			## .. fitting with up-limit in lgfb (high mass-bin, younger bin)
			# ## NFW 
			# if mass_id == False:
			# 	if (mm == 0) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.8, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 15, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 4, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -4.0, -3.2, n_walk ) # lgbf
			# '''
			# if mass_id == True:
			# 	if (mm == 1) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.5, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 25, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 4, 10, n_walk ) # n
			# 		put_x3 = np.random.uniform( -4.0, -3.2, n_walk ) # lgbf
			# '''
			# ## Hernquist
			# if mass_id == True:
			# 	if (mm == 1) & (band_str == 'gi'):
			# 		put_x0 = np.random.uniform( 7.5, 8.5, n_walk ) # lgI
			# 		put_x1 = np.random.uniform( 5, 25, n_walk ) # Re
			# 		put_x2 = np.random.uniform( 2, 8, n_walk ) # n
			# 		put_x3 = np.random.uniform( -4.0, -3.2, n_walk ) # lgbf

			L_chains = 2e4

			param_labels = ['$lg\\Sigma_{e}$', '$R_{e}$', '$n$', '$lgf_{b}$']

			pos = np.array( [put_x0, put_x1, put_x2, put_x3] ).T
			n_dim = pos.shape[1]

			print(pos.shape)
			print(n_dim)

			if NFW_on == True:
				file_name = out_path + '%s_%s-band-based_mass-profile_NFW_mcmc_fit.h5' % (cat_lis[mm], band_str) # NFW
			if NFW_on == False:
				file_name = out_path + '%s_%s-band-based_mass-profile_Hern_mcmc_fit.h5' % (cat_lis[mm], band_str) # Hernquist

			## ip-limit fitting
			# if NFW_on == True:
			# 	file_name = out_path + '%s_%s-band-based_mass-profile_NFW_up-limit-fb_mcmc_fit.h5' % (cat_lis[mm], band_str) # NFW
			# if NFW_on == False:
			# 	file_name = out_path + '%s_%s-band-based_mass-profile_Hern_up-limit-fb_mcmc_fit.h5' % (cat_lis[mm], band_str) # Hernquist

			backend = emcee.backends.HDFBackend( file_name )
			backend.reset( n_walk, n_dim )

			with Pool( 70 ) as pool:
				## normal
				sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = ( obs_R, lg_M, put_params, lg_M_err), pool = pool, backend = backend,)
				
				## age-bin
				# sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = ( obs_R, lg_M, put_params, lg_M_err, band_str), pool = pool, backend = backend,)

				## mass-bin
				# sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = ( obs_R, lg_M, put_params, lg_M_err, mm, band_str), pool = pool, backend = backend,)

				sampler.run_mcmc(pos, L_chains, progress = True, )

			# sampler = emcee.backends.HDFBackend( file_name )

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
			fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
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

			if NFW_on == True:
				plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_NFW_mcmc_params.png' % (cat_lis[mm], band_str), dpi = 300) # NFW
			if NFW_on == False:
				plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_Hern_mcmc_params.png' % (cat_lis[mm], band_str), dpi = 300) # Hernquist
			plt.close()

			Ie_fit, Re_fit, ne_fit, bf_fit = mc_fits[:]

			_cen_M = sersic_func( obs_R, 10**mc_fits[0], mc_fits[1], mc_fits[2] )

			_out_M = np.log10( 10**lg_M_sigma * 10**mc_fits[3] )

			_sum_fit = np.log10( _cen_M + 10**lg_M_sigma * 10**mc_fits[3] )


			cov_inv = np.linalg.pinv( cov_arr )
			delta = _sum_fit - lg_M
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


			## .. save params
			keys = ['Ie', 'Re', 'ne',  'M2L']
			values = []
			for jj in range( n_dim ):

				values.append( mc_fits[jj] )

			fill = dict( zip( keys, values) )
			out_data = pds.DataFrame( fill, index = ['k', 'v'])
			
			if NFW_on == True:
				out_data.to_csv( out_path + '%s_%s-band-based_mass-profile_NFW_mcmc_fit.csv' % (cat_lis[mm], band_str),) # NFW
			if NFW_on == False:
				out_data.to_csv( out_path + '%s_%s-band-based_mass-profile_Hern_mcmc_mcmc_fit.csv' % (cat_lis[mm], band_str),) # Hernquist

			## ip-limit fitting
			# if NFW_on == True:
			# 	out_data.to_csv( out_path + '%s_%s-band-based_mass-profile_NFW_up-limit-fb_mcmc_fit.csv' % (cat_lis[mm], band_str),) # NFW
			# if NFW_on == False:
			# 	out_data.to_csv( out_path + '%s_%s-band-based_mass-profile_Hern_up-limit-fb_mcmc_fit.csv' % (cat_lis[mm], band_str),) # Hernquist


			plt.figure()
			ax = plt.subplot(111)
			ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', 
				alpha = 0.5, mec = 'r', mfc = 'r', label = 'signal')

			ax.plot( obs_R, np.log10( _cen_M ), ls = '--', color = 'c', alpha = 0.5, label = 'sersic')

			if NFW_on == True:
				ax.plot( obs_R, _out_M, ls = '-.', color = 'g', alpha = 0.5, label = '$ NFW_{mis} $')
			if NFW_on == False:
				ax.plot( obs_R, _out_M, ls = '-.', color = 'g', alpha = 0.5, label = '$ Hern_{mis} $')

			ax.plot( obs_R, _sum_fit, ls = '-', color = 'b', alpha = 0.5,)

			ax.text( 2e1, 5.0, s = '$ lg\\Sigma_{e} = %.2f$' % Ie_fit + '\n' + '$R_{e} = %.2f$' % Re_fit + 
				'\n' + '$ n = %.2f $' % ne_fit, color = 'b',)

			ax.text( 2e1, 4.5, s = '$ lgf_{b} = %.2f $' % bf_fit, color = 'b',)
			ax.text( 2e1, 4.0, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k',)
			ax.text( 2e1, 3.6, s = '$\\chi^{2} / \\nu[R>=200] = %.5f$' % cut_chi2nu, color = 'k',)

			ax.legend( loc = 1, )
			ax.set_ylim( 3, 8.5)
			ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

			ax.set_xlabel( 'R [kpc]')
			ax.set_xscale( 'log' )

			if NFW_on == True:
				plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_NFW_mcmc_test.png' % (cat_lis[mm], band_str), dpi = 300) # NFW
			if NFW_on == False:
				plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_Hern_mcmc_test.png' % (cat_lis[mm], band_str), dpi = 300) # Hernquist
			plt.close()


