"""
this file use for test the fitting on mid mass profile
1. it's sensitive to the central or not
2. change initial value of fitting, how it will be like
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

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

### === ### cosmology
rad2asec = U.rad.to( U.arcsec )
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

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)


### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### func.s
def sersic_func(r, Ie, re, n):

	belta = 2 * n - 0.324
	fn = -1 * belta * ( r / re )**(1 / n) + belta
	Ir = Ie * np.exp( fn )

	return Ir

def lg_sersic_func(r, lg_Ie, re, n):

	Ie = 10**lg_Ie

	belta = 2 * n - 0.324
	fn = -1 * belta * ( r / re )**(1 / n) + belta
	Ir = Ie * np.exp( fn )

	return np.log10( Ir )

def fixn_sersic_err_fit_f(p, x, y, params, yerr):

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

def freen_sersic_err_fit_f(p, x, y, params, yerr):

	cov_mx = params[:]
	_Ie, _Re, _ne = p[:]

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

##.. middle
def SM_log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * (np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_M = lg_SM0 - lg_A0 - lg_A1

	return 10**lg_M

def log_norm_func( r, Am, Rt, sigm_tt ):

	mf0 = r * sigm_tt * np.sqrt( 2 * np.pi )
	mf1 = -0.5 * ( np.log(r) - np.log(Rt) )**2 / sigm_tt**2
	Pdf = Am * np.exp( mf1 ) / mf0

	return Pdf

def lg_norm_err_fit_f(p, x, y, yerr):

	Am0, _R_t, _sigm_tt = p[:]

	_mpdf = log_norm_func( x, Am0, _R_t, _sigm_tt )

	delta = _mpdf - y

	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf


#### ==== #### data load
# ... DM mass profile
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


# ... observed surface mass profile
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'

band_str = 'gri'

#. mass estimation with deredden or not
id_dered = True
dered_str = 'with-dered_'


#. adjust on central sersic fitting

id_fixn = True
# id_fixn = False


#. parameters of scaled relation
out_lim_R = 350 # 350, 400

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % (dered_str,out_lim_R),)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]


dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )

_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
tmp_tot_M_f = interp.interp1d( _cp_R, _cp_SM, kind = 'linear', fill_value = 'extrapolate',)

obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )


##.. cov_arr
with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5', 'r') as f:
	cov_arr = np.array( f['cov_MX'] )
	cor_arr = np.array( f['cor_MX'] )

if id_fixn == True:
	id_rx = obs_R >= 10    ## fixed n case

if id_fixn == False:
	id_rx = obs_R >= 3     ## free n case


obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

id_cov = np.where( id_rx )[0][0]
cov_arr = cov_arr[id_cov:, id_cov:]

lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )


### ... fitting test
cen_lim_R = [20, 30, 40, 50]

for ll in range( len( cen_lim_R ) ):

	idx_lim = obs_R <= cen_lim_R[ ll ]

	##... central fit test
	if id_fixn == True:

		id_dex = np.where( idx_lim == True)[0][-1]
		cut_cov = cov_arr[:id_dex+1, :id_dex+1]
		fit_R, fit_SM, fit_SM_err = obs_R[idx_lim], lg_M[idx_lim], lg_M_err[idx_lim]

		put_param = [ cut_cov, 4 ]
		po = [ 6, 10 ]

		bounds = [ [4.5, 9.5], [5, 50] ]

		E_return = optimize.minimize( fixn_sersic_err_fit_f, x0 = np.array( po ), args = (fit_R, fit_SM, put_param, fit_SM_err), 
								method = 'L-BFGS-B', bounds = bounds,)

		print( E_return )

		popt = E_return.x
		Ie_fit, Re_fit = popt
		Ne_fit = 4

		## .. save
		keys = [ 'Ie', 'Re', 'ne' ]
		values = [ Ie_fit, Re_fit, Ne_fit ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill, index = ['k', 'v'])
		out_data.to_csv( fit_path + 
						'%stotal-sample_%s-band-based_SM_cen-deV_R-%dkpc-lim.csv' % (dered_str, band_str, cen_lim_R[ll]),)

		_cen_M = sersic_func( _cp_R, 10**Ie_fit, Re_fit, Ne_fit) - sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)


		plt.figure()
		ax = plt.subplot(111)

		ax.plot( _cp_R, np.log10( _cp_SM ), 'r.-', label = 'Obs')
		ax.fill_between( _cp_R, y1 = np.log10( _cp_SM - _cp_SM_err), y2 = np.log10( _cp_SM + _cp_SM_err),
						color = 'r', alpha = 0.5,)

		ax.plot( _cp_R, np.log10( _cen_M ), ls = '--', color = 'b', alpha = 0.75, label = 'Sersic',)
		ax.text( 1e1, 4.5, s = '$ lg\\Sigma_{e} = %.2f$' % Ie_fit + '\n' + '$R_{e} = %.2f$' % Re_fit + 
			'\n' + '$ n = %.2f $' % Ne_fit, color = 'b',)
		ax.legend( loc = 1, )
		ax.set_ylim( 6, 9.5 )
		ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

		ax.set_xlim( 1e0, 2e2 )
		ax.set_xlabel( 'R [kpc]')
		ax.set_xscale( 'log' )

		plt.savefig('/home/xkchen/' + 
					'%stotal-sample_%s-band_based_fit-compare_R-%dkpc_limit.png' % (dered_str, band_str, cen_lim_R[ll]), dpi = 300)
		plt.close()

	if id_fixn == False:

		id_dex = np.where( idx_lim == True)[0][-1]
		cut_cov = cov_arr[:id_dex+1, :id_dex+1]

		fit_R, fit_SM, fit_SM_err = obs_R[idx_lim], lg_M[idx_lim], lg_M_err[idx_lim]
		put_param = [ cut_cov ]


		# po = [ 6, 10, 4 ]
		# bounds = [ [3.5, 9.5], [5, 30], [0.5, 10] ]
		# E_return = optimize.minimize( freen_sersic_err_fit_f, x0 = np.array( po ), args = (fit_R, fit_SM, put_param, fit_SM_err), 
		# 								method = 'L-BFGS-B', bounds = bounds,)

		# print( E_return )
		# popt = E_return.x
		# Ie_fit, Re_fit, Ne_fit = popt


		po = [ 4, 9, 5.5 ]
		popt, pcov = optimize.curve_fit( lg_sersic_func, fit_R, fit_SM, p0 = np.array( po ), bounds = ([3.5, 5, 0.5], [9.5, 20, 10]), sigma = fit_SM_err,)
		Ie_fit, Re_fit, Ne_fit = popt

		## .. save
		keys = [ 'Ie', 'Re', 'ne' ]
		values = [ Ie_fit, Re_fit, Ne_fit ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill, index = ['k', 'v'])
		out_data.to_csv( fit_path + 
						'%stotal-sample_%s-band-based_SM_cen-deV_R-%dkpc-lim_free-n.csv' % (dered_str, band_str, cen_lim_R[ll]),)


		_cen_M = sersic_func( _cp_R, 10**Ie_fit, Re_fit, Ne_fit) - sersic_func( 2e3, 10**Ie_fit, Re_fit, Ne_fit)

		plt.figure()
		ax = plt.subplot(111)

		ax.errorbar( _cp_R, _cp_SM, yerr = _cp_SM_err, xerr = None, color = 'k', marker = 'o', ms = 4, ls = 'none', 
			ecolor = 'k', mec = 'k', mfc = 'none', capsize = 3,)

		ax.plot( fit_R, 10**fit_SM, 'g*',)

		ax.plot( _cp_R, _cen_M, ls = '-.', color = 'r', alpha = 0.55, label = 'Sersic',)
		ax.plot( _cp_R, _cen_M + ( xi_to_Mf( _cp_R ) - sigma_2Mpc ) * 10**lg_fb_gi, ls = '--', color = 'r', alpha = 0.5,)

		ax.text( 5e0, 10**5.2, s = '$ lg\\Sigma_{e} = %.2f$' % Ie_fit + '\n' + '$R_{e} = %.2f$' % Re_fit + 
			'\n' + '$ n = %.2f $' % Ne_fit, color = 'b',)

		ax.legend( loc = 1, )
		ax.set_ylim( 1e5, 4e9 )
		ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )
		ax.set_yscale( 'log' )

		ax.set_xlim( 1e0, 5e2 )
		ax.set_xlabel( 'R [kpc]')
		ax.set_xscale( 'log' )

		plt.savefig('/home/xkchen/' + 
					'%stotal-sample_%s-band_based_fit-compare_R-%dkpc_free-n.png' % (dered_str, band_str, cen_lim_R[ll]), dpi = 300)
		plt.close()


	##... median fit test
	if id_fixn == True:
		p_dat = pds.read_csv( fit_path + 
							'%stotal-sample_%s-band-based_SM_cen-deV_R-%dkpc-lim.csv' % (dered_str, band_str, cen_lim_R[ll]),)
		c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

	if id_fixn == False:
		p_dat = pds.read_csv( fit_path + 
							'%stotal-sample_%s-band-based_SM_cen-deV_R-%dkpc-lim_free-n.csv' % (dered_str, band_str, cen_lim_R[ll]),)
		c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]


	cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
	fit_cen_M = sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc
	fit_out_SM = ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi
	fit_sum = fit_cen_M + fit_out_SM

	#. middle mass and mass ratio
	devi_M = surf_M - fit_sum

	id_vx = devi_M < 0
	devi_M[ id_vx ] = 0.

	id_Rx = obs_R >= 20
	id_lim = ( id_vx == False ) & id_Rx

	#.. SM ratio
	mid_R = obs_R[ id_lim ]
	mid_M_eta = devi_M[ id_lim ] / tmp_tot_M_f( mid_R )
	mid_M_eta_err = surf_M_err[ id_lim ] / tmp_tot_M_f( mid_R )

	mid_M = devi_M[ id_lim ]
	mid_M_err = surf_M_err[ id_lim ]


	#.. fitting test
	po = [ 1.5, 100, 0.5 ]

	bounds = [ [1e-2, 1e2], [10, 500], [0.1, 3] ]
	E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = (mid_R, mid_M_eta, mid_M_eta_err), 
									method = 'L-BFGS-B', bounds = bounds,)
	popt = E_return.x

	print( popt )
	Am_fit, Rt_fit, sigm_tt_fit = popt

	R_mode = np.exp( np.log( Rt_fit ) - sigm_tt_fit**2 )

	fit_cross = log_norm_func( obs_R, Am_fit, Rt_fit, sigm_tt_fit)
	fit_mid_SM = fit_cross * ( fit_sum ) / ( 1 - fit_cross )


	chi_SM0 = log_norm_func( mid_R, Am_fit, Rt_fit, sigm_tt_fit ) * tmp_tot_M_f( mid_R )
	delta = chi_SM0 - mid_M
	chi2_0 = np.sum( delta**2 / mid_M_err**2 )
	n_free = len( mid_R ) - 3
	chi2nv_0 = chi2_0 / n_free


	#... save the fitting results
	keys = ['Am', 'Rt', 'sigma_t', 'R_mode', 'chi2nv']
	values = [ Am_fit, Rt_fit, sigm_tt_fit, R_mode, chi2nv_0 ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])

	if id_fixn == True:	
		out_data.to_csv( fit_path + 
						'%stotal_%s-band-based_mid-region_Lognorm_ratio-based_R-%dkpc_lim.csv' % (dered_str, band_str, cen_lim_R[ll]),)

	if id_fixn == False:
		out_data.to_csv( fit_path + 
						'%stotal_%s-band-based_mid-region_Lognorm_ratio-based_R-%dkpc_lim_free-n.csv' % (dered_str, band_str, cen_lim_R[ll]),)


	plt.figure()
	plt.plot( obs_R, fit_mid_SM, ls = '--', color = 'r', label = 'Lognormal')

	plt.errorbar( mid_R, mid_M, yerr = mid_M_err, xerr = None, color = 'k', marker = 'o', ms = 4, ls = 'none', 
		ecolor = 'k', mec = 'k', mfc = 'none', capsize = 3,)

	plt.legend( loc = 2 )
	plt.xlim( 1e1, 5e2)
	plt.xscale( 'log' )
	plt.xlabel('R [kpc]')
	plt.ylim( 1e3, 2e6 )
	plt.yscale( 'log' )
	plt.ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} / kpc^{2} ]$')
	plt.savefig( '/home/xkchen/%stotal-sample_mid-region_R-%dkpc_lim.png' % (dered_str, cen_lim_R[ll]), dpi = 300)
	plt.close()


### ... figs
def fixn_compare():

	new_R = np.logspace( 0, np.log10(2.5e3), 100 )
	mod_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi

	fixn_cen_params = []
	fixn_cen_SM_arr = []

	fixn_mid_params = []
	fixn_mid_SM_arr = []
	fixn_diffi_SM_arr = []

	for ll in range( 2 ):
		if ll == 0:
			### === fiducal fit (fixed n, 10~20kpc)
			p_dat = pds.read_csv( fit_path + '%stotal-sample_gri-band-based_mass-profile_cen-deV_fit.csv' % dered_str,)

		if ll == 1:
			##... fixed_n case (10~30kpc)
			p_dat = pds.read_csv( fit_path + 
						'%stotal-sample_gri-band-based_SM_cen-deV_R-30kpc-lim.csv' % dered_str,)

		Ie_kk, Re_kk, ne_kk = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

		cen_M_2Mpc_kk = sersic_func( 2e3, 10**Ie_kk, Re_kk, ne_kk)
		cen_M_kk = sersic_func( new_R, 10**Ie_kk, Re_kk, ne_kk) - cen_M_2Mpc_kk

		out_M_kk = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi
		fit_sum_kk = out_M_kk + cen_M_kk


		fixn_cen_params.append( [ Ie_kk, Re_kk, ne_kk ] )
		fixn_cen_SM_arr.append( cen_M_kk )


		if ll == 0:
			mid_pat = pds.read_csv( fit_path + 
							'%stotal_%s-band-based_mid-region_Lognorm_ratio-based_fit.csv' % (dered_str, band_str),)

		if ll == 1:
			mid_pat = pds.read_csv( fit_path + 
							'%stotal_gri-band-based_mid-region_Lognorm_ratio-based_R-%dkpc_lim.csv' % (dered_str, cen_lim_R[ll]),)
		Am_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['Am'] )[0], np.array( mid_pat['Rt'] )[0], np.array( mid_pat['sigma_t'] )[0]
		R_mode = np.array( mid_pat['R_mode'] )[0]

		fit_cross = log_norm_func( new_R, Am_fit, Rt_fit, sigm_tt_fit)
		fit_mid_SM = fit_cross * ( fit_sum_kk ) / ( 1 - fit_cross )


		fixn_mid_params.append( [ Am_fit, Rt_fit, sigm_tt_fit, R_mode ] )
		fixn_mid_SM_arr.append( fit_mid_SM )


		#. middle SM(r)
		devi_M = surf_M - ( xi_to_Mf( obs_R ) - sigma_2Mpc) * 10**lg_fb_gi - ( sersic_func( obs_R, 10**Ie_kk, Re_kk, ne_kk) - cen_M_2Mpc_kk)
		devi_err = surf_M_err

		devi_lgM = np.log10( devi_M )
		id_nan = np.isnan( devi_lgM )

		id_M_lim = devi_lgM < 3
		id_R_x0 = obs_R < 20
		id_R_x1 = obs_R > 300
		id_R_lim = id_R_x0 | id_R_x1

		id_lim = (id_nan | id_M_lim) | id_R_lim

		devi_R = obs_R[ id_lim == False ]
		devi_M = devi_M[ id_lim == False ]
		devi_err = devi_err[ id_lim == False ]

		fixn_diffi_SM_arr.append( [ devi_R, devi_M, devi_err ] )


	##... figs
	line_c = ['b', 'r']
	mark_s = ['s', 'o']

	fig = plt.figure( figsize = (5.8, 5.4) )
	ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
		capsize = 3, mec = 'k', mfc = 'none', label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $',)
	ax1.plot( new_R / 1e3, mod_out_M, ls = '-', color = 'k', alpha = 0.65, label = '$ \\gamma \, \\Sigma_{m} $',)

	l_ = []

	for ll in range( 2 ):

		fit_sum = fixn_cen_SM_arr[ll] + mod_out_M

		ax1.plot( new_R / 1e3, fixn_cen_SM_arr[ll], ls = ':', color = line_c[ll], label = '$\\Sigma_{\\ast}^{ \\mathrm{deV} }$')

		l_kk, = ax1.plot( new_R / 1e3, fit_sum, ls = '--', color = line_c[ll], alpha = 0.75, linewidth = 2.5, 
			label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\gamma \, \\Sigma_{m} $')

		ax1.plot( new_R / 1e3, fixn_mid_SM_arr[ll], ls = '-.', color = line_c[ll], alpha = 0.75, 
			label = '$\\Sigma_{\\ast}^{tran} \, (\\mathrm{Eqn.\,5})$')

		ax1.errorbar( fixn_diffi_SM_arr[ll][0] / 1e3, fixn_diffi_SM_arr[ll][1], yerr = fixn_diffi_SM_arr[ll][2], 
			ls = 'none', marker = 's', ms = 8, mec = line_c[ll], mfc = 'none', ecolor = line_c[ll], alpha = 0.75, capsize = 3,
			label = '$\\Sigma_{\\ast}^{tran} $')

		tmp_tot_M_f = interp.interp1d( obs_R, surf_M, kind = 'linear', fill_value = 'extrapolate',)

		sub_ax1.errorbar( fixn_diffi_SM_arr[ll][0] / 1e3, fixn_diffi_SM_arr[ll][1] / tmp_tot_M_f( fixn_diffi_SM_arr[ll][0] ), 
			yerr = fixn_diffi_SM_arr[ll][2] / tmp_tot_M_f( fixn_diffi_SM_arr[ll][0] ), ls = 'none', marker = 's', ms = 8, 
			mec = line_c[ll], mfc = 'none', ecolor = line_c[ll], alpha = 0.75, capsize = 3,)
		
		sub_ax1.plot( new_R / 1e3, fixn_mid_SM_arr[ll] / ( fixn_mid_SM_arr[ll] + fit_sum ), ls = '-.', color = line_c[ll], alpha = 0.75,)

		l_.append( l_kk )

		if ll == 0:
			handles, label_lis = ax1.get_legend_handles_labels()

	cc_legend = ax1.legend( [ l_[0], l_[1] ], ['$R_{cen}^{limit}$: 10$\\sim$20 kpc', '$R_{cen}^{limit}$: 10$\\sim$30 kpc'], 
		loc = 'upper center', frameon = False, fontsize = 12)
	ax1.legend( handles, label_lis, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)
	ax1.add_artist( cc_legend )

	ax1.set_ylim( 9e3, 3e8 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

	ax1.set_xlim( 9e-3, 2e0 )
	ax1.set_xscale( 'log' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax1.text( 0.01, 5e5, s = '$\\lg\\Sigma_{\\ast, \, e}=%.2f \, [M_{\\odot}]$' % fixn_cen_params[0][0] + '\n' + 
							'$R_{e}=%.2f \, [kpc]$' % fixn_cen_params[0][1], color = 'b', fontsize = 8)
	ax1.text( 0.01, 1e5, s = '$\\lg\\Sigma_{\\ast, \, e}=%.2f \, [M_{\\odot}]$' % fixn_cen_params[1][0] + '\n' + 
							'$R_{e}=%.2f \, [kpc]$' % fixn_cen_params[1][1], color = 'r', fontsize = 8)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)
	sub_ax1.set_xscale( 'log' )
	sub_ax1.set_ylim( -0.075, 0.50 )
	sub_ax1.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)

	sub_ax1.set_xticks([1e-2, 1e-1, 1e0, 2e0])
	sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'] ) )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	sub_ax1.axhline( y = 0.255, ls = ':',)

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/test_fixn_DM_Ng_compare.png', dpi = 300)
	plt.close()

	return

fixn_compare()

raise


def freen_compare():

	##... free_n case
	new_R = np.logspace( 0, np.log10(2.5e3), 100 )
	mod_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi

	cen_lim_R = [ 20, 30 ]

	freen_cen_params = []
	freen_cen_SM_arr = []

	freen_mid_params = []
	freen_mid_SM_arr = []
	freen_diffi_SM_arr = []

	for ll in range( 2 ):

		p_dat = pds.read_csv( fit_path + 
								'%stotal-sample_gri-band-based_SM_cen-deV_R-%dkpc-lim_free-n.csv' % (dered_str, cen_lim_R[ll]),)
		Ie_kk, Re_kk, ne_kk = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

		cen_M_2Mpc_kk = sersic_func( 2e3, 10**Ie_kk, Re_kk, ne_kk)
		cen_M_kk = sersic_func( new_R, 10**Ie_kk, Re_kk, ne_kk) - cen_M_2Mpc_kk

		out_M_kk = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi
		fit_sum_kk = out_M_kk + cen_M_kk


		freen_cen_params.append( [Ie_kk, Re_kk, ne_kk] )
		freen_cen_SM_arr.append( cen_M_kk )


		#. middle region fitting test
		mid_pat = pds.read_csv( fit_path + 
						'%stotal_gri-band-based_mid-region_Lognorm_ratio-based_R-%dkpc_lim_free-n.csv' % (dered_str, cen_lim_R[ll]),)
		Am_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['Am'] )[0], np.array( mid_pat['Rt'] )[0], np.array( mid_pat['sigma_t'] )[0]
		R_mode = np.array( mid_pat['R_mode'] )[0]

		fit_cross = log_norm_func( new_R, Am_fit, Rt_fit, sigm_tt_fit)
		fit_mid_SM = fit_cross * ( fit_sum_kk ) / ( 1 - fit_cross )


		freen_mid_params.append( [ Am_fit, Rt_fit, sigm_tt_fit, R_mode ] )
		freen_mid_SM_arr.append( fit_mid_SM )


		devi_M = surf_M - ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi - (sersic_func( obs_R, 10**Ie_kk, Re_kk, ne_kk) - cen_M_2Mpc_kk )
		devi_err = surf_M_err

		devi_lgM = np.log10( devi_M )
		id_nan = np.isnan( devi_lgM )

		id_M_lim = devi_lgM < 3
		id_R_x0 = obs_R < 30
		id_R_x1 = obs_R > 300
		id_R_lim = id_R_x0 | id_R_x1

		id_lim = (id_nan | id_M_lim) | id_R_lim

		devi_R_kk = obs_R[ id_lim == False ]
		devi_M_kk = devi_M[ id_lim == False ]
		devi_err_kk = devi_err[ id_lim == False ]


		freen_diffi_SM_arr.append( [ devi_R_kk, devi_M_kk, devi_err_kk ] )


	##... figs
	line_c = ['b', 'r']
	mark_s = ['s', 'o']

	fig = plt.figure( figsize = (5.8, 5.4) )
	ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
		capsize = 3, mec = 'k', mfc = 'none', label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $',)
	ax1.plot( new_R / 1e3, mod_out_M, ls = '-', color = 'k', alpha = 0.65, label = '$ \\gamma \, \\Sigma_{m} $',)

	l_ = []

	for ll in range( 2 ):

		fit_sum = freen_cen_SM_arr[ll] + mod_out_M

		ax1.plot( new_R / 1e3, freen_cen_SM_arr[ll], ls = ':', color = line_c[ll], label = '$\\Sigma_{\\ast}^{ \\mathrm{deV} }$')

		l_kk, = ax1.plot( new_R / 1e3, fit_sum, ls = '--', color = line_c[ll], alpha = 0.75, linewidth = 2.5, 
			label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\gamma \, \\Sigma_{m} $')

		ax1.plot( new_R / 1e3, freen_mid_SM_arr[ll], ls = '-.', color = line_c[ll], alpha = 0.75, 
			label = '$\\Sigma_{\\ast}^{tran} \, (\\mathrm{Eqn.\,5})$')

		ax1.errorbar( freen_diffi_SM_arr[ll][0] / 1e3, freen_diffi_SM_arr[ll][1], yerr = freen_diffi_SM_arr[ll][2], 
			ls = 'none', marker = 's', ms = 8, mec = line_c[ll], mfc = 'none', ecolor = line_c[ll], alpha = 0.75, capsize = 3,
			label = '$\\Sigma_{\\ast}^{tran} $')

		tmp_tot_M_f = interp.interp1d( obs_R, surf_M, kind = 'linear', fill_value = 'extrapolate',)

		sub_ax1.errorbar( freen_diffi_SM_arr[ll][0] / 1e3, freen_diffi_SM_arr[ll][1] / tmp_tot_M_f( freen_diffi_SM_arr[ll][0] ), 
			yerr = freen_diffi_SM_arr[ll][2] / tmp_tot_M_f( freen_diffi_SM_arr[ll][0] ), ls = 'none', marker = 's', ms = 8, 
			mec = line_c[ll], mfc = 'none', ecolor = line_c[ll], alpha = 0.75, capsize = 3,)
		
		sub_ax1.plot( new_R / 1e3, freen_mid_SM_arr[ll] / ( freen_mid_SM_arr[ll] + fit_sum ), ls = '-.', color = line_c[ll], alpha = 0.75,)

		l_.append( l_kk )

		if ll == 0:
			handles, label_lis = ax1.get_legend_handles_labels()

	cc_legend = ax1.legend( [ l_[0], l_[1] ], ['$R_{cen}^{limit}$: 10$\\sim$20 kpc', '$R_{cen}^{limit}$: 10$\\sim$30 kpc'], 
		loc = 'upper center', frameon = False, fontsize = 12)
	ax1.legend( handles, label_lis, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)
	ax1.add_artist( cc_legend )

	ax1.set_ylim( 9e3, 3e8 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

	ax1.set_xlim( 9e-3, 2e0 )
	ax1.set_xscale( 'log' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax1.text( 0.01, 5e5, s = '$\\lg\\Sigma_{\\ast, \, e}=%.2f \, [M_{\\odot}]$' % freen_cen_params[0][0] + '\n' + 
							'$R_{\, e}=%.2f \, [kpc]$' % freen_cen_params[0][1] + '\n' + 
							'$n_{e}=%.2f \, $' % freen_cen_params[0][2], color = 'b', fontsize = 8)
	ax1.text( 0.01, 1e5, s = '$\\lg\\Sigma_{\\ast, \, e}=%.2f \, [M_{\\odot}]$' % freen_cen_params[1][0] + '\n' + 
							'$R_{ e}=%.2f \, [kpc]$' % freen_cen_params[1][1] + '\n' + 
							'$n_{e}=%.2f \, $' % freen_cen_params[0][2], color = 'r', fontsize = 8)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)
	sub_ax1.set_xscale( 'log' )
	sub_ax1.set_ylim( -0.075, 0.50 )
	sub_ax1.set_yticks([ 0, 0.2, 0.4 ])
	sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.0}$','$\\mathrm{0.2}$', '$\\mathrm{0.4}$'])
	sub_ax1.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)

	sub_ax1.set_xticks([1e-2, 1e-1, 1e0, 2e0])
	sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'] ) )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	sub_ax1.axhline( y = 0.255, color = 'g', ls = ':')
	sub_ax1.axhline( y = 0.23, color = 'r', ls = ':')
	sub_ax1.axhline( y = 0.155, color = 'b', ls = ':')

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/test_freen_DM_Ng_compare.png', dpi = 300)
	plt.close()

freen_compare()

