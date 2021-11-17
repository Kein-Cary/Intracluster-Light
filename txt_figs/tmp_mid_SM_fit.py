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

### === ### median SM fit
def Drude_F(x, lg_Am, L_w, x_0 ):

	mf0 = (L_w / x_0)**2
	mf1 = (x / x_0 - x_0 / x)**2
	mf = mf0 / ( mf1 + mf0 )
	Am = 10**lg_Am

	return Am * mf

def Drude_err_fit_f(p, x, y, params, yerr):

	cov_mx = params[0]

	lg_AM_0, Lw_0, xc_0 = p[:]

	_mass_cen = Drude_F( x, lg_AM_0, Lw_0, xc_0 )
	_mass_2Mpc = Drude_F( 2e3, lg_AM_0, Lw_0, xc_0 )

	_sum_mass = _mass_cen - _mass_2Mpc

	delta = _sum_mass - y

	# cov_inv = np.linalg.pinv( cov_mx )
	# chi2 = delta.T.dot( cov_inv ).dot(delta)

	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf


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

	_sum_mass = _mass_cen - _mass_2Mpc

	delta = _sum_mass - y

	# cov_inv = np.linalg.pinv( cov_mx )
	# chi2 = delta.T.dot( cov_inv ).dot(delta)

	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === MCMC function


### === dataload
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

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


"""
### === subsamples
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

out_lim_R = 350 # 400

for mm in range( 2 ):

	if id_dered == False:
		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['medi_correct_surf_M']), np.array(dat['surf_M_err'])
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

	if id_dered == True:
		dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str) )
		_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['medi_correct_surf_M']), np.array(dat['surf_M_err'])
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

		##.. cov_arr
		with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )


	id_rx = obs_R >= 10.
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

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
	devi_M = surf_M - _out_M - ( _cen_M - _cen_M_2Mpc )

	devi_lgM = np.log10( devi_M )
	devi_err = lg_M_err
	devi_R = obs_R

	id_nan = np.isnan( devi_lgM )

	id_M_lim = devi_lgM < 4.0
	id_R_x0 = obs_R < 10
	id_R_x1 = obs_R > 300
	id_R_lim = id_R_x0 | id_R_x1

	id_lim = (id_nan | id_M_lim) | id_R_lim
	lis_x = np.where( id_lim )[0]

	mid_cov = np.delete( cov_arr, tuple(lis_x), axis = 1)
	mid_cov = np.delete( mid_cov, tuple(lis_x), axis = 0)

	fit_R = obs_R[ id_lim == False ]
	fit_SM = devi_M[ id_lim == False ]
	fit_SM_err = surf_M_err[ id_lim == False ]


	#. pre-fitting
	po_param = [ mid_cov ]

	#... Log-norm
	po = [ 6, 100, 1 ]
	bounds = [ [3.5, 9.5], [10, 500], [0.1, 3] ]
	E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = ( fit_R, fit_SM, po_param, fit_SM_err), 
									method = 'L-BFGS-B', bounds = bounds,)

	popt = E_return.x

	lg_SM_fit, Rt_fit, sigm_tt_fit = popt
	fit_cross = log_norm_func( obs_R, lg_SM_fit, Rt_fit, sigm_tt_fit) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit)

	#. save fitting
	keys = ['lg_M0', 'R_t', 'sigma_t']
	values = [ lg_SM_fit, Rt_fit, sigm_tt_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[mm], band_str, dered_str),)


	#. 
	po = [ 6, 30, 100]
	bounds = [ [3.5, 8.5], [10, 100], [ 50, 200 ] ]
	E_return = optimize.minimize( Drude_err_fit_f, x0 = np.array( po ), args = (fit_R, fit_SM, po_param, fit_SM_err), 
									method = 'L-BFGS-B', bounds = bounds,)

	popt = E_return.x

	lg_AM_fit, Lw_fit, xc_fit = popt
	fit_cross_1 = Drude_F( obs_R, lg_AM_fit, Lw_fit, xc_fit) - Drude_F( 2e3, lg_AM_fit, Lw_fit, xc_fit )

	##... save the fitting
	keys = ['lg_Am', 'Lw', 'R_cen']
	values = [ lg_AM_fit, Lw_fit, xc_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Drude-mcmc-fit%s.csv' % (cat_lis[mm], band_str, dered_str),)


	plt.figure()

	plt.plot( devi_R, fit_cross, ls = '--', color = 'r', label = 'Lognormal')
	plt.plot( devi_R, fit_cross_1, ls = '--', color = 'b', label = 'Drude')

	plt.plot( fit_R, fit_SM, 'gs', markersize = 5,)

	plt.errorbar( devi_R, devi_M, yerr = surf_M_err, xerr = None, color = 'k', marker = 'o', ms = 4, ls = 'none', 
		ecolor = 'k', mec = 'k', mfc = 'none', capsize = 3,)
	plt.legend( loc = 2 )
	plt.xlim( 1e1, 5e2)
	plt.xscale( 'log' )
	plt.xlabel('R [kpc]')
	plt.ylim( 1e4, 2e6 )
	plt.yscale( 'log' )
	plt.ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} / kpc^{2} ]$')
	plt.savefig('/home/xkchen/%s_%s-band-based_mass-profile_mid-region_fit-test%s.png' % (cat_lis[mm], band_str, dered_str), dpi = 300)
	plt.close()

"""


### === all samples
#. flux scaling correction
BG_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

band_str = 'gri'

#. mass estimation with deredden or not
id_dered = True
dered_str = 'with-dered_'

# id_dered = False
# dered_str = ''

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
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % band_str, 'r') as f:
		cov_arr = np.array( f['cov_MX'] )
		cor_arr = np.array( f['cor_MX'] )

if id_dered == True:

	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )
	_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

	obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	##.. cov_arr
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % band_str, 'r') as f:
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

id_nan = np.isnan( devi_lgM )

id_M_lim = devi_lgM < 4.0
id_R_x0 = obs_R < 10
id_R_x1 = obs_R > 300
id_R_lim = id_R_x0 | id_R_x1

id_lim = (id_nan | id_M_lim) | id_R_lim
lis_x = np.where( id_lim )[0]

mid_cov = np.delete( cov_arr, tuple(lis_x), axis = 1)
mid_cov = np.delete( mid_cov, tuple(lis_x), axis = 0)

fit_R = obs_R[ id_lim == False ]
fit_SM = devi_M[ id_lim == False ]
fit_SM_err = surf_M_err[ id_lim == False ]


#. pre-fitting test
po_param = [ mid_cov ]

#.
po = [ 7, 100, 0.8 ]
bounds = [ [3.5, 9.5], [10, 500], [0.1, 3] ]
E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = (fit_R, fit_SM, po_param, fit_SM_err), 
								method = 'L-BFGS-B', bounds = bounds,)

popt = E_return.x
lg_SM_fit, Rt_fit, sigm_tt_fit = popt
fit_cross = log_norm_func( obs_R, lg_SM_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit )

##... save the fitting
keys = ['lg_M0', 'R_t', 'sigma_t']
values = [ lg_SM_fit, Rt_fit, sigm_tt_fit ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
out_data.to_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (dered_str, band_str),)


#. 
po = [ 6, 30, 100]
bounds = [ [3.5, 8.5], [10, 100], [ 50, 200 ] ]
E_return = optimize.minimize( Drude_err_fit_f, x0 = np.array( po ), args = (fit_R, fit_SM, po_param, fit_SM_err), 
								method = 'L-BFGS-B', bounds = bounds,)

popt = E_return.x

lg_AM_fit, Lw_fit, xc_fit = popt
fit_cross_1 = Drude_F( obs_R, lg_AM_fit, Lw_fit, xc_fit) - Drude_F( 2e3, lg_AM_fit, Lw_fit, xc_fit )

##... save the fitting
keys = ['lg_Am', 'Lw', 'R_cen']
values = [ lg_AM_fit, Lw_fit, xc_fit ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
out_data.to_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Drude-mcmc-fit.csv' % (dered_str, band_str),)


plt.figure()

plt.plot( devi_R, fit_cross, ls = '--', color = 'r', label = 'Lognormal')
plt.plot( devi_R, fit_cross_1, ls = '--', color = 'b', label = 'Drude')

plt.plot( fit_R, fit_SM, 'gs', markersize = 5,)

plt.errorbar( devi_R, devi_M, yerr = surf_M_err, xerr = None, color = 'k', marker = 'o', ms = 4, ls = 'none', 
	ecolor = 'k', mec = 'k', mfc = 'none', capsize = 3,)

plt.legend( loc = 2 )
plt.xlim( 1e1, 5e2)
plt.xscale( 'log' )
plt.xlabel('R [kpc]')
plt.ylim( 1e4, 2e6 )
plt.yscale( 'log' )
plt.ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} / kpc^{2} ]$')
plt.savefig( '/home/xkchen/%stotal-sample_mid-region_fit-test.png' % dered_str, dpi = 300)
plt.close()

