import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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


### === ### middle SM ratio fit
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


#. overall sample
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

sigma_2Mpc = xi_to_Mf( 2e3 )


### === subsamples
BG_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

band_str = 'gri'

id_dered = True
dered_str = '_with-dered'

for mm in range( 2 ):

	# SM(r)
	dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str),)

	_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array( dat['mean_correct_surf_M'] ), np.array(dat['surf_M_err'])

	obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	tmp_tot_M_f = interp.interp1d( _cp_R, _cp_SM, kind = 'linear', fill_value = 'extrapolate',)

	##.. cov_arr
	with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'r') as f:
		cov_arr = np.array( f['cov_MX'] )
		cor_arr = np.array( f['cor_MX'] )

	id_rx = obs_R >= 10
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

	id_cov = np.where( id_rx )[0][0]
	cov_arr = cov_arr[id_cov:, id_cov:]

	lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )


	#.. use params of total sample for large scale
	out_lim_R = 350

	c_dat = pds.read_csv( fit_path + 'with-dered_total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R )
	lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
	lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
	lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

	if mm == 0:
		fit_out_SM = ( lo_interp_F( obs_R ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi

	if mm == 1:
		fit_out_SM = ( hi_interp_F( obs_R ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi


	## .. centeral deV profile
	p_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[mm], band_str, dered_str),)
	c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

	#...trans mass part
	cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
	fit_cen_M = sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc
	fit_sum = fit_cen_M + fit_out_SM

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


	po = [ 1.5, 100, 0.5 ]

	# popt, pcov = optimize.curve_fit( log_norm_func, mid_R, mid_M_eta, p0 = np.array( po ), sigma = mid_M_eta_err,)
	bounds = [ [1e-2, 1e2], [10, 500], [0.1, 3] ]
	E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = (mid_R, mid_M_eta, mid_M_eta_err), 
									method = 'L-BFGS-B', bounds = bounds,)
	popt = E_return.x

	print( popt )
	Am_fit, Rt_fit, sigm_tt_fit = popt
	R_mode = np.exp( np.log( Rt_fit ) - sigm_tt_fit**2 )

	#...
	fit_cross = log_norm_func( obs_R, Am_fit, Rt_fit, sigm_tt_fit)
	# fit_mid_SM_0 = fit_cross * tmp_tot_M_f( obs_R )
	fit_mid_SM_0 = fit_cross * ( fit_sum ) / ( 1 - fit_cross )

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
	out_data.to_csv( fit_path + '%s_%s-band-based_mid-region_Lognorm_ratio-based_fit%s.csv' % (cat_lis[mm], band_str, dered_str),)


	#. fitting with SM(r)
	mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[mm], band_str, dered_str),)
	cp_lgSM_fit, cp_Rt_fit, cp_sigmt_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
	chi2nv_1 = np.array( mid_dat['chi2nv'] )[0]

	fit_mid_SM_1 = SM_log_norm_func( obs_R, cp_lgSM_fit, cp_Rt_fit, cp_sigmt_fit ) - SM_log_norm_func( 2e3, cp_lgSM_fit, cp_Rt_fit, cp_sigmt_fit )


	plt.figure()
	ax = plt.subplot( 111 )

	ax.errorbar( mid_R, mid_M_eta, yerr = mid_M_eta_err, ls = 'none', marker = 's', ms = 8, mec = 'k', 
				mfc = 'none', ecolor = 'k', alpha = 0.75, capsize = 3,)
	ax.plot( obs_R, fit_cross, 'r-',)
	ax.set_ylim( -0.2, 0.4 )

	ax.text( 10, 0.3, s = '$\\sigma_{t} = %.3f$' % sigm_tt_fit + '\n' + '$R_{t} = %.3f$' % Rt_fit + 
		'\n' + '$A_{m} = %.3f $' % Am_fit,)
	ax.text( 10, 0.1, s = '$R_{mode} = %.3f$' % R_mode )

	ax.axvline( x = Rt_fit, ls = '--', color = 'r', label = '$R_{t}$')
	ax.axvline( x = R_mode, ls = ':', color = 'r', label = '$R_{mode}$')

	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	plt.savefig('/home/xkchen/%s_SM_ratio_fit.png' % cat_lis[mm], dpi = 300)
	plt.close()


	fig = plt.figure( figsize = (5.8, 5.4) )
	ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax1.errorbar( obs_R, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
		capsize = 3, mec = 'k', mfc = 'none', label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $',)

	# ax1.plot( obs_R, fit_out_SM, ls = '-', color = 'k', alpha = 0.65,
	# 	label = '$ \\gamma \, \\Sigma_{m} $',)

	# ax1.plot( obs_R, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{ \\mathrm{deV} }$')

	ax1.plot( obs_R, fit_sum, ls = '--', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
		label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\gamma \, \\Sigma_{m} $')

	ax1.plot( obs_R, fit_mid_SM_0, ls = '-.', color = 'k', alpha = 0.75, label = 'SM ratio fit',)
	ax1.plot( obs_R, fit_mid_SM_1, ls = '-.', color = 'r', alpha = 0.75, label = 'SM fit',)

	ax1.errorbar( mid_R, mid_M, yerr = mid_M_err, ls = 'none', marker = 's', ms = 8, mec = 'k', mfc = 'none', 
		ecolor = 'k', alpha = 0.75, capsize = 3, label = '$\\Sigma_{\\ast}^{tran} $')

	ax1.text( 1e1, 1e6, s = '$\\chi^{2}/\\nu=%.2f$' % chi2nv_0, color = 'k',)
	ax1.text( 1e1, 3e6, s = '$\\chi^{2}/\\nu=%.2f$' % chi2nv_1, color = 'r',)

	ax1.legend(loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

	ax1.set_ylim( 1e4, 3e8 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

	ax1.set_xlim( 9e0, 2e3 )
	ax1.set_xscale( 'log' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	sub_ax1.errorbar( mid_R, mid_M_eta, yerr = mid_M_eta_err, ls = 'none', marker = 's', ms = 8, mec = 'k', 
				mfc = 'none', ecolor = 'k', alpha = 0.75, capsize = 3,)
	sub_ax1.plot( obs_R, fit_cross, 'k-',)
	sub_ax1.plot( obs_R, fit_mid_SM_1 / (fit_mid_SM_1 + fit_sum), 'r-',)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)
	sub_ax1.set_xscale( 'log' )
	sub_ax1.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)

	sub_ax1.set_ylim( -0.075, 0.50 )
	sub_ax1.set_yticks([ 0, 0.2, 0.4 ])
	sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.0}$','$\\mathrm{0.2}$', '$\\mathrm{0.4}$'])	
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_mid_SM_ratio_fit_test.png' % cat_lis[mm], dpi = 300)
	plt.close()

raise



### === all samples
#.
BG_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'


band_str = 'gri'

#. mass estimation with deredden or not
id_dered = True
dered_str = 'with-dered_'


# SM(r)
dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )

_cp_R, _cp_SM, _cp_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
tmp_tot_M_f = interp.interp1d( _cp_R, _cp_SM, kind = 'linear', fill_value = 'extrapolate',)

obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

##.. cov_arr
with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % band_str, 'r') as f:
	cov_arr = np.array( f['cov_MX'] )
	cor_arr = np.array( f['cor_MX'] )

id_rx = obs_R >= 9 # 9, 10
obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

id_cov = np.where( id_rx )[0][0]
cov_arr = cov_arr[id_cov:, id_cov:]

lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )


# central part
p_dat = pds.read_csv( fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % (dered_str,band_str),)
c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]


##.. parameters of scaled relation
out_lim_R = 350
c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % (dered_str,out_lim_R),)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]


#...trans mass part
cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
fit_cen_M = sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc

fit_out_SM = ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi
fit_sum = fit_cen_M + fit_out_SM


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


po = [ 1.5, 100, 0.5 ]

# popt, pcov = optimize.curve_fit( log_norm_func, mid_R, mid_M_eta, p0 = np.array( po ), sigma = mid_M_eta_err,)

bounds = [ [1e-2, 1e2], [10, 500], [0.1, 3] ]
E_return = optimize.minimize( lg_norm_err_fit_f, x0 = np.array( po ), args = (mid_R, mid_M_eta, mid_M_eta_err), 
								method = 'L-BFGS-B', bounds = bounds,)
popt = E_return.x

print( popt )
Am_fit, Rt_fit, sigm_tt_fit = popt

R_mode = np.exp( np.log( Rt_fit ) - sigm_tt_fit**2 )

#...
fit_cross = log_norm_func( obs_R, Am_fit, Rt_fit, sigm_tt_fit)
# fit_mid_SM_0 = fit_cross * tmp_tot_M_f( obs_R )
fit_mid_SM_0 = fit_cross * ( fit_sum ) / ( 1 - fit_cross )

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
out_data.to_csv( fit_path + '%stotal_%s-band-based_mid-region_Lognorm_ratio-based_fit.csv' % (dered_str, band_str),)


#. SM(r) fit
mid_dat = pds.read_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (dered_str, band_str),)

cp_lgSM_fit, cp_Rt_fit, cp_sigmt_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
chi2nv_1 = np.array( mid_dat['chi2nv'] )[0]

fit_mid_SM_1 = SM_log_norm_func( obs_R, cp_lgSM_fit, cp_Rt_fit, cp_sigmt_fit ) - SM_log_norm_func( 2e3, cp_lgSM_fit, cp_Rt_fit, cp_sigmt_fit )



plt.figure()
ax = plt.subplot( 111 )

ax.errorbar( mid_R, mid_M_eta, yerr = mid_M_eta_err, ls = 'none', marker = 's', ms = 8, mec = 'k', 
			mfc = 'none', ecolor = 'k', alpha = 0.75, capsize = 3,)
ax.plot( obs_R, fit_cross, 'r-',)
ax.set_ylim( -0.2, 0.4 )

ax.text( 10, 0.2, s = '$\\sigma_{t} = %.3f$' % sigm_tt_fit + '\n' + '$R_{t} = %.3f$' % Rt_fit + 
	'\n' + '$A_{m} = %.3f $' % Am_fit,)

ax.text( 10, 0.1, s = '$R_{mode} = %.3f$' % R_mode )

ax.axvline( x = Rt_fit, ls = '--', color = 'r', label = '$R_{t}$')
ax.axvline( x = R_mode, ls = ':', color = 'r', label = '$R_{mode}$')

ax.legend( loc = 1)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
plt.savefig('/home/xkchen/total_SM_ratio_fit.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (5.8, 5.4) )
ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

ax1.errorbar( obs_R, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
	capsize = 3, mec = 'k', mfc = 'none', label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $',)

# ax1.plot( obs_R, fit_out_SM * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
# 	label = '$ \\gamma \, \\Sigma_{m} $',)

# ax1.plot( obs_R, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{ \\mathrm{deV} }$')

ax1.plot( obs_R, fit_sum, ls = '--', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
	label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\gamma \, \\Sigma_{m} $')

ax1.plot( obs_R, fit_mid_SM_0, ls = '-.', color = 'k', alpha = 0.75, label = 'SM ratio fit',)
ax1.plot( obs_R, fit_mid_SM_1, ls = '-.', color = 'r', alpha = 0.75, label = 'SM fit',)

ax1.plot( obs_R, fit_mid_SM_1, ls = '-.', color = 'r',)

ax1.errorbar( mid_R, mid_M, yerr = mid_M_err, ls = 'none', marker = 's', ms = 8, mec = 'k', mfc = 'none', 
	ecolor = 'k', alpha = 0.75, capsize = 3, label = '$\\Sigma_{\\ast}^{tran} $')

# handles,labels = ax1.get_legend_handles_labels()
# handles = [ handles[3], handles[2], handles[0], handles[1], handles[5], handles[4] ]
# labels = [ labels[3], labels[2], labels[0], labels[1], labels[5], labels[4] ]
# ax1.legend( handles, labels, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

ax1.legend(loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

ax1.text( 1e1, 1e6, s = '$\\chi^{2}/\\nu=%.2f$' % chi2nv_0, color = 'k',)
ax1.text( 1e1, 3e6, s = '$\\chi^{2}/\\nu=%.2f$' % chi2nv_1, color = 'r',)

ax1.set_ylim( 1e4, 3e8 )
ax1.set_yscale('log')
ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

ax1.set_xlim( 9e0, 2e3 )
ax1.set_xscale( 'log' )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


sub_ax1.errorbar( mid_R, mid_M_eta, yerr = mid_M_eta_err, ls = 'none', marker = 's', ms = 8, mec = 'k', 
			mfc = 'none', ecolor = 'k', alpha = 0.75, capsize = 3,)
sub_ax1.plot( obs_R, fit_cross, 'k-',)
sub_ax1.plot( obs_R, fit_mid_SM_1 / (fit_mid_SM_1 + fit_sum), 'r-',)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)
sub_ax1.set_xscale( 'log' )
sub_ax1.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)

sub_ax1.set_ylim( -0.075, 0.50 )
sub_ax1.set_yticks([ 0, 0.2, 0.4 ])
sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.0}$','$\\mathrm{0.2}$', '$\\mathrm{0.4}$'])	
sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/total_mid_SM_ratio_fit_test.png', dpi = 300)
plt.close()

