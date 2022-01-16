import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import signal
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set

import emcee

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
def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )

		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

def aveg_sigma_func(rp, sigma_arr, N_grid = 100):

	NR = len( rp )
	aveg_sigma = np.zeros( NR, dtype = np.float32 )

	tR = rp
	intep_sigma_F = interp.interp1d( tR , sigma_arr, kind = 'cubic', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( tR[ii] ), N_grid)
		new_sigma = intep_sigma_F( new_rp )

		cumu_sigma = integ.simps( new_rp * new_sigma, new_rp)

		aveg_sigma[ii] = 2 * cumu_sigma / tR[ii]**2

	return aveg_sigma

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * (np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_M = lg_SM0 - lg_A0 - lg_A1

	return 10**lg_M

def pdf_log_norm_func( r, Am, Rt, sigm_tt ):

	mf0 = r * sigm_tt * np.sqrt( 2 * np.pi )
	mf1 = -0.5 * ( np.log(r) - np.log(Rt) )**2 / sigm_tt**2
	Pdf = Am * np.exp( mf1 ) / mf0

	return Pdf

def norm_func(r, Am, Rt, sigm_tt ):

	mf0 = -0.5 * ( np.log10(r) - np.log10(Rt) )**2 / sigm_tt**2

	# Pdf = Am * np.exp( mf0 ) / sigm_tt
	Pdf = Am * np.exp( mf0 )

	return  Pdf


### === loda obs data
color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
mark_s = ['s', 'o']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

## ... DM mass profile
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

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


## ... amplitude fitting on large scale
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'

out_lim_R = 350 # 350, 400
c_dat = pds.read_csv( fit_path + 'with-dered_total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R,)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]


const = 10**(-1 * lg_fb_gi)


### === ### subsamples 
def subset_func():

	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
	fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

	file_s = 'BCG_Mstar_bin'
	band_str = 'gri'
	out_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'

	#. mass estimation with deredden or not
	id_dered = True
	dered_str = '_with-dered'


	#. surface mass profiles
	dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[0], band_str),)
	lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[1], band_str),)
	hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	lo_tmp_M_f = interp.interp1d( lo_R, lo_surf_M, kind = 'linear', fill_value = 'extrapolate',)
	hi_tmp_M_f = interp.interp1d( hi_R, hi_surf_M, kind = 'linear', fill_value = 'extrapolate',)



	#. mass profile for central region
	cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[0], band_str, dered_str) )
	lo_Ie, lo_Re, lo_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]
	lo_cen_M = sersic_func( lo_rp, 10**lo_Ie, lo_Re, lo_Ne) - sersic_func( 2e3, 10**lo_Ie, lo_Re, lo_Ne)

	cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[1], band_str, dered_str) )
	hi_Ie, hi_Re, hi_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]
	hi_cen_M = sersic_func( hi_rp, 10**hi_Ie, hi_Re, hi_Ne) - sersic_func( 2e3, 10**hi_Ie, hi_Re, hi_Ne)


	#. mass profile on large scale
	lo_out_SM = ( lo_interp_F( lo_rp ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi
	hi_out_SM = ( hi_interp_F( hi_rp ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi



	#... mass profile for the middle region
	#. SM(r) fitting
	# mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[0], band_str, dered_str),)
	# lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
	# lo_mid_mass = log_norm_func( lo_rp, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit ) - log_norm_func( 2e3, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit )

	# mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[1], band_str, dered_str),)
	# hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
	# hi_mid_mass = log_norm_func( hi_rp, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit ) - log_norm_func( 2e3, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit )


	#. SM(r) ratio fitting (log-normal)
	# mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_mid-region_Lognorm_ratio-based_fit%s.csv' % (cat_lis[0], band_str, dered_str),)
	# lo_Am_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['Am'])[0], np.array( mid_dat['Rt'] )[0], np.array( mid_dat['sigma_t'] )[0]

	# lo_mid_pdf = pdf_log_norm_func( lo_rp, lo_Am_fit, lo_Rt_fit, lo_sigm_t_fit)
	# lo_fit_sum = lo_out_SM + lo_cen_M
	# lo_mid_mass = lo_mid_pdf * lo_fit_sum / ( 1 - lo_mid_pdf )


	# mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_mid-region_Lognorm_ratio-based_fit%s.csv' % (cat_lis[1], band_str, dered_str),)
	# hi_Am_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['Am'])[0], np.array( mid_dat['Rt'] )[0], np.array( mid_dat['sigma_t'] )[0]

	# hi_mid_pdf = pdf_log_norm_func( hi_rp, hi_Am_fit, hi_Rt_fit, hi_sigm_t_fit)
	# hi_fit_sum = hi_out_SM + hi_cen_M
	# hi_mid_mass = hi_mid_pdf * hi_fit_sum / ( 1 - hi_mid_pdf )


	#. SM(r) ratio fitting (log-normal)
	mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_mid-region_ratio-based_norm-fit%s.csv' % (cat_lis[0], band_str, dered_str),)
	lo_Am_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['Am'])[0], np.array( mid_dat['Rt'] )[0], np.array( mid_dat['sigma_t'] )[0]

	lo_mid_pdf = norm_func( lo_rp, lo_Am_fit, lo_Rt_fit, lo_sigm_t_fit)
	lo_fit_sum = lo_out_SM + lo_cen_M
	lo_mid_mass = lo_mid_pdf * lo_fit_sum / ( 1 - lo_mid_pdf )


	mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_mid-region_ratio-based_norm-fit%s.csv' % (cat_lis[1], band_str, dered_str),)
	hi_Am_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['Am'])[0], np.array( mid_dat['Rt'] )[0], np.array( mid_dat['sigma_t'] )[0]

	hi_mid_pdf = norm_func( hi_rp, hi_Am_fit, hi_Rt_fit, hi_sigm_t_fit)
	hi_fit_sum = hi_out_SM + hi_cen_M
	hi_mid_mass = hi_mid_pdf * hi_fit_sum / ( 1 - hi_mid_pdf )



	#. trans mass
	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/BCGM_bin_reRbin/SM_pros/' + 
						'%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[0], band_str),)
	cp_lo_R, cp_lo_surf_M, cp_lo_surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/BCGM_bin_reRbin/SM_pros/' + 
						'%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[1], band_str),)
	cp_hi_R, cp_hi_surf_M, cp_hi_surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	_cc_lo_out_M = ( lo_interp_F( cp_lo_R ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi
	_cc_lo_cen_M = sersic_func( cp_lo_R, 10**lo_Ie, lo_Re, lo_Ne) - sersic_func( 2e3, 10**lo_Ie, lo_Re, lo_Ne)

	lo_devi_M = cp_lo_surf_M - ( _cc_lo_out_M + _cc_lo_cen_M )

	id_M_lim = lo_devi_M > 10**4.6 # mass density limit
	id_rx = cp_lo_R > 20
	id_lim = id_M_lim & id_rx

	lo_devi_R = cp_lo_R[ id_lim ]
	lo_devi_M = lo_devi_M[ id_lim ]
	lo_devi_Merr = cp_lo_surf_M_err[ id_lim ]


	_cc_hi_out_M = ( hi_interp_F( cp_hi_R ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi
	_cc_hi_cen_M = sersic_func( cp_hi_R, 10**hi_Ie, hi_Re, hi_Ne) - sersic_func( 2e3, 10**hi_Ie, hi_Re, hi_Ne)

	hi_devi_M = cp_hi_surf_M - ( _cc_hi_out_M + _cc_hi_cen_M )

	id_M_lim = hi_devi_M > 10**2
	id_rx = cp_hi_R > 30
	id_lim = id_M_lim & id_rx

	hi_devi_R = cp_hi_R[ id_lim ]
	hi_devi_M = hi_devi_M[ id_lim ]
	hi_devi_Merr = cp_hi_surf_M_err[ id_lim ]



	#. lensing profile
	hi_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm.txt')
	hi_obs_R, hi_obs_Detsigm = np.array( hi_obs_dat['R'] ), np.array( hi_obs_dat['delta_sigma'] )

	hi_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm_covmat.txt')
	hi_obs_err = np.sqrt( np.diag( hi_obs_cov ) )

	hi_obs_Rp = hi_obs_R / (1 + z_ref) / h
	hi_obs_Detsigm, hi_obs_err = hi_obs_Detsigm * h * (1 + z_ref)**2, hi_obs_err * h * (1 + z_ref)**2


	lo_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm.txt')
	lo_obs_R, lo_obs_Detsigm = np.array( lo_obs_dat['R'] ), np.array( lo_obs_dat['delta_sigma'] )

	lo_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm_covmat.txt')
	lo_obs_err = np.sqrt( np.diag( lo_obs_cov ) )

	lo_obs_Rp = lo_obs_R / (1 + z_ref) / h
	lo_obs_Detsigm, lo_obs_err = lo_obs_Detsigm * h * (1 + z_ref)**2, lo_obs_err * h * (1 + z_ref)**2


	#. cross corelation function to delta_sigma (physical coordinate)
	N_grid = 250

	lo_aveg_xi2mis_sigma = aveg_sigma_func( lo_rp, lo_rho_m, N_grid = N_grid,)
	lo_delta_xi2mis_sigma = (lo_aveg_xi2mis_sigma - lo_rho_m) * 1e-6 # M_sun / pc^2

	hi_aveg_xi2mis_sigma = aveg_sigma_func( hi_rp, hi_rho_m, N_grid = N_grid,)
	hi_delta_xi2mis_sigma = (hi_aveg_xi2mis_sigma - hi_rho_m) * 1e-6

	lo_intep_xi2sigm_F = interp.interp1d( lo_rp, lo_delta_xi2mis_sigma, kind = 'linear', fill_value = 'extrapolate',)
	hi_intep_xi2sigm_F = interp.interp1d( hi_rp, hi_delta_xi2mis_sigma, kind = 'linear', fill_value = 'extrapolate',)


	#. central mass overdensity
	# lo_cen_aveg_sm = aveg_sigma_func( lo_R, _cc_lo_cen_M, )
	# lo_cen_deta_sigm = ( lo_cen_aveg_sm - _cc_lo_cen_M ) * 1e-6

	# hi_cen_aveg_sm = aveg_sigma_func( hi_R, _cc_hi_cen_M, )
	# hi_cen_deta_sigm = ( hi_cen_aveg_sm - _cc_hi_cen_M ) * 1e-6

	new_R = np.logspace( 0.203, 3.03, 42)

	_cc_lo_cen_M = sersic_func( new_R, 10**lo_Ie, lo_Re, lo_Ne) - sersic_func( 2e3, 10**lo_Ie, lo_Re, lo_Ne)
	_cc_hi_cen_M = sersic_func( new_R, 10**hi_Ie, hi_Re, hi_Ne) - sersic_func( 2e3, 10**hi_Ie, hi_Re, hi_Ne)

	lo_cen_aveg_sm = aveg_sigma_func( new_R, _cc_lo_cen_M,)
	lo_cen_deta_sigm = ( lo_cen_aveg_sm - _cc_lo_cen_M ) * 1e-6

	hi_cen_aveg_sm = aveg_sigma_func( new_R, _cc_hi_cen_M,)
	hi_cen_deta_sigm = ( hi_cen_aveg_sm - _cc_hi_cen_M ) * 1e-6



	#. satellites number density
	# lo_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/' + 'g2r_all_sample/data/g-r_deext/low-BCG_g-r_deext_allinfo_noRG.csv')
	lo_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/extend_bcgM/low_BCG_star-Mass_sigma-g_profile.csv')   ## extended catalog

	lo_n_rp, lo_Ng, lo_Ng_err = np.array(lo_Ng_dat['rbins']), np.array(lo_Ng_dat['sigma']), np.array(lo_Ng_dat['sigma_err'])
	lo_Ng, lo_Ng_err = lo_Ng * h**2 / a_ref**2 / 1e6, lo_Ng_err * h**2 / a_ref**2 / 1e6 # unit, '/kpc^{-2}'
	lo_n_rp = lo_n_rp / h / (1 + z_ref)


	# hi_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/' + 'g2r_all_sample/data/g-r_deext/hi-BCG_g-r_deext_allinfo_noRG.csv')
	hi_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/extend_bcgM/high_BCG_star-Mass_sigma-g_profile.csv')  ## extended catalog

	hi_n_rp, hi_Ng, hi_Ng_err = np.array(hi_Ng_dat['rbins']), np.array(hi_Ng_dat['sigma']), np.array(hi_Ng_dat['sigma_err'])
	hi_Ng, hi_Ng_err = hi_Ng * h**2 / a_ref**2 / 1e6, hi_Ng_err * h**2 / a_ref**2 / 1e6
	hi_n_rp = hi_n_rp / h / (1 + z_ref)

	lo_intep_ng_F = interp.interp1d( lo_n_rp, lo_Ng, kind = 'linear', fill_value = 'extrapolate')
	hi_intep_ng_F = interp.interp1d( hi_n_rp, hi_Ng, kind = 'linear', fill_value = 'extrapolate')

	lo_Ng_2Mpc = lo_intep_ng_F( 2 )
	hi_Ng_2Mpc = hi_intep_ng_F( 2 )



	fig = plt.figure( figsize = (15.0, 4.8) )

	# ax0 = fig.add_axes([0.06, 0.33, 0.275, 0.64])
	# bot_ax0 = fig.add_axes([0.06, 0.12, 0.275, 0.21])

	ax0 = fig.add_axes([0.06, 0.12, 0.275, 0.85])

	ax1 = fig.add_axes([0.39, 0.12, 0.275, 0.85])
	ax2 = fig.add_axes([0.72, 0.12, 0.275, 0.85])

	ax0.plot( lo_rp / 1e3, lo_cen_M, ls = ':', color = 'b', alpha = 0.75,)
	ax0.plot( hi_rp / 1e3, hi_cen_M, ls = ':', color = 'r', alpha = 0.75, label = '$\\Sigma_{\\ast}^{\\mathrm{deV}}$',)

	ax0.plot( lo_rp / 1e3, lo_out_SM, ls = '-', color = 'b', alpha = 0.75,)
	ax0.plot( hi_rp / 1e3, hi_out_SM, ls = '-', color = 'r', alpha = 0.75, label = '$ \\gamma \, \\Sigma_{m} $',)

	ax0.plot( lo_rp / 1e3, lo_mid_mass, ls = '--', color = 'b', alpha = 0.75,)
	ax0.plot( hi_rp / 1e3, hi_mid_mass, ls = '--', color = 'r', alpha = 0.75, label = '$\\Sigma_{\\ast}^{tran}$',)

	ax0.errorbar( lo_R / 1e3, lo_surf_M, yerr = lo_surf_M_err, xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', 
		mec = 'b', mfc = 'none', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{\\mathrm{ \\tt{B} {+} \\tt{I} } }$, ' + fig_name[0],)

	ax0.errorbar( hi_R / 1e3, hi_surf_M, yerr = hi_surf_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r',
		mec = 'r', mfc = 'none', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{\\mathrm{ \\tt{B} {+} \\tt{I} } }$, ' + fig_name[1],)


	ax0.errorbar( lo_devi_R / 1e3, lo_devi_M, yerr = lo_devi_Merr, xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', 
		mec = 'b', mfc = 'b', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{tran}$, ' + fig_name[0],)

	ax0.errorbar( hi_devi_R / 1e3, hi_devi_M, yerr = hi_devi_Merr, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r',
		mec = 'r', mfc = 'r', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{tran}$, ' + fig_name[1],)

	# ax0.plot( lo_devi_R / 1e3, lo_devi_M, 'bs', markersize = 7, label = '$\\Sigma_{\\ast}^{tran}$, ' + fig_name[0],)
	# ax0.fill_between( lo_devi_R / 1e3, y1 = lo_devi_M - lo_devi_Merr, y2 = lo_devi_M + lo_devi_Merr, color = 'b', alpha = 0.15,)

	# ax0.plot( hi_devi_R / 1e3, hi_devi_M, 'ro', markersize = 7, label = '$\\Sigma_{\\ast}^{tran}$, ' + fig_name[1],)
	# ax0.fill_between( hi_devi_R / 1e3, y1 = hi_devi_M - hi_devi_Merr, y2 = hi_devi_M + hi_devi_Merr, color = 'r', alpha = 0.15,)


	ax0.annotate( text = 'BCG+ICL', xy = (0.03, 0.05), xycoords = 'axes fraction', fontsize = 15,)

	_handles, _labels = ax0.get_legend_handles_labels()
	ax0.legend( handles = _handles[::-1], labels = _labels[::-1], loc = 1, frameon = False, fontsize = 12, markerfirst = False,)

	ax0.set_xlim( 9e-3, 2e0 )
	ax0.set_xscale('log')
	ax0.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)
	ax0.set_yscale('log')
	ax0.set_ylim( 1e4, 3e8 )
	ax0.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^{2}]$', fontsize = 15,)

	ax0.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
	ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	# #. ratio plot
	# bot_ax0.errorbar( lo_devi_R / 1e3, lo_devi_M / lo_tmp_M_f( lo_devi_R ), yerr = lo_devi_Merr / lo_tmp_M_f( lo_devi_R ), 
	# 	ls = 'none', marker = 's', ms = 7, mec = 'b', mfc = 'b', ecolor = 'b', alpha = 0.75, capsize = 3,)
	# bot_ax0.plot( lo_rp / 1e3, lo_mid_mass / (lo_mid_mass + lo_cen_M + lo_out_SM), ls = '--', color = 'b',)

	# bot_ax0.errorbar( hi_devi_R / 1e3, hi_devi_M / hi_tmp_M_f( hi_devi_R ), yerr = hi_devi_Merr / hi_tmp_M_f( hi_devi_R ), 
	# 	ls = 'none', marker = 'o', ms = 7, mec = 'r', mfc = 'r', ecolor = 'r', alpha = 0.75, capsize = 3,)
	# bot_ax0.plot( hi_rp / 1e3, hi_mid_mass / (hi_mid_mass + hi_cen_M + hi_out_SM), ls = '--', color = 'r',)

	# bot_ax0.set_xlim( ax0.get_xlim() )
	# bot_ax0.set_xscale('log')
	# bot_ax0.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)
	# bot_ax0.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
	# bot_ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])

	# bot_ax0.set_ylim( -0.075, 0.50 )
	# bot_ax0.set_yticks([ 0, 0.2, 0.4 ])
	# bot_ax0.set_yticklabels( labels = ['$\\mathrm{0.0}$','$\\mathrm{0.2}$', '$\\mathrm{0.4}$'])

	# bot_ax0.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)
	# bot_ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	# bot_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	# ax0.set_xticklabels( labels = [] )


	ax1.plot( new_R / 1e3, lo_cen_deta_sigm, ls = ':', color = 'b',)
	ax1.plot( new_R / 1e3, hi_cen_deta_sigm, ls = ':', color = 'r', label = '$\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} }$')

	ax1.plot( lo_rp / 1e3, lo_delta_xi2mis_sigma, ls = '-', color = 'b',)
	ax1.plot( hi_rp / 1e3, hi_delta_xi2mis_sigma, ls = '-', color = 'r', label = '$\\Delta \\Sigma_{m}$')

	ax1.plot( new_R / 1e3, lo_cen_deta_sigm + lo_intep_xi2sigm_F( new_R ), ls = '--', color = 'b',)
	ax1.plot( new_R / 1e3, hi_cen_deta_sigm + hi_intep_xi2sigm_F( new_R ), ls = '--', color = 'r', 
			label = '$\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} } \, {+} \, \\Delta \\Sigma_{m}$')

	ax1.errorbar( lo_obs_R[2:] / (1 + z_ref) / h, lo_obs_Detsigm[2:] * h * (1 + z_ref)**2, yerr = lo_obs_err[2:] * h * (1 + z_ref)**2, 
		xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', mec = 'b', mfc = 'none', capsize = 2, markersize = 7, label = fig_name[0],)

	ax1.errorbar( hi_obs_R[2:] / (1 + z_ref) / h, hi_obs_Detsigm[2:] * h * (1 + z_ref)**2, yerr = hi_obs_err[2:] * h * (1 + z_ref)**2, 
		xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', mec = 'r', mfc = 'none', capsize = 2, markersize = 7, label = fig_name[1],)

	ax1.annotate( text = 'Weak Lensing', xy = (0.03, 0.05), xycoords = 'axes fraction', fontsize = 15,)

	ax1.set_xlim( 9e-3, 2e0 )
	ax1.set_xscale('log')
	ax1.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)

	ax1.set_yscale('log')
	ax1.set_ylim( 2e1, 5e2)
	ax1.set_ylabel('$\\Delta \\Sigma \; [M_{\\odot} \, / \, pc^{2}]$', fontsize = 15,)

	ax1.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
	ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])

	_handles, _labels = ax1.get_legend_handles_labels()
	ax1.legend( handles = _handles[::-1], labels = _labels[::-1], loc = 1, frameon = False, fontsize = 12, markerfirst = False, borderaxespad = 0.2,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	ax2.errorbar( lo_n_rp, lo_Ng * 1e6, yerr = lo_Ng_err * 1e6, xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', mec = 'b', 
		mfc = 'none', capsize = 2, markersize = 7, label = fig_name[0],)

	ax2.errorbar( hi_n_rp, hi_Ng * 1e6, yerr = hi_Ng_err * 1e6, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', mec = 'r', 
		mfc = 'none', capsize = 2, markersize = 7, label = fig_name[1],)

	ax2.annotate( text = 'Galaxies ($M_{i}{<}{-}%.2f$)' % np.abs( (-19.43 + 5 * np.log10(h) ) ), xy = (0.03, 0.05), xycoords = 'axes fraction', fontsize = 15,)

	ax2.set_xlim( 9e-3, 2e0 )
	ax2.set_xscale('log')
	ax2.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)
	ax2.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
	ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])

	ax2.set_yscale('log')
	ax2.set_ylim( 1e0, 3e2 )
	ax2.legend( loc = 1, frameon = False, fontsize = 15,)
	ax2.set_ylabel('$ \\Sigma_{g} \; [ \\# \, / \, \\mathrm{M}pc^{2} ]$', fontsize = 15,)
	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	# plt.savefig('/home/xkchen/subsamples_SB_SM%s.png' % dered_str, dpi = 300)
	plt.savefig('/home/xkchen/subsamples_SB_SM.pdf', dpi = 300)
	plt.close()


subset_func()
raise


### === ### total sample
def tot_samp_SM_fig():

	## ... satellite number density
	bin_R, siglow, errsiglow, sighig, errsighig, highoverlow, errhighoverlow = np.genfromtxt(
																			 '/home/xkchen/tmp_run/data_files/figs/result_high_over_low.txt', unpack = True)
	bin_R = bin_R * 1e3 * a_ref / h
	siglow, errsiglow, sighig, errsighig = np.array( [siglow * h**2 / 1e6, errsiglow * h**2 / 1e6, sighig * h**2 / 1e6, errsighig * h**2 / 1e6] ) / a_ref**2

	id_nan = np.isnan( bin_R )
	bin_R = bin_R[ id_nan == False]
	siglow, errsiglow, sighig, errsighig = siglow[ id_nan == False], errsiglow[ id_nan == False], sighig[ id_nan == False], errsighig[ id_nan == False]

	lo_Ng_int_F = interp.interp1d( bin_R, siglow, kind = 'linear', fill_value = 'extrapolate')
	hi_Ng_int_F = interp.interp1d( bin_R, sighig, kind = 'linear', fill_value = 'extrapolate')


	##... mass profile
	BG_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
	band_str = 'gri'

	##... mass estimation with deredden or not
	id_dered = True
	dered_str = 'with-dered_'


	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
	obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

	## re-divide radial bins
	new_R = np.logspace( 0, np.log10(2.5e3), 100)

	#... central part
	p_dat = pds.read_csv( fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % (dered_str, band_str),)
	c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

	cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
	fit_cen_M = sersic_func( new_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc


	#... signal on large scale
	xi_rp = (lo_xi + hi_xi) / 2
	tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
	xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

	misNFW_sigma = xi_to_Mf( lo_rp )
	sigma_2Mpc = xi_to_Mf( 2e3 )
	lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	fit_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi
	fit_sum = fit_out_M + fit_cen_M


	#... trans part
	devi_M = surf_M - ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi - ( sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc )
	devi_err = surf_M_err

	idx_lim = devi_M >= 0 # 10**4.6
	idR_lim = obs_R >= 30 # kpc
	id_lim = idx_lim & idR_lim

	devi_R = obs_R[ id_lim ]
	devi_M = devi_M[ id_lim ]
	devi_err = devi_err[ id_lim ]


	##... trans part fitting
	#. SM(r) fitting
	# mid_pat = pds.read_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (dered_str, band_str),)
	# lg_SM_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['lg_M0'] )[0], np.array( mid_pat['R_t'] )[0], np.array( mid_pat['sigma_t'] )[0]
	# fit_cross = log_norm_func( new_R, lg_SM_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit )


	#. SM(r) ratio fitting ( log-normal )
	# mid_pat = pds.read_csv( fit_path + '%stotal_%s-band-based_mid-region_Lognorm_ratio-based_fit.csv' % (dered_str, band_str),)
	# Am_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['Am'] )[0], np.array( mid_pat['Rt'] )[0], np.array( mid_pat['sigma_t'] )[0]

	# fit_mid_pdf = pdf_log_norm_func( new_R, Am_fit, Rt_fit, sigm_tt_fit)
	# fit_cross = fit_mid_pdf * ( fit_sum ) / ( 1 - fit_mid_pdf )


	#. SM(r) ratio fitting (normal)
	mid_pat = pds.read_csv( fit_path + '%stotal_%s-band-based_mid-region_ratio-based_norm-fit.csv' % (dered_str, band_str),)
	Am_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['Am'] )[0], np.array( mid_pat['Rt'] )[0], np.array( mid_pat['sigma_t'] )[0]

	fit_mid_pdf = norm_func( new_R, Am_fit, Rt_fit, sigm_tt_fit)
	fit_cross = fit_mid_pdf * ( fit_sum ) / ( 1 - fit_mid_pdf )



	### === ### figs
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
		capsize = 3, mec = 'k', mfc = 'none', label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $',)

	ax1.plot( new_R / 1e3, fit_sum, ls = '-.', color = 'Gray', alpha = 0.95, 
		label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\gamma \, \\Sigma_{m} $')

	ax1.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
		label = '$ \\gamma \, \\Sigma_{m} $',)

	ax1.plot( new_R / 1e3, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{ \\mathrm{deV} }$')

	ax1.plot( new_R / 1e3, fit_cross, ls = '--', color = 'k', alpha = 0.75, label = '$\\Sigma_{\\ast}^{tran}$')
	ax1.errorbar( devi_R / 1e3, devi_M, yerr = devi_err, ls = 'none', marker = 's', ms = 8, mec = 'k', mfc = 'none', 
		ecolor = 'k', alpha = 0.75, capsize = 3, label = '$\\Sigma_{\\ast}^{tran} $')

	# handles,labels = ax1.get_legend_handles_labels()
	# handles = [ handles[3], handles[2], handles[0], handles[1], handles[5], handles[4] ]
	# labels = [ labels[3], labels[2], labels[0], labels[1], labels[5], labels[4] ]
	# ax1.legend( handles, labels, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

	ax1.legend(loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

	ax1.set_ylim( 1e4, 3e8 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

	ax1.set_xlim( 9e-3, 2e0 )
	ax1.set_xscale( 'log' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	tmp_tot_M_f = interp.interp1d( obs_R, surf_M, kind = 'linear', fill_value = 'extrapolate',)

	sub_ax1.errorbar( devi_R / 1e3, devi_M / tmp_tot_M_f( devi_R ), yerr = devi_err / tmp_tot_M_f( devi_R ), 
		ls = 'none', marker = 's', ms = 8, mec = 'k', mfc = 'none', ecolor = 'k', alpha = 0.75, capsize = 3,)

	sub_ax1.plot( new_R / 1e3, fit_cross / ( fit_cross + fit_sum ), ls = '--', color = 'k', alpha = 0.75,)
	# sub_ax1.axhline( y = 0.255, ls = '-.', color = 'b',)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)
	sub_ax1.set_xscale( 'log' )
	sub_ax1.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)

	sub_ax1.set_ylim( -0.075, 0.40 )
	sub_ax1.set_yticks([ 0, 0.1, 0.2, 0.3 ])
	sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.0}$', '$\\mathrm{0.1}$', '$\\mathrm{0.2}$', '$\\mathrm{0.3}$'])	

	sub_ax1.set_xticks([1e-2, 1e-1, 1e0, 2e0])
	sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'] ) )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/DM_Ng_compare.pdf', dpi = 300)
	# plt.savefig('/home/xkchen/DM_Ng_compare.png', dpi = 300)
	plt.close()

	raise

tot_samp_SM_fig()

