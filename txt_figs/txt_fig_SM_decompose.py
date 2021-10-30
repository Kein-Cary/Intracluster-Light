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
from color_2_mass import get_c2mass_func, gi_band_c2m_func
from multiprocessing import Pool

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
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

out_lim_R = 350 #350, 400

c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R,)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_SG_N-fit.csv' % out_lim_R,)
lg_Ng_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_Ng_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_Ng_ri = np.array( c_dat['lg_fb_ri'] )[0]

const = 10**(-1 * lg_fb_gi)


### === ### subsamples
def subset_func():

	#. miscenterinf parameters
	# v_m = 200 # rho_mean = 200 * rho_c * omega_m
	# c_mass = [5.87, 6.95]
	# Mh0 = [14.24, 14.24]
	# off_set = [230, 210] # in unit kpc / h
	# f_off = [0.37, 0.20]


	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
	fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

	file_s = 'BCG_Mstar_bin'
	out_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
	band_str = 'gri'

	#. mass estimation with deredden or not
	# id_dered = True
	# dered_str = '_with-dered'

	id_dered = False
	dered_str = ''

	if id_dered == False:
		#. surface mass profiles
		dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str),)
		lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

		dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str),)
		hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	if id_dered == True:
		#. surface mass profiles
		dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[0], band_str),)
		lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

		dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[1], band_str),)
		hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )


	#. mass profile on large scale
	lo_out_SM = ( lo_interp_F( lo_rp ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi
	hi_out_SM = ( hi_interp_F( hi_rp ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi

	#. mass profile for central region
	cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[0], band_str, dered_str) )
	lo_Ie, lo_Re, lo_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]
	lo_cen_M = sersic_func( lo_R, 10**lo_Ie, lo_Re, lo_Ne) - sersic_func( 2e3, 10**lo_Ie, lo_Re, lo_Ne)

	cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[1], band_str, dered_str) )
	hi_Ie, hi_Re, hi_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]
	hi_cen_M = sersic_func( hi_R, 10**hi_Ie, hi_Re, hi_Ne) - sersic_func( 2e3, 10**hi_Ie, hi_Re, hi_Ne)


	#... mass profile for the middle region
	mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[0], band_str, dered_str),)
	lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
	lo_mid_mass = log_norm_func( lo_R, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit ) - log_norm_func( 2e3, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit )


	mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[1], band_str, dered_str),)
	hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
	hi_mid_mass = log_norm_func( hi_R, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit ) - log_norm_func( 2e3, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit )


	#. trans mass
	_cc_lo_out_M = ( lo_interp_F( lo_R ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi
	lo_devi_M = lo_surf_M - ( _cc_lo_out_M + lo_cen_M )

	id_M_lim = lo_devi_M > 10**4.6 # mass density limit
	id_rx = lo_R > 20 # 50
	id_lim = id_M_lim & id_rx

	lo_devi_R = lo_R[ id_lim ]
	lo_devi_M = lo_devi_M[ id_lim ]
	lo_devi_Merr = lo_surf_M_err[ id_lim ]


	_cc_hi_out_M = ( hi_interp_F( hi_R ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi
	hi_devi_M = hi_surf_M - ( _cc_hi_out_M + hi_cen_M ) 

	id_M_lim = hi_devi_M > 10**4
	id_rx = hi_R > 20 # 51
	id_lim = id_M_lim & id_rx

	hi_devi_R = hi_R[ id_lim ]
	hi_devi_M = hi_devi_M[ id_lim ]
	hi_devi_Merr = hi_surf_M_err[ id_lim ]



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
	lo_cen_aveg_sm = aveg_sigma_func( lo_R, lo_cen_M )
	lo_cen_deta_sigm = ( lo_cen_aveg_sm - lo_cen_M ) * 1e-6

	hi_cen_aveg_sm = aveg_sigma_func( hi_R, hi_cen_M )
	hi_cen_deta_sigm = ( hi_cen_aveg_sm - hi_cen_M ) * 1e-6


	# lo_r_avegM = cumu_mass_func( lo_R, lo_cen_M, N_grid = N_grid)
	# hi_r_avegM = cumu_mass_func( hi_R, hi_cen_M, N_grid = N_grid)

	# _cc_lo_avegM = ( lo_r_avegM / ( np.pi * lo_R**2 ) ) * 1e-6
	# _cc_hi_avegM = ( hi_r_avegM / ( np.pi * hi_R**2 ) ) * 1e-6

	# plt.figure()
	# ax1 = plt.subplot(111)

	# ax1.plot( lo_R / 1e3, lo_cen_deta_sigm, ls = '-', color = 'b', label = fig_name[0] + ', $\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} }$')
	# ax1.plot( hi_R / 1e3, hi_cen_deta_sigm, ls = '-', color = 'r', label = fig_name[1] + ', $\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} }$')

	# ax1.plot( lo_R / 1e3, _cc_lo_avegM, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] + ', $M_{\\ast}[<R] / (\\pi R^{2})$')
	# ax1.plot( hi_R / 1e3, _cc_hi_avegM, ls = '--', color = 'r', alpha = 0.75, label = fig_name[1] + ', $M_{\\ast}[<R] / (\\pi R^{2})$')

	# ax1.legend( loc = 3,)

	# ax1.set_xlim( 5e-3, 1e-1 )
	# ax1.set_xscale('log')
	# ax1.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)

	# ax1.set_yscale('log')
	# ax1.set_ylim( 2e1, 5e2)
	# ax1.set_ylabel('$\\Delta \\Sigma \; [M_{\\odot} \, / \, pc^{2}]$', fontsize = 15,)
	# ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	# plt.savefig('/home/xkchen/delta_sigma_compare.png', dpi = 300)
	# plt.close()


	#. satellites number density
	lo_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/' + 'g2r_all_sample/data/g-r_deext/low-BCG_g-r_deext_allinfo_noRG.csv')
	lo_n_rp, lo_Ng, lo_Ng_err = np.array(lo_Ng_dat['rbins']), np.array(lo_Ng_dat['sigma']), np.array(lo_Ng_dat['sigma_err'])
	lo_Ng, lo_Ng_err = lo_Ng * h**2 / a_ref**2 / 1e6, lo_Ng_err * h**2 / a_ref**2 / 1e6 # unit, '/kpc^{-2}'
	lo_n_rp = lo_n_rp / h / (1 + z_ref)

	hi_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/' + 'g2r_all_sample/data/g-r_deext/hi-BCG_g-r_deext_allinfo_noRG.csv')
	hi_n_rp, hi_Ng, hi_Ng_err = np.array(hi_Ng_dat['rbins']), np.array(hi_Ng_dat['sigma']), np.array(hi_Ng_dat['sigma_err'])
	hi_Ng, hi_Ng_err = hi_Ng * h**2 / a_ref**2 / 1e6, hi_Ng_err * h**2 / a_ref**2 / 1e6
	hi_n_rp = hi_n_rp / h / (1 + z_ref)

	lo_intep_ng_F = interp.interp1d( lo_n_rp, lo_Ng, kind = 'linear', fill_value = 'extrapolate')
	hi_intep_ng_F = interp.interp1d( hi_n_rp, hi_Ng, kind = 'linear', fill_value = 'extrapolate')

	lo_Ng_2Mpc = lo_intep_ng_F( 2 )
	hi_Ng_2Mpc = hi_intep_ng_F( 2 )



	fig = plt.figure( figsize = (15.0, 4.8) )
	ax0 = fig.add_axes([0.05, 0.12, 0.275, 0.85])
	ax1 = fig.add_axes([0.38, 0.12, 0.275, 0.85])
	ax2 = fig.add_axes([0.71, 0.12, 0.275, 0.85])

	ax0.plot( lo_R / 1e3, lo_cen_M, ls = ':', color = 'b', alpha = 0.75,)
	ax0.plot( hi_R / 1e3, hi_cen_M, ls = ':', color = 'r', alpha = 0.75, label = '$\\Sigma_{\\ast}^{\\mathrm{deV}}$',)

	ax0.plot( lo_rp / 1e3, lo_out_SM, ls = '-', color = 'b', alpha = 0.75,)
	# ax0.plot( hi_rp / 1e3, hi_out_SM, ls = '-', color = 'r', alpha = 0.75, label = '$ \\Sigma_{m} \, / \, {%.0f} $' % const,)
	ax0.plot( hi_rp / 1e3, hi_out_SM, ls = '-', color = 'r', alpha = 0.75, label = '$ \\gamma \, \\Sigma_{m} $',)

	ax0.plot( lo_R / 1e3, lo_mid_mass, ls = '--', color = 'b', alpha = 0.75,)
	ax0.plot( hi_R / 1e3, hi_mid_mass, ls = '--', color = 'r', alpha = 0.75, label = '$\\Sigma_{\\ast}^{tran} \, (\\mathrm{Eqn. \, 5})$',)

	ax0.errorbar( lo_R / 1e3, lo_surf_M, yerr = lo_surf_M_err, xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', 
		mec = 'b', mfc = 'none', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{\\mathrm{ \\tt{B} {+} \\tt{I} } }$, ' + fig_name[0],)

	ax0.errorbar( hi_R / 1e3, hi_surf_M, yerr = hi_surf_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r',
		mec = 'r', mfc = 'none', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{\\mathrm{ \\tt{B} {+} \\tt{I} } }$, ' + fig_name[1],)

	ax0.errorbar( lo_devi_R / 1e3, lo_devi_M, yerr = lo_devi_Merr, xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', 
		mec = 'b', mfc = 'b', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{tran}$, ' + fig_name[0],)

	ax0.errorbar( hi_devi_R / 1e3, hi_devi_M, yerr = hi_devi_Merr, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r',
		mec = 'r', mfc = 'r', capsize = 2, markersize = 7, label = '$\\Sigma_{\\ast}^{tran}$, ' + fig_name[1],)


	_handles, _labels = ax0.get_legend_handles_labels()
	ax0.legend( handles = _handles[::-1], labels = _labels[::-1], loc = 1, frameon = False, fontsize = 12, markerfirst = False,)

	ax0.set_xlim( 9e-3, 2e0 )
	ax0.set_xscale('log')
	ax0.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)
	ax0.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
	ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])

	ax0.set_yscale('log')
	ax0.set_ylim( 1e4, 3e8 )

	ax0.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^{2}]$', fontsize = 15,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	ax1.plot( lo_R / 1e3, lo_cen_deta_sigm, ls = ':', color = 'b',)
	ax1.plot( hi_R / 1e3, hi_cen_deta_sigm, ls = ':', color = 'r', label = '$\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} }$')

	ax1.plot( lo_rp / 1e3, lo_delta_xi2mis_sigma, ls = '-', color = 'b',)
	ax1.plot( hi_rp / 1e3, hi_delta_xi2mis_sigma, ls = '-', color = 'r', label = '$\\Delta \\Sigma_{m}$')

	ax1.plot( lo_R / 1e3, lo_cen_deta_sigm + lo_intep_xi2sigm_F( lo_R ), ls = '--', color = 'b',)
	ax1.plot( hi_R / 1e3, hi_cen_deta_sigm + hi_intep_xi2sigm_F( hi_R ), ls = '--', color = 'r', 
			label = '$\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} } \, {+} \, \\Delta \\Sigma_{m}$')

	ax1.errorbar( lo_obs_R[2:] / (1 + z_ref) / h, lo_obs_Detsigm[2:] * h * (1 + z_ref)**2, yerr = lo_obs_err[2:] * h * (1 + z_ref)**2, 
		xerr = None, color = 'b', marker = 's', ls = 'none', ecolor = 'b', mec = 'b', mfc = 'none', capsize = 2, markersize = 7, label = fig_name[0],)

	ax1.errorbar( hi_obs_R[2:] / (1 + z_ref) / h, hi_obs_Detsigm[2:] * h * (1 + z_ref)**2, yerr = hi_obs_err[2:] * h * (1 + z_ref)**2, 
		xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', mec = 'r', mfc = 'none', capsize = 2, markersize = 7, label = fig_name[1],)

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

	ax2.set_xlim( 9e-3, 2e0 )
	ax2.set_xscale('log')
	ax2.set_xlabel('$R \; [\\mathrm{M}pc]$', fontsize = 15,)
	ax2.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
	ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])

	ax2.set_yscale('log')
	ax2.set_ylim( 1e0, 3e2 )
	ax2.legend( loc = 3, frameon = False, fontsize = 15,)
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
	BG_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
	band_str = 'gri'

	##... mass estimation with deredden or not
	id_dered = True
	dered_str = 'with-dered_'

	# id_dered = False
	# dered_str = ''

	if id_dered == False:
		dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
		obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

	if id_dered == True:
		dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
		obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


	#... central part
	p_dat = pds.read_csv( fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % (dered_str, band_str),)
	c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

	#... trans part fitting
	mid_pat = pds.read_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (dered_str, band_str),)
	lg_SM_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['lg_M0'] )[0], np.array( mid_pat['R_t'] )[0], np.array( mid_pat['sigma_t'] )[0]


	## re-divide radial bins
	new_R = np.logspace( 0, np.log10(2.5e3), 100)

	fit_cross = log_norm_func( new_R, lg_SM_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit )

	cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
	fit_cen_M = sersic_func( new_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc

	#...
	xi_rp = (lo_xi + hi_xi) / 2
	tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
	xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

	misNFW_sigma = xi_to_Mf( lo_rp )
	sigma_2Mpc = xi_to_Mf( 2e3 )
	lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	#...
	sig_aveg = (siglow + sighig) / 2
	err_aveg = np.sqrt( errsiglow**2 / 4 + errsighig**2 / 4)

	sig_rho_f = interp.interp1d( bin_R, sig_aveg, kind = 'linear', fill_value = 'extrapolate',)

	Ng_sigma = sig_rho_f( bin_R )
	Ng_2Mpc = sig_rho_f( 2e3 )
	lg_Ng_sigma = np.log10( Ng_sigma - Ng_2Mpc )

	#... trans part
	devi_M = surf_M - ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi - ( sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc )
	devi_err = surf_M_err

	idx_lim = devi_M >= 10**4.6
	idR_lim = obs_R >= 30 # kpc
	id_lim = idx_lim & idR_lim

	devi_R = obs_R[ id_lim ]
	devi_M = devi_M[ id_lim ]
	devi_err = devi_err[ id_lim ]

	fit_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi
	fit_sum = fit_out_M + fit_cen_M


	### === ### figs
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
		capsize = 3, mec = 'k', mfc = 'none', label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $',)

	# ax1.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
	# 	label = '$ \\Sigma_{m} \, / \, {%.0f} $' % const,)
	ax1.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
		label = '$ \\gamma \, \\Sigma_{m} $',)

	ax1.plot( new_R / 1e3, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{ \\mathrm{deV} }$')
	# ax1.plot( new_R / 1e3, fit_sum, ls = '--', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
	# 	label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\Sigma_{m} \, / \, {%.0f} $' % const)
	ax1.plot( new_R / 1e3, fit_sum, ls = '--', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
		label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{deV} } } {+} \\gamma \, \\Sigma_{m} $')

	ax1.plot( new_R / 1e3, fit_cross, ls = '-.', color = 'k', alpha = 0.75, label = '$\\Sigma_{\\ast}^{tran} \, (\\mathrm{Eqn.\,5})$')
	ax1.errorbar( devi_R / 1e3, devi_M, yerr = devi_err, ls = 'none', marker = 's', ms = 8, mec = 'k', mfc = 'none', 
		ecolor = 'k', alpha = 0.75, capsize = 3, label = '$\\Sigma_{\\ast}^{tran} $')

	handles,labels = ax1.get_legend_handles_labels()
	handles = [ handles[3], handles[2], handles[0], handles[1], handles[5], handles[4] ]
	labels = [ labels[3], labels[2], labels[0], labels[1], labels[5], labels[4] ]
	ax1.legend( handles, labels, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

	# ax1.legend(loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

	ax1.set_ylim( 1e4, 3e8 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

	ax1.set_xlim( 9e-3, 2e0 )
	ax1.set_xscale( 'log' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	tmp_tot_M_f = interp.interp1d( obs_R, surf_M, kind = 'linear', fill_value = 'extrapolate',)

	sub_ax1.errorbar( devi_R / 1e3, devi_M / tmp_tot_M_f( devi_R ), yerr = devi_err / tmp_tot_M_f( devi_R ), 
		ls = 'none', marker = 's', ms = 8, mec = 'k', mfc = 'none', ecolor = 'k', alpha = 0.75, capsize = 3,)
	sub_ax1.plot( new_R / 1e3, fit_cross / ( fit_cross + fit_sum ), ls = '-.', color = 'k', alpha = 0.75,)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)
	sub_ax1.set_xscale( 'log' )
	sub_ax1.set_ylabel('$\\Sigma_{\\ast}^{tran} \, / \, \\Sigma_{\\ast}^{ \\mathrm{ \\tt{B} {+} \\tt{I} } } $', fontsize = 15)

	sub_ax1.set_xticks([1e-2, 1e-1, 1e0, 2e0])
	sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'] ) )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/DM_Ng_compare.pdf', dpi = 300)
	# plt.savefig('/home/xkchen/%sDM_Ng_compare.png' % dered_str, dpi = 300)
	plt.close()


	# fig_tx = plt.figure( figsize = (5.4, 5.4) )
	# ax1 = fig_tx.add_axes( [0.15, 0.11, 0.83, 0.83] )

	# ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
	# 	label = '$\\Sigma_{\\ast}^{ \\mathrm{BCG}{+}\\mathrm{ICL} } $', capsize = 3, )

	# # ax1.plot( bin_R / 1e3, (Ng_sigma - Ng_2Mpc) * 10**lg_Ng_gi, ls = '--', color = 'k', alpha = 0.65, 
	# # 	label = '$ 10^{\\mathrm{%.0f}}\, M_{\\odot} \\times \\mathrm{N}_{g} $' % lg_Ng_gi,)
	# ax1.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
	# 	label = '$ \\Sigma_{m} / {%.0f} $' % const,)

	# ax1.plot( new_R / 1e3, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{ \\mathrm{BCG} }$')
	# ax1.plot( new_R / 1e3, fit_sum, ls = '--', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
	# 	label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{BCG} } } {+} \\Sigma_{m} / {%.0f} $' % const)

	# ax1.errorbar( devi_R / 1e3, devi_M, yerr = devi_err, ls = 'none', marker = 'o', ms = 8, mec = 'k', mfc = 'none', 
	# 	ecolor = 'k', alpha = 0.75, capsize = 3,
	# 	label = '$\\Sigma_{\\ast}^{tran} \, = \, \\Sigma_{\\ast}^{ \\mathrm{BCG}{+}\\mathrm{ICL} } {-} $' + 
	# 			'$(\\Sigma_{\\ast}^{ \\mathrm{BCG} } {+} \\Sigma_{m} / \\mathrm{%.0f} )$' % const)

	# handles,labels = ax1.get_legend_handles_labels()

	# handles = [ handles[3], handles[4], handles[0], handles[2], handles[1] ]
	# labels = [ labels[3], labels[4], labels[0], labels[2], labels[1] ]
	# ax1.legend( handles, labels, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

	# ax1.set_ylim( 1e4, 3e8 )
	# ax1.set_yscale('log')
	# ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

	# ax1.set_xlim( 9e-3, 2e0 )
	# ax1.set_xscale( 'log' )
	# ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)

	# ax1.set_xticks([ 1e-2, 1e-1, 1e0])
	# ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

	# ax1.set_xticks([ 2e0 ], minor = True,)
	# ax1.set_xticklabels( labels = ['$\\mathrm{2}$'], minor = True,)

	# ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	# plt.savefig('/home/xkchen/DM_Ng_compare.pdf', dpi = 300)
	# # plt.savefig('/home/xkchen/DM_Ng_compare.png', dpi = 300)
	# plt.close()

tot_samp_SM_fig()
