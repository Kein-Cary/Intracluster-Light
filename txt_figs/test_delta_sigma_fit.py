import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.signal as signal

from scipy import stats as sts
from scipy import integrate as integ
import seaborn as sns

from surface_mass_density import sigmam, sigmac
from surface_mass_density import input_cosm_model, cosmos_param, rhom_set

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
psf_FWHM = 1.32 # arcsec

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

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

### === ### data load
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)


## amplitude adjust factor for the Sigma_g and Sigma_m profiles
out_lim_R = 350
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R,)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_SG_N-fit.csv' % out_lim_R,)
lg_Ng_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_Ng_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_Ng_ri = np.array( c_dat['lg_fb_ri'] )[0]

const = 10**(-1 * lg_fb_gi)


##.. xi-profile of Zu et al. 2021
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3, comoving case

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1] ## xi in unit Mpc / h, comoving coordinate
lo_rho_m = ( lo_xi * 1e3 * rho_m ) * h / a_ref**2
lo_rp = lo_rp * 1e3 * a_ref / h

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) * h / a_ref**2
hi_rp = hi_rp * 1e3 * a_ref / h

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'linear', fill_value = 'extrapolate')
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'linear', fill_value = 'extrapolate')

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )


##.. galaxy number density
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


##.. BCG+ICL surface mass density
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
file_s = 'BCG_Mstar_bin'

out_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'

dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )
lo_R = lo_R / 1e3

dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )
hi_R = hi_R / 1e3


#. subsample large radii surface mass fitting
c_dat = pds.read_csv(fit_path + '%s_all-color-to-M_SG_N-fit.csv' % cat_lis[0],)
sub_lg_Ng_fb = [ np.array( c_dat['lg_fb_300'] )[0], np.array( c_dat['lg_fb_350'] )[0], np.array( c_dat['lg_fb_400'] )[0] ]

c_dat = pds.read_csv(fit_path + '%s_all-color-to-M_xi2M-fit.csv' % cat_lis[0],)
sub_lg_xi2M_fb = [ np.array( c_dat['lg_fb_300'] )[0], np.array( c_dat['lg_fb_350'] )[0], np.array( c_dat['lg_fb_400'] )[0] ]


def pre_compare_func():
	fig = plt.figure()
	ax1 = fig.add_axes( [0.14, 0.11, 0.80, 0.20] )
	ax0 = fig.add_axes( [0.14, 0.31, 0.80, 0.60] )

	l1, = ax0.plot( lo_R, lo_surf_M, ls = '-', color = 'b', alpha = 0.75, 
		label = '$\\Sigma_{\\ast}^{ \\mathrm{BCG+ICL} }[M_{\\odot}]$',)
	ax0.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = 'b', alpha = 0.12,)
	l2, = ax0.plot( hi_R, hi_surf_M, ls = '-', color = 'r', alpha = 0.75,)
	ax0.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = 'r', alpha = 0.12,)

	# af_0 = 2.65e4
	ax0.plot( lo_n_rp, (lo_Ng - lo_Ng_2Mpc) * 10**lg_Ng_gi, ls = '--', color = 'b', alpha = 0.75, 
		label = '$ 10^{\\mathrm{%.1f}} \, M_{\\odot} \\times \\Sigma_{g} $' % lg_Ng_gi,)
	ax0.plot( hi_n_rp, (hi_Ng - hi_Ng_2Mpc) * 10**lg_Ng_gi, ls = '--', color = 'r', alpha = 0.75, )

	# af_1 = 2e-3
	ax0.plot( lo_rp / 1e3, (lo_rho_m - lo_xi2M_2Mpc) * 10**lg_fb_gi, ls = ':', color = 'b', alpha = 0.75, 
		label = '$ \\Sigma_{m} / {%.0f} $' % const,)
	ax0.plot( hi_rp / 1e3, (hi_rho_m - hi_xi2M_2Mpc) * 10**lg_fb_gi, ls = ':', color = 'r', alpha = 0.75, )

	legend_0 = ax0.legend( handles = [ l1, l2 ], labels = [ fig_name[0], fig_name[1] ], loc = 1, frameon = False, fontsize = 13,)
	ax0.legend( loc = 3, frameon = False, fontsize = 13,)
	ax0.add_artist( legend_0 )

	ax0.set_xlim( 1e-2, 2e0 )
	ax0.set_xscale('log')
	# ax0.set_xlabel('$R \; [\mathrm{M}pc]$')
	ax0.set_yscale('log')
	ax0.set_ylim( 5e3, 3e8 )
	ax0.set_ylabel('$ \\Sigma \; [ kpc^{-2}]$', fontsize = 15,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax1.plot( lo_R, lo_surf_M / lo_surf_M, ls = '-', color = 'b', alpha = 0.75,)
	ax1.plot( hi_R, hi_surf_M / lo_surf_M, ls = '-', color = 'r', alpha = 0.75,)

	ax1.plot( lo_rp / 1e3, (lo_rho_m - lo_xi2M_2Mpc) / (lo_rho_m - lo_xi2M_2Mpc), ls = ':', color = 'b', alpha = 0.75,)
	ax1.plot( hi_rp / 1e3, (hi_rho_m - hi_xi2M_2Mpc) / (lo_interp_F( hi_rp ) - lo_xi2M_2Mpc), ls = ':', color = 'r', alpha = 0.75,)

	ax1.plot( lo_n_rp, ( lo_Ng - lo_Ng_2Mpc) / (lo_Ng - lo_Ng_2Mpc), ls = '--', color = 'b', alpha = 0.75,)
	ax1.plot( hi_n_rp, (hi_Ng - hi_Ng_2Mpc) / (lo_intep_ng_F( hi_n_rp ) - lo_Ng_2Mpc), ls = '--', color = 'r', alpha = 0.75,)

	ax1.set_xlim( 1e-2, 2e0 )
	ax1.set_xscale('log')
	ax1.set_xlabel('$R \; [\mathrm{M}pc]$', fontsize = 15,)

	ax1.set_ylabel('$ \\Sigma / \\Sigma_{ \\mathrm{low} } $', fontsize = 15,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax0.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/pros_ratio.png', dpi = 300)
	plt.close()


	fig = plt.figure()
	ax = fig.add_axes([0.14, 0.12, 0.80, 0.80])

	l1, = ax.plot( lo_R, lo_surf_M, ls = '-', color = 'b', alpha = 0.75, 
		label = '$\\Sigma_{\\ast}^{ \\mathrm{BCG+ICL} }[M_{\\odot}]$',)
	ax.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = 'b', alpha = 0.12,)
	l2, = ax.plot( hi_R, hi_surf_M, ls = '-', color = 'r', alpha = 0.75,)
	ax.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = 'r', alpha = 0.12,)

	af_0 = 10**lg_Ng_gi
	af_1 = 10**lg_fb_gi

	# af_0 = 2.65e4
	ax.plot( lo_n_rp, (lo_Ng - lo_Ng_2Mpc) * af_0, ls = '--', color = 'b', alpha = 0.75, 
		label = '$\\Sigma_{g} \\times 10^{%.1f} \; [M_{\\odot}]$' % lg_Ng_gi,)
	ax.plot( hi_n_rp, (hi_Ng - hi_Ng_2Mpc) * af_0, ls = '--', color = 'r', alpha = 0.75, )

	# af_1 = 2e-3
	ax.plot( lo_rp / 1e3, (lo_rho_m - lo_xi2M_2Mpc) * af_1, ls = ':', color = 'b', alpha = 0.75, 
		label = '$\\Sigma_{m}[M_{\\odot}] \\times 10^{%.1f}$' % lg_fb_gi,)
	ax.plot( hi_rp / 1e3, (hi_rho_m - hi_xi2M_2Mpc) * af_1, ls = ':', color = 'r', alpha = 0.75, )

	legend_0 = ax.legend( handles = [ l1, l2 ], labels = [ fig_name[0], fig_name[1] ], loc = 1, frameon = False, fontsize = 15,)
	ax.legend( loc = 3, frameon = False, fontsize = 15,)
	ax.add_artist( legend_0 )

	ax.set_xlim( 1e-2, 2e0 )
	ax.set_xscale('log')
	ax.set_xlabel('$R \; [\mathrm{M}pc]$')
	ax.set_yscale('log')
	ax.set_ylim( 5e3, 3e8 )
	ax.set_ylabel('$ \\Sigma \; [ kpc^{-2}]$', fontsize = 15,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/Sigma_pros_compare.png', dpi = 300)
	plt.close()


	fig = plt.figure( figsize = (15.40, 4.8) )
	ax0 = fig.add_axes([0.05, 0.12, 0.275, 0.85])
	ax1 = fig.add_axes([0.38, 0.12, 0.275, 0.85])
	ax2 = fig.add_axes([0.71, 0.12, 0.275, 0.85])

	ax0.plot( lo_R, lo_surf_M, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0],)
	ax0.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = 'b', alpha = 0.12,)
	ax0.plot( hi_R, hi_surf_M, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1],)
	ax0.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = 'r', alpha = 0.12,)

	ax0.set_xlim( 1e-2, 2e0 )
	ax0.set_xscale('log')
	ax0.set_xlabel('$R \; [\mathrm{M}pc]$')
	ax0.set_yscale('log')
	ax0.set_ylim( 5e3, 3e8 )
	ax0.legend( loc = 3, frameon = False, fontsize = 15,)
	ax0.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, kpc^{-2}]$', fontsize = 15,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	ax1.plot( lo_rp / 1e3, lo_rho_m, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0],)
	ax1.plot( hi_rp / 1e3, hi_rho_m, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1],)

	ax1.set_xlim( 1e-2, 2e0 )
	ax1.set_xscale('log')
	ax1.set_xlabel('$R \; [\mathrm{M}pc]$')
	ax1.set_yscale('log')
	ax1.set_ylim( 8e6, 2e9 )
	ax1.legend( loc = 3, frameon = False, fontsize = 15,)
	ax1.set_ylabel('$ \\Sigma_{m} \; [ M_{\\odot} \, kpc^{-2} ]$', fontsize = 15,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


	ax2.plot( lo_n_rp, lo_Ng * 1e6, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0],)
	ax2.fill_between( lo_n_rp, y1 = (lo_Ng - lo_Ng_err) * 1e6, y2 = (lo_Ng + lo_Ng_err) * 1e6, color = 'b', alpha = 0.12,)
	ax2.plot( hi_n_rp, hi_Ng * 1e6, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1],)
	ax2.fill_between( hi_n_rp, y1 = (hi_Ng - hi_Ng_err) * 1e6, y2 = (hi_Ng + hi_Ng_err) * 1e6, color = 'r', alpha = 0.12,)

	ax2.set_xlim( 1e-2, 2e0 )
	ax2.set_xscale('log')
	ax2.set_xlabel('$R \; [\mathrm{M}pc]$')
	ax2.set_yscale('log')
	ax2.set_ylim( 1e0, 3e2 )
	ax2.legend( loc = 3, frameon = False, fontsize = 15,)
	ax2.set_ylabel('$ \\Sigma_{g} \; [ Mpc^{-2} ]$', fontsize = 15,)
	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/BCG-Mstar_overview.png', dpi = 300)
	plt.close()

	return

# pre_compare_func()


N_grid = 250

##.. central BCG surface density profile
cen_dat = pds.read_csv( fit_path + 'total-sample_gri-band-based_mass-profile_cen-deV_fit.csv' )
cen_Ie, cen_Re, cen_ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]
cen_surf_M = sersic_func( lo_rp, 10**cen_Ie, cen_Re, cen_ne)

cen_aveg_sigm = aveg_sigma_func( lo_rp, cen_surf_M )
cen_deta_sigm = (cen_aveg_sigm - cen_surf_M ) * 1e-6


##.. lensing profile (measured on overall sample)
tot_sigm_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/gglensing_decals_dr8_kNN_cluster_lowhigh.cat.cov')

tot_sigma_dat = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/gglensing_decals_dr8_kNN_cluster_lowhigh.cat')
tt_Rc, tt_sigma, tt_sigma_err = tot_sigma_dat[:,0], tot_sigma_dat[:,1], tot_sigma_dat[:,2]
tt_calib_f = tot_sigma_dat[:,3]
tt_boost = tot_sigma_dat[:,4]

#. weak lensing in physical coordinate
tt_Rp = tt_Rc / (1 + z_ref) / h
tt_sigma_p, tt_sigma_perr = tt_sigma * h * (1 + z_ref)**2, tt_sigma_err * h * (1 + z_ref)**2


#. subsample case (comoving coordinate)
hi_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm.txt')
hi_obs_R, hi_obs_Detsigm = np.array( hi_obs_dat['R'] ), np.array( hi_obs_dat['delta_sigma'] )

hi_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm_covmat.txt')
hi_obs_err = np.sqrt( np.diag( hi_obs_cov ) )

lo_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm.txt')
lo_obs_R, lo_obs_Detsigm = np.array( lo_obs_dat['R'] ), np.array( lo_obs_dat['delta_sigma'] )

lo_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm_covmat.txt')
lo_obs_err = np.sqrt( np.diag( lo_obs_cov ) )

aveg_Delta_sigm = ( hi_obs_Detsigm / hi_obs_err**2 + lo_obs_Detsigm / lo_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )
aveg_delta_err = np.sqrt( lo_obs_err**2 + hi_obs_err**2 )


##.. miscen params for high mass
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24] # in unit M_sun / h
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]


##.. cross corelation function to delta_sigma (physical coordinate)
lo_aveg_xi2mis_sigma = aveg_sigma_func( lo_rp, lo_rho_m, N_grid = N_grid,)
lo_delta_xi2mis_sigma = (lo_aveg_xi2mis_sigma - lo_rho_m) * 1e-6 # M_sun / pc^2

hi_aveg_xi2mis_sigma = aveg_sigma_func( hi_rp, hi_rho_m, N_grid = N_grid,)
hi_delta_xi2mis_sigma = (hi_aveg_xi2mis_sigma - hi_rho_m) * 1e-6

xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) * h / a_ref**2 #. physical coordinate, M_sun / kpc^2
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'linear',)
xi2mis_sigma = xi_to_Mf( lo_rp )

aveg_xi2mis_sigma = aveg_sigma_func( lo_rp, xi2mis_sigma, N_grid = N_grid,)
delta_xi2mis_sigma = (aveg_xi2mis_sigma - xi2mis_sigma) * 1e-6


#. inverse var weit
_cp_lo_xi2sigma = interp.interp1d( lo_rp * h / ( a_ref * 1e3), lo_delta_xi2mis_sigma * a_ref**2 / h, kind = 'linear', fill_value = 'extrapolate',)
_cp_hi_xi2sigma = interp.interp1d( hi_rp * h / ( a_ref * 1e3), hi_delta_xi2mis_sigma * a_ref**2 / h, kind = 'linear', fill_value = 'extrapolate',)

weit_delta_sigma = ( _cp_lo_xi2sigma( lo_obs_R ) / lo_obs_err**2 + _cp_hi_xi2sigma( hi_obs_R ) / hi_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )


# #.. delta sigma (with 1-halo term only)
# mis_sigma = obs_sigma_func( lo_rp * h, np.mean(f_off), np.mean(off_set), z_ref, np.mean(c_mass), np.mean(Mh0), v_m)
# mis_sigma = mis_sigma * h # M_sun / kpc^2
# mean_mis_sigma = aveg_sigma_func( lo_rp, mis_sigma, N_grid = N_grid,)
# mis_delta_sigma = ( mean_mis_sigma - mis_sigma ) * 1e-6 # M_sun / pc^2

# lo_mis_sigma = obs_sigma_func( lo_rp * h, f_off[0], off_set[0], z_ref, c_mass[0], Mh0[0], v_m)
# lo_mis_sigma = lo_mis_sigma * h
# lo_mean_sigma = aveg_sigma_func( lo_rp, lo_mis_sigma, N_grid = N_grid,)
# lo_mis_delta_sigm = (lo_mean_sigma - lo_mis_sigma) * 1e-6

# hi_mis_sigma = obs_sigma_func( hi_rp * h, f_off[1], off_set[1], z_ref, c_mass[1], Mh0[1], v_m)
# hi_mis_sigma = hi_mis_sigma * h
# hi_mean_sigma = aveg_sigma_func( hi_rp, hi_mis_sigma, N_grid = N_grid,)
# hi_mis_delta_sigm = (hi_mean_sigma - hi_mis_sigma) * 1e-6



plt.figure()
ax = plt.subplot(111)

ax.errorbar( tt_Rc[1:], tt_sigma[1:] * tt_boost[1:], yerr = tt_sigma_err[1:], xerr = None, color = 'k', marker = 's', ls = 'none', 
	ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = 'All sample (observed)')

ax.errorbar( lo_obs_R, aveg_Delta_sigm, yerr = aveg_delta_err, xerr = None, color = 'g', marker = 's', ls = 'none', 
	ecolor = 'g', alpha = 0.75, mec = 'g', mfc = 'none', capsize = 2, label = '$1/ \\sigma$ Weighted average of subsamples')

# ax.errorbar( lo_obs_R[2:], lo_obs_Detsigm[2:], yerr = lo_obs_err[2:], xerr = None, color = 'b', marker = '.', ls = 'none', ms = 5, 
# 	ecolor = 'b', alpha = 0.75, mec = 'b', capsize = 2, label = fig_name[0],)
# ax.errorbar( hi_obs_R[2:], hi_obs_Detsigm[2:], yerr = hi_obs_err[2:], xerr = None, color = 'r', marker = '.', ls = 'none', ms = 5, 
# 	ecolor = 'r', alpha = 0.75, mec = 'r', capsize = 2, label = fig_name[1],)

# ax.plot( lo_rp * h / 1e3 / a_ref, lo_delta_xi2mis_sigma * a_ref**2 / h, 'b:', label = fig_name[0] + ', fitting',)
# ax.plot( hi_rp * h / 1e3 / a_ref, hi_delta_xi2mis_sigma * a_ref**2 / h, 'r--', label = fig_name[1] + ', fitting',)

l1, = ax.plot( lo_rp * h / ( 1e3 * a_ref), delta_xi2mis_sigma * a_ref**2 / h, 'b-', alpha = 0.75,)
l2, = ax.plot( lo_obs_R, weit_delta_sigma, 'r-', alpha = 0.75,)

legend_0 = ax.legend( handles = [ l1, l2 ], labels = [ 'Mean', 'Inverse error weighted average' ], loc = 1, frameon = False,)
ax.legend( loc = 3, frameon = False, )
ax.add_artist( legend_0 )

ax.legend( loc = 3, frameon = False, )
ax.set_xlabel('$R \; [h^{-1} \, \\mathrm{M}pc]$')
ax.set_ylabel('$\\Delta \\Sigma \; [h \, M_{\\odot} \, / \, pc^{2}]$')
ax.set_ylim( 9e-1, 5e2 )
ax.set_xlim( 1e-2, 2e1 )
ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig('/home/xkchen/shape_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax2 = plt.subplot(111)

# ax2.plot( lo_rp / 1e3, mis_delta_sigma, 'r-', alpha = 0.75,)

ax2.errorbar( tt_Rp[1:], tt_sigma_p[1:] * tt_boost[1:], yerr = tt_sigma_perr[1:], xerr = None, color = 'k', marker = 's', ls = 'none', 
	ecolor = 'k', alpha = 0.75, mec = 'k', capsize = 2, label = 'Observed')

ax2.errorbar( lo_obs_R / (1 + z_ref) / h, aveg_Delta_sigm * h / a_ref**2, yerr = aveg_delta_err * h / a_ref**2, xerr = None, color = 'g', 
	marker = 's', ls = 'none', ecolor = 'g', alpha = 0.75, mec = 'g', mfc = 'none', capsize = 2, label = '$1/ \\sigma$ Weighted average of subsamples',)

l1, = ax2.plot( lo_rp / 1e3, delta_xi2mis_sigma, 'b-', alpha = 0.75, label = 'Mean of fitting on subsamples',)
# l2, = ax2.plot( lo_obs_R * a_ref / h, weit_delta_sigma * h / a_ref**2, 'r-', alpha = 0.75,)

# legend_0 = ax2.legend( handles = [ l1, l2 ], labels = [ 'Mean', 'Inverse error weighted average' ], loc = 1, frameon = False,)
# ax2.legend( loc = 3, frameon = False, fontsize = 13, )
# ax2.add_artist( legend_0 )

ax2.legend( loc = 3, fontsize = 13, )

ax2.set_xlim( 3e-3, 2e0)
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax2.set_yscale('log')

ax2.set_ylim( 2e1, 4e2)
ax2.set_ylabel('$\\Delta \\Sigma \; [M_{\\odot} \, / \, pc^2]$', fontsize = 15,)

ax2.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'])
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/weak_lensing_compare.png', dpi = 300)
plt.close()

