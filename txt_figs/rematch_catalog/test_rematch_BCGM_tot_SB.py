import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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
from scipy.interpolate import splev, splrep

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func, sub_color_func, fit_surfM_func
from fig_out_module import color_func

from color_2_mass import jk_sub_SB_func
from color_2_mass import get_c2mass_func

from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

band = ['r', 'g', 'i']

psf_FWHM = 1.32 # arcsec

Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === 
def sersic_func(r, Ie, re, ndex):

	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )

	return Ir

### === ### data load
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

BG_path = '/home/xkchen/figs/extend_bcgM_cat/BGs/'
path = '/home/xkchen/figs/extend_bcgM_cat/SBs/'


### === ### BG estimate and BG-subtraction (based on diag-err only)

"""
for kk in range( 1,2 ):

	with h5py.File( 
		path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

	p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
	bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

	R_psf = 10

	lo_R_lim = 500

	hi_R_lim = 1.4e3
	trunk_R = 2e3

	out_params_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]
	out_pros_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_diag-fit.csv' % band[kk]

	clust_SB_fit_func(
		tt_r, tt_sb, tt_err, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, out_params_file, out_pros_file, trunk_R = trunk_R,)

	## fig
	p_dat = pds.read_csv( out_params_file )
	( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD, I_e, R_e) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0],
															np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0], 
															np.array(p_dat['offD'])[0], np.array(p_dat['I_e'])[0], np.array(p_dat['R_e'])[0] )

	fit_rnd_sb = cc_rand_sb_func( tt_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)  
	sign_fit = sersic_func( tt_r, I_e, R_e, 2.1)
	BG_pros = fit_rnd_sb - offD
	comb_F = BG_pros + sign_fit

	sb_2Mpc = sersic_func( trunk_R, I_e, R_e, 2.1)
	norm_sign = sign_fit - sb_2Mpc
	norm_BG = comb_F - norm_sign

	c_dat = pds.read_csv( out_pros_file )
	chi_ov_nu = np.array( c_dat['chi2nu'] )[0]
	chi_inner_m = np.array( c_dat['chi2nu_inner'] )[0]

	plt.figure()
	ax = plt.subplot(111)

	ax.plot( tt_r, tt_sb, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
	ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.12)

	ax.plot(tt_r, comb_F, ls = '-', color = 'k', alpha = 0.5, label = 'Best fitting',)
	ax.plot(tt_r, norm_sign, ls = '-.', color = 'k', alpha = 0.5, label = 'signal (model)',)
	ax.plot(tt_r, norm_BG, ls = '--', color = 'k', alpha = 0.5, label = 'BackGround')

	ax.axvline(x = lo_R_lim, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

	ax.annotate(text = '$\\chi^2 / \\nu = %.5f$' % chi_ov_nu, xy = (0.60, 0.60), xycoords = 'axes fraction', color = 'k',)

	ax.set_xlim(1e2, 4e3)
	ax.set_xscale('log')

	if kk == 1:
		ax.set_ylim( 2e-3, 5e-3)
	else:
		ax.set_ylim(2e-3, 7e-3)

	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / arcsec^2]')
	ax.legend( loc = 1,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.subplots_adjust(left = 0.15, right = 0.9,)
	plt.savefig('/home/xkchen/tot-BCG-star-Mass_%s-band_SB_n=2.1-sersic.png' % band[kk], dpi = 300)
	plt.close()

N_bin = 30

for kk in range( 3 ):
	jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_%s-band_' % band[kk] + 'jack-sub-%d_SB-pro_z-ref.h5'

	sb_out_put = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk]
	BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]
	BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)

"""

def color_profile():

	N_bin = 30

	#... color of jack-sub sample and mean of jack average
	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_%s-band_' % band[kk] + 'jack-sub-%d_SB-pro_z-ref.h5'
		BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]
		out_sub_sb = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_' % band[kk] + 'jack-sub-%d_BG-sub_SB.csv'

		jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, out_sub_sb )

	sub_r_file = BG_path + 'photo-z_tot-BCG-star-Mass_r-band_jack-sub-%d_BG-sub_SB.csv'
	sub_g_file = BG_path + 'photo-z_tot-BCG-star-Mass_g-band_jack-sub-%d_BG-sub_SB.csv'
	sub_i_file = BG_path + 'photo-z_tot-BCG-star-Mass_i-band_jack-sub-%d_BG-sub_SB.csv'

	sub_c_file = BG_path + 'tot-BCG-star-Mass_jack-sub-%d_color_profile.csv'
	aveg_c_file = BG_path + 'tot-BCG-star-Mass_color_profile.csv'

	sub_color_func( N_bin, sub_r_file, sub_g_file, sub_i_file, sub_c_file, aveg_c_file )


	#... color profiles with extinction correction
	p_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'high_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	AL_r0, AL_g0, AL_i0 = np.array( p_dat['Al_r'] ), np.array( p_dat['Al_g'] ), np.array( p_dat['Al_i'] )

	p_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'low_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	AL_r1, AL_g1, AL_i1 = np.array( p_dat['Al_r'] ), np.array( p_dat['Al_g'] ), np.array( p_dat['Al_i'] )

	AL_r = np.r_[ AL_r0, AL_r1 ]
	AL_g = np.r_[ AL_g0, AL_g1 ]
	AL_i = np.r_[ AL_i0, AL_i1 ]

	AL_arr = [ AL_r, AL_g, AL_i ]

	x_dered = True

	sub_r_file = BG_path + 'photo-z_tot-BCG-star-Mass_r-band_' + 'jack-sub-%d_BG-sub_SB.csv'
	sub_g_file = BG_path + 'photo-z_tot-BCG-star-Mass_g-band_' + 'jack-sub-%d_BG-sub_SB.csv'
	sub_i_file = BG_path + 'photo-z_tot-BCG-star-Mass_i-band_' + 'jack-sub-%d_BG-sub_SB.csv'

	sub_c_file = BG_path + 'photo-z_tot-BCG-star-Mass_%d_dered_color_profile.csv'
	aveg_c_file = BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv'

	sub_color_func( N_bin, sub_r_file, sub_g_file, sub_i_file, sub_c_file, aveg_c_file, id_dered = x_dered, Al_arr = AL_arr )

# color_profile()


def SM_profile():

	out_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
	band_str = 'gri'
	fit_file = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/least-square_M-to-i-band-Lumi&color.csv'

	N_samples = 30
	low_R_lim, up_R_lim = 1e0, 1.2e3

	##.. SM(r)
	sub_sb_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_jack-sub-%d_BG-sub_SB.csv'
	sub_sm_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv'

	aveg_jk_sm_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str
	lgM_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str
	M_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % band_str

	fit_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_ref, z_ref,
					aveg_jk_sm_file, lgM_cov_file, fit_file, M_cov_file = M_cov_file )


	##. SM(r) with deredden color profiles
	## lgM = a(g-r) + b(r-i) + c*lg_Li + d
	pfit_dat = pds.read_csv( fit_file )
	a_fit = np.array( pfit_dat['a'] )[0]
	b_fit = np.array( pfit_dat['b'] )[0]
	c_fit = np.array( pfit_dat['c'] )[0]
	d_fit = np.array( pfit_dat['d'] )[0]

	#.
	p_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'high_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	AL_r0, AL_g0, AL_i0 = np.array( p_dat['Al_r'] ), np.array( p_dat['Al_g'] ), np.array( p_dat['Al_i'] )

	p_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'low_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	AL_r1, AL_g1, AL_i1 = np.array( p_dat['Al_r'] ), np.array( p_dat['Al_g'] ), np.array( p_dat['Al_i'] )

	AL_r = np.r_[ AL_r0, AL_r1 ]
	AL_g = np.r_[ AL_g0, AL_g1 ]
	AL_i = np.r_[ AL_i0, AL_i1 ]

	mA_g = np.median( AL_g )
	mA_r = np.median( AL_r )
	mA_i = np.median( AL_i )

	tmp_r, tmp_SM, tmp_lgSM, tmp_Li = [], [], [], []

	for nn in range( N_samples ):

		_sub_dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv' % nn)
		_nn_R, _nn_surf_M, _nn_Li = np.array( _sub_dat['R'] ), np.array( _sub_dat['surf_mass'] ), np.array( _sub_dat['lumi'] )

		mf0 = a_fit * ( mA_r - mA_g )
		mf1 = b_fit * ( mA_i - mA_r )
		mf2 = c_fit * 0.4 * mA_i

		modi_surf_M = _nn_surf_M * 10**( mf0 + mf1 + mf2 )
		modi_Li = _nn_Li * 10** mf2

		keys = ['R', 'surf_mass', 'lumi']
		values = [ _nn_R, modi_surf_M, modi_Li ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi_with-dered.csv' % nn )

		tmp_r.append( _nn_R )
		tmp_SM.append( modi_surf_M )
		tmp_lgSM.append( np.log10( modi_surf_M ) )
		tmp_Li.append( modi_Li )

	aveg_R_0, aveg_surf_m, aveg_surf_m_err = arr_jack_func( tmp_SM, tmp_r, N_samples )[:3]
	aveg_R_1, aveg_lumi, aveg_lumi_err = arr_jack_func( tmp_Li, tmp_r, N_samples )[:3]
	aveg_R_2, aveg_lgM, aveg_lgM_err = arr_jack_func( tmp_lgSM, tmp_r, N_samples )[:3]

	keys = ['R', 'surf_mass', 'surf_mass_err', 'lumi', 'lumi_err', 'lg_M', 'lg_M_err']
	values = [ aveg_R_0, aveg_surf_m, aveg_surf_m_err, aveg_lumi, aveg_lumi_err, aveg_lgM, aveg_lgM_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str )

	#. covmatrix
	R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_lgSM, id_jack = True)
	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5' % band_str, 'w') as f:
		f['R_kpc'] = np.array( R_mean )
		f['cov_MX'] = np.array( cov_MX )
		f['cor_MX'] = np.array( cor_MX )

	R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_SM, id_jack = True)
	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % band_str, 'w') as f:
		f['R_kpc'] = np.array( R_mean )
		f['cov_MX'] = np.array( cov_MX )
		f['cor_MX'] = np.array( cor_MX )


	dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
	aveg_R = np.array(dat['R'])
	aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

	dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
	cc_aveg_R = np.array(dat['R'])
	cc_aveg_surf_m, cc_aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


	fig = plt.figure()
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax.plot( aveg_R, aveg_surf_m, ls = '-', color = 'r', alpha = 0.5, label = 'w/o deredden')
	ax.fill_between( aveg_R, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'r', alpha = 0.12,)

	ax.plot( cc_aveg_R, cc_aveg_surf_m, ls = '--', color = 'r', alpha = 0.5, label = 'deredden')
	ax.fill_between( cc_aveg_R, y1 = cc_aveg_surf_m - cc_aveg_surf_m_err, 
					y2 = cc_aveg_surf_m + cc_aveg_surf_m_err, color = 'r', alpha = 0.12,)

	ax.legend( loc = 3, frameon = False,)
	ax.set_ylim( 6e3, 3e9 )
	ax.set_yscale( 'log' )
	ax.set_xlim( 1e0, 1e3 )
	ax.set_xscale('log')

	ax.set_ylabel('$M_{\\ast} [M_{\\odot}]$', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax.plot( aveg_R, aveg_surf_m / cc_aveg_surf_m, ls = '--', color = 'r', alpha = 0.75,)

	sub_ax.set_ylim( 0.985, 1.015 )
	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xscale('log')

	sub_ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
	sub_ax.set_xlabel('R [kpc]', fontsize = 12,)
	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/tot_%s-band_based_surface_mass_profile.png' % band_str, dpi = 300)
	plt.close()

	return

# SM_profile()
raise


### === figs
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3

#. SB profiles
nbg_tot_r = []
nbg_tot_mag, nbg_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_tot_r.append( tt_r )
	nbg_tot_mag.append( tt_mag )
	nbg_tot_mag_err.append( tt_mag_err )

#. SM
dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
					'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )
obs_R, obs_SM, obs_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

#. color
c_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv' )
dered_R, dered_g2r, dered_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
sm_dered_g2r = signal.savgol_filter( dered_g2r, 7, 3)
dered_R = dered_R / 1e3

##... Z05, color
pdat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_g-r.csv')
r_05_0, g2r_05 = np.array(pdat_0['(R)^(1/4)']), np.array(pdat_0['g-r'])
r_05_0 = r_05_0**4



##... previous sample
pre_BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'

pre_tot_r = []
pre_tot_mag, pre_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( pre_BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	pre_tot_r.append( tt_r )
	pre_tot_mag.append( tt_mag )
	pre_tot_mag_err.append( tt_mag_err )

#. color
c_dat = pds.read_csv( pre_BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv' )
pre_dered_R, pre_dered_g2r, pre_dered_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
sm_pre_dered_g2r = signal.savgol_filter( pre_dered_g2r, 7, 3)
pre_dered_R = pre_dered_R / 1e3

#. 
p_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
					  'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv',)
p_obs_R, p_obs_SM, p_obs_SM_err = np.array(p_dat['R']), np.array(p_dat['surf_mass']), np.array(p_dat['surf_mass_err'])


plt.figure()
ax = plt.subplot( 111 )

ax.plot( p_obs_R, p_obs_SM, ls = '--', color = 'b', alpha = 0.5, label = 'Previous',)
ax.fill_between( p_obs_R, y1 = p_obs_SM - p_obs_SM_err, y2 = p_obs_SM + p_obs_SM_err, color = 'b', alpha = 0.12,)

ax.plot( obs_R, obs_SM, ls = '-', color = 'r', alpha = 0.5, label = 'Now',)
ax.fill_between( obs_R, y1 = obs_SM - obs_SM_err, y2 = obs_SM + obs_SM_err, color = 'r', alpha = 0.12,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 6e3, 3e9 )
ax.set_yscale( 'log' )
ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('$M_{\\ast} \; [M_{\\odot}]$', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/all_sample_SM_compare.png', dpi = 300)
plt.close()


raise


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( r_05_0 / 1e3, g2r_05, ls = ':', color = 'k', label = 'Z05')

ax.plot( dered_R, sm_dered_g2r, ls = '-', color = 'r', alpha = 0.75, label = 'Now')
ax.fill_between( dered_R, y1 = sm_dered_g2r - dered_g2r_err, y2 = sm_dered_g2r + dered_g2r_err, color = 'r', alpha = 0.12,)

ax.plot( pre_dered_R, sm_pre_dered_g2r, ls = '--', color = 'r', alpha = 0.75, label = 'Previous')
ax.fill_between( pre_dered_R, y1 = sm_pre_dered_g2r - pre_dered_g2r_err, 
				y2 = sm_pre_dered_g2r + pre_dered_g2r_err, color = 'r', ls = '--', alpha = 0.12,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 0.95, 1.55 )

ax.set_xlim(3e-3, 1e0)
ax.set_xscale('log')
ax.set_ylabel('g - r', fontsize = 12,)
ax.set_xlabel('R [Mpc]', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/total-sample_g2r_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

for kk in range( 3 ):

	l1, = ax.plot( nbg_tot_r[kk], nbg_tot_mag[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax.fill_between( nbg_tot_r[kk], y1 = nbg_tot_mag[kk] - nbg_tot_mag_err[kk], 
					y2 = nbg_tot_mag[kk] + nbg_tot_mag_err[kk], color = color_s[kk], alpha = 0.15,)

	l2, = ax.plot( pre_tot_r[kk], pre_tot_mag[kk], ls = '--', color = color_s[kk], alpha = 0.75,)

ax.axvline( x = phyR_psf, ls = ':', color = 'k', alpha = 0.5, ymin = 0.7, ymax = 1.0, linewidth = 1.5,)

legend_2 = plt.legend( handles = [ l1, l2 ], labels = ['Now', 'Previous'], loc = 3, frameon = False, fontsize = 15,)
legend_20 = ax.legend( loc = 1, frameon = False, fontsize = 15,)
ax.add_artist( legend_2 )

ax.set_xlim( 1e0, 1e3)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]', fontsize = 15,)

ax.set_ylim( 20, 34 )
ax.invert_yaxis()
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/total_sample_SB_compare.png', dpi = 300)
plt.close()


