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
from scipy import interpolate as interp
from scipy import integrate as integ

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.signal as signal
from scipy.interpolate import splev, splrep

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func, sub_color_func
from img_BG_sub_SB_measure import fit_surfM_func
from color_2_mass import jk_sub_SB_func

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
Da_ref = Test_model.angular_diameter_distance( z_ref ).value

psf_FWHM = 1.32 # arcsec

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === ### funcs
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

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === ### data load
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

cat_lis = ['low-lgM20', 'hi-lgM20']
fig_name = ['Low $ M_{\\ast, \, 20}$', 'High $ M_{\\ast, \, 20}$']
file_s = 'lgM20-binned'

# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'BCG_Mstar_bin'


BG_path = '/home/xkchen/figs/Pcen_cut/BGs/'
path = '/home/xkchen/figs/Pcen_cut/SBs/'

### === ### BG-sub SB and color
"""
#. Background estimate
for mm in range( 2 ):

	for kk in range( 3 ):

		with h5py.File( path + 'photo-z_match_gri-common_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

		p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
		bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

		R_psf = 10

		lo_R_lim = 500 # 450, 500, 400

		hi_R_lim = 1.4e3
		trunk_R = 2e3

		out_params_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
		out_pros_file = BG_path + 'photo-z_%s_%s-band_BG-profile_diag-fit.csv' % (cat_lis[ mm ], band[kk])

		clust_SB_fit_func( tt_r, tt_sb, tt_err, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, out_params_file, 
							out_pros_file, trunk_R = trunk_R,)

		## fig
		p_dat = pds.read_csv( out_params_file )
		( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD, I_e, R_e) = ( 
																np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], 
																np.array(p_dat['e_x0'])[0], np.array(p_dat['e_A'])[0], 
																np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0], 
																np.array(p_dat['offD'])[0], np.array(p_dat['I_e'])[0], 
																np.array(p_dat['R_e'])[0] )

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
		ax.set_title( fig_name[ mm ] + ', %s band' % band[kk] )

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
		plt.savefig('/home/xkchen/%s_%s-band_SB_n=2.1-sersic.png' % (cat_lis[ mm ], band[kk]), dpi = 300)
		plt.close()

raise
"""

"""
N_bin = 30

#. BG-sub SB profiles
for mm in range( 2 ):

	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % (cat_lis[ mm ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref.h5'

		sb_out_put = BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[ mm ], band[kk])
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
		BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)

#. color profiles
for mm in range( 2 ):

	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % (cat_lis[ mm ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref.h5'
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
		out_sub_sb = BG_path + '%s_%s-band_' % (cat_lis[mm], band[kk]) + 'jack-sub-%d_BG-sub_SB.csv'

		jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, out_sub_sb )

for mm in range( 2 ):

	sub_r_file = BG_path + '%s_r-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv'
	sub_g_file = BG_path + '%s_g-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv'
	sub_i_file = BG_path + '%s_i-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv'

	sub_c_file = BG_path + '%s_jack-sub-' % cat_lis[mm] + '%d_color_profile.csv'
	aveg_c_file = BG_path + '%s_color_profile.csv' % cat_lis[mm]

	sub_color_func( N_bin, sub_r_file, sub_g_file, sub_i_file, sub_c_file, aveg_c_file )
"""


### === ### surface mass profile and mass ratio relative to the total cluster sample
def sub_sample_SM():

	#. surface mass profile
	low_R_lim, up_R_lim = 1e0, 1.2e3

	N_samples = 30

	band_str = 'gri'
	fit_file = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv'
	out_path = '/home/xkchen/figs/Pcen_cut/surf_M/'

	for mm in range( 2 ):

		sub_sb_file = BG_path + '%s_' % cat_lis[mm] + '%s-band_jack-sub-%d_BG-sub_SB.csv'
		sub_sm_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'

		aveg_jk_sm_file = out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str)
		lgM_cov_file = out_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)
		M_cov_file = out_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)

		fit_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_ref, z_ref,
						aveg_jk_sm_file, lgM_cov_file, fit_file, M_cov_file = M_cov_file )	

	#. surface mass profile with deredden color profile
	## lgM = a(g-r) + b(r-i) + c*lg_Li + d
	pfit_dat = pds.read_csv( fit_file )
	a_fit = np.array( pfit_dat['a'] )[0]
	b_fit = np.array( pfit_dat['b'] )[0]
	c_fit = np.array( pfit_dat['c'] )[0]
	d_fit = np.array( pfit_dat['d'] )[0]

	for mm in range( 2 ):

		E_dat = pds.read_csv( '/home/xkchen/figs/Pcen_cut/cat/' + '%s_photo-z-match_Pcen-lim_cat_dust-value.csv' % cat_lis[mm],)
		AL_r, AL_g, AL_i = np.array( E_dat['Al_r'] ), np.array( E_dat['Al_g'] ), np.array( E_dat['Al_i'] )

		mA_r = np.median( AL_r )
		mA_g = np.median( AL_g )
		mA_i = np.median( AL_i )

		tmp_r = []
		tmp_SM = []
		tmp_lgSM = []
		tmp_Li = []

		for nn in range( N_samples ):

			_sub_dat = pds.read_csv( out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv' % nn)
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
			out_data.to_csv( out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi_with-dered.csv' % nn )

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
		out_data.to_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str) )

		#. covmatrix
		R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_lgSM, id_jack = True)
		with h5py.File( out_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'w') as f:
			f['R_kpc'] = np.array( R_mean )
			f['cov_MX'] = np.array( cov_MX )
			f['cor_MX'] = np.array( cor_MX )	

		R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_SM, id_jack = True)
		with h5py.File( out_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'w') as f:
			f['R_kpc'] = np.array( R_mean )
			f['cov_MX'] = np.array( cov_MX )
			f['cor_MX'] = np.array( cor_MX )

	##... mass profile compare
	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
	lo_m_R, lo_surf_M, lo_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )

	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
	hi_m_R, hi_surf_M, hi_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )


	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' % cat_lis[0] )
	cc_lo_m_R, cc_lo_surf_M, cc_lo_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )

	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' % cat_lis[1] )
	cc_hi_m_R, cc_hi_surf_M, cc_hi_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )


	fig = plt.figure()
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax.plot( hi_m_R, hi_surf_M, ls = '-', color = 'r', alpha = 0.75, linewidth = 1, label = fig_name[1])
	ax.fill_between( hi_m_R, y1 = hi_surf_M - hi_SM_err, y2 = hi_surf_M + hi_SM_err, color = 'r', alpha = 0.15,)

	ax.plot( lo_m_R, lo_surf_M, ls = '-', color = 'b', alpha = 0.75, linewidth = 1, label = fig_name[0])
	ax.fill_between( lo_m_R, y1 = lo_surf_M - lo_SM_err, y2 = lo_surf_M + lo_SM_err, color = 'b', alpha = 0.15,)


	ax.plot( cc_hi_m_R, cc_hi_surf_M, ls = '--', color = 'r', alpha = 0.45, linewidth = 3, label = fig_name[1] + ', deredden')
	ax.fill_between( cc_hi_m_R, y1 = cc_hi_surf_M - cc_hi_SM_err, y2 = cc_hi_surf_M + cc_hi_SM_err, color = 'r', alpha = 0.15,)

	ax.plot( cc_lo_m_R, cc_lo_surf_M, ls = '--', color = 'b', alpha = 0.45, linewidth = 3, label = fig_name[0] + ', deredden')
	ax.fill_between( cc_lo_m_R, y1 = cc_lo_surf_M - cc_lo_SM_err, y2 = cc_lo_surf_M + cc_lo_SM_err, color = 'b', alpha = 0.15,)

	ax.legend( loc = 3, frameon = False,)
	ax.set_ylim( 6e3, 3e9 )
	ax.set_yscale( 'log' )
	ax.set_xlim( 1e0, 1e3 )
	ax.set_xscale('log')

	ax.set_ylabel('$M_{\\ast} [M_{\\odot}]$', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax.plot( hi_m_R, cc_hi_surf_M / hi_surf_M, ls = '--', color = 'r', alpha = 0.75,)
	sub_ax.plot( lo_m_R, cc_lo_surf_M / lo_surf_M, ls = '--', color = 'b', alpha = 0.75,)

	sub_ax.set_ylim( 0.99, 1.02 )
	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xscale('log')

	sub_ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
	sub_ax.set_xlabel('R [kpc]', fontsize = 12,)
	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_Pcen-lim_surf-M_compare.png' % file_s, dpi = 300)
	plt.close()

	return

# sub_sample_SM()


def sub_SM_ratio():

	out_path = '/home/xkchen/figs/Pcen_cut/surf_M/'
	cat_path = '/home/xkchen/figs/Pcen_cut/cat/'

	band_str = 'gri'
	N_samples = 30

	#. mass estimation with deredden correction or not
	# id_dered = False
	# dered_str = ''

	id_dered = True
	dered_str = '_with-dered'

	fix_R_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
								'total-sample_%s-band_based_BCGM-match_R%s.csv' % (band_str, dered_str),)

	R_fixed_M = np.array( fix_R_dat[ 'R_fixed_medi_M' ] )[0]

	cali_factor = np.array( [] )

	for mm in range( 2 ):

		#... mass profile
		m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi%s.csv' % (cat_lis[mm], band_str, dered_str),)
		jk_R = np.array(m_dat['R'])
		surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
		surf_L = np.array( m_dat['lumi'] )

		N_grid = 250
		up_lim_R = R_fixed_M

		cumu_M = cumu_mass_func( jk_R, surf_m, N_grid = N_grid )
		intep_Mf = interp.interp1d( jk_R, cumu_M, kind = 'cubic',)

		M_c0 = intep_Mf( up_lim_R )

		#...sample properties
		p_dat = pds.read_csv( cat_path + '%s_gri-common_P-cen_lim_params.csv' % cat_lis[mm] )
		p_lgM = np.array( p_dat['lg_Mbcg'] ) # Mass unit : M_sun

		lg_Mean = np.log10( np.mean( 10**p_lgM ) )
		lg_Medi = np.log10( np.median( 10**p_lgM ) )

		devi_Mean = lg_Mean - np.log10( M_c0 )
		devi_Medi = lg_Medi - np.log10( M_c0 )

		medi_off_surf_M = surf_m * 10**( devi_Medi )
		mean_off_surf_M = surf_m * 10**( devi_Mean )

		#. save the calibrated SM profiles
		keys = ['R', 'medi_correct_surf_M', 'mean_correct_surf_M', 'surf_M_err']
		values = [ jk_R, medi_off_surf_M, mean_off_surf_M, surf_m_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[mm], band_str, dered_str),)

		cali_factor = np.r_[ cali_factor, [ devi_Medi, devi_Mean ] ]

	#.
	keys = ['low_medi_devi', 'low_mean_devi', 'high_medi_devi', 'high_mean_devi']
	values = list( cali_factor )
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv(out_path + '%s_%s-band-based_M_calib-f%s.csv' % (file_s, band_str, dered_str),)


	#. relative ratio to the overall cluster sample
	dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi%s.csv' % dered_str)

	tot_R = np.array(dat['R'])
	tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

	interp_M_f = interp.interp1d( tot_R, tot_surf_m, kind = 'linear',)


	calib_cat = pds.read_csv( out_path + '%s_gri-band-based_M_calib-f%s.csv' % (file_s, dered_str),)

	lo_shift, hi_shift = np.array(calib_cat['low_medi_devi'])[0], np.array(calib_cat['high_medi_devi'])[0]
	M_offset = [ lo_shift, hi_shift ]

	#. ratio of mass profile with fixed-R mass correction
	for mm in range( 2 ):

		jk_sub_m_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi' + '%s.csv' % dered_str

		tmp_r, tmp_ratio = [], []

		for nn in range( N_samples ):

			o_dat = pds.read_csv( jk_sub_m_file % nn,)

			tt_r = np.array( o_dat['R'] )
			tt_M = np.array( o_dat['surf_mass'] )

			tt_M = tt_M * 10**M_offset[mm]

			idx_lim = ( tt_r >= np.nanmin( tot_R ) ) & ( tt_r <= np.nanmax( tot_R ) )

			lim_R = tt_r[ idx_lim ]
			lim_M = tt_M[ idx_lim ]

			com_M = interp_M_f( lim_R )

			sub_ratio = np.zeros( len(tt_r),)
			sub_ratio[ idx_lim ] = lim_M / com_M

			sub_ratio[ idx_lim == False ] = np.nan

			tmp_r.append( tt_r )
			tmp_ratio.append( sub_ratio )

		aveg_R, aveg_ratio, aveg_ratio_err = arr_jack_func( tmp_ratio, tmp_r, N_samples)[:3]

		keys = ['R', 'M/M_tot', 'M/M_tot-err']
		values = [ aveg_R, aveg_ratio, aveg_ratio_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[mm], band_str, dered_str),)


	###... mass profile compare to all clusters
	dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[0],dered_str),)
	lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[1], dered_str),)
	hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )


	lo_eat_dat = pds.read_csv( out_path + '%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], dered_str),)
	lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['M/M_tot']), np.array(lo_eat_dat['M/M_tot-err'])

	hi_eat_dat = pds.read_csv( out_path + '%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], dered_str),)
	hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['M/M_tot']), np.array(hi_eat_dat['M/M_tot-err'])


	fig = plt.figure()
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax = fig.add_axes( [0.13, 0.32, 0.83, 0.63] )
	sub_ax = fig.add_axes( [0.13, 0.11, 0.83, 0.21] )

	ax.plot( tot_R, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
	ax.fill_between( tot_R, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.15,)

	ax.plot( hi_R, hi_surf_M, ls = '-', color = 'r', alpha = 0.75, linewidth = 1, label = fig_name[1])
	ax.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = 'r', alpha = 0.15,)

	ax.plot( lo_R, lo_surf_M, ls = '-', color = 'b', alpha = 0.75, linewidth = 1, label = fig_name[0])
	ax.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = 'b', alpha = 0.15,)

	ax.legend( loc = 3, frameon = False,)
	ax.set_ylim( 6e3, 2e8 )
	ax.set_yscale( 'log' )
	ax.set_xlim( 1e1, 1e3 )
	ax.set_xscale('log')

	ax.set_ylabel('$M_{\\ast} [M_{\\odot}]$', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax.plot( lo_eta_R, lo_eta, ls = '--', color = line_c[0], alpha = 0.75,)
	sub_ax.fill_between( lo_eta_R, y1 = lo_eta - lo_eta_err, y2 = lo_eta + lo_eta_err, color = line_c[0], alpha = 0.15,)
	sub_ax.plot( hi_eta_R, hi_eta, ls = '-', color = line_c[1], alpha = 0.75,)
	sub_ax.fill_between( hi_eta_R, y1 = hi_eta - hi_eta_err, y2 = hi_eta + hi_eta_err, color = line_c[1], alpha = 0.15,)
	sub_ax.plot( tot_R, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)

	sub_ax.set_ylim( 0.45, 1.65 )
	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xscale('log')

	sub_ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
	sub_ax.set_xlabel('R [kpc]', fontsize = 12,)
	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_surf-M_ratio_compare%s_Pcen-lim.png' % (file_s, dered_str), dpi = 300)
	plt.close()

	return

sub_SM_ratio()


raise

### === ### figs
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3

#... surface brightness profile compare
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []
for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( tt_mag )
	nbg_low_err.append( tt_mag_err )

nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( tt_mag )
	nbg_hi_err.append( tt_mag_err )


#. subsamples (without P_cen cut)
if file_s == 'lgM20-binned':
	cc_path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/BGs/'
	cc_fig_name = ['Low $ M_{\\ast, \, 20}$', 'High $ M_{\\ast, \, 20}$']

if file_s == 'BCG_Mstar_bin':
	cc_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
	cc_fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']


cc_low_r, cc_low_sb, cc_low_err = [], [], []
for kk in range( 3 ):
	with h5py.File( cc_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	cc_low_r.append( tt_r )
	cc_low_sb.append( tt_mag )
	cc_low_err.append( tt_mag_err )

cc_hi_r, cc_hi_sb, cc_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( cc_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	cc_hi_r.append( tt_r )
	cc_hi_sb.append( tt_mag )
	cc_hi_err.append( tt_mag_err )


#... color profile comparison
mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
hi_gr = signal.savgol_filter( hi_gr, 7, 3)

mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
lo_gr = signal.savgol_filter( lo_gr, 7, 3)

mu_dat = pds.read_csv( cc_path + '%s_color_profile.csv' % cat_lis[0] )
cc_hi_c_r, cc_hi_gr, cc_hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
cc_hi_gr = signal.savgol_filter( cc_hi_gr, 7, 3)

mu_dat = pds.read_csv( cc_path + '%s_color_profile.csv' % cat_lis[1] )
cc_lo_c_r, cc_lo_gr, cc_lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
cc_lo_gr = signal.savgol_filter( cc_lo_gr, 7, 3)


fig = plt.figure( figsize = (5.4, 5.4) )
ax0 = fig.add_axes( [0.12, 0.12, 0.83, 0.83] )

for kk in range( 3 ):

	ax0.plot( cc_low_r[kk] / 1e3, cc_low_sb[kk] - 3.5, ls = '--', color = color_s[kk], alpha = 0.75, linewidth = 2,)
	ax0.fill_between( cc_low_r[kk] / 1e3, y1 = cc_low_sb[kk] - 3.5 - cc_low_err[kk], 
		y2 = cc_low_sb[kk] - 3.5 + cc_low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( cc_hi_r[kk] / 1e3, cc_hi_sb[kk] - 3.5, ls = '-', color = color_s[kk], alpha = 0.75, linewidth = 2,)
	ax0.fill_between( cc_hi_r[kk] / 1e3, y1 = cc_hi_sb[kk] - 3.5 - cc_hi_err[kk], 
		y2 = cc_hi_sb[kk] - 3.5 + cc_hi_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( nbg_low_r[kk] / 1e3, nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75, linewidth = 1,)
	ax0.fill_between( nbg_low_r[kk] / 1e3, y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
		y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.15, )

	ax0.plot( nbg_hi_r[kk] / 1e3, nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, linewidth = 1,
		label = '%s band' % band[kk],)
	ax0.fill_between( nbg_hi_r[kk] / 1e3, y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
		y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.15,)

legend_1 = ax0.legend( [ cc_fig_name[0] + ' w/o $P_{cen}$ cut', cc_fig_name[1] + ' w/o $P_{cen}$ cut', 
						fig_name[0] + ' with $P_{cen}$ cut', fig_name[1] + ' with $P_{cen}$ cut'], 
						loc = 3, frameon = False, fontsize = 13,)
legend_0 = ax0.legend( loc = 1, frameon = False, fontsize = 13,)
ax0.add_artist( legend_1 )

ax0.fill_betweenx( y = np.linspace( 10, 36, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax0.text( 3e-3, 27, s = 'PSF', fontsize = 16,)
ax0.axvline( x = 0.3, ls = ':', color = 'k', alpha = 0.75, ymin = 0.0, ymax = 0.65)

ax0.set_xlim( 3e-3, 1e0)
ax0.set_ylim( 17, 33.5 )
ax0.invert_yaxis()

ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
ax0.set_xticks([ 1e-2, 1e-1, 1e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax0.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/%s_SB_compare.png' % file_s, dpi = 300)
plt.close()


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( hi_c_r, hi_gr, ls = '-', color = 'r', alpha = 0.75, linewidth = 1, label = fig_name[0] + ' with $P_{cen}$ cut')
ax.fill_between( hi_c_r, y1 = hi_gr - hi_gr_err, y2 = hi_gr + hi_gr_err, color = 'r', alpha = 0.15,)

ax.plot( lo_c_r, lo_gr, ls = '-', color = 'b', alpha = 0.75, linewidth = 1, label = fig_name[1] + ' with $P_{cen}$ cut')
ax.fill_between( lo_c_r, y1 = lo_gr - lo_gr_err, y2 = lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)

ax.plot( cc_lo_c_r, cc_lo_gr, ls = ':', color = 'k', alpha = 0.55, label = cc_fig_name[0] + ' w/o $P_{cen}$ cut',)
ax.plot( cc_hi_c_r, cc_hi_gr, ls = '-.', color = 'k', alpha = 0.55, label = cc_fig_name[1] + ' w/o $P_{cen}$ cut',)

legend_1 = ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 0.5, 1.7 )

ax.set_xlim(1e0, 1e3)
ax.set_xscale('log')
ax.set_ylabel('g - r', fontsize = 12,)
ax.set_xlabel('R [kpc]', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/%s_g2r_compare.png' % file_s, dpi = 300)
plt.close()


#... surface mass profile compare
out_path = '/home/xkchen/figs/Pcen_cut/surf_M/'

if file_s == 'lgM20-binned':
	cc_SM_path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/surface_M/'
if file_s == 'BCG_Mstar_bin':
	cc_SM_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'


#. surface mass profile
surf_M_dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0],)
lo_sm_R = np.array( surf_M_dat['R'] )
lo_sm_M, lo_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )

surf_M_dat = pds.read_csv(out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1],)
hi_sm_R = np.array( surf_M_dat['R'] )
hi_sm_M, hi_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )


dat = pds.read_csv( cc_SM_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
cc_lo_R, cc_lo_surf_M, cc_lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( cc_SM_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
cc_hi_R, cc_hi_surf_M, cc_hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )


#. mass profile ratios (to the integral sample)
eta_dat = pds.read_csv(out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[0],)
lo_eta_R, lo_eta_M, lo_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )

eta_dat = pds.read_csv(out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[1],)
hi_eta_R, hi_eta_M, hi_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )

eta_dat = pds.read_csv( cc_SM_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[0] )
cc_lo_eta_R, cc_lo_eta_M, cc_lo_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'])

eta_dat = pds.read_csv( cc_SM_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[1] )
cc_hi_eta_R, cc_hi_eta_M, cc_hi_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'])


#. overall sample
dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
					'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv')
aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


fig = plt.figure( figsize = (5.4, 5.4) )
ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

ax1.plot( lo_sm_R / 1e3, lo_sm_M, ls = '--', color = line_c[0], alpha = 0.45, linewidth = 3.5, 
	label = fig_name[0] + ' with $P_{cen}$ cut',)
# ax1.fill_between( lo_sm_R / 1e3, y1 = lo_sm_M - lo_sm_err, y2 = lo_sm_M + lo_sm_err, color = line_c[0], alpha = 0.15,)

ax1.plot( cc_lo_R / 1e3, cc_lo_surf_M, ls = '--', color = 'b', label = cc_fig_name[0] + ' w/o $P_{cen}$ cut',)
ax1.fill_between( cc_lo_R / 1e3, y1 = cc_lo_surf_M - cc_lo_surf_M_err, y2 = cc_lo_surf_M + cc_lo_surf_M_err, 
	color = line_c[0], alpha = 0.15,)

ax1.plot( hi_sm_R / 1e3, hi_sm_M, ls = '-', color = line_c[1], alpha = 0.45, linewidth = 3.5, 
	label = fig_name[1] + ' with $P_{cen}$ cut',)
# ax1.fill_between( hi_sm_R / 1e3, y1 = hi_sm_M - hi_sm_err, y2 = hi_sm_M + hi_sm_err, color = line_c[1], alpha = 0.15,)

ax1.plot( cc_hi_R / 1e3, cc_hi_surf_M, ls = '-', color = 'r', label = cc_fig_name[1] + ' w/o $P_{cen}$ cut',)
ax1.fill_between( cc_hi_R / 1e3, y1 = cc_hi_surf_M - cc_hi_surf_M_err, y2 = cc_hi_surf_M + cc_hi_surf_M_err, 
	color = line_c[1], alpha = 0.15,)

ax1.plot( aveg_R / 1e3, aveg_surf_m, ls = '-.', color = 'k', alpha = 0.85, label = 'All clusters',)
# ax1.fill_between( aveg_R / 1e3, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'k', alpha = 0.15,)

ax1.fill_betweenx( y = np.logspace(3, 10, 250), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax1.text( 3.5e-3, 1e7, s = 'PSF', fontsize = 13,)

ax1.set_xlim( 3e-3, 1e0 )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim( 5e3, 2e9 )
ax1.legend( loc = 3, frameon = False, fontsize = 12,)
ax1.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 15,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


sub_ax1.plot( lo_eta_R / 1e3, lo_eta_M, ls = '--', color = line_c[0], alpha = 0.45, linewidth = 3.5,)
# sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta_M - lo_eta_M_err, y2 = lo_eta_M + lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta_M, ls = '-', color = line_c[1], alpha = 0.45, linewidth = 3.5,)
# sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta_M - hi_eta_M_err, y2 = hi_eta_M + hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( cc_lo_eta_R / 1e3, cc_lo_eta_M, ls = '--', color = line_c[0],)
sub_ax1.fill_between( cc_lo_eta_R / 1e3, y1 = cc_lo_eta_M - cc_lo_eta_M_err, 
		y2 = cc_lo_eta_M + cc_lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( cc_hi_eta_R / 1e3, cc_hi_eta_M, ls = '-', color = line_c[1],)
sub_ax1.fill_between( cc_hi_eta_R / 1e3, y1 = cc_hi_eta_M - cc_hi_eta_M_err, 
		y2 = cc_hi_eta_M + cc_hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.fill_betweenx( y = np.logspace(-5, 5), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
sub_ax1.axhline( y = 1, ls = '-.', color = 'k', alpha = 0.85,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
sub_ax1.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax1.set_ylabel('$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{\\mathrm{All \; clusters} }$', fontsize = 15,)
sub_ax1.set_ylim( 0.5, 1.6 )
sub_ax1.set_yticks([0.5, 1, 1.5 ])
sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.5}$','$\\mathrm{1.0}$', '$\\mathrm{1.5}$'] )

sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.set_xticklabels( labels = [],)

plt.savefig('/home/xkchen/%s_SB_SM_compare.png' % file_s, dpi = 300)
plt.close()

