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


##... cosmology model
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
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir


### === ### data load
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']
file_s = 'BCG_Mstar_bin'

rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/BGs/'
path  = '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SBs/'


"""
##... Background subtraction
for kk in range( 3 ):

	##. entire samples
	aveg_SB_file = path + 'photo-z_match_tot-BCG-star-Mass_Pcen-cut_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[kk]
	out_params_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_Pcen_BG-profile_params_diag-fit.csv' % band[kk]
	out_pros_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_Pcen_BG-profile_diag-fit.csv' % band[kk]


	with h5py.File( aveg_SB_file, 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

	p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
	bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

	R_psf = 10

	lo_R_lim = 500

	hi_R_lim = 1.4e3 # 1.8e3 for i-band
	trunk_R = 2e3

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
	# ax.set_title( fig_name[ mm ] + ', %s band' % band[kk] )

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

	# plt.savefig('/home/xkchen/%s_%s-band_SB_n=2.1-sersic.png' % (cat_lis[ mm ], band[kk]), dpi = 300)
	plt.savefig('/home/xkchen/total_%s-band_SB_n=2.1-sersic.png' % band[kk], dpi = 300)
	plt.close()

N_bin = 30

#. BG-sub SB profiles
for kk in range( 3 ):

	##. entire sample
	jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_Pcen-cut_%s-band_' % band[kk] + 'jack-sub-%d_SB-pro_z-ref.h5'
	sb_out_put = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_Pcen_BG-sub_SB.h5' % band[kk]
	BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_Pcen_BG-profile_params_diag-fit.csv' % band[kk]

	BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)

##.. BG-sub SB profiles of jack-subsample
for kk in range( 3 ):

	##. entire sample
	jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_Pcen-cut_%s-band_' % band[kk] + 'jack-sub-%d_SB-pro_z-ref.h5'
	BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_Pcen_BG-profile_params_diag-fit.csv' % band[kk]
	out_sub_sb = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_' % band[kk] + 'Pcen_jack-sub-%d_BG-sub_SB.csv'

	jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, out_sub_sb )

"""

##... apply Pcen cut on entire sample, and recompute the SM ratio of subsamples
def tot_Pcen_cut_SM():

	### === SM(r) of the entire sample with Pcen cut
	low_R_lim, up_R_lim = 1e0, 1.2e3

	N_samples = 30
	band_str = 'gri'

	cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/'
	out_path = '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/'
	fit_file = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/least-square_M-to-i-band-Lumi&color.csv'


	##.. SM profile
	sub_sb_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_Pcen_jack-sub-%d_BG-sub_SB.csv'
	sub_sm_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'Pcen_jack-sub-%d_mass-Lumi.csv'

	aveg_jk_sm_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_mass-Lumi.csv' % band_str
	lgM_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_log-surf-mass_cov_arr.h5' % band_str
	M_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_surf-mass_cov_arr.h5' % band_str

	fit_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_ref, z_ref,
					aveg_jk_sm_file, lgM_cov_file, fit_file, M_cov_file = M_cov_file )


	##.. surface mass profile with deredden color profile
	## lgM = a(g-r) + b(r-i) + c*lg_Li + d
	pfit_dat = pds.read_csv( fit_file )
	a_fit = np.array( pfit_dat['a'] )[0]
	b_fit = np.array( pfit_dat['b'] )[0]
	c_fit = np.array( pfit_dat['c'] )[0]
	d_fit = np.array( pfit_dat['d'] )[0]


	E_dat = pds.read_csv( cat_path + 'total_gri-common_P-cen_lim_cat_params.csv',)
	AL_r, AL_g, AL_i = np.array( E_dat['Al_r'] ), np.array( E_dat['Al_g'] ), np.array( E_dat['Al_i'] )

	mA_r = np.median( AL_r )
	mA_g = np.median( AL_g )
	mA_i = np.median( AL_i )

	tmp_r = []
	tmp_SM = []
	tmp_lgSM = []
	tmp_Li = []

	for nn in range( N_samples ):

		_sub_dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'Pcen_jack-sub-%d_mass-Lumi.csv' % nn)
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
		out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'Pcen_jack-sub-%d_mass-Lumi_with-dered.csv' % nn )

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
	out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)

	#. covmatrix
	R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_lgSM, id_jack = True)
	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_log-surf-mass_cov_arr_with-dered.h5' % band_str, 'w') as f:
		f['R_kpc'] = np.array( R_mean )
		f['cov_MX'] = np.array( cov_MX )
		f['cor_MX'] = np.array( cor_MX )

	R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_SM, id_jack = True)
	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_surf-mass_cov_arr_with-dered.h5' % band_str, 'w') as f:
		f['R_kpc'] = np.array( R_mean )
		f['cov_MX'] = np.array( cov_MX )
		f['cor_MX'] = np.array( cor_MX )


	##... mass profile compare
	m_dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_mass-Lumi.csv' % band_str )
	m_R, surf_M, SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )

	m_dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_mass-Lumi_with-dered.csv' % band_str )
	cc_m_R, cc_surf_M, cc_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )

	#. all samples
	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
						'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
	tot_R = np.array(dat['R'])
	tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


	fig = plt.figure()
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax.plot( m_R, surf_M, ls = '-', color = 'b', alpha = 0.75, linewidth = 1, label = '$\\Sigma_{\\ast}$')
	ax.fill_between( m_R, y1 = surf_M - SM_err, y2 = surf_M + SM_err, color = 'b', alpha = 0.15,)

	ax.plot( cc_m_R, cc_surf_M, ls = '--', color = 'b', alpha = 0.45, linewidth = 3, label = '$\\Sigma_{\\ast}$, deredden')
	ax.fill_between( cc_m_R, y1 = cc_surf_M - cc_SM_err, y2 = cc_surf_M + cc_SM_err, color = 'b', alpha = 0.15,)

	ax.plot( tot_R, tot_surf_m, ls = '-.', color = 'k', label = 'All clusters')

	ax.legend( loc = 3, frameon = False,)
	ax.set_ylim( 6e3, 3e9 )
	ax.set_yscale( 'log' )
	ax.set_xlim( 1e0, 1e3 )
	ax.set_xscale('log')

	ax.set_ylabel('$M_{\\ast} [M_{\\odot}]$', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax.plot( m_R, cc_surf_M / surf_M, ls = '--', color = 'b', alpha = 0.75,)

	sub_ax.set_ylim( 0.99, 1.02 )
	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xscale('log')

	sub_ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
	sub_ax.set_xlabel('R [kpc]', fontsize = 12,)
	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/total_Pcen-lim_surf-M_compare.png', dpi = 300)
	plt.close()	


	### === SM(r) mass correction



	### === SM(r) ratio of subsamples



	return

# tot_Pcen_cut_SM()


def sub_set_SM_ratio():

	out_path = '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/'
	cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/'

	band_str = 'gri'
	N_samples = 30

	id_dered = True
	dered_str = '_with-dered'

	##.. aperture around BCG stellar mass
	fix_R_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
								'total-sample_%s-band_based_BCGM-match_R_with-dered.csv' % band_str )
	R_fixed_M = [ np.array( fix_R_dat[ 'R_fixed_mean_M' ] )[0], np.array( fix_R_dat[ 'R_fixed_medi_M' ] )[0] ]


	cali_factor = np.array( [] )

	#... mass profile
	m_dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_aveg-jack_mass-Lumi%s.csv' % (band_str, dered_str),)
	jk_R = np.array(m_dat['R'])
	surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
	surf_L = np.array( m_dat['lumi'] )

	N_grid = 250

	up_lim_R_0 = R_fixed_M[0]  ## mean Mstar calibrated
	up_lim_R_1 = R_fixed_M[1]  ## median Mstar calibrated

	cumu_M = cumu_mass_func( jk_R, surf_m, N_grid = N_grid )
	intep_Mf = interp.interp1d( jk_R, cumu_M, kind = 'cubic',)

	M_c0 = intep_Mf( up_lim_R_0 )
	M_c1 = intep_Mf( up_lim_R_1 )


	#...sample properties
	p_dat = pds.read_csv( cat_path + 'total_gri-common_P-cen_lim_cat_params.csv',)
	p_lgM = np.array( p_dat['lg_Mstar'] ) # Mass unit : M_sun / h^2

	lg_Mean = np.log10( np.mean(10**p_lgM / h**2) )
	lg_Medi = np.log10( np.median(10**p_lgM / h**2) )

	devi_Mean = lg_Mean - np.log10( M_c0 )
	devi_Medi = lg_Medi - np.log10( M_c1 )

	medi_off_surf_M = surf_m * 10**( devi_Medi )
	mean_off_surf_M = surf_m * 10**( devi_Mean )


	#. save the calibrated SM profiles
	keys = ['R', 'medi_correct_surf_M', 'mean_correct_surf_M', 'surf_M_err']
	values = [ jk_R, medi_off_surf_M, mean_off_surf_M, surf_m_err ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_corrected_aveg-jack_mass-Lumi%s.csv' % (band_str, dered_str),)


	cali_factor = np.r_[ cali_factor, [ devi_Medi, devi_Mean ] ]

	#.
	keys = ['Pcut_medi_devi', 'Pcut_mean_devi']
	values = list( cali_factor )
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_path + 'total_%s-band-based_Pcen_M_calib-f%s.csv' % (band_str, dered_str),)


	##. SM ratio of subsamples to the total sample with Pcen cut
	dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_Pcen_corrected_aveg-jack_mass-Lumi%s.csv' % (band_str, dered_str),)
	tot_R = np.array(dat['R'])
	
	tot_medi_calib_SM = np.array( dat['medi_correct_surf_M'])
	tot_mean_calib_SM = np.array( dat['mean_correct_surf_M'])

	medi_interp_Mf = interp.interp1d( tot_R, tot_medi_calib_SM, kind = 'linear',)
	mean_interp_Mf = interp.interp1d( tot_R, tot_mean_calib_SM, kind = 'linear',)


	calib_cat = pds.read_csv( out_path + '%s_%s-band-based_M_calib-f%s.csv' % (file_s, band_str, dered_str),)
	medi_M_offset = [ np.array(calib_cat['low_medi_devi'])[0], np.array(calib_cat['high_medi_devi'])[0] ]
	mean_M_offset = [ np.array(calib_cat['low_mean_devi'])[0], np.array(calib_cat['high_mean_devi'])[0] ]


	for mm in range( 2 ):

		jk_sub_m_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi' + '%s.csv' % dered_str

		tmp_r, tmp_mean_ratio = [], []
		tmp_medi_ratio = []

		for nn in range( N_samples ):

			o_dat = pds.read_csv( jk_sub_m_file % nn,)

			tt_r = np.array( o_dat['R'] )
			tt_M = np.array( o_dat['surf_mass'] )

			tt_medi_off_M = tt_M * 10**medi_M_offset[ mm ]
			tt_mean_off_M = tt_M * 10**mean_M_offset[ mm ]

			idx_lim = ( tt_r >= np.nanmin( tot_R ) ) & ( tt_r <= np.nanmax( tot_R ) )

			lim_R = tt_r[ idx_lim ]
			lim_medi_off_M = tt_medi_off_M[ idx_lim ]
			lim_mean_off_M = tt_mean_off_M[ idx_lim ]


			sub_ratio = np.zeros( len(tt_r),)
			sub_ratio[ idx_lim ] = lim_medi_off_M / medi_interp_Mf( lim_R )
			sub_ratio[ idx_lim == False ] = np.nan

			sub_ratio_1 = np.zeros( len(tt_r),)
			sub_ratio_1[ idx_lim ] = lim_mean_off_M / mean_interp_Mf( lim_R )
			sub_ratio_1[ idx_lim == False ] = np.nan

			tmp_r.append( tt_r )
			tmp_medi_ratio.append( sub_ratio )
			tmp_mean_ratio.append( sub_ratio_1 )

		aveg_R, aveg_medi_ratio, aveg_medi_ratio_err = arr_jack_func( tmp_medi_ratio, tmp_r, N_samples)[:3]
		aveg_R, aveg_mean_ratio, aveg_mean_ratio_err = arr_jack_func( tmp_mean_ratio, tmp_r, N_samples)[:3]

		keys = ['R', 'medi_ref_M/M_tot', 'medi_ref_M/M_tot-err', 'mean_ref_M/M_tot', 'mean_ref_M/M_tot-err']
		values = [ aveg_R, aveg_medi_ratio, aveg_medi_ratio_err, aveg_mean_ratio, aveg_mean_ratio_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )

		out_data.to_csv( out_path + '%s_%s-band_Pcen_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[mm], band_str, dered_str),)


	##... figs
	cc_lo_eat_dat = pds.read_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], band_str, dered_str),)
	cc_lo_eta_R, cc_lo_eta, cc_lo_eta_err = np.array(cc_lo_eat_dat['R']), np.array(cc_lo_eat_dat['mean_ref_M/M_tot']), np.array(cc_lo_eat_dat['mean_ref_M/M_tot-err'])

	cc_hi_eat_dat = pds.read_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], band_str, dered_str),)
	cc_hi_eta_R, cc_hi_eta, cc_hi_eta_err = np.array(cc_hi_eat_dat['R']), np.array(cc_hi_eat_dat['mean_ref_M/M_tot']), np.array(cc_hi_eat_dat['mean_ref_M/M_tot-err'])


	lo_eat_dat = pds.read_csv( out_path + '%s_%s-band_Pcen_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], band_str, dered_str),)
	lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['mean_ref_M/M_tot']), np.array(lo_eat_dat['mean_ref_M/M_tot-err'])

	hi_eat_dat = pds.read_csv( out_path + '%s_%s-band_Pcen_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], band_str, dered_str),)
	hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['mean_ref_M/M_tot']), np.array(hi_eat_dat['mean_ref_M/M_tot-err'])	


	plt.figure()
	ax = plt.subplot( 111 )

	ax.plot( lo_eta_R, lo_eta, ls = '-', color = line_c[0], alpha = 0.75, label = fig_name[0] + '(/ All with $P_{cen} cut$)')
	ax.fill_between( lo_eta_R, y1 = lo_eta - lo_eta_err, y2 = lo_eta + lo_eta_err, color = line_c[0], alpha = 0.15,)
	ax.plot( hi_eta_R, hi_eta, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1] + '(/ All) with $P_{cen} cut$')
	ax.fill_between( hi_eta_R, y1 = hi_eta - hi_eta_err, y2 = hi_eta + hi_eta_err, color = line_c[1], alpha = 0.15,)


	ax.plot( cc_lo_eta_R, cc_lo_eta, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0] + '(/ All)')
	ax.fill_between( cc_lo_eta_R, y1 = cc_lo_eta - cc_lo_eta_err, y2 = cc_lo_eta + cc_lo_eta_err, color = line_c[0], alpha = 0.15,)
	ax.plot( cc_hi_eta_R, cc_hi_eta, ls = '--', color = line_c[1], alpha = 0.75, label = fig_name[1] + '(/ All)')
	ax.fill_between( cc_hi_eta_R, y1 = cc_hi_eta - cc_hi_eta_err, y2 = cc_hi_eta + cc_hi_eta_err, color = line_c[1], alpha = 0.15,)

	ax.legend( loc = 1)
	ax.set_ylim( 0.45, 1.65 )
	ax.set_xlim( 1e1, 1e3 )
	ax.set_xscale('log')

	ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
	ax.set_xlabel('R [kpc]', fontsize = 12,)
	ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_surf-M_ratio_compare%s_Pcen-lim.png' % (file_s, dered_str), dpi = 300)
	plt.close()


sub_set_SM_ratio()

