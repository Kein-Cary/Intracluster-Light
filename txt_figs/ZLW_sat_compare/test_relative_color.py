import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

import scipy.signal as signal
import scipy.interpolate as interp
from scipy import optimize
from astropy.coordinates import SkyCoord
from astropy import cosmology as apcy

from img_random_SB_fit import random_SB_fit_func
from img_random_SB_fit import clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import arr_jack_func, cc_grid_img
from color_2_mass import jk_sub_SB_func, color_func

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
L_wave = np.array( [ 6166, 4686, 7480 ] )
Mag_sun = [ 4.65, 5.11, 4.53 ]

pixel = 0.396
z_ref = 0.25

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

###===### data load
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 
# 			'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'

# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
# 			'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
			 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'


###... BG estimation and color profiles
rand_path = '/home/xkchen/tmp_run/data_files/jupyter/random_ref_SB/'

# BG_path = '/home/xkchen/figs/half_z_color/equal_size/BGs/'
# path = '/home/xkchen/figs/half_z_color/equal_size/SBs/'

BG_path = '/home/xkchen/figs/half_z_color/identi_z_cut/BGs/'
path = '/home/xkchen/figs/half_z_color/identi_z_cut/SBs/'

z_samp = [ 'low-z', 'hi-z' ]

line_s = [ '--', '-' ]
line_c = [ 'r', 'g', 'b']


#... sample properties over view
cat_path = '/home/xkchen/tmp_run/data_files/figs/'
hi_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )
hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['BCG_age'] )

hi_ra, hi_dec = np.array( hi_dat['ra']), np.array( hi_dat['dec'] )
hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg,)

lo_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )
lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['BCG_age'] )

lo_ra, lo_dec = np.array( lo_dat['ra']), np.array( lo_dat['dec'] )
lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg,)

'''
fig = plt.figure( figsize = (15.40, 4.8) )
ax0 = fig.add_axes([0.05, 0.12, 0.275, 0.85])
ax1 = fig.add_axes([0.38, 0.12, 0.275, 0.85])
ax2 = fig.add_axes([0.71, 0.12, 0.275, 0.85])

qq = 1

for mm in range( 2 ):

	# tt_dat = pds.read_csv('/home/xkchen/figs/half_z_color/equal_size/half_z_cat/' + 
	# 						'%s_bin_gri-common-cat_r-band_%s_zref-cat.csv' % (cat_lis[qq], z_samp[mm]),)

	tt_dat = pds.read_csv('/home/xkchen/figs/half_z_color/identi_z_cut/half_z_cat/' + 
							'%s_bin_gri-common-cat_r-band_%s_zref-cat.csv' % (cat_lis[qq], z_samp[mm]),)

	tt_ra, tt_dec, tt_z = np.array( tt_dat['ra'] ), np.array( tt_dat['dec'] ), np.array( tt_dat['z'] )
	tt_coord = SkyCoord( ra = tt_ra * U.deg, dec = tt_dec * U.deg,)

	if qq == 0:
		_idx_, d2d, d3d = tt_coord.match_to_catalog_sky( lo_coord )
		id_lim = d2d.value < 2.7e-4
		tt_rich, tt_lgM, tt_age = lo_rich[ _idx_[ id_lim ] ], lo_lgM[ _idx_[ id_lim ] ], lo_age[ _idx_[ id_lim ] ]

	if qq == 1:
		_idx_, d2d, d3d = tt_coord.match_to_catalog_sky( hi_coord )
		id_lim = d2d.value < 2.7e-4
		tt_rich, tt_lgM, tt_age = hi_rich[ _idx_[ id_lim ] ], hi_lgM[ _idx_[ id_lim ] ], hi_age[ _idx_[ id_lim ] ]		

	ax0.hist( tt_rich, bins = 35, density = True, histtype = 'step', color = line_c[mm], label = z_samp[mm], alpha = 0.5,)
	ax0.axvline( np.median(tt_rich), ls = '-', color = line_c[mm], ymin = 0.0, ymax = 0.35, alpha = 0.5, 
		label = z_samp[mm] + ', median')
	ax0.axvline( np.mean(tt_rich), ls = '--', color = line_c[mm], ymin = 0.0, ymax = 0.35, alpha = 0.5, 
		label = z_samp[mm] + ', mean')
	ax0.set_xlabel('$\\lambda$')
	ax0.set_xscale('log')
	ax0.set_yscale('log')
	ax0.set_ylabel('pdf')
	ax0.legend( loc = 1, frameon = False,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in',)

	ax1.hist( tt_lgM, bins = 35, density = True, histtype = 'step', color = line_c[mm], label = z_samp[mm], alpha = 0.5,)
	ax1.axvline( np.median(tt_lgM), ls = '-', color = line_c[mm], ymin = 0.0, ymax = 0.35, alpha = 0.5, 
		label = z_samp[mm] + ', median')
	ax1.axvline( np.mean(tt_lgM), ls = '--', color = line_c[mm], ymin = 0.0, ymax = 0.35, alpha = 0.5, 
		label = z_samp[mm] + ', mean')
	ax1.set_xlabel('$\\lg M_{\\ast} \; [M_{\\odot} / h^{2}]$')
	ax1.set_ylabel('pdf')
	ax1.legend( loc = 2, frameon = False,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in',)

	ax2.hist( tt_age, bins = 35, density = True, histtype = 'step', color = line_c[mm], label = z_samp[mm], alpha = 0.5,)
	ax2.axvline( np.median(tt_age), ls = '-', color = line_c[mm], ymin = 0.0, ymax = 0.35, alpha = 0.5, 
		label = z_samp[mm] + ', median')
	ax2.axvline( np.mean(tt_age), ls = '--', color = line_c[mm], ymin = 0.0, ymax = 0.35, alpha = 0.5, 
		label = z_samp[mm] + ', mean')
	ax2.set_xlabel('$t_{\\mathrm{age}} \; [\\mathrm{G}yr]$')
	ax2.set_ylabel('pdf')
	ax2.legend( loc = 2, frameon = False,)
	ax2.tick_params( axis = 'both', which = 'both', direction = 'in',)
	ax2.annotate( text = fig_name[qq], xy = (0.05, 0.45), xycoords = 'axes fraction', fontsize = 13,)

plt.savefig('/home/xkchen/%s_z-divid_sample_compare.png' % cat_lis[qq], dpi = 300)
plt.close()
'''

"""
#... background fitting
for mm in range( 2 ):

	for ll in range( 2 ):

		for kk in range( 3 ):

			with h5py.File( path + 
				'photo-z_match_gri-common_%s_%s-band_%s_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk], z_samp[ll]), 'r') as f:
				tt_r = np.array(f['r'])
				tt_sb = np.array(f['sb'])
				tt_err = np.array(f['sb_err'])

			params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

			p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
			bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

			R_psf = 10
			lo_R_lim = 400 # 400, 500

			hi_R_lim = 1.4e3
			trunk_R = 2e3

			out_params_file = BG_path + 'photo-z_%s_%s-band_%s_BG-profile_params.csv' % (cat_lis[ mm ], band[kk], z_samp[ll])
			out_pros_file = BG_path + 'photo-z_%s_%s-band_%s_BG-profile.csv' % (cat_lis[ mm ], band[kk], z_samp[ll])

			clust_SB_fit_func( tt_r, tt_sb, tt_err, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, 
								out_params_file, out_pros_file, trunk_R = trunk_R,)

			## fig
			p_dat = pds.read_csv( out_params_file )
			( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD, I_e, R_e) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], 
				np.array(p_dat['e_x0'])[0], np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0], 
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

			ax.set_title( fig_name[ mm ] + ', %s band' % band[kk] + ', %s' % z_samp[ll] )

			ax.plot( tt_r, tt_sb, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
			ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.12)

			ax.plot(tt_r, comb_F, ls = '-', color = 'k', alpha = 0.5, label = 'Best fitting',)
			ax.plot(tt_r, norm_sign, ls = '-.', color = 'k', alpha = 0.5, label = 'signal (model)',)
			ax.plot(tt_r, norm_BG, ls = '--', color = 'k', alpha = 0.5, label = 'BackGround')

			ax.axvline(x = lo_R_lim, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

			ax.annotate(text = '$\\chi^2 / \\nu = %.5f$' % chi_ov_nu, xy = (0.60, 0.60), 
					xycoords = 'axes fraction', color = 'k',)
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
			plt.savefig(
				'/home/xkchen/%s_%s-band_%s_SB_n=2.1-sersic.png' % (cat_lis[ mm ], band[kk], z_samp[ll]), dpi = 300)
			plt.close()


#... subsample BG_subtracted surface brightness
N_bin = 30

for mm in range( 2 ):
	for ll in range( 2 ):
		for kk in range( 3 ):

			sub_sb = [ path + 'photo-z_match_gri-common_%s_%s-band_%s_' % (cat_lis[mm], band[kk], z_samp[ll]) + 
						'jack-sub-%d_SB-pro_z-ref.h5' ][0]

			sb_out_put = BG_path + 'photo-z_match_gri-common_%s_%s-band_%s_BG-sub_SB.h5' % (cat_lis[mm], band[kk], z_samp[ll])
			BG_file = BG_path + 'photo-z_%s_%s-band_%s_BG-profile_params.csv' % (cat_lis[ mm ], band[kk], z_samp[ll])

			BG_sub_sb_func( N_bin, sub_sb, sb_out_put, band[ kk ], BG_file,)

#... over view
for mm in range( 2 ):
	
	fig = plt.figure()
	ax = fig.add_axes( [0.12, 0.11, 0.80, 0.80] )
	ax.set_title( fig_name[mm] )

	for ll in range( 2 ):

		for kk in range( 3 ):

			# with h5py.File( path + 
			# 	'photo-z_match_gri-common_%s_%s-band_%s_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk], z_samp[ll]), 'r') as f:

			with h5py.File( BG_path + 
				'photo-z_match_gri-common_%s_%s-band_%s_BG-sub_SB.h5' % (cat_lis[mm], band[kk], z_samp[ll]), 'r') as f:

				tt_r = np.array( f['r'] )
				tt_sb = np.array( f['sb'] )
				tt_err = np.array( f['sb_err'] )

			ax.plot( tt_r, tt_sb, ls = line_s[ll], color = line_c[kk], alpha = 0.75, label = z_samp[ll] + ', %s' % band[kk])
			ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = line_c[kk], alpha = 0.13)

	ax.set_xlim(1e0, 4e3)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')

	ax.set_ylabel('SB [nanomaggies / arcsec^2]')
	ax.set_yscale('log')
	# ax.set_ylim( 1e-3, 2e1 )
	ax.set_ylim( 1e-5, 2e1 )

	ax.legend( loc = 1, frameon = False,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.savefig('/home/xkchen/%s_SB_compare.png' % cat_lis[ mm ], dpi = 300)
	plt.close()
"""

#... subsample color and surfce mass profile
def sub_color_f( cat_str, sample_str ):

	cat_lis = cat_str
	z_samp = sample_str

	N_samples = 30

	for mm in range( 2 ):
		for ll in range( 2 ):
			for kk in range( 3 ):

				jk_sub_sb = [ path + 'photo-z_match_gri-common_%s_%s-band_%s_' % (cat_lis[mm], band[kk], z_samp[ll]) + 
						'jack-sub-%d_SB-pro_z-ref.h5' ][0]
				BG_file = BG_path + 'photo-z_%s_%s-band_%s_BG-profile_params.csv' % (cat_lis[ mm ], band[kk], z_samp[ll])
				out_sub_sb = BG_path + '%s_%s-band_%s_' % (cat_lis[mm], band[kk], z_samp[ll]) + 'jack-sub-%d_BG-sub_SB.csv'
				jk_sub_SB_func( N_samples, jk_sub_sb, BG_file, out_sub_sb )

	### average jack-sub sample color
	for mm in range( 2 ):

		for ll in range( 2 ):

			tmp_r, tmp_gr, tmp_gi, tmp_ri = [], [], [], []
		
			for kk in range( N_samples ):

				p_r_dat = pds.read_csv( BG_path + '%s_r-band_%s_' % (cat_lis[mm], z_samp[ll]) + 'jack-sub-%d_BG-sub_SB.csv' % kk )
				tt_r_R, tt_r_sb, tt_r_err = np.array( p_r_dat['R'] ), np.array( p_r_dat['BG_sub_SB'] ), np.array( p_r_dat['sb_err'] )

				p_g_dat = pds.read_csv( BG_path + '%s_g-band_%s_' % (cat_lis[mm], z_samp[ll]) + 'jack-sub-%d_BG-sub_SB.csv' % kk )
				tt_g_R, tt_g_sb, tt_g_err = np.array( p_g_dat['R'] ), np.array( p_g_dat['BG_sub_SB'] ), np.array( p_g_dat['sb_err'] )

				p_i_dat = pds.read_csv( BG_path + '%s_i-band_%s_' % (cat_lis[mm], z_samp[ll]) + 'jack-sub-%d_BG-sub_SB.csv' % kk )
				tt_i_R, tt_i_sb, tt_i_err = np.array( p_i_dat['R'] ), np.array( p_i_dat['BG_sub_SB'] ), np.array( p_i_dat['sb_err'] )


				idR_lim = tt_r_R <= 1.2e3
				tt_r_R, tt_r_sb, tt_r_err = tt_r_R[ idR_lim], tt_r_sb[ idR_lim], tt_r_err[ idR_lim]

				idR_lim = tt_g_R <= 1.2e3
				tt_g_R, tt_g_sb, tt_g_err = tt_g_R[ idR_lim], tt_g_sb[ idR_lim], tt_g_err[ idR_lim] 

				idR_lim = tt_i_R <= 1.2e3
				tt_i_R, tt_i_sb, tt_i_err = tt_i_R[ idR_lim], tt_i_sb[ idR_lim], tt_i_err[ idR_lim] 

				gr_arr, gr_err = color_func( tt_g_sb, tt_g_err, tt_r_sb, tt_r_err )
				gi_arr, gi_err = color_func( tt_g_sb, tt_g_err, tt_i_sb, tt_i_err )
				ri_arr, ri_err = color_func( tt_r_sb, tt_r_err, tt_i_sb, tt_i_err )

				keys = [ 'R', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err']
				values = [ tt_g_R, gr_arr, gr_err, gi_arr, gi_err, ri_arr, ri_err ]
				fill = dict(zip( keys, values) )
				out_data = pds.DataFrame( fill )
				out_data.to_csv( BG_path + '%s_%s_' % (cat_lis[mm], z_samp[ll]) + '_jack-sub-%d_color_profile.csv' % kk,)

				tmp_r.append( tt_g_R )
				tmp_gr.append( gr_arr )
				tmp_gi.append( gi_arr )
				tmp_ri.append( ri_arr )

			aveg_R_0, aveg_gr, aveg_gr_err = arr_jack_func( tmp_gr, tmp_r, N_samples )[:3]
			aveg_R_1, aveg_gi, aveg_gi_err = arr_jack_func( tmp_gi, tmp_r, N_samples )[:3]
			aveg_R_2, aveg_ri, aveg_ri_err = arr_jack_func( tmp_ri, tmp_r, N_samples )[:3]

			Len_x = np.max( [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ] )
			id_L = [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ].index( Len_x )

			cc_aveg_R = [ aveg_R_0, aveg_R_1, aveg_R_2 ][ id_L ]

			cc_aveg_gr = np.zeros( Len_x,)
			cc_aveg_gr[ :len(aveg_gr) ] = aveg_gr
			cc_aveg_gr_err = np.zeros( Len_x,)
			cc_aveg_gr_err[ :len(aveg_gr) ] = aveg_gr_err

			cc_aveg_gi = np.zeros( Len_x,)
			cc_aveg_gi[ :len(aveg_gi) ] = aveg_gi
			cc_aveg_gi_err = np.zeros( Len_x,)
			cc_aveg_gi_err[ :len(aveg_gi) ] = aveg_gi_err

			cc_aveg_ri = np.zeros( Len_x,)
			cc_aveg_ri[ :len(aveg_ri) ] = aveg_ri
			cc_aveg_ri_err = np.zeros( Len_x,)
			cc_aveg_ri_err[ :len(aveg_ri) ] = aveg_ri_err

			keys = [ 'R_kpc', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err' ]
			values = [ cc_aveg_R, cc_aveg_gr, cc_aveg_gr_err, cc_aveg_gi, cc_aveg_gi_err, cc_aveg_ri, cc_aveg_ri_err ]
			fill = dict( zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( BG_path + '%s_%s_color_profile.csv' % ( cat_lis[mm], z_samp[ll] ),)

	return

def smooth_slope_func(r_arr, color_arr, wind_L, order_dex, delta_x):

	id_nn = np.isnan( color_arr )

	if np.sum( id_nn ) == 0:
		dc_dr = signal.savgol_filter( color_arr, wind_L, order_dex, deriv = 1, delta = delta_x)
	else:
		_cp_r = r_arr[ id_nn == False ]
		_cp_color = color_arr[ id_nn == False ]

		_cp_color_F = interp.interp1d( _cp_r, _cp_color, kind = 'linear', fill_value = 'extrapolate',)

		color_arr[id_nn] = _cp_color_F( r_arr[id_nn] )
		dc_dr = signal.savgol_filter( color_arr, wind_L, order_dex, deriv = 1, delta = delta_x)
	return dc_dr

def color_slope_f( cat_str, sample_str ):

	cat_lis = cat_str
	z_samp = sample_str

	N_samples = 30

	### average jack-sub sample color
	for mm in range( 2 ):

		for ll in range( 2 ):

			tmp_r, tmp_dgr, tmp_dgi, tmp_dri = [], [], [], []

			for kk in range( N_samples ):

				pdat = pds.read_csv( BG_path + '%s_%s_' % (cat_lis[mm], z_samp[ll]) + '_jack-sub-%d_color_profile.csv' % kk,)
				tt_r, tt_gr, tt_gr_err = np.array( pdat['R'] ), np.array( pdat['g-r'] ), np.array( pdat['g-r_err'] )
				tt_gi, tt_gi_err = np.array( pdat['g-i'] ), np.array( pdat['g-i_err'] )
				tt_ri, tt_ri_err = np.array( pdat['r-i'] ), np.array( pdat['r-i_err'] )

				WL, p_order = 13, 1
				delt_x = 0.0635

				d_gr_dlgr = smooth_slope_func(tt_r, tt_gr, WL, p_order, delt_x)
				d_gi_dlgr = smooth_slope_func(tt_r, tt_gi, WL, p_order, delt_x)
				d_ri_dlgr = smooth_slope_func(tt_r, tt_ri, WL, p_order, delt_x)

				keys = [ 'R', 'd_gr_dlgr', 'd_gi_dlgr', 'd_ri_dlgr' ]
				values = [ tt_r, d_gr_dlgr, d_gi_dlgr, d_ri_dlgr, ]

				fill = dict(zip( keys, values) )
				out_data = pds.DataFrame( fill )
				out_data.to_csv( BG_path + '%s_%s_' % (cat_lis[mm], z_samp[ll]) + '_jack-sub-%d_color_slope.csv' % kk,)

				tmp_r.append( tt_r )
				tmp_dgr.append( d_gr_dlgr )
				tmp_dgi.append( d_gi_dlgr )
				tmp_dri.append( d_ri_dlgr )

			aveg_R_0, aveg_dgr, aveg_dgr_err = arr_jack_func( tmp_dgr, tmp_r, N_samples )[:3]
			aveg_R_1, aveg_dgi, aveg_dgi_err = arr_jack_func( tmp_dgi, tmp_r, N_samples )[:3]
			aveg_R_2, aveg_dri, aveg_dri_err = arr_jack_func( tmp_dri, tmp_r, N_samples )[:3]

			Len_x = np.max( [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ] )
			id_L = [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ].index( Len_x )

			cc_aveg_R = [ aveg_R_0, aveg_R_1, aveg_R_2 ][ id_L ]

			cc_aveg_dgr = np.zeros( Len_x,)
			cc_aveg_dgr[ :len(aveg_dgr) ] = aveg_dgr
			cc_aveg_dgr_err = np.zeros( Len_x,)
			cc_aveg_dgr_err[ :len(aveg_dgr) ] = aveg_dgr_err

			cc_aveg_dgi = np.zeros( Len_x,)
			cc_aveg_dgi[ :len(aveg_dgi) ] = aveg_dgi
			cc_aveg_dgi_err = np.zeros( Len_x,)
			cc_aveg_dgi_err[ :len(aveg_dgi) ] = aveg_dgi_err

			cc_aveg_dri = np.zeros( Len_x,)
			cc_aveg_dri[ :len(aveg_dri) ] = aveg_dri
			cc_aveg_dri_err = np.zeros( Len_x,)
			cc_aveg_dri_err[ :len(aveg_dri) ] = aveg_dri_err

			keys = [ 'R_kpc', 'd_gr', 'd_gr_err', 'd_gi', 'd_gi_err', 'd_ri', 'd_ri_err' ]
			values = [ cc_aveg_R, cc_aveg_dgr, cc_aveg_dgr_err, cc_aveg_dgi, cc_aveg_dgi_err, cc_aveg_dri, cc_aveg_dri_err ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( BG_path + '%s_%s_color_slope.csv' % (cat_lis[mm], z_samp[ll]),)

	return

# sub_color_f( cat_lis, z_samp )

# color_slope_f( cat_lis, z_samp )


###===### figures
#. color profile of overall sample
mu_dat = pds.read_csv( pre_path + '%s_color_profile.csv' % cat_lis[1] )
all_hi_R, all_hi_gr, all_hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
all_hi_gr = signal.savgol_filter( all_hi_gr, 7, 3)

mu_dat = pds.read_csv( pre_path + '%s_color_profile.csv' % cat_lis[0] )
all_lo_R, all_lo_gr, all_lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
all_lo_gr = signal.savgol_filter( all_lo_gr, 7, 3)


tt_lo_R, tt_lo_gr, tt_lo_gr_err = [], [], []
tt_hi_R, tt_hi_gr, tt_hi_gr_err = [], [], []

for mm in range( 2 ):

	c_dat = pds.read_csv( BG_path + '%s_%s_color_profile.csv' % ( cat_lis[1], z_samp[mm] ) )
	hi_c_r, hi_gr, hi_gr_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
	hi_gr = signal.savgol_filter( hi_gr, 7, 3)

	c_dat = pds.read_csv( BG_path + '%s_%s_color_profile.csv' % ( cat_lis[0], z_samp[mm] ) )
	lo_c_r, lo_gr, lo_gr_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
	lo_gr = signal.savgol_filter( lo_gr, 7, 3)

	tt_hi_R.append( hi_c_r )
	tt_hi_gr.append( hi_gr )
	tt_hi_gr_err.append( hi_gr_err )

	tt_lo_R.append( lo_c_r )
	tt_lo_gr.append( lo_gr )
	tt_lo_gr_err.append( lo_gr_err )


tt_lo_dcR, tt_lo_dgr, tt_lo_dgr_err = [], [], []
tt_hi_dcR, tt_hi_dgr, tt_hi_dgr_err = [], [], []

for mm in range( 2 ):

	c_dat = pds.read_csv( BG_path + '%s_%s_color_slope.csv' % (cat_lis[0], z_samp[mm]),)
	tt_c_r = np.array( c_dat['R_kpc'] )
	tt_dgr, tt_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )

	tt_lo_dcR.append( tt_c_r )
	tt_lo_dgr.append( tt_dgr )
	tt_lo_dgr_err.append( tt_dgr_err )

	c_dat = pds.read_csv( BG_path + '%s_%s_color_slope.csv' % (cat_lis[1], z_samp[mm]),)
	tt_c_r = np.array( c_dat['R_kpc'] )
	tt_dgr, tt_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )

	tt_hi_dcR.append( tt_c_r )
	tt_hi_dgr.append( tt_dgr )
	tt_hi_dgr_err.append( tt_dgr_err )

'''
fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.08, 0.13, 0.38, 0.80])
ax1 = fig.add_axes([0.58, 0.13, 0.38, 0.80])

ax0.plot( tt_lo_R[1], tt_lo_gr[1], ls = '-', color = 'b', alpha = 0.75, label = fig_name[0] + ', %s' % z_samp[1] )
# ax0.fill_between( tt_lo_R[1], y1 = tt_lo_gr[1] - tt_lo_gr_err[1], y2 = tt_lo_gr[1] + tt_lo_gr_err[1], color = 'b', alpha = 0.15,)

ax0.plot( tt_hi_R[1], tt_hi_gr[1], ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] + ', %s' % z_samp[1] )
# ax0.fill_between( tt_hi_R[1], y1 = tt_hi_gr[1] - tt_hi_gr_err[1], y2 = tt_hi_gr[1] + tt_hi_gr_err[1], color = 'r', alpha = 0.15,)

ax0.plot( tt_lo_R[0], tt_lo_gr[0], ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] + ', %s' % z_samp[0] )
ax0.fill_between( tt_lo_R[0], y1 = tt_lo_gr[0] - tt_lo_gr_err[0], y2 = tt_lo_gr[0] + tt_lo_gr_err[0], color = 'b', alpha = 0.15,)

ax0.plot( tt_hi_R[0], tt_hi_gr[0], ls = '--', color = 'r', alpha = 0.75, label = fig_name[1] + ', %s' % z_samp[0] )
ax0.fill_between( tt_hi_R[0], y1 = tt_hi_gr[0] - tt_hi_gr_err[0], y2 = tt_hi_gr[0] + tt_hi_gr_err[0], color = 'r', alpha = 0.15,)

ax0.legend( loc = 3, fontsize = 13, frameon = False,)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 13)
ax0.set_xlim( 1e0, 1.1e3)

ax0.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax0.set_ylim( 0.8, 1.65 )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)


ax1.plot( tt_lo_dcR[1], tt_lo_dgr[1], ls = '-', color = 'b', alpha = 0.75, label = fig_name[0] + ', %s' % z_samp[1] )
# ax1.fill_between( tt_lo_dcR[1], y1 = tt_lo_dgr[1] - tt_lo_dgr_err[1], y2 = tt_lo_dgr[1] + tt_lo_dgr_err[1], 
# 				color = 'b', alpha = 0.15,)

ax1.plot( tt_hi_dcR[1], tt_hi_dgr[1], ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] + ', %s' % z_samp[1] )
# ax1.fill_between( tt_hi_dcR[1], y1 = tt_hi_dgr[1] - tt_hi_dgr_err[1], y2 = tt_hi_dgr[1] + tt_hi_dgr_err[1], 
# 				color = 'r', alpha = 0.15,)

ax1.plot( tt_lo_dcR[0], tt_lo_dgr[0], ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] + ', %s' % z_samp[0] )
ax1.fill_between( tt_lo_dcR[0], y1 = tt_lo_dgr[0] - tt_lo_dgr_err[0], y2 = tt_lo_dgr[0] + tt_lo_dgr_err[0], 
				color = 'b', alpha = 0.15,)

ax1.plot( tt_hi_dcR[0], tt_hi_dgr[0], ls = '--', color = 'r', alpha = 0.75, label = fig_name[1] + ', %s' % z_samp[0] )
ax1.fill_between( tt_hi_dcR[0], y1 = tt_hi_dgr[0] - tt_hi_dgr_err[0], y2 = tt_hi_dgr[0] + tt_hi_dgr_err[0], 
				color = 'r', alpha = 0.15,)

ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 13)
ax1.set_xlim( 1e0, 1.1e3)

ax1.set_ylabel( '$ {\\rm d} (g-r) \, / \, {\\rm d} \\mathcal{lg}R $', fontsize = 13,)
ax1.set_ylim( -0.6, 0.)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

plt.savefig('/home/xkchen/z-divide_sample_color_compare.png', dpi = 300)
plt.close()
'''

#. ratio of color profile
fig = plt.figure()
ax1 = fig.add_axes( [0.14, 0.11, 0.80, 0.20] )
ax0 = fig.add_axes( [0.14, 0.31, 0.80, 0.60] )

for mm in range( 2 ):

	lo_intep_gr_F = interp.interp1d( tt_lo_R[mm], tt_lo_gr[mm], kind = 'linear', fill_value = 'extrapolate',)

	ax0.plot( tt_lo_R[mm], tt_lo_gr[mm], ls = line_s[mm], color = 'b', alpha = 0.75, label = fig_name[0] + ', %s' % z_samp[mm] )
	ax0.fill_between( tt_lo_R[mm], y1 = tt_lo_gr[mm] - tt_lo_gr_err[mm], y2 = tt_lo_gr[mm] + tt_lo_gr_err[mm], color = 'b', alpha = 0.15,)

	ax0.plot( tt_hi_R[mm], tt_hi_gr[mm], ls = line_s[mm], color = 'r', alpha = 0.75, label = fig_name[1] + ', %s' % z_samp[mm] )
	ax0.fill_between( tt_hi_R[mm], y1 = tt_hi_gr[mm] - tt_hi_gr_err[mm], y2 = tt_hi_gr[mm] + tt_hi_gr_err[mm], color = 'r', alpha = 0.15,)

	ax1.plot( tt_hi_R[mm], tt_hi_gr[mm] / lo_intep_gr_F( tt_hi_R[mm] ), ls = line_s[mm], color = 'r', )

ax0.plot( all_lo_R, all_lo_gr, '*-', color = 'k', alpha = 0.5, label = 'All sample, %s' % fig_name[0],)
ax0.plot( all_hi_R, all_hi_gr, 'o-', color = 'k', alpha = 0.5, label = 'All sample, %s' % fig_name[1],)

lo_intep_gr_F = interp.interp1d( all_lo_R, all_lo_gr, kind = 'linear', fill_value = 'extrapolate',)
ax1.plot( all_hi_R, all_hi_gr / lo_intep_gr_F( all_hi_R), ls = ':', color = 'k', alpha = 0.65,)
ax1.axhline( y = 1, ls = '--', color = 'b', alpha = 0.5,)

ax0.legend( loc = 3, frameon = False,)
ax0.set_xscale('log')
# ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 13)
ax0.set_xlim( 1e0, 1.1e3)

ax0.set_ylabel('$ g \; - \; r $', fontsize = 15,)
ax0.set_ylim( 0.8, 1.65 )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 13)
ax1.set_xlim( 1e0, 1.1e3)

# ax1.set_ylim( 0.8, 1.2 )
# ax1.set_ylabel('$ High \, M_{\\ast}^{\\mathrm{BCG}} \; / \; Low \, M_{\\ast}^{\\mathrm{BCG}} $', fontsize = 13,)

# ax1.set_ylim( 0.8, 1.2 )
# ax1.set_ylabel('$ High \, \\lambda \; / \; Low \, \\lambda $', fontsize = 13,)

ax1.set_ylim( 0.9, 1.35 )
ax1.set_ylabel('$ High \, t_{\\mathrm{age} } \; / \; Low \, t_{\\mathrm{age} } $', fontsize = 13,)

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax0.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/color_difference.png', dpi = 300)
plt.close()
