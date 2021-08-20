import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import scipy.signal as signal

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from fig_out_module import arr_jack_func

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
band = ['r', 'g', 'i']
# psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec
psf_FWHM = 1.32

### === ### data load and figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

## fixed richness samples
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# file_s = 'BCG_Mstar_bin'
# cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'

# cat_lis = ['younger', 'older']
# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
# file_s = 'BCG_age_bin'
# cat_path = '/home/xkchen/mywork/ICL/data/cat_z_form/age_bin_cat/gri_common_cat/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/BGs/'


## fixed BCG Mstar samples
# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'rich_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
file_s = 'age_bin_fixed_BCG_M'
cat_path = '/home/xkchen/tmp_run/data_files/figs/'
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'


#... SB profile
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

nbg_low_r = np.array( nbg_low_r )
nbg_low_r = nbg_low_r / 1e3

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

nbg_hi_r = np.array( nbg_hi_r )
nbg_hi_r = nbg_hi_r / 1e3


#...color profile
mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
hi_gi, hi_gi_err = np.array( mu_dat['g-i'] ), np.array( mu_dat['g-i_err'] )

hi_gr = signal.savgol_filter( hi_gr, 7, 3)
hi_gi = signal.savgol_filter( hi_gi, 7, 3)
hi_c_r = hi_c_r / 1e3

mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
lo_gi, lo_gi_err = np.array( mu_dat['g-i'] ), np.array( mu_dat['g-i_err'] )

lo_gr = signal.savgol_filter( lo_gr, 7, 3)
lo_gi = signal.savgol_filter( lo_gi, 7, 3)
lo_c_r = lo_c_r / 1e3


#...color slope
c_dat = pds.read_csv( BG_path + '%s_color_slope.csv' % cat_lis[0],)
lo_dgr, lo_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )
lo_dc_r = np.array( c_dat['R_kpc'] )
lo_dc_r = lo_dc_r / 1e3

c_dat = pds.read_csv( BG_path + '%s_color_slope.csv' % cat_lis[1],)
hi_dgr, hi_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )
hi_dc_r = np.array( c_dat['R_kpc'] )
hi_dc_r = hi_dc_r / 1e3


#... sample properties
hi_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )
hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['BCG_age'] )

lo_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )
lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['BCG_age'] )


#... fitting test
# band_str = 'gri'
out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'

#...Mass profile
# dat = pds.read_csv( BG_path + '%s_gi-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )
lo_R = lo_R / 1e3

# dat = pds.read_csv( BG_path + '%s_gi-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )
hi_R = hi_R / 1e3


#... total sample for comparison
# dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/' + 
# 	'photo-z_tot-BCG-star-Mass_gi-band-based_corrected_aveg-jack_mass-Lumi.csv')
# tot_R, tot_surf_m, tot_surf_m_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv',)
tot_R = np.array(dat['R'])
tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

tot_R  = tot_R / 1e3
#... estimate ratio between sub-sample and total samples
interp_M_f = interp.interp1d( tot_R, tot_surf_m, kind = 'linear',)

# calib_cat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_gi-band-based_M_calib-f.csv' % file_s )
# lo_shift, hi_shift = np.array(calib_cat['low_mean_devi'])[0], np.array(calib_cat['high_mean_devi'])[0]

calib_cat = pds.read_csv( out_path + '%s_gri-band-based_M_calib-f.csv' % file_s )
lo_shift, hi_shift = np.array(calib_cat['low_medi_devi'])[0], np.array(calib_cat['high_medi_devi'])[0]

M_offset = [ lo_shift, hi_shift ]
"""
for mm in range( 2 ):

	N_samples = 30

	# jk_sub_m_file = BG_path + '%s_gi-band-based_' % cat_lis[mm] + 'jack-sub-%d_mass-Lumi.csv'
	jk_sub_m_file = out_path + '%s_gri-band-based_' % cat_lis[mm] + 'jack-sub-%d_mass-Lumi.csv'

	tmp_r, tmp_ratio = [], []

	for nn in range( N_samples ):

		o_dat = pds.read_csv( jk_sub_m_file % nn,)

		tt_r = np.array( o_dat['R'] )
		tt_M = np.array( o_dat['surf_mass'] )

		tt_r = tt_r / 1e3
		tt_M = tt_M * 10**M_offset[mm]

		idx_lim = ( tt_r >= np.nanmin( tot_R ) ) & ( tt_r <= np.nanmax( tot_R ) )

		lim_R = tt_r[ idx_lim ]
		lim_M = tt_M[ idx_lim ]

		com_M = interp_M_f( lim_R )

		sub_ratio = np.zeros( len(tt_r),)
		sub_ratio[ idx_lim ] = lim_M / com_M

		sub_ratio[ idx_lim == False ] = np.nan

		tmp_r.append( tt_r * 1e3 )
		tmp_ratio.append( sub_ratio )

	aveg_R, aveg_ratio, aveg_ratio_err = arr_jack_func( tmp_ratio, tmp_r, N_samples)[:3]

	keys = ['R', 'M/M_tot', 'M/M_tot-err']
	values = [ aveg_R, aveg_ratio, aveg_ratio_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	# out_data.to_csv(BG_path + '%s_gi-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[mm],)
	out_data.to_csv( out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[mm],)
"""

# lo_eat_dat = pds.read_csv( BG_path + '%s_gi-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[0] )
lo_eat_dat = pds.read_csv( out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[0] )

lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['M/M_tot']), np.array(lo_eat_dat['M/M_tot-err'])


# hi_eat_dat = pds.read_csv( BG_path + '%s_gi-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[1] )
hi_eat_dat = pds.read_csv( out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[1] )

hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['M/M_tot']), np.array(hi_eat_dat['M/M_tot-err'])


Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3

#...figs
fig = plt.figure( figsize = (12.628, 12.628) )
ax0 = fig.add_axes([0.08, 0.56, 0.42, 0.42])
ax1 = fig.add_axes([0.57, 0.56, 0.42, 0.42])

# ax2 = fig.add_axes([0.08, 0.07, 0.41, 0.41])
# ax3 = fig.add_axes([0.56, 0.07, 0.41, 0.41])

ax2 = fig.add_axes([0.08, 0.21, 0.42, 0.28])
bot_ax2 = fig.add_axes([0.08, 0.07, 0.42, 0.14])

# ax3 = fig.add_axes([0.57, 0.07, 0.42, 0.42])
ax3 = fig.add_axes([0.57, 0.21, 0.42, 0.28])
bot_ax3 = fig.add_axes([0.57, 0.07, 0.42, 0.14])


if file_s == 'BCG_Mstar_bin':

	sub_ax0 = fig.add_axes( [ 0.35, 0.63, 0.12, 0.12] )
	sub_ax1 = fig.add_axes( [ 0.35, 0.62, 0.12, 0.01] )
	# BCG-M bin, fixed richness
	_point_rich = np.array([20, 30, 40, 50, 100, 200])
	line_divi = 0.446 * np.log10( _point_rich ) + 10.518

	ax0.scatter( lo_rich, lo_lgM, s = 15, marker = '.', c = lo_age, cmap = 'bwr', alpha = 0.75, vmin = 1, vmax = 11,)
	ax0.scatter( hi_rich, hi_lgM, s = 15, marker = '.', c = hi_age, cmap = 'bwr', alpha = 0.75, vmin = 1, vmax = 11,)
	ax0.plot( _point_rich, line_divi, ls = '-', color = 'k', alpha = 0.75,)

	age_edgs = np.logspace( 0, np.log10(12), 50)
	sub_ax0.hist( lo_age, bins = age_edgs, density = True, histtype = 'step', color = 'b', ls = '--', alpha = 0.75, )
	sub_ax0.hist( hi_age, bins = age_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.75, )
	sub_ax0.set_xlim( 1, 11 )

	cmap = mpl.cm.bwr
	norm = mpl.colors.Normalize( vmin = 1, vmax = 11 )

	c_ticks = np.array([1, 3, 5, 7, 9, 11])
	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( '$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 14,)

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.0f' % ll for ll in c_ticks ] )

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.set_xticks( [] )

	ax0.set_xscale( 'log' )
	ax0.set_xlabel( '$ \\lambda $' , fontsize = 18,)
	ax0.set_ylabel( '$ {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} \; [M_{\\odot} \, / \, h^2] $', fontsize = 18,)

	ax0.text( 97, 11.55, s = 'High $ M_{\\ast}^{\\mathrm{BCG}} $', fontsize = 18, rotation = 9,)
	ax0.text( 100, 11.25, s = 'Low $ M_{\\ast}^{\\mathrm{BCG}} $', fontsize = 18, rotation = 9,)

	ax0.set_xlim( 20, 200 )
	xtick_arr = [20, 30, 40, 50, 100, 200]
	tick_lis = [ '%d' % ll for ll in xtick_arr ]
	ax0.set_xticks( xtick_arr, minor = True,)
	ax0.set_xticklabels( labels = tick_lis, fontsize = 18, minor = True,)

	ax0.set_ylim( 10.31, 12.0 )
	ytick_arr = [10.5, 11, 11.5, 12]
	tick_lis = [ '%.1f' % ll for ll in ytick_arr ]
	ax0.set_yticks( ytick_arr, )
	ax0.set_yticklabels( labels = tick_lis, fontsize = 18,)	

	ax0.set_xticks( [ 100 ] )
	ax0.set_xticklabels( labels = ['100'], fontsize = 18,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in',)

if file_s == 'BCG_age_bin':

	sub_ax0 = fig.add_axes( [ 0.35, 0.63, 0.12, 0.12] )
	sub_ax1 = fig.add_axes( [ 0.35, 0.62, 0.12, 0.01] )
	# BCG-age bin, fixed richness
	div_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/age-bin_fixed-rich_divid_line.csv' )
	_point_rich, line_divi = np.array( div_dat['rich'] ), np.array( div_dat['age'] )

	ax0.scatter( lo_rich, lo_age, s = 15, c = lo_lgM, marker = '.', cmap = 'bwr', alpha = 0.75, vmin = 10, vmax = 12,)
	ax0.scatter( hi_rich, hi_age, s = 15, c = hi_lgM, marker = '.', cmap = 'bwr', alpha = 0.75, vmin = 10, vmax = 12,)

	mass_edgs = np.linspace( 10, 12, 50)
	sub_ax0.hist( lo_lgM, bins = mass_edgs, density = True, histtype = 'step', color = 'b',  ls = '--', alpha = 0.75, )
	sub_ax0.hist( hi_lgM, bins = mass_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.75, )
	sub_ax0.set_xlim( 10, 12 )

	# color bar
	cmap = mpl.cm.bwr
	norm = mpl.colors.Normalize( vmin = 10, vmax = 12 )

	c_ticks = sub_ax0.get_xticks()

	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( '$ {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} \; [M_{\\odot} \, / \, h^2] $', fontsize = 14,)

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.1f' % ll for ll in c_ticks ] )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.set_xticks( [] )

	ax0.plot( _point_rich, line_divi, ls = '-', color = 'k', alpha = 0.75,)

	ax0.set_xlim( 20, 200 )
	ax0.set_ylim( 1.69, 11)

	ax0.set_ylabel('$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 18,)
	ax0.set_xlabel('$\\lambda$', fontsize = 18,)
	ax0.set_xscale( 'log' )

	x_ticks = np.array( [ 20, 30, 40, 50, 100, 200] )
	ax0.set_xticks( x_ticks, minor = True,)
	ax0.set_xticklabels( labels = [ '%d' % ll for ll in x_ticks ], minor = True, fontsize = 18,)

	ax0.set_xticks( [ 100 ] )
	ax0.set_xticklabels( labels = [ '100' ],fontsize = 18,)

	ax0.text( 1.1e2, 8.0, s = 'High $ t_{\\mathrm{age}} $', fontsize = 18,)
	ax0.text( 1.1e2, 7.0, s = 'Low $ t_{\\mathrm{age}} $', fontsize = 18,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

if file_s == 'rich_bin_fixed_BCG_M':

	sub_ax0 = fig.add_axes( [ 0.12, 0.83, 0.13, 0.13] )
	sub_ax1 = fig.add_axes( [ 0.12, 0.82, 0.13, 0.01] )

	div_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/rich-bin_fixed-BCG-M_divid_line.csv' )
	lgM_x, line_divi = np.array( div_dat['lgM'] ), np.array( div_dat['lg_rich'] )

	ax0.scatter( lo_lgM, lo_rich, s = 15, c = lo_age, marker = '.', cmap = 'bwr', alpha = 0.75, vmin = 1, vmax = 11,)
	ax0.scatter( hi_lgM, hi_rich, s = 15, c = hi_age, marker = '.', cmap = 'bwr', alpha = 0.75, vmin = 1, vmax = 11,)

	age_edgs = np.logspace( 0, np.log10(12), 50)

	sub_ax0.hist( lo_age, bins = age_edgs, density = True, histtype = 'step', color = 'b',  ls = '--', )
	sub_ax0.hist( hi_age, bins = age_edgs, density = True, histtype = 'step', color = 'r', )

	sub_ax0.set_xlim( 1, 11 )

	cmap = mpl.cm.bwr
	norm = mpl.colors.Normalize( vmin = 1, vmax = 11 )

	c_ticks = np.array([1, 3, 5, 7, 9, 11])
	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( '$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 14,)

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.0f' % ll for ll in c_ticks ] )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.set_xticks( [] )

	ax0.plot( lgM_x, 10**line_divi, ls = '-', color = 'k', alpha = 0.75,)

	ax0.text( 11.65, 48, s = 'High $ \\lambda $', fontsize = 18, rotation = 32,)
	ax0.text( 11.7, 32, s = 'Low $ \\lambda $', fontsize = 18, rotation = 32,)

	ax0.set_ylim( 20, 200 )
	ax0.set_yscale( 'log' )

	ax0.set_xlim( 10.44, 12 )

	xtick_arr = [ 10.5, 11, 11.5, 12 ]
	tick_lis = [ '%.1f' % ll for ll in xtick_arr ]
	ax0.set_xticks( xtick_arr,)
	ax0.set_xticklabels( labels = tick_lis, fontsize = 18,)

	y_ticks = np.array( [ 20, 30, 40, 50, 100, 200] )
	ax0.set_yticks( y_ticks, minor = True,)
	ax0.set_yticklabels( labels = [ '%d' % ll for ll in y_ticks ], minor = True, fontsize = 18,)

	ax0.set_yticks( [ 100 ] )
	ax0.set_yticklabels( labels = [ '100' ],fontsize = 18,)

	ax0.set_ylabel('$ \\lambda $', fontsize = 18,)
	ax0.set_xlabel('$ {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} \; [M_{\\odot} \, / \, h^2] $', fontsize = 18,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

if file_s == 'age_bin_fixed_BCG_M':

	from matplotlib import ticker

	sub_ax0 = fig.add_axes( [ 0.362, 0.605, 0.114, 0.113] )
	sub_ax1 = fig.add_axes( [ 0.362, 0.595, 0.114, 0.010] )

	# BCG-age bin, fixed richness
	div_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/age-bin_fixed-BCG-M_divid_line.csv' )
	lgM_x, line_divi = np.array( div_dat['lgM'] ), np.array( div_dat['lg_age'] )

	#... set color for points
	_c_str = []
	for jj in range( 10 ):
		_c_str.append( mpl.cm.bwr( jj / 9 ) )

	_c_lis_lo = np.zeros( len(lo_rich) )
	_c_lis_hi = np.zeros( len(hi_rich) )
	edg_arr = [20, 23, 25, 30, 40, 50, 200]

	for ii in range( 6 ):

		if ii <= 2:
			id_x0 = (lo_rich >= edg_arr[ii] ) & (lo_rich < edg_arr[ii+1] )
			_c_lis_lo[ id_x0 ] = ii

			id_x1 = (hi_rich >= edg_arr[ii] ) & (hi_rich < edg_arr[ii+1] )
			_c_lis_hi[ id_x1 ] = ii
		else:
			id_x0 = (lo_rich >= edg_arr[ii] ) & (lo_rich < edg_arr[ii+1] )
			_c_lis_lo[ id_x0 ] = ii + 5

			id_x1 = (hi_rich >= edg_arr[ii] ) & (hi_rich < edg_arr[ii+1] )
			_c_lis_hi[ id_x1 ] = ii + 5

	_c_lis_lo = _c_lis_lo / 9
	_c_lis_hi = _c_lis_hi / 9

	ax0.scatter( lo_lgM, lo_age, s = 15, c = _c_lis_lo, marker = '.', cmap = 'bwr', alpha = 0.75, )
	ax0.scatter( hi_lgM, hi_age, s = 15, c = _c_lis_hi, marker = '.', cmap = 'bwr', alpha = 0.75, )

	lgrich_edgs = np.logspace( 1.30, 2.30, 29)

	sub_ax0.hist( lo_rich, bins = lgrich_edgs, density = True, histtype = 'step', color = 'b', ls = '--',)
	sub_ax0.hist( hi_rich, bins = lgrich_edgs, density = True, histtype = 'step', color = 'r', )

	sub_ax0.set_xlim( 2e1, 2e2 )
	sub_ax0.set_xscale('log')
	sub_ax0.set_yscale('log')

	sub_ax0.set_xticklabels( labels = [], minor = True,)
	sub_ax0.set_xticks( [100] )
	sub_ax0.set_xticklabels( labels = [],)

	sub_ax0.set_yticks( [1e-3, 1e-2] )
	sub_ax0.set_yticklabels( labels = ['$10^{-3}$', '$10^{-2}$'],)

	#... colorbar
	c_bounds = edg_arr
	me_map = mpl.colors.ListedColormap( _c_str )
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = me_map, norm = norm, extend = 'neither', ticks = edg_arr, 
		spacing = 'proportional', orientation = 'horizontal', )
	cbs.set_label( '$ \\lambda $', fontsize = 14, labelpad = -4, )
	cbs.ax.set_xticklabels( ['%d' % ll for ll in edg_arr ], fontsize = 14,)
	cbs.ax.set_xscale( 'log' )

	sub_ax1.set_xticklabels( labels = [], minor = True,)
	sub_ax1.set_xticklabels( labels = [],)

	_ticks = np.array( [20, 40, 100, 200] )
	label_str = ['%d' % ll for ll in _ticks ]
	sub_ax1.xaxis.set_major_locator( ticker.FixedLocator( _ticks ) )
	sub_ax1.xaxis.set_major_formatter( ticker.FixedFormatter( label_str ) )

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)

	ax0.plot( lgM_x, 10**line_divi, ls = '-', color = 'k',)

	ax0.set_ylabel('$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 18,)
	ax0.set_xlabel('$ {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} \; [M_{\\odot} \, / \, h^2] $', fontsize = 18,)

	ax0.set_xlim( 10.5, 12 )
	ax0.set_ylim( 1.65, 10.80 )

	x_ticks = np.array( [ 10.5, 11, 11.5, 12 ] )
	ax0.set_xticks( x_ticks )
	ax0.set_xticklabels( labels = [ '%.1f' % ll for ll in x_ticks ], fontsize = 18,)

	ax0.text( 11.62, 8.6, s = 'High $ t_{\\mathrm{age}} $', fontsize = 16, rotation = 25,)
	ax0.text( 11.67, 7.8, s = 'Low $ t_{\\mathrm{age}} $', fontsize = 16, rotation = 25,)

	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)


for kk in ( 2, 0, 1 ):

	ax1.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75,)
	ax1.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
		y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax1.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk])
	ax1.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
		y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.15,)

legend_1 = ax1.legend( [ fig_name[0], fig_name[1] ], loc = 3, frameon = False, fontsize = 18,)
legend_0 = ax1.legend( loc = 1, frameon = False, fontsize = 18,)
ax1.add_artist( legend_1 )

ax1.fill_betweenx( y = np.linspace( 19, 36, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax1.text( 3e-3, 29.5, s = 'PSF', fontsize = 18,)

ax1.set_ylim( 20, 33.5 )
ax1.invert_yaxis()

ax1.set_xlim( 3e-3, 1e0 )
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 18,)

ax1.set_xticks([ 1e-2, 1e-1, 1e0])
ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
ax1.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 18,)


# if file_s == 'rich_bin_fixed_BCG_M':

ax2.plot( lo_R, lo_surf_M, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0],)
ax2.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = line_c[0], alpha = 0.12,)
ax2.plot( hi_R, hi_surf_M, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1],)
ax2.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = line_c[1], alpha = 0.12,)

ax2.plot( tot_R, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
ax2.fill_between( tot_R, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.12,)

ax2.fill_betweenx( y = np.logspace(3, 10, 250), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)

ax2.set_xlim( 3e-3, 1e0 )
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim( 5e3, 2e9 )
ax2.legend( loc = 3, frameon = False, fontsize = 18,)
ax2.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 18,)
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

bot_ax2.plot( lo_eta_R / 1e3, lo_eta, ls = '--', color = line_c[0], alpha = 0.75,)
bot_ax2.fill_between( lo_eta_R / 1e3, y1 = lo_eta - lo_eta_err, y2 = lo_eta + lo_eta_err, color = line_c[0], alpha = 0.12,)

bot_ax2.plot( hi_eta_R / 1e3, hi_eta, ls = '-', color = line_c[1], alpha = 0.75,)
bot_ax2.fill_between( hi_eta_R / 1e3, y1 = hi_eta - hi_eta_err, y2 = hi_eta + hi_eta_err, color = line_c[1], alpha = 0.12,)

bot_ax2.plot( tot_R, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)
# bot_ax2.fill_between( tot_R, y1 = (tot_surf_m - tot_surf_m_err) / tot_surf_m, 
# 	y2 = (tot_surf_m + tot_surf_m_err) / tot_surf_m, color = 'k', alpha = 0.12,)

bot_ax2.fill_betweenx( y = np.linspace(-10, 10, 250), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)

bot_ax2.set_xlim( ax2.get_xlim() )
bot_ax2.set_xscale( 'log' )
bot_ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 18,)

bot_ax2.set_xticks([ 1e-2, 1e-1, 1e0])
bot_ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

bot_ax2.set_ylim( 0.50, 1.55 )
bot_ax2.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{ \\mathrm{All \; clusters} } $', fontsize = 18,)
bot_ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
ax2.set_xticklabels( [] )

# else:
# ax2.plot( lo_R, lo_surf_M, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0],)
# ax2.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = line_c[0], alpha = 0.12,)
# ax2.plot( hi_R, hi_surf_M, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1],)
# ax2.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = line_c[1], alpha = 0.12,)

# ax2.fill_betweenx( y = np.logspace(3, 10, 250), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)

# ax2.set_xlim( 3e-3, 1e0 )
# ax2.set_xscale('log')
# ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 18,)

# ax2.set_yscale('log')
# ax2.set_ylim( 1e4, 2e9 )
# ax2.legend( loc = 3, frameon = False, fontsize = 18,)
# ax2.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 18,)
# ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)


ax3.plot( lo_c_r, lo_gr, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax3.fill_between( lo_c_r, y1 = lo_gr - lo_gr_err, y2 = lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)

ax3.plot( hi_c_r, hi_gr, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax3.fill_between( hi_c_r, y1 = hi_gr - hi_gr_err, y2 = hi_gr + hi_gr_err, color = 'r', alpha = 0.15,)

ax3.legend( loc = 3, frameon = False, fontsize = 18,)
ax3.set_ylim( 0.8, 1.55 )
ax3.set_ylabel('$ g \; - \; r $', fontsize = 20,)

ax3.fill_betweenx( y = np.linspace(0, 5, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)

ax3.set_xlim( 3e-3, 1e0 )
ax3.set_xscale('log')
ax3.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 18,)

ax3.set_xticks([ 1e-2, 1e-1, 1e0])
ax3.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )
ax3.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

bot_ax3.plot( lo_dc_r, lo_dgr, ls = '--', color = 'b', alpha = 0.75,)
bot_ax3.fill_between( lo_dc_r, y1 = lo_dgr - lo_dgr_err, y2 = lo_dgr + lo_dgr_err, color = 'b', alpha = 0.15,)

bot_ax3.plot( hi_dc_r, hi_dgr, ls = '-', color = 'r', alpha = 0.75,)
bot_ax3.fill_between( hi_dc_r, y1 = hi_dgr - hi_dgr_err, y2 = hi_dgr + hi_dgr_err, color = 'r', alpha = 0.15,)

bot_ax3.set_ylim( -0.6, -0.0099)
bot_ax3.set_ylabel('$ {\\rm d} (g-r) \, / \, {\\rm d} \\mathcal{lg}R $', fontsize = 15, labelpad = -0.2)

bot_ax3.fill_betweenx( y = np.linspace(-1, 1, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)

bot_ax3.set_xlim( 3e-3, 1e0 )
bot_ax3.set_xscale('log')
bot_ax3.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 18,)

bot_ax3.set_xticks([ 1e-2, 1e-1, 1e0])
bot_ax3.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )
bot_ax3.tick_params( axis = 'x', which = 'both', direction = 'in', labelsize = 18,)
bot_ax3.tick_params( axis = 'y', which = 'both', direction = 'in', labelsize = 15,)
ax3.set_xticklabels( [] )

# plt.savefig('/home/xkchen/%s_result.png' % file_s, dpi = 300)
plt.savefig('/home/xkchen/%s_result.pdf' % file_s, dpi = 300)
plt.close()

