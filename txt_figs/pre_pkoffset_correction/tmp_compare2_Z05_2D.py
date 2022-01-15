import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

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

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
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
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_pros/'
out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/BG_estimate/'

color_s = ['r', 'g', 'b']

cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass', 'tot-BCG-star-Mass' ]

fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$', 'low $M_{\\ast}$ + high $M_{\\ast}$']

### === ### figs and compare
# low mass bin
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []
for kk in range( 3 ):

	with h5py.File( out_path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( mag_sb )
	nbg_low_err.append( mag_err )

low_r, low_sb, low_err = [], [], [] 
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_low_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	low_r.append( tt_r )
	low_sb.append( mag_sb )
	low_err.append( mag_err )

# higher mass sample SB profiles
nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( out_path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( mag_sb )
	nbg_hi_err.append( mag_err )

hi_r, hi_sb, hi_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_high_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	hi_r.append( tt_r )
	hi_sb.append( mag_sb )
	hi_err.append( mag_err )

# low + high mass bin
nbg_tot_r, nbg_tot_sb, nbg_tot_err = [], [], []

for kk in range( 3 ):

	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band_diag-fit-BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_tot_r.append( tt_r )
	nbg_tot_sb.append( mag_sb )
	nbg_tot_err.append( mag_err )

# Z05 result
Z05_r, Z05_sb = [], []
for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4

	Z05_r.append( R_obs )
	Z05_sb.append( SB_obs )

last_Z05_r, last_Z05_sb = [], []
for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_sub_unmask.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4

	last_Z05_r.append( R_obs )
	last_Z05_sb.append( SB_obs )

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
L_pix = Da_ref * 10**3 * pixel / rad2asec
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
"""
for kk in range( 3 ):

	plt.figure()
	ax = plt.subplot(111)

	ax.plot(nbg_tot_r[kk], nbg_tot_sb[kk], ls = '-', color = color_s[kk], alpha = 0.5, label = 'this work')
	ax.fill_between(nbg_tot_r[kk], y1 = nbg_tot_sb[kk] - nbg_tot_err[kk], y2 = nbg_tot_sb[kk] + nbg_tot_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(Z05_r[kk], Z05_sb[kk], ls = '-.', color = 'k', alpha = 0.5, label = 'Z05, Raw BCG + ICL',)
	ax.plot( last_Z05_r[kk], last_Z05_sb[kk], ls = '--', color = 'k', alpha = 0.5, label = 'Z05, Pure BCG + ICL',)

	ax.annotate(text = '%s band' % band[kk], xy = (0.80, 0.60), xycoords = 'axes fraction', color = 'k', fontsize = 15,)
	ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.75, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')

	ax.set_xlim( 1e0, 1e3)
	ax.set_ylim( 20, 34 )

	ax.set_xscale('log')
	ax.invert_yaxis()
	ax.set_xlabel('R [kpc]', fontsize = 15, )
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax.set_ylabel('SB [mag / $arcsec^2$]', fontsize = 15, )
	ax.legend( loc = 3, fontsize = 15, frameon = False)

	plt.savefig('/home/xkchen/figs/mass-bin_%s-band_sample_BG-sub-SB_compare.png' % band[kk], dpi = 300)
	plt.close()
"""
raise

### === ### 2D flux and signal
from scipy import ndimage

rand_path = '/home/xkchen/mywork/ICL/code/ref_BG_profile/'
img_path = '/home/xkchen/mywork/ICL/code/photo_z_match_SB/'

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
L_pix = Da_ref * 10**3 * pixel / rad2asec

R1Mpc = 1000 / L_pix
R2Mpc = 2000 / L_pix
R3Mpc = 3000 / L_pix

"""
## r+i image case
for mass_dex in range( 3 ):

	## r band img
	with h5py.File( img_path + 'photo-z_match_%s_r-band_Mean_jack_img_z-ref.h5' % cat_lis[mass_dex], 'r') as f:
		r_band_img = np.array( f['a'] )
	with h5py.File( img_path + '%s_r-band_stack_test_rms.h5' % cat_lis[mass_dex], 'r') as f:
		r_band_rms = np.array( f['a'] )

	inves_r_rms2 = 1 / r_band_rms**2 

	## i band img
	with h5py.File( img_path + 'photo-z_match_%s_i-band_Mean_jack_img_z-ref.h5' % cat_lis[mass_dex], 'r') as f:
		i_band_img = np.array( f['a'] )
	with h5py.File( img_path + '%s_i-band_stack_test_rms.h5' % cat_lis[mass_dex], 'r') as f:
		i_band_rms = np.array( f['a'] )

	## random imgs
	with h5py.File( rand_path + 'random_r-band_rand-stack_Mean_jack_img_z-ref-aveg.h5', 'r') as f:
		r_rand_img = np.array( f['a'])

	BG_file = out_path + 'photo-z_%s_r-band_BG-profile_params_diag-fit.csv' % cat_lis[ mass_dex ]
	cat = pds.read_csv( BG_file )
	r_offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	r_sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	off_r_band_rand_img = r_rand_img / pixel**2 - r_offD + r_sb_2Mpc

	with h5py.File( rand_path + 'random_i-band_rand-stack_Mean_jack_img_z-ref-aveg.h5', 'r') as f:
		i_rand_img = np.array( f['a'])

	BG_file = out_path + 'photo-z_%s_i-band_BG-profile_params_diag-fit.csv' % cat_lis[ mass_dex ]
	cat = pds.read_csv( BG_file )
	i_offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	i_sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	off_i_band_rand_img = i_rand_img / pixel**2 - i_offD + i_sb_2Mpc

	inves_i_rms2 = 1 / i_band_rms**2

	r_BG_sub_img = r_band_img / pixel**2 - off_r_band_rand_img
	i_BG_sub_img = i_band_img / pixel**2 - off_i_band_rand_img

	cen_x, cen_y = np.int( r_band_img.shape[1] / 2 ), np.int( r_band_img.shape[0] / 2 )
	weit_img = ( r_BG_sub_img * inves_r_rms2 + i_BG_sub_img * inves_i_rms2 ) / ( inves_r_rms2 + inves_i_rms2 )

	###
	#cut_L = np.int( 2e3 / L_pix )
	cut_L = np.int( 1e3 / L_pix )

	cut_img = weit_img[ cen_y - cut_L: cen_y + cut_L, cen_x - cut_L: cen_x + cut_L ]

	filt_img_0 = ndimage.gaussian_filter( cut_img, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( cut_img, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( cut_img, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( cut_img, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( cut_img, sigma = 21,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	## color_lis
	color_str = []
	for jj in range( 7 ):
		color_str.append( mpl.cm.autumn_r(jj / 6) )

	me_map = mpl.colors.ListedColormap( color_str )
	c_bounds = [ 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 32.5]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	fig = plt.figure()
	ax = fig.add_axes([ 0.05, 0.10, 0.90, 0.80])
	ax1 = fig.add_axes([ 0.82, 0.10, 0.02, 0.80])

	ax.set_title( '%s, r+i stacking image' % fig_name[mass_dex] )

	ax.imshow( cut_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax.contour( mag_map_0, origin = 'lower', levels = [26, 100], alpha = 0.75,
		colors = [ color_str[0], color_str[-1] ] )
	#plt.clabel( cs, inline = True, fontsize = 6, fmt = '%.1f', )

	cs = ax.contour( mag_map_1, origin = 'lower', levels = [27, 100], alpha = 0.75, 
		colors = [ color_str[1], color_str[-1] ] )
	#plt.clabel( cs, inline = True, fontsize = 6, fmt = '%.1f',)

	cs = ax.contour( mag_map_2, origin = 'lower', levels = [28, 100], alpha = 0.75, 
		colors = [ color_str[2], color_str[-1] ] )
	#plt.clabel( cs, inline = True, fontsize = 6, fmt = '%.1f', colors = 'k')

	cs = ax.contour( mag_map_3, origin = 'lower', levels = [29, 100], alpha = 0.75, 
		colors = [ color_str[3], color_str[-1] ] )
	#plt.clabel( cs, inline = True, fontsize = 6, fmt = '%.1f', colors = 'k')

	cs = ax.contour( mag_map_4, origin = 'lower', levels = [30, 32, 100,], alpha = 0.75, 
		colors = [ color_str[4], color_str[5], color_str[-1] ] )
	#plt.clabel( cs, inline = True, fontsize = 6, fmt = '%.1f', colors = 'k')

	cbs = mpl.colorbar.ColorbarBase( ax = ax1, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 32],
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '32'] )

	clust = Circle(xy = (cut_L, cut_L), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax.add_patch(clust)
	clust = Circle(xy = (cut_L, cut_L), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5, label = '0.5Mpc')
	ax.add_patch(clust)

	ax.set_xlim(0, cut_L * 2)
	ax.set_ylim(0, cut_L * 2)

	## # of pixels pre 100kpc
	ax.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax.set_yticklabels( labels = [] )

	n200 = 200 / L_pix

	ticks_0 = np.arange( cut_L, 0, -1 * n200)
	ticks_1 = np.arange( cut_L, cut_L * 2, n200)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	tick_R = np.r_[ np.arange(800, 0, -200), np.arange(0, 1000, 200) ]
	tick_lis = [ '%d' % ll for ll in tick_R ]

	ax.set_xticks( ticks, minor = True, )
	ax.set_xticklabels( labels = tick_lis, minor = True,)

	ax.set_yticks( ticks, minor = True )
	ax.set_yticklabels( labels = tick_lis, minor = True,)
	ax.tick_params( axis = 'both', which = 'major', direction = 'in',)

	ax.set_xlabel( 'kpc' )
	ax.set_ylabel( 'kpc' )

	ax.legend( loc = 1, fontsize = 8)

	plt.savefig('/home/xkchen/figs/%s_r+i_stacking-img.png' % cat_lis[mass_dex], dpi = 300)
	plt.close()
"""

mass_dex = 1 # 0, 1

for kk in range( 3 ):

	## flux imgs
	with h5py.File( '/home/xkchen/mywork/ICL/code/photo_z_match_SB/' + 
		'photo-z_match_%s_%s-band_Mean_jack_img_z-ref.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
		tmp_img = np.array( f['a'])

	cen_x, cen_y = np.int( tmp_img.shape[1] / 2 ), np.int( tmp_img.shape[0] / 2 )

	## random imgs
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[kk], 'r') as f:
		rand_img = np.array( f['a'])
	xn, yn = np.int( rand_img.shape[1] / 2 ), np.int( rand_img.shape[0] / 2 )

	## BG-estimate params
	BG_file = out_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mass_dex ], band[kk])
	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	## 2D signal contour
	targ_R = np.array( [50, 100, 300, 500, 1000] )

	if mass_dex == 0:

		isnan = np.isnan( nbg_low_sb[kk] )
		id_lim = ( isnan == False ) & ( nbg_low_r[kk] <= 2e3 )
		use_r, use_sb = nbg_low_r[kk][ id_lim ], nbg_low_sb[kk][ id_lim ]
		use_sb_err = nbg_low_err[kk][ id_lim ]

		pre_r, pre_sb, pre_sb_err = low_r[kk], low_sb[kk], low_err[kk]

	if mass_dex == 1:

		isnan = np.isnan( nbg_hi_sb[kk] )
		id_lim = ( isnan == False ) & ( nbg_hi_r[kk] <= 2e3 )
		use_r, use_sb = nbg_hi_r[kk][ id_lim ], nbg_hi_sb[kk][ id_lim ]
		use_sb_err = nbg_hi_err[kk][ id_lim ]

		pre_r, pre_sb, pre_sb_err = hi_r[kk], hi_sb[kk], hi_err[kk]

	intep_F_0 = interp.interp1d( pre_r, pre_sb, kind = 'cubic')
	levels_0 = intep_F_0( targ_R )

	intep_F_1 = interp.interp1d( use_r, use_sb, kind = 'cubic',)
	levels_1 = intep_F_1( targ_R )

	color_lis = []
	for jj in (0, 2, 4, 6, 10):
		color_lis.append( mpl.cm.autumn_r( jj / 10) )

	color_lis_1 = []
	for jj in (0, 1, 2, 3, 4, 6, 10):
		color_lis_1.append( mpl.cm.autumn_r( jj / 10) )

	levels = [26, 28, 30, 32, 34]

	## 2D signal contour
	cut_L = np.int( 1.5e3 / L_pix )

	cut_img = tmp_img[ cen_y - cut_L: cen_y + cut_L, cen_x - cut_L: cen_x + cut_L ] / pixel**2
	cut_BG_sub_img = BG_sub_img[ cen_y - cut_L: cen_y + cut_L, cen_x - cut_L: cen_x + cut_L ]

	filt_img = ndimage.gaussian_filter( cut_img, sigma = 45,)
	filt_mag = 22.5 - 2.5 * np.log10( filt_img )


	filt_img_0 = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 45,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	## figs
	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])
	ax2 = fig.add_axes([0.40, 0.10, 0.02, 0.80])

	ax0.set_title( '2D signal distribution')
	ax1.set_title( 'SB profile',)

	tf = ax0.imshow( cut_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax0.contour( filt_mag, origin = 'lower', levels = levels_0, colors = color_lis, alpha = 0.75,)
	# plt.clabel( cs, inline = False, fontsize = 6, fmt = '%.2f', colors = 'k',)

	me_map = mpl.colors.ListedColormap( color_lis )
	medi_lines = 0.5 * ( levels_0[1:] + levels_0[:-1] )

	c_bounds = np.r_[ levels_0[0] - 0.05, medi_lines, levels_0[-1] + 0.05 ]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = levels_0,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in levels_0] )

	clust = Circle( xy = (cut_L, cut_L), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax0.add_patch(clust)
	clust = Circle( xy = (cut_L, cut_L), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5, label = '0.5Mpc')
	ax0.add_patch(clust)
	clust = Circle( xy = (cut_L, cut_L), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5, label = '2Mpc')
	ax0.add_patch(clust)

	ax0.legend( loc = 3, fontsize = 8)
	ax0.set_xlim(0, cut_L * 2)
	ax0.set_ylim(0, cut_L * 2)

	## # of pixels pre 100kpc
	ax0.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax0.set_yticklabels( labels = [] )

	n300 = 300 / L_pix

	ticks_0 = np.arange( cut_L, 0, -1 * n300)
	ticks_1 = np.arange( cut_L, cut_L * 2, n300)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	tick_R = np.r_[ np.arange(1200, 0, -300), np.arange(0, 1500, 300) ]
	tick_lis = [ '%d' % ll for ll in tick_R ]

	ax0.set_xticks( ticks, minor = True, )
	ax0.set_xticklabels( labels = tick_lis, minor = True,)

	ax0.set_yticks( ticks, minor = True )
	ax0.set_yticklabels( labels = tick_lis, minor = True,)
	ax0.tick_params( axis = 'both', which = 'major', direction = 'in',)

	ax0.set_xlabel( 'kpc' )
	ax0.set_ylabel( 'kpc' )

	ax1.plot( pre_r, pre_sb, ls = '-', color = color_s[kk], alpha = 0.45, label = fig_name[ mass_dex ] + ',' + band[kk],)
	ax1.fill_between( pre_r, y1 = pre_sb - pre_sb_err, y2 = pre_sb + pre_sb_err, color = color_s[kk], alpha = 0.12,)

	ax1.set_xlim( 1e1, 2e3)
	ax1.set_ylim( 22, 30,)
	ax1.invert_yaxis()

	ax1.legend( loc = 1)
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')
	ax1.set_ylabel('SB [mag / $arcsec^2$]')
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.savefig('/home/xkchen/figs/%s_%s-band_pre-BG-sub_signal.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()


	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])
	ax2 = fig.add_axes([0.40, 0.10, 0.02, 0.80])

	ax0.set_title( '2D signal distribution')
	ax1.set_title( 'SB profile',)

	tf = ax0.imshow( cut_BG_sub_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax0.contour( mag_map_0, origin = 'lower', levels = [26, 100], alpha = 0.75, colors = [ color_lis_1[0], 'w'])
	#plt.clabel( cs, inline = False, fontsize = 6, fmt = '%.1f', colors = 'k',)

	cs = ax0.contour( mag_map_0, origin = 'lower', levels = [27, 100], alpha = 0.75, colors = [ color_lis_1[1], 'w'])
	#plt.clabel( cs, inline = False, fontsize = 6, fmt = '%.1f', colors = 'k',)

	cs = ax0.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis_1[2], 'w'], alpha = 0.75,)
	#plt.clabel( cs, inline = False, fontsize = 6, fmt = '%.1f', colors = 'k',)

	cs = ax0.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis_1[3], 'w'], alpha = 0.75,)
	#plt.clabel( cs, inline = False, fontsize = 6, fmt = '%.1f', colors = 'k',)

	cs = ax0.contour( mag_map_4, origin = 'lower', levels = [30, 32, 34, 100,], 
		colors = [ color_lis_1[4], color_lis_1[5], color_lis_1[6], 'w'], alpha = 0.75,)
	#plt.clabel( cs, inline = False, fontsize = 6, fmt = '%.1f', colors = 'k',)


	me_map = mpl.colors.ListedColormap( color_lis_1 )
	c_bounds = [25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 32.5, 34.5]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )
	cbs = mpl.colorbar.ColorbarBase( ax = ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 32, 34], 
		spacing = 'proportional', orientation = 'vertical')
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '32', '34'] )

	clust = Circle( xy = (cut_L, cut_L), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax0.add_patch(clust)
	clust = Circle( xy = (cut_L, cut_L), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5, label = '0.5Mpc')
	ax0.add_patch(clust)
	clust = Circle( xy = (cut_L, cut_L), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5, label = '2Mpc')
	ax0.add_patch(clust)

	ax0.legend( loc = 3, fontsize = 8)
	ax0.set_xlim(0, cut_L * 2)
	ax0.set_ylim(0, cut_L * 2)

	## # of pixels pre 100kpc
	ax0.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax0.set_yticklabels( labels = [] )

	n300 = 300 / L_pix

	ticks_0 = np.arange( cut_L, 0, -1 * n300)
	ticks_1 = np.arange( cut_L, cut_L * 2, n300)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	tick_R = np.r_[ np.arange(1200, 0, -300), np.arange(0, 1500, 300) ]
	tick_lis = [ '%d' % ll for ll in tick_R ]

	ax0.set_xticks( ticks, minor = True, )
	ax0.set_xticklabels( labels = tick_lis, minor = True,)

	ax0.set_yticks( ticks, minor = True )
	ax0.set_yticklabels( labels = tick_lis, minor = True,)
	ax0.tick_params( axis = 'both', which = 'major', direction = 'in',)

	ax0.set_xlabel( 'kpc' )
	ax0.set_ylabel( 'kpc' )


	ax1.plot(use_r, use_sb, ls = '-', color = color_s[kk], alpha = 0.5, label = fig_name[ mass_dex ] + ',' + band[kk],)
	ax1.fill_between(use_r, y1 = use_sb - use_sb_err, y2 = use_sb + use_sb_err, color = color_s[kk], alpha = 0.12,)

	ax1.set_xlim( 1e1, 2e3)
	ax1.set_ylim( 22, 34,)
	ax1.invert_yaxis()

	ax1.legend( loc = 1)
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')
	ax1.set_ylabel('SB [mag / $arcsec^2$]')
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.savefig('/home/xkchen/figs/%s_%s-band_pos-BG-sub_signal.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()

raise

for kk in range( 3 ):

	## flux imgs
	with h5py.File( '/home/xkchen/mywork/ICL/code/photo_z_match_SB/' + 
		'photo-z_match_%s_%s-band_Mean_jack_img_z-ref.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
		tmp_img = np.array( f['a'])
	cen_x, cen_y = np.int( tmp_img.shape[1] / 2 ), np.int( tmp_img.shape[0] / 2 )

	idnn = np.isnan( tmp_img )
	idy_lim, idx_lim = np.where(idnn == False)
	x_lo_lim, x_up_lim = idx_lim.min(), idx_lim.max()
	y_lo_lim, y_up_lim = idy_lim.min(), idy_lim.max()

	## random imgs
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[kk], 'r') as f:
		rand_img = np.array( f['a'])
	xn, yn = np.int( rand_img.shape[1] / 2 ), np.int( rand_img.shape[0] / 2 )

	idnn = np.isnan( rand_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
	y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()

	## BG-estimate params
	BG_file = out_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mass_dex ], band[kk])
	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	idnn = np.isnan( BG_sub_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()


	cut_img = tmp_img[ y_lo_lim: y_up_lim + 1, x_lo_lim: x_up_lim + 1 ] / pixel**2
	id_nan = np.isnan( cut_img )
	cut_img[id_nan] = 0.

	cut_rand = rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ] / pixel**2
	id_nan = np.isnan( cut_rand )
	cut_rand[id_nan] = 0.

	cut_off_rand = shift_rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ]
	id_nan = np.isnan( cut_off_rand )
	cut_off_rand[id_nan] = 0.

	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_BG_sub_img )
	cut_BG_sub_img[id_nan] = 0.

	### figs of 2D signal
	color_str = []
	for jj in range( 9 ):
		color_str.append( mpl.cm.autumn_r( jj / 9 ) )

	color_lis = []
	for jj in np.arange(0, 90, 10):
		color_lis.append( mpl.cm.rainbow_r( jj / 80 ) )


	filt_rand = ndimage.gaussian_filter( cut_rand, sigma = 65,)
	filt_rand_mag = 22.5 - 2.5 * np.log10( filt_rand )

	filt_off_rand = ndimage.gaussian_filter( cut_off_rand, sigma = 65,)
	filt_off_rand_mag = 22.5 - 2.5 * np.log10( filt_off_rand )


	filt_img = ndimage.gaussian_filter( cut_img, sigma = 65,)
	filt_mag = 22.5 - 2.5 * np.log10( filt_img )

	filt_BG_sub_img = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 65,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )


	fig = plt.figure( figsize = (18, 12) )
	ax0 = fig.add_axes( [0.03, 0.55, 0.40, 0.40] )
	cb_ax0 = fig.add_axes( [0.41, 0.55, 0.02, 0.40] )

	ax1 = fig.add_axes( [0.52, 0.55, 0.40, 0.40] )
	cb_ax1 = fig.add_axes( [0.90, 0.55, 0.02, 0.40] )

	ax2 = fig.add_axes( [0.03, 0.05, 0.40, 0.40] )
	cb_ax2 = fig.add_axes( [0.41, 0.05, 0.02, 0.40] )

	ax3 = fig.add_axes( [0.52, 0.05, 0.40, 0.40] )
	cb_ax3 = fig.add_axes( [0.90, 0.05, 0.02, 0.40] )

	# if kk == 0:
	# 	levels_0 = np.linspace( 28.54, 28.64, 6)
	# if kk == 1:
	# 	levels_0 = np.linspace( 28.77, 28.86, 6)
	# if kk == 2:
	# 	levels_0 = np.linspace( 28.15, 28.45, 6)

	levels_0 = np.linspace(28, 29, 6)

	## cluster imgs before BG subtract
	ax0.set_title( 'stacking cluster image' )
	tf = ax0.imshow( cut_img, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax0.contour( filt_mag, origin = 'lower',  levels = levels_0, colors = color_str[:6], extent = (0, x_up_lim + 1 - x_lo_lim, 0, y_up_lim + 1 - y_lo_lim ), )

	#c_bounds = np.r_[ levels_0[0] - 0.01, levels_0 + 0.01 ]
	c_bounds = np.r_[ levels_0[0] - 0.1, levels_0 + 0.1 ]
	me_map = mpl.colors.ListedColormap( color_str[:6] )
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax0, cmap = me_map, norm = norm, extend = 'neither', ticks = levels_0,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in levels_0] )

	clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)

	ax0.set_xlim( 0, x_up_lim + 1 - x_lo_lim )
	ax0.set_ylim( 0, y_up_lim + 1 - y_lo_lim )

	ax0.set_xticklabels( labels = [] )
	ax0.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_lim, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_lim, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax0.set_xticks( x_ticks, minor = True, )
	ax0.set_xticklabels( labels = tick_lis, minor = True,)
	ax0.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_lim, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_lim, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax0.set_yticks( y_ticks, minor = True )
	ax0.set_yticklabels( labels = tick_lis, minor = True,)
	ax0.set_ylabel( 'Mpc' )
	ax0.tick_params( axis = 'both', which = 'major', direction = 'in',)

	## cluster imgs after BG-subtraction
	ax2.set_title( 'stacking cluster image - background image')	
	tf = ax2.imshow( cut_BG_sub_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	# dd_lis = np.arange(26, 35, 1)
	# dt_lis = np.r_[ dd_lis[:5], 32, 34 ]

	# tp_c_lis = color_lis[:5]
	# tp_c_lis.append( color_lis[6] )
	# tp_c_lis.append( color_lis[8] )

	# cs_0 = ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = dt_lis, colors = tp_c_lis, alpha = 0.35, 
	# 	extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut), )

	# me_map = mpl.colors.ListedColormap( color_lis )
	# c_bounds = np.r_[ dd_lis[0] - 0.5, dd_lis + 0.5]
	# norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )
	# cbs = mpl.colorbar.ColorbarBase( ax = cb_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 32, 34], 
	# 	spacing = 'proportional', orientation = 'vertical')
	# cbs.set_label( 'SB [mag / $arcsec^2$]' )
	# cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '32', '34'] )

	cs = ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = levels_0, colors = color_str[:6], 
		extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut ), )

	#c_bounds = np.r_[ levels_0[0] - 0.01, levels_0 + 0.01 ]
	c_bounds = np.r_[ levels_0[0] - 0.1, levels_0 + 0.1 ]
	me_map = mpl.colors.ListedColormap( color_str[:6] )
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = levels_0,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in levels_0] )


	clust = Circle( xy = (cen_x - x_lo_cut, cen_y - y_lo_cut), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax2.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_cut, cen_y - y_lo_cut), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5,label = '0.5Mpc')
	ax2.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_cut, cen_y - y_lo_cut), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5, label = '2Mpc')
	ax2.add_patch(clust)

	ax2.legend( loc = 1 )
	ax2.set_xlim( 0, x_up_cut + 1 - x_lo_cut )
	ax2.set_ylim( 0, y_up_cut + 1 - y_lo_cut )

	## # of pixels pre 100kpc
	ax2.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax2.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_cut, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_cut, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax2.set_xticks( x_ticks, minor = True, )
	ax2.set_xticklabels( labels = tick_lis, minor = True,)
	ax2.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_cut, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_cut, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax2.set_yticks( y_ticks, minor = True )
	ax2.set_yticklabels( labels = tick_lis, minor = True,)
	ax2.set_ylabel( 'Mpc' )
	ax2.tick_params( axis = 'both', which = 'major', direction = 'in',)

	ax2.set_xlabel( 'Mpc' )
	ax2.set_ylabel( 'Mpc' )

	### random image
	if kk == 0:
		tt_lis = np.linspace( 28.54, 28.64, 6)
	if kk == 1:
		tt_lis = np.linspace( 28.77, 28.86, 6)
	if kk == 2:
		tt_lis = np.linspace( 28.15, 28.45, 6)

	ax1.set_title( 'stacking random image' )
	ax1.imshow( cut_rand, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax1.contour( filt_rand_mag, origin = 'lower',  levels = tt_lis, colors = color_str[:6], 
		extent = (0, x_up_eff + 1 - x_lo_eff, 0, y_up_eff + 1 - y_lo_eff ), )

	me_map = mpl.colors.ListedColormap( color_str[:6] )
	c_bounds = np.r_[ tt_lis[0] - 0.01, tt_lis + 0.01]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax1, cmap = me_map, norm = norm, extend = 'neither', ticks = tt_lis,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in tt_lis] )

	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax1.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5, label = '0.5Mpc')
	ax1.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5, label = '2Mpc')
	ax1.add_patch(clust)

	ax1.set_xlim( 0, x_up_eff + 1 - x_lo_eff )
	ax1.set_ylim( 0, y_up_eff + 1 - y_lo_eff )

	# ticks set
	ax1.set_xticklabels( labels = [] )
	ax1.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_eff, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_eff, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax1.set_xticks( x_ticks, minor = True, )
	ax1.set_xticklabels( labels = tick_lis, minor = True,)
	ax1.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_eff, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_eff, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax1.set_yticks( y_ticks, minor = True )
	ax1.set_yticklabels( labels = tick_lis, minor = True,)
	ax1.set_ylabel( 'Mpc' )
	ax1.tick_params( axis = 'both', which = 'major', direction = 'in',)


	### shift random ( based on BG estimate fitting)
	ax3.set_title( 'background image [stacking random image - C]' )
	ax3.imshow( cut_off_rand, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax3.contour( filt_off_rand_mag, origin = 'lower',  levels = tt_lis, colors = color_str[:6], 
		extent = (0, x_up_eff + 1 - x_lo_eff, 0, y_up_eff + 1 - y_lo_eff),)

	me_map = mpl.colors.ListedColormap( color_str[:6] )
	c_bounds = np.r_[ tt_lis[0] - 0.01, tt_lis + 0.01]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax3, cmap = me_map, norm = norm, extend = 'neither', ticks = tt_lis,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in tt_lis] )


	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5,)
	ax3.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5,)
	ax3.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5,)
	ax3.add_patch(clust)

	ax3.set_xlim( 0, x_up_eff + 1 - x_lo_eff )
	ax3.set_ylim( 0, y_up_eff + 1 - y_lo_eff )

	# ticks set
	ax3.set_xticklabels( labels = [] )
	ax3.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_eff, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_eff, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax3.set_xticks( x_ticks, minor = True, )
	ax3.set_xticklabels( labels = tick_lis, minor = True,)
	ax3.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_eff, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_eff, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax3.set_yticks( y_ticks, minor = True )
	ax3.set_yticklabels( labels = tick_lis, minor = True,)
	ax3.set_ylabel( 'Mpc' )
	ax3.tick_params( axis = 'both', which = 'major', direction = 'in',)

	plt.savefig('/home/xkchen/figs/%s_%s-band_2D_flux_compare.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()

raise
