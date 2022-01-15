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

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage
import scipy.signal as signal

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov

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

### === data load
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# file_s = 'BCG_Mstar_bin'
# BG_path = '/home/xkchen/jupyter/fixed_rich/BCG_M_bin/BGs/'
# img_path = '/home/xkchen/jupyter/fixed_rich/BCG_M_bin/'

# cat_lis = ['younger', 'older']
# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
# file_s = 'BCG_age_bin'
# BG_path = '/home/xkchen/jupyter/fixed_rich/age_bin_SBs/BGs/'
# img_path = '/home/xkchen/jupyter/fixed_rich/age_bin_SBs/'

## fixed BCG Mstar samples
# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'rich_bin_fixed_BCG_M'
# BG_path = '/home/xkchen/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'
# img_path = '/home/xkchen/jupyter/fixed_BCG_M/rich_bin_SBs/'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
file_s = 'age_bin_fixed_BCG_M'
BG_path = '/home/xkchen/jupyter/fixed_BCG_M/age_bin/BGs/'
img_path = '/home/xkchen/jupyter/fixed_BCG_M/age_bin/'

rand_path = '/home/xkchen/jupyter/random_ref_SB/'

L_pix = Da_ref * 10**3 * pixel / rad2asec

n500 = 500 / L_pix
n1000 = 1000 / L_pix
n900 = 900 / L_pix
n200 = 200 / L_pix

smooth_file = '/home/xkchen/figs/smooth_arr/'

'''
for ll in range( 2 ):

	for kk in range( 3 ):

		## flux imgs
		with h5py.File( img_path + 'photo-z_match_gri-common_%s_%s-band_Mean_jack_img_z-ref_pk-off.h5' % (cat_lis[ll], band[kk]), 'r') as f:
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
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ll], band[kk])

		cat = pds.read_csv( BG_file )
		offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
		sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

		#... change flux to SB in each pixel
		shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
		BG_sub_img = tmp_img / pixel**2 - shift_rand_img

		## save the BG-sub image
		with h5py.File( smooth_file + '%s_%s-band_BG-sub_img.h5' % (cat_lis[ll], band[kk]), 'w') as f:
			f['a'] = np.array( BG_sub_img )
'''

### 2D color
band_str = 'gr'
# band_str = 'gi'

fig = plt.figure( figsize = (10.20, 4.8) )
ax1 = fig.add_axes( [0.08, 0.12, 0.40, 0.85] )
ax2 = fig.add_axes( [0.485, 0.12, 0.40, 0.85] )
sub_ax2 = fig.add_axes( [0.885, 0.12, 0.02, 0.85] )

for ll in range( 2 ):

	#.. direct flux compare
	with h5py.File( smooth_file + '%s_g-band_BG-sub_img.h5' % cat_lis[ll], 'r') as f:
		g_img = np.array( f['a'] )

	if band_str == 'gr':
		with h5py.File( smooth_file + '%s_r-band_BG-sub_img.h5' % cat_lis[ll], 'r') as f:
			r_img = np.array( f['a'] )

	if band_str == 'gi':
		with h5py.File( smooth_file + '%s_i-band_BG-sub_img.h5' % cat_lis[ll], 'r') as f:
			r_img = np.array( f['a'] )

	xn, yn = r_img.shape[1] / 2, r_img.shape[0] / 2

	idnn = np.isnan( r_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()

	idnn = np.isnan( g_img)
	g_img[idnn] = 0

	idnn = np.isnan( r_img)
	r_img[idnn] = 0

	cut_r_img = r_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	cut_g_img = g_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]


	x_peak, y_peak = np.int( r_img.shape[1] / 2 ), np.int( r_img.shape[0] / 2 )
	cen_r_img = r_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]
	cen_g_img = g_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]


	#... smooth pixel value
	filt_g_img = ndimage.gaussian_filter( cen_g_img, sigma = 31,)
	filt_r_img = ndimage.gaussian_filter( cen_r_img, sigma = 31,)

	r_mag_map = 22.5 - 2.5 * np.log10( filt_r_img )
	g_mag_map = 22.5 - 2.5 * np.log10( filt_g_img )

	# gr_2D = -2.5 * np.log10( filt_g_img / filt_r_img )
	# gr_2D = g_mag_map - r_mag_map

	# idnn = np.isnan( gr_2D )
	# gr_2D[ idnn ] = 0.


	#... smooth Mag_map
	_gr_pix = -2.5 * np.log10( cut_g_img / cut_r_img )
	# r_mag_map = 22.5 - 2.5 * np.log10( cut_r_img )
	# g_mag_map = 22.5 - 2.5 * np.log10( cut_g_img )
	# _gr_pix = g_mag_map - r_mag_map

	idnn = np.isnan( _gr_pix )
	_gr_pix[idnn] = 0.

	gr_2D = ndimage.gaussian_filter( _gr_pix, sigma = 3,)


	color_lis = []
	for jj in np.arange( 11 ):
		color_lis.append( mpl.cm.rainbow( jj / 10) )

	# if band_str == 'gr':
	# 	dd_lis = np.linspace( 0.7, 1.4, 8)
	# if band_str == 'gi':
	# 	dd_lis = np.linspace( 1.3, 2.0, 8)

	dd_lis = np.linspace(0.7, 2.1, 8)

	if ll == 0:
		ax = ax1
	if ll == 1:
		ax = ax2

	cut_px = x_peak - x_lo_cut
	cut_py = y_peak - y_lo_cut

	ax.imshow( _gr_pix, cmap = 'Greys', vmin = 0.0, vmax = 1.5, alpha = 0.75,)

	ax.contour( gr_2D, origin = 'lower',  levels = dd_lis, colors = color_lis[-8:], 
		extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut), )

	# ax.contour( gr_2D, origin = 'lower',  levels = [ dd_lis[0], 100], colors = [ color_lis[0], 'w'], 
	# 	extent = (cut_px - 3 * n500, cut_px + 3 * n500 + 1, cut_py - 3 * n500, cut_py + 3 * n500 + 1), )

	# ax.contour( gr_2D, origin = 'lower',  levels = [ dd_lis[1], 100], colors = [ color_lis[2], 'w'], 
	# 	extent = (cut_px - 3 * n500, cut_px + 3 * n500 + 1, cut_py - 3 * n500, cut_py + 3 * n500 + 1), )

	# ax.contour( gr_2D, origin = 'lower',  levels = [ dd_lis[2], 100], colors = [ color_lis[3], 'w'], 
	# 	extent = (cut_px - 3 * n500, cut_px + 3 * n500 + 1, cut_py - 3 * n500, cut_py + 3 * n500 + 1), )

	# ax.contour( gr_2D, origin = 'lower',  levels = [ dd_lis[3], 100], colors = [ color_lis[4], 'w'], 
	# 	extent = (cut_px - 3 * n500, cut_px + 3 * n500 + 1, cut_py - 3 * n500, cut_py + 3 * n500 + 1), )

	# ax.contour( gr_2D, origin = 'lower',  levels = [ dd_lis[4], 100], colors = [ color_lis[6], 'w'], 
	# 	extent = (cut_px - 3 * n500, cut_px + 3 * n500 + 1, cut_py - 3 * n500, cut_py + 3 * n500 + 1), )

	# ax.contour( gr_2D, origin = 'lower',  levels = [ dd_lis[5], dd_lis[6], dd_lis[7] ], colors = [color_lis[8], color_lis[9], color_lis[10] ], 
	# 	extent = (cut_px - 3 * n500, cut_px + 3 * n500 + 1, cut_py - 3 * n500, cut_py + 3 * n500 + 1), )

	clust = Circle( xy = (cut_px, cut_py), radius = n1000 * 0.3, fill = False, ec = 'k', ls = '--', linewidth = 1, alpha = 0.75,)
	ax.add_patch(clust)

	ax.set_xticklabels( labels = [] )
	ax.set_yticklabels( labels = [] )

	x_ticks_0 = np.arange( xn - x_lo_cut, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_cut, cut_r_img.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( -(len(x_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax.set_xticks( x_ticks, minor = True, )
	ax.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
	ax.xaxis.set_ticks_position('bottom')
	ax.set_xlabel( 'X [Mpc]', fontsize = 15, )

	if ll == 0:

		y_ticks_0 = np.arange( yn - y_lo_cut, 0, -1 * n500)
		y_ticks_1 = np.arange( yn - y_lo_cut, cut_r_img.shape[0], n500)
		y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( -(len(y_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax.set_yticks( y_ticks, minor = True )
		ax.set_yticklabels( labels = tick_lis, minor = True, fontsize = 15,)
		ax.set_ylabel( 'Y [Mpc]', fontsize = 15,)

	ax.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
	ax.set_xlim( cut_px - np.ceil(1.1*n1000), cut_px + np.ceil(1.1*n1000) )
	ax.set_ylim( cut_py - np.ceil(1.1*n1000), cut_py + np.ceil(1.1*n1000) )

	ax.annotate( text = fig_name[ll], xy = (0.60, 0.02), xycoords = 'axes fraction', fontsize = 14, color = 'k',)

me_map = mpl.colors.ListedColormap( 
	[ color_lis[0], color_lis[2], color_lis[3], color_lis[4], color_lis[6], color_lis[8], color_lis[9], color_lis[10] ] )

c_bounds = np.r_[ dd_lis[0] - 0.05, dd_lis + 0.05]
norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

cbs = mpl.colorbar.ColorbarBase( ax = sub_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = list( dd_lis	), 
	spacing = 'proportional', orientation = 'vertical',)
cbs.ax.set_yticklabels( ['%.1f' % pp for pp in dd_lis ], fontsize = 15)

if band_str == 'gr':
	cbs.set_label( 'g - r', fontsize = 15,)
	plt.savefig('/home/xkchen/%s_g-r_color_2D_compare.png' % file_s, dpi = 300)
	# plt.savefig('/home/xkchen/%s_g-r_color_2D_compare.pdf' % file_s, dpi = 100)

if band_str == 'gi':
	cbs.set_label( 'g - i', fontsize = 15,)
	plt.savefig('/home/xkchen/%s_g-i_color_2D_compare.png' % file_s, dpi = 300)
	# plt.savefig('/home/xkchen/%s_g-i_color_2D_compare.pdf' % file_s, dpi = 100)

plt.close()
