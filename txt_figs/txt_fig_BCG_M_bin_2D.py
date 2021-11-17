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

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === data load
Da_ref = Test_model.angular_diameter_distance( z_ref ).value

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
file_s = 'BCG_Mstar_bin'

#. flux scaling correction
BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
img_path = '/home/xkchen/figs/re_measure_SBs/SBs/'
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'


### === load data
rand_r, rand_sb, rand_err = [], [], []

for ii in range( 3 ):

	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_SB-pro_z-ref-aveg.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])
	rand_r.append( tt_r )
	rand_sb.append( tt_sb )
	rand_err.append( tt_err )

### === 2D signal contour (focus on center region)
L_pix = Da_ref * 10**3 * pixel / rad2asec

n500 = 500 / L_pix
n1000 = 1000 / L_pix
n900 = 900 / L_pix
n200 = 200 / L_pix

### === total sample
def tot_signal_f():

	kk = 0

	tot_img_path = '/home/xkchen/figs/re_measure_SBs/SBs/'
	tot_BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'


	with h5py.File( tot_img_path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:
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
	BG_file = tot_BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]

	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	idnn = np.isnan( BG_sub_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()

	## 2D signal
	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_BG_sub_img )
	cut_BG_sub_img[id_nan] = 0.

	y_peak, x_peak = np.where( cut_BG_sub_img == np.nanmax( cut_BG_sub_img ) )
	y_peak, x_peak = y_peak[0], x_peak[0]
	cen_region = cut_BG_sub_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]

	filt_img_0 = ndimage.gaussian_filter( cen_region, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( cen_region, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( cen_region, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( cen_region, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( cen_region, sigma = 21,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	filt_BG_sub_img = ndimage.gaussian_filter( cen_region, sigma = 29,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )

	return mag_map_0, mag_map_1, mag_map_2, mag_map_3, mag_map_4, filt_BG_sub_mag

#... signal of total sample
tot_mag_0, tot_mag_1, tot_mag_2, tot_mag_3, tot_mag_4, tot_mag_5 = tot_signal_f()

### === sub-sample
fig = plt.figure( figsize = (10.20, 4.8) )
ax1 = fig.add_axes( [0.08, 0.12, 0.40, 0.85] )
ax2 = fig.add_axes( [0.485, 0.12, 0.40, 0.85] )
sub_ax2 = fig.add_axes( [0.885, 0.12, 0.02, 0.85] )

for ll in ( 1, 0 ):

	kk = 0 ## 0, 1, 2 --> r, g, i band

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

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	idnn = np.isnan( BG_sub_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()

	cut_rand = rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ] / pixel**2
	id_nan = np.isnan( cut_rand )
	cut_rand[id_nan] = 0.
	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_BG_sub_img )
	cut_BG_sub_img[id_nan] = 0.

	y_peak, x_peak = np.where( cut_BG_sub_img == np.nanmax( cut_BG_sub_img ) )
	y_peak, x_peak = y_peak[0], x_peak[0]
	cen_region = cut_BG_sub_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]

	filt_img_0 = ndimage.gaussian_filter( cen_region, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( cen_region, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( cen_region, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( cen_region, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( cen_region, sigma = 21,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	# filt_BG_sub_img = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 21,)
	# filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )
	# xp, yp = np.int( cut_BG_sub_img.shape[1] / 2 ), np.int( cut_BG_sub_img.shape[0] / 2 )

	filt_BG_sub_img = ndimage.gaussian_filter( cen_region, sigma = 29,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )
	xp, yp = np.int( cen_region.shape[1] / 2 ), np.int( cen_region.shape[0] / 2 )

	dd_lis = np.arange(26, 33, 1)

	color_lis = []
	for jj in np.arange( 11 ):
		color_lis.append( mpl.cm.rainbow_r( jj / 10) )

	color_str = []
	for jj in np.arange( 11 ):
		color_str.append( mpl.cm.rainbow_r( jj / 10) )

	if ll == 0:
		ax = ax1
	if ll == 1:
		ax = ax2

	# ax.imshow( cen_region, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)
	ax.imshow( cen_region, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 2e-2, alpha = 0.65,)

	cntr1 = ax.contour( mag_map_0, origin = 'lower', levels = [26, 100], colors = [ color_lis[0], 'w'], linewidths = 1.0, linestyles = '--',)

	ax.contour( mag_map_1, origin = 'lower', levels = [27, 100], colors = [ color_lis[2], 'w'], linewidths = 1.0, linestyles = '--',)

	ax.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis[3], 'w'], linewidths = 1.0, linestyles = '--',)

	ax.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis[4], 'w'], linewidths = 1.0, linestyles = '--',)

	ax.contour( mag_map_4, origin = 'lower', levels = [30, 100], colors = [ color_lis[6], 'w'], linewidths = 1.0, linestyles = '--',)

	ax.contour( filt_BG_sub_mag, origin = 'lower',  levels = [31, 32], colors = [ color_lis[8], color_lis[10] ], linewidths = 1.0, linestyles = '--',)

	# ax.contour( mag_map_0, origin = 'lower', levels = dd_lis[:5], colors = color_lis[:5],)
	# ax.contour( filt_BG_sub_mag, origin = 'lower', levels = dd_lis, colors = color_lis[:len(dd_lis)],)

	#... total sample
	# cntr2 = ax.contour( tot_mag_0, origin = 'lower', levels = [26, 100], colors = [ color_str[0], 'w'], linewidths = 0.75, linestyles = '--',)
	# ax.contour( tot_mag_1, origin = 'lower', levels = [27, 100], colors = [ color_str[2], 'w'], linewidths = 0.75, linestyles = '--',)
	# ax.contour( tot_mag_2, origin = 'lower', levels = [28, 100], colors = [ color_str[3], 'w'], linewidths = 0.75, linestyles = '--',)
	# ax.contour( tot_mag_3, origin = 'lower', levels = [29, 100], colors = [ color_str[4], 'w'], linewidths = 0.75, linestyles = '--',)
	# ax.contour( tot_mag_4, origin = 'lower', levels = [30, 100], colors = [ color_str[6], 'w'], linewidths = 0.75, linestyles = '--',)
	# ax.contour( tot_mag_5, origin = 'lower', levels = [31, 32], colors = [ color_str[8], color_str[10] ], linewidths = 0.75, linestyles = '--',)

	cntr2 = ax.contour( tot_mag_0, origin = 'lower', levels = [26, 100], colors = [ color_str[0], 'w'],  linestyles = '-', alpha = 0.65,)
	ax.contour( tot_mag_1, origin = 'lower', levels = [27, 100], colors = [ color_str[2], 'w'],  linestyles = '-', alpha = 0.65,)
	ax.contour( tot_mag_2, origin = 'lower', levels = [28, 100], colors = [ color_str[3], 'w'],  linestyles = '-', alpha = 0.65,)
	ax.contour( tot_mag_3, origin = 'lower', levels = [29, 100], colors = [ color_str[4], 'w'],  linestyles = '-', alpha = 0.65,)
	ax.contour( tot_mag_4, origin = 'lower', levels = [30, 100], colors = [ color_str[6], 'w'],  linestyles = '-', alpha = 0.65,)
	ax.contour( tot_mag_5, origin = 'lower', levels = [31, 32], colors = [ color_str[8], color_str[10] ],  linestyles = '-', alpha = 0.65,)


	ax.set_xticklabels( labels = [] )
	ax.set_yticklabels( labels = [] )

	x_ticks_0 = np.arange( xp, 0, -1 * n500)
	x_ticks_1 = np.arange( xp, cen_region.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( -(len(x_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax.set_xticks( x_ticks, minor = True, )
	ax.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
	ax.xaxis.set_ticks_position('bottom')
	ax.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

	if ll == 0:
		y_ticks_0 = np.arange( yp, 0, -1 * n500)
		y_ticks_1 = np.arange( yp, cen_region.shape[0], n500)
		y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( -(len(y_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax.set_yticks( y_ticks, minor = True )
		ax.set_yticklabels( labels = tick_lis, minor = True, fontsize = 15,)
		ax.set_ylabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15,)

	ax.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
	ax.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
	ax.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)

	# if ll == 0:
	# 	ax.annotate( text = fig_name[ll], xy = (0.67, 0.02), xycoords = 'axes fraction', fontsize = 14, color = 'w',)
	# if ll == 1:
	# 	ax.annotate( text = fig_name[ll], xy = (0.03, 0.02), xycoords = 'axes fraction', fontsize = 14, color = 'w',)

	h1_, l1_ = cntr1.legend_elements()
	h2_, l2_ = cntr2.legend_elements()
	ax.legend( [ h2_[0], h1_[0] ], [ '$\\mathrm{All} \; \\mathrm{clusters}$', fig_name[ll] ], loc = 4, fontsize = 14, 
		edgecolor = 'w', facecolor = 'w', labelcolor = 'k',)

me_map = mpl.colors.ListedColormap( 
	[ color_lis[0], color_lis[2], color_lis[3], color_lis[4], color_lis[6], color_lis[8], color_lis[10] ] )

c_bounds = np.r_[ dd_lis[0] - 0.5, dd_lis + 0.5]
norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

cbs = mpl.colorbar.ColorbarBase( ax = sub_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 31, 32], 
	spacing = 'proportional', orientation = 'vertical',)
cbs.set_label( '$ \\mu_{%s} \; [mag \, / \, arcsec^2] $' % band[kk], fontsize = 15,)
cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '31', '32'], fontsize = 15)
cbs.ax.invert_yaxis()

# plt.savefig( '/home/xkchen/%s_%s-band_2D_compare.png' % (file_s, band[kk]), dpi = 300)
plt.savefig( '/home/xkchen/%s_%s-band_2D_signal.pdf' % (file_s, band[kk]), dpi = 100)
plt.close()
