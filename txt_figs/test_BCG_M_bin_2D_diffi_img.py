import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.patches import Circle
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

### === 
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

def mu_1D_compare():

	#. 1D profile of subsamples
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


	fig = plt.figure( figsize = (5.8, 5.4) )
	ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	for kk in ( 2, 0, 1 ):

		ax1.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75,)
		ax1.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
			y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.15,)

		ax1.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk])
		ax1.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
			y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.15,)

		_tt_mu_F = interp.interp1d( nbg_hi_r[kk], nbg_hi_sb[kk], kind = 'linear', fill_value = 'extrapolate',)

		sub_ax1.plot( nbg_low_r[kk], _tt_mu_F( nbg_low_r[kk] ) - nbg_low_sb[kk], ls = ':', color = color_s[kk], alpha = 0.75,)

	legend_1 = ax1.legend( [ fig_name[0], fig_name[1] ], loc = 1, frameon = False, fontsize = 14,)
	legend_0 = ax1.legend( loc = 3, frameon = False, fontsize = 14, )
	ax1.add_artist( legend_1 )

	ax1.set_ylim( 21.5, 33.5 )
	ax1.invert_yaxis()
	ax1.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 14,)

	ax1.set_xlim( 1e-2, 1e0 )
	ax1.set_xscale('log')
	ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale( 'log' )
	sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 14,)
	sub_ax1.set_ylabel('$ \\mu_{High} \, - \, \\mu_{Low} $', fontsize = 14,)

	x_tick_arr = [ 1e-2, 1e-1, 1e0]
	tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
	sub_ax1.set_xticks( x_tick_arr )
	sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/BCG-M_SB_diffi.png', dpi = 300)
	plt.close()

	return

# mu_1D_compare()


#.
L_pix = Da_ref * 10**3 * pixel / rad2asec

n500 = 500 / L_pix
n1000 = 1000 / L_pix
n900 = 900 / L_pix
n200 = 200 / L_pix

#. estimate Background subtract image
subset_img = []

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
	cen_region = cut_BG_sub_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, 
								x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]
	subset_img.append( cen_region )

xp, yp = np.int( cen_region.shape[1] / 2 ), np.int( cen_region.shape[0] / 2 )
diffi_img = subset_img[0] - subset_img[1]
ratio_img = subset_img[0] / subset_img[1] # flux ratio between subsamples


#. figs
id_diffi = True
# id_diffi = False

fig = plt.figure( figsize = (12.40, 4.8 ) )
ax0 = fig.add_axes( [0.08, 0.12, 0.275, 0.85] )
ax1 = fig.add_axes( [0.36, 0.12, 0.275, 0.85] )
ax2 = fig.add_axes( [0.64, 0.12, 0.275, 0.85] )
sub_ax = fig.add_axes( [0.92, 0.12, 0.02, 0.85] )

axes = [ ax0, ax1, ax2 ]
dd_lis = np.arange(26, 33, 1)

color_lis = []
for jj in np.arange( 11 ):
	color_lis.append( mpl.cm.rainbow_r( jj / 10) )

for ll in range( 2 ):

	_ll_img = subset_img[ ll ]

	filt_img_0 = ndimage.gaussian_filter( _ll_img, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( _ll_img, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( _ll_img, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( _ll_img, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( _ll_img, sigma = 21,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	filt_BG_sub_img = ndimage.gaussian_filter( _ll_img, sigma = 29,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )


	ax = axes[ ll ]
	ax.imshow( _ll_img, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 5e-1, 
				norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)

	ax.contour( mag_map_0, origin = 'lower', levels = [26, 100], colors = [ color_lis[0], 'w'], linewidths = 1.0, linestyles = '-',)
	ax.contour( mag_map_1, origin = 'lower', levels = [27, 100], colors = [ color_lis[2], 'w'], linewidths = 1.0, linestyles = '-',)
	ax.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis[3], 'w'], linewidths = 1.0, linestyles = '-',)
	ax.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis[4], 'w'], linewidths = 1.0, linestyles = '-',)
	ax.contour( mag_map_4, origin = 'lower', levels = [30, 100], colors = [ color_lis[6], 'w'], linewidths = 1.0, linestyles = '-',)
	ax.contour( filt_BG_sub_mag, origin = 'lower',  levels = [31, 32], colors = [ color_lis[8], color_lis[10] ], 
				linewidths = 1.0, linestyles = '-',)

clust = Circle( xy = (xp, yp), radius = n200, fill = False, ec = 'k', ls = '--', linewidth = 1, label = '0.2 Mpc',)
ax0.add_patch(clust)

ax0.set_xticklabels( labels = [] )
ax0.set_yticklabels( labels = [] )

x_ticks_0 = np.arange( xp, 0, -1 * n500)
x_ticks_1 = np.arange( xp, cen_region.shape[1], n500)
x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

tick_R = np.r_[ np.arange( -(len(x_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

ax0.set_xticks( x_ticks, minor = True, )
ax0.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
ax0.xaxis.set_ticks_position('bottom')
ax0.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

y_ticks_0 = np.arange( yp, 0, -1 * n500)
y_ticks_1 = np.arange( yp, cen_region.shape[0], n500)
y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

tick_R = np.r_[ np.arange( -(len(y_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

ax0.set_yticks( y_ticks, minor = True )
ax0.set_yticklabels( labels = tick_lis, minor = True, fontsize = 15,)
ax0.set_ylabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15,)

ax0.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
ax0.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
ax0.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
ax0.annotate( text = fig_name[1], xy = (0.05, 0.90), fontsize = 14, xycoords = 'axes fraction', color = 'k',
				bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 0.75, 'pad': 2 },)

clust = Circle( xy = (xp, yp), radius = n200, fill = False, ec = 'k', ls = '--', linewidth = 1, label = '0.2 Mpc')
ax1.add_patch(clust)

ax1.set_xticklabels( labels = [] )
ax1.set_yticklabels( labels = [] )

ax1.set_xticks( x_ticks, minor = True, )
ax1.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

ax1.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
ax1.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
ax1.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
ax1.annotate( text = fig_name[0], xy = (0.05, 0.90), fontsize = 14, xycoords = 'axes fraction', color = 'k',
				bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 0.75, 'pad': 2 },)

if id_diffi == True:
	#. diffi image
	_ll_img = diffi_img

	filt_img_0 = ndimage.gaussian_filter( _ll_img, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( _ll_img, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( _ll_img, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( _ll_img, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( _ll_img, sigma = 21,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	filt_BG_sub_img = ndimage.gaussian_filter( _ll_img, sigma = 29,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )

	ax2.imshow( diffi_img, origin = 'lower', cmap = 'PiYG_r', vmin = -2e-1, vmax = 2e-1,
				norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)

	ax2.contour( mag_map_0, origin = 'lower', levels = [26, 100], colors = [ color_lis[0], 'w'], linewidths = 1.0, linestyles = '-',)
	ax2.contour( mag_map_1, origin = 'lower', levels = [27, 100], colors = [ color_lis[2], 'w'], linewidths = 1.0, linestyles = '-',)
	ax2.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis[3], 'w'], linewidths = 1.0, linestyles = '-',)
	ax2.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis[4], 'w'], linewidths = 1.0, linestyles = '-',)
	ax2.contour( mag_map_4, origin = 'lower', levels = [30, 100], colors = [ color_lis[6], 'w'], linewidths = 1.0, linestyles = '-',)
	ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = [31, 32], colors = [ color_lis[8], color_lis[10] ], 
				linewidths = 1.0, linestyles = '-',)

	clust = Circle( xy = (xp, yp), radius = n200, fill = False, ec = 'k', ls = '--', linewidth = 1, label = '0.2 Mpc')
	ax2.add_patch(clust)

	ax2.legend( loc = 3, frameon = True, fontsize = 15,)

	ax2.set_xticklabels( labels = [] )
	ax2.set_yticklabels( labels = [] )

	ax2.set_xticks( x_ticks, minor = True, )
	ax2.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
	ax2.xaxis.set_ticks_position('bottom')
	ax2.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

	ax2.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
	ax2.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
	ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
	ax2.annotate( text = fig_name[1] + ' - ' + fig_name[0], xy = (0.05, 0.90), fontsize = 14, xycoords = 'axes fraction', color = 'k',
				bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 0.75, 'pad': 2 },)

	#. color bar
	me_map = mpl.colors.ListedColormap( 
		[ color_lis[0], color_lis[2], color_lis[3], color_lis[4], color_lis[6], color_lis[8], color_lis[10] ] )

	c_bounds = np.r_[ dd_lis[0] - 0.5, dd_lis + 0.5]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 31, 32], 
		spacing = 'proportional', orientation = 'vertical',)
	cbs.set_label( '$ \\mu_{%s} \; [mag \, / \, arcsec^2] $' % band[kk], fontsize = 15,)
	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '31', '32'], fontsize = 15)
	cbs.ax.invert_yaxis()

	plt.savefig('/home/xkchen/BCG-M_bin_diffi_img.pdf', dpi = 100)
	plt.close()

if id_diffi == False:

	#. ratio img
	delta_mu = np.linspace(0.2, 0.8, 7)
	delta_mu = -1 * delta_mu[::-1]

	color_str = []
	for jj in np.arange( 11 ):
		color_str.append( mpl.cm.autumn_r( jj / 10) )

	# filt_img_0 = ndimage.gaussian_filter( subset_img[0], sigma = 3 ) / ndimage.gaussian_filter( subset_img[1], sigma = 3 )
	# mag_map_0 = - 2.5 * np.log10( filt_img_0 )

	# filt_img_1 = ndimage.gaussian_filter( subset_img[0], sigma = 7 ) / ndimage.gaussian_filter( subset_img[1], sigma = 7 )
	# mag_map_1 = - 2.5 * np.log10( filt_img_1 )

	# filt_img_2 = ndimage.gaussian_filter( subset_img[0], sigma = 11 ) / ndimage.gaussian_filter( subset_img[1], sigma = 11 )
	# mag_map_2 = - 2.5 * np.log10( filt_img_2 )

	# filt_img_3 = ndimage.gaussian_filter( subset_img[0], sigma = 17 ) / ndimage.gaussian_filter( subset_img[1], sigma = 17 )
	# mag_map_3 = - 2.5 * np.log10( filt_img_3 )

	# filt_img_4 = ndimage.gaussian_filter( subset_img[0], sigma = 21 ) / ndimage.gaussian_filter( subset_img[1], sigma = 21 )
	# mag_map_4 = - 2.5 * np.log10( filt_img_4 )

	filt_BG_sub_img = ndimage.gaussian_filter( subset_img[0], sigma = 25,) / ndimage.gaussian_filter( subset_img[1], sigma = 25,)
	filt_BG_sub_mag = - 2.5 * np.log10( filt_BG_sub_img )

	ax2.imshow( ratio_img, origin = 'lower', cmap = 'Greys', vmin = 1, vmax = 5,
				norm = mpl.colors.SymLogNorm( linthresh = 0.1, linscale = 1, base = 10),)

	# ax2.contour( mag_map_0, origin = 'lower', levels = [ delta_mu[0], 100], colors = [ color_str[0], 'w'], linewidths = 1.0, linestyles = '-',)
	# ax2.contour( mag_map_1, origin = 'lower', levels = [ delta_mu[1], 100], colors = [ color_str[2], 'w'], linewidths = 1.0, linestyles = '-',)
	# ax2.contour( mag_map_2, origin = 'lower', levels = [ delta_mu[2], 100], colors = [ color_str[3], 'w'], linewidths = 1.0, linestyles = '-',)
	# ax2.contour( mag_map_3, origin = 'lower', levels = [ delta_mu[3], 100], colors = [ color_str[4], 'w'], linewidths = 1.0, linestyles = '-',)
	# ax2.contour( mag_map_4, origin = 'lower', levels = [ delta_mu[4], 100], colors = [ color_str[6], 'w'], linewidths = 1.0, linestyles = '-',)
	# ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = [ delta_mu[5], delta_mu[6] ], colors = [ color_str[8], color_str[10] ], 
	# 			linewidths = 1.0, linestyles = '-',)

	clies = ax2.contour( filt_BG_sub_mag, origin = 'lower', levels = delta_mu, colors = [ 
				color_str[0], color_str[2], color_str[3], color_str[4], color_str[6], color_str[8], color_str[10] ],)
	plt.clabel( clies, fontsize = 6, fmt = '%.1f', inline = False, colors = 'b',)

	clust = Circle( xy = (xp, yp), radius = n200, fill = False, ec = 'k', ls = '--', linewidth = 1, label = '0.2 Mpc')
	ax2.add_patch(clust)

	ax2.legend( loc = 1, frameon = True, fontsize = 15,)

	ax2.set_xticklabels( labels = [] )
	ax2.set_yticklabels( labels = [] )

	ax2.set_xticks( x_ticks, minor = True, )
	ax2.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
	ax2.xaxis.set_ticks_position('bottom')
	ax2.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

	ax2.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
	ax2.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
	ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
	ax2.annotate( text = '$ \\mu_{High \, M_{\\ast}^{BCG} }\, - \, \\mu_{Low \, M_{\\ast}^{BCG} } $', xy = (0.15, 0.05), fontsize = 14, 
					xycoords = 'axes fraction', color = 'k', 
					bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 1.0, 'pad': 2}, zorder = 10000,)

	#. color bar
	me_map = mpl.colors.ListedColormap( 
		[ color_lis[0], color_lis[2], color_lis[3], color_lis[4], color_lis[6], color_lis[8], color_lis[10] ] )

	c_bounds = np.r_[ dd_lis[0] - 0.5, dd_lis + 0.5]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 31, 32], 
		spacing = 'proportional', orientation = 'vertical',)
	cbs.set_label( '$ \\mu_{%s} \; [mag \, / \, arcsec^2] $' % band[kk], fontsize = 15,)
	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '31', '32'], fontsize = 15)
	cbs.ax.invert_yaxis()

	plt.savefig('/home/xkchen/BCG-M_bin_ratio_img.pdf', dpi = 100)
	plt.close()

