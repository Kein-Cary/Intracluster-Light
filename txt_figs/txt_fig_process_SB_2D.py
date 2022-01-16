"""
this file use for figure adjust
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import ndimage

from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

#..
from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
from img_cat_param_match import match_func
from img_pre_selection import extra_match_func


# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

psf_FWHM = 1.32 # arcsec
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec


### === ### Background subtraction img
path = '/home/xkchen/figs/extend_bcgM_cat/SBs/'
img_path = '/home/xkchen/figs/extend_bcgM_cat/SBs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat/BGs/'

rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'


# random profile
rand_r, rand_sb, rand_err = [], [], []
for ii in range( 3 ):
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_SB-pro_z-ref-aveg.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])
	rand_r.append( tt_r )
	rand_sb.append( tt_sb )
	rand_err.append( tt_err )

# before subtraction
tot_r, tot_sb, tot_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tot_r.append( tt_r )
	tot_sb.append( tt_sb )
	tot_err.append( tt_err )

# after subtraction
nbg_tot_r, nbg_tot_sb, nbg_tot_err = [], [], []
for kk in range( 3 ):

	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_tot_r.append( tt_r )
	nbg_tot_sb.append( tt_sb )
	nbg_tot_err.append( tt_err )


L_pix = Da_ref * 10**3 * pixel / rad2asec

R1Mpc = 1000 / L_pix
R2Mpc = 2000 / L_pix
R3Mpc = 3000 / L_pix


### === ### 2D signal compare
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

n500 = 500 / L_pix
n1000 = 1000 / L_pix
n900 = 900 / L_pix

##... full area of stacked image
def full_stack_image_2D():

	for kk in range( 0,1 ):

		## flux imgs
		with h5py.File( img_path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:
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
		BG_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]

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

		y_peak, x_peak = np.where( cut_BG_sub_img == np.nanmax( cut_BG_sub_img ) )
		y_peak, x_peak = y_peak[0], x_peak[0]
		cen_region = cut_BG_sub_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]

		#... smooth image
		filt_off_rand = ndimage.gaussian_filter( cut_off_rand, sigma = 105,)
		filt_off_rand_mag = 22.5 - 2.5 * np.log10( filt_off_rand )

		filt_img = ndimage.gaussian_filter( cut_img, sigma = 115,)
		filt_mag = 22.5 - 2.5 * np.log10( filt_img )

		filt_img_1 = ndimage.gaussian_filter( cen_region, sigma = 7,)
		mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

		filt_img_2 = ndimage.gaussian_filter( cen_region, sigma = 11,)
		mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

		filt_img_3 = ndimage.gaussian_filter( cen_region, sigma = 17,)
		mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

		filt_img_4 = ndimage.gaussian_filter( cen_region, sigma = 21,)
		mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

		filt_BG_sub_img = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 105)#sigma = 81,)
		filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )

		#... large scale only array
		Nx = cut_img.shape[1]
		Ny = cut_img.shape[0]

		lx = np.linspace( 0, Nx - 1, Nx)
		ly = np.linspace( 0, Ny - 1, Ny)
		pix_xy = np.array( np.meshgrid( lx, ly ) )

		pc_x, pc_y = cen_x - x_lo_lim, cen_y - y_lo_lim
		cen_dr = np.sqrt( ( (2*pix_xy[0] + 1) / 2 - (2*pc_x + 1) / 2)**2 + ( (2*pix_xy[1] + 1) / 2 - (2*pc_y + 1) / 2)**2 )

		id_cen_arr = cen_dr <= n900
		cp_cut_img = cut_img.copy()
		cp_cut_img[ id_cen_arr ] = np.nan

		idnan = np.isnan( cp_cut_img )
		cp_cut_img[ idnan ] = 0.

		filt_cp_img = ndimage.gaussian_filter( cp_cut_img, sigma = 115,)
		filt_cp_mag = 22.5 - 2.5 * np.log10( filt_cp_img )


		## SB profiles in mag / arcsec^2
		modi_rand_sb = rand_sb[kk] - offD + sb_2Mpc
		modi_rand_mag = 22.5 - 2.5 * np.log10( modi_rand_sb )
		modi_rand_mag_err = 2.5 * rand_err[kk] / ( np.log(10) * modi_rand_sb )

		clus_mag = 22.5 - 2.5 * np.log10( tot_sb[kk] )
		clus_mag_err = 2.5 * tot_err[kk] / ( np.log(10) * tot_sb[kk] )

		nbg_clus_mag = 22.5 - 2.5 * np.log10( nbg_tot_sb[kk] )
		nbg_clus_mag_err = 2.5 * nbg_tot_err[kk] / ( np.log(10) * nbg_tot_sb[kk] )

		idnn = np.isnan( nbg_clus_mag )
		nbg_clus_mag = nbg_clus_mag[ idnn == False ]
		nbg_clus_mag_err = nbg_clus_mag_err[ idnn == False ]


		color_str = []
		for jj in range( 12 ):
			# color_str.append( mpl.cm.rainbow_r( jj / 11) )
			color_str.append( mpl.cm.coolwarm_r( jj / 11) )

		color_lis = []
		for jj in np.arange( 11 ):
			# color_lis.append( mpl.cm.rainbow_r( jj / 10) )
			color_lis.append( mpl.cm.coolwarm_r( jj / 10) )

		# levels_0 = np.linspace(28, 29, 11)
		# levels_0 = np.linspace(28.5, 29.0, 9)
		levels_0 = np.linspace(28.7, 29.0, 7)


		## figs
		fig = plt.figure( figsize = (14.2, 9.6) )
		cb_ax2 = fig.add_axes( [0.863, 0.07, 0.02, 0.40] )
		ax2 = fig.add_axes( [0.48, 0.07, 0.40, 0.40] )

		ax3 = fig.add_axes( [0.07, 0.07, 0.36, 0.40] )

		ax0 = fig.add_axes( [0.05, 0.55, 0.40, 0.40] )

		cb_ax1 = fig.add_axes( [0.866, 0.55, 0.02, 0.40] )
		ax1 = fig.add_axes( [0.48, 0.55, 0.40, 0.40] )


		## cluster imgs before BG subtract
		ax0.imshow( cut_img, origin = 'lower', cmap = 'Greys', vmin = -3e-2, vmax = 3e-2, alpha = 0.75,)
		#... use levels 28~29
		edge_levels = np.array( [28.55, 28.6, 28.65 ] )
		ax0.contour( filt_cp_mag, origin = 'lower',  levels = edge_levels, colors = color_str[1:4], 
			extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut), )

		ax0.contour( filt_mag, origin = 'lower',  levels = levels_0, colors = color_str[4:], 
			extent = (0, x_up_lim + 1 - x_lo_lim, 0, y_up_lim + 1 - y_lo_lim ), )

		clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = n500 * 2, fill = False, ec = 'k', ls = '--', alpha = 1.0,)
		ax0.add_patch(clust)

		ax0.set_xlim( 0, x_up_lim + 1 - x_lo_lim )
		ax0.set_ylim( 0, y_up_lim + 1 - y_lo_lim )

		ax0.set_xticklabels( labels = [] )
		ax0.set_yticklabels( labels = [] )

		x_ticks_0 = np.arange( xn - x_lo_lim, 0, -1 * n500)
		x_ticks_1 = np.arange( xn - x_lo_lim, cut_rand.shape[1], n500)
		x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax0.set_xticks( x_ticks, minor = True, )
		ax0.set_xticklabels( labels = tick_lis, minor = True, fontsize = 14,)
		ax0.xaxis.set_ticks_position('bottom')
		ax0.set_xlabel('Mpc', fontsize = 16,)

		y_ticks_0 = np.arange( yn - y_lo_lim, 0, -1 * n500)
		y_ticks_1 = np.arange( yn - y_lo_lim, cut_rand.shape[0], n500)
		y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax0.set_yticks( y_ticks, minor = True )
		ax0.set_yticklabels( labels = tick_lis, minor = True, fontsize = 14,)
		ax0.set_ylabel( 'Mpc', fontsize = 16,)
		ax0.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False,) #labelsize = 16,)

		ax0.text( 100, 100, 'ICL + BCG + Background', fontsize = 15, bbox = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad':5.0},)


		## cluster imgs after BG-subtraction
		ax2.imshow( cut_BG_sub_img, origin  ='lower', cmap = 'Greys', vmin = -3e-2, vmax = 3e-2, alpha = 0.75,)

		dd_lis = np.arange(27, 34, 1)

		ax2.contour( mag_map_1, origin = 'lower', levels = [27, 100], colors = [ color_lis[0], 'w'], 
			extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis[2], 'w'], 
			extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis[3], 'w'], 
			extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_4, origin = 'lower', levels = [30, 100], colors = [ color_lis[4], 'w'], 
			extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

		ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = [31, 32, 33, 100], colors = [ color_lis[6], color_lis[8], color_lis[10], 'w'], 
			extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut), )

		clust = Circle( xy = (xn - x_lo_cut, yn - y_lo_cut), radius = n500 * 2, fill = False, ec = 'k', ls = '--', alpha = 1.0,)
		ax2.add_patch(clust)

		dt_color = [ color_lis[0], color_lis[2], color_lis[3], color_lis[4], color_lis[6], color_lis[8], color_lis[10] ]
		me_map = mpl.colors.ListedColormap( dt_color )
		c_bounds = np.r_[ dd_lis[0] - 0.5, dd_lis + 0.5]
		norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

		cbs = mpl.colorbar.ColorbarBase( ax = cb_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = [27, 28, 29, 30, 31, 32, 33], 
			spacing = 'proportional', orientation = 'vertical')
		cbs.set_label( '$ SB \; [mag \, / \, arcsec^2] $', fontsize = 15,)
		cbs.ax.set_yticklabels( ['27', '28', '29', '30', '31', '32', '33'], fontsize = 14)
		cbs.ax.tick_params( axis = 'both', which = 'major', direction = 'in')
		cbs.ax.invert_yaxis()

		ax2.set_xlim( 0, x_up_cut + 1 - x_lo_cut )
		ax2.set_ylim( 0, y_up_cut + 1 - y_lo_cut )

		ax2.set_xticklabels( labels = [] )
		ax2.set_yticklabels( labels = [] )

		x_ticks_0 = np.arange( xn - x_lo_cut, 0, -1 * n500)
		x_ticks_1 = np.arange( xn - x_lo_cut, cut_rand.shape[1], n500)
		x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax2.set_xticks( x_ticks, minor = True, )
		ax2.set_xticklabels( labels = tick_lis, minor = True, fontsize = 14,)
		ax2.xaxis.set_ticks_position('bottom')
		ax2.set_xlabel( 'Mpc', fontsize = 16,)

		y_ticks_0 = np.arange( yn - y_lo_cut, 0, -1 * n500)
		y_ticks_1 = np.arange( yn - y_lo_cut, cut_rand.shape[0], n500)
		y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax2.set_yticks( y_ticks, minor = True )
		ax2.set_yticklabels( labels = tick_lis, minor = True, fontsize = 14,)
		ax2.set_ylabel( 'Mpc', fontsize = 16,)

		ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False,) #labelsize = 16,)
		ax2.text( 100, 100, 'ICL + BCG', fontsize = 15, bbox = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad':5.0},)


		### random image
		ax1.imshow( cut_off_rand, origin = 'lower', cmap = 'Greys', vmin = -3e-2, vmax = 3e-2, alpha = 0.75,)

		# .. use levels 28 ~ 29
		ax1.contour( filt_off_rand_mag, origin = 'lower', levels = edge_levels, colors = color_str[1:4], 
			extent = (0, x_up_eff + 1 - x_lo_eff, 0, y_up_eff + 1 - y_lo_eff), )

		ax1.contour( filt_off_rand_mag, origin = 'lower', levels = levels_0, colors = color_str[4:], 
			extent = (0, x_up_eff + 1 - x_lo_eff, 0, y_up_eff + 1 - y_lo_eff), )

		c_bounds = np.r_[ edge_levels[0] - 0.025, edge_levels + 0.025, levels_0 + 0.025 ]
		me_map = mpl.colors.ListedColormap( color_str[1:] )
		norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

		cbs = mpl.colorbar.ColorbarBase( ax = cb_ax1, cmap = me_map, norm = norm, extend = 'neither', 
			ticks = np.r_[ edge_levels, levels_0],
			spacing = 'proportional', orientation = 'vertical', )
		cbs.set_label( '$ SB \; [mag \, / \, arcsec^2] $', fontsize = 15)
		cbs.ax.set_yticklabels( ['%.2f' % ll for ll in np.r_[ edge_levels, levels_0] ], fontsize = 14)
		cbs.ax.tick_params( axis = 'both', which = 'major', direction = 'in')
		cbs.ax.invert_yaxis()

		clust = Circle( xy = (xn - x_lo_eff, yn - y_lo_eff), radius = n500 * 2, fill = False, ec = 'k', ls = '--', alpha = 1.0,)
		ax1.add_patch(clust)

		ax1.set_xlim( 0, x_up_eff + 1 - x_lo_eff )
		ax1.set_ylim( 0, y_up_eff + 1 - y_lo_eff )

		ax1.set_yticklabels( labels = [] )
		ax1.set_xticklabels( labels = [] )

		x_ticks_0 = np.arange( xn - x_lo_eff, 0, -1 * n500)
		x_ticks_1 = np.arange( xn - x_lo_eff, cut_rand.shape[1], n500)
		x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax1.set_xticks( x_ticks, minor = True, )
		ax1.set_xticklabels( labels = tick_lis, minor = True, fontsize = 14,)
		ax1.xaxis.set_ticks_position('bottom')
		ax1.set_xlabel( 'Mpc', fontsize = 16,)

		y_ticks_0 = np.arange( yn - y_lo_eff, 0, -1 * n500)
		y_ticks_1 = np.arange( yn - y_lo_eff, cut_rand.shape[0], n500)
		y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax1.set_yticks( y_ticks, minor = True )
		ax1.set_yticklabels( labels = tick_lis, minor = True, fontsize = 14,)
		ax1.set_ylabel( 'Mpc', fontsize = 16,)

		ax1.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False,) #labelsize = 16,)
		ax1.text( 100, 100, 'Background', fontsize = 15, bbox = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad':5.0},)


		## 1D profile
		ax3.plot( tot_r[kk], clus_mag, ls = '--', color = 'b', alpha = 0.85, label = 'ICL + BCG + Background',)
		ax3.fill_between( tot_r[kk], y1 = clus_mag - clus_mag_err, y2 = clus_mag + clus_mag_err, color = 'b', alpha = 0.18,)

		ax3.plot( rand_r[kk], modi_rand_mag, ls = '--', color = 'k', alpha = 0.85, label = 'Background',)
		ax3.fill_between( rand_r[kk], y1 = modi_rand_mag - modi_rand_mag_err, y2 = modi_rand_mag + modi_rand_mag_err, color = 'k', alpha = 0.18,)

		ax3.plot(nbg_tot_r[kk][ idnn == False ], nbg_clus_mag, ls = '-', color = 'r', alpha = 0.85, label = 'ICL + BCG',)
		ax3.fill_between(nbg_tot_r[kk][ idnn == False ], y1 = nbg_clus_mag - nbg_clus_mag_err, 
			y2 = nbg_clus_mag + nbg_clus_mag_err, color = 'r', alpha = 0.18,)

		ax3.set_xlim( 3e0, 1.1e3 )
		ax3.set_ylim( 20, 34 )

		ax3.invert_yaxis()
		ax3.set_xscale('log')
		ax3.set_xlabel('$ R \; [kpc] $', fontsize = 16,)
		ax3.set_ylabel('$ SB \; [mag \, / \, arcsec^2] $', fontsize = 16,)
		ax3.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14.5,)
		ax3.legend( loc = 3, fontsize = 18, frameon = False)
		ax3.set_aspect( 'auto' )

		plt.savefig('/home/xkchen/mass-bin_%s-band_BG-sub_process.pdf' % band[kk], dpi = 100)
		plt.close()

	return

# full_stack_image_2D()


##... center region
def ri_aveg_img():

	## r band img
	with h5py.File( img_path + 'photo-z_match_tot-BCG-star-Mass_r-band_Mean_jack_img_z-ref.h5', 'r') as f:
		r_band_img = np.array( f['a'] )
	with h5py.File( img_path + 'photo-z_match_tot-BCG-star-Mass_r-band_stack_test_rms.h5', 'r') as f:
		r_band_rms = np.array( f['a'] )

	inves_r_rms2 = 1 / r_band_rms**2 

	## i band img
	with h5py.File( img_path + 'photo-z_match_tot-BCG-star-Mass_i-band_Mean_jack_img_z-ref.h5', 'r') as f:
		i_band_img = np.array( f['a'] )
	with h5py.File( img_path + 'photo-z_match_tot-BCG-star-Mass_i-band_stack_test_rms.h5', 'r') as f:
		i_band_rms = np.array( f['a'] )

	inves_i_rms2 = 1 / i_band_rms**2

	## random imgs
	with h5py.File( rand_path + 'random_r-band_rand-stack_Mean_jack_img_z-ref-aveg.h5', 'r') as f:
		r_rand_img = np.array( f['a'])

	BG_file = out_path + 'photo-z_tot-BCG-star-Mass_r-band_BG-profile_params_diag-fit.csv'
	cat = pds.read_csv( BG_file )
	r_offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	r_sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	off_r_band_rand_img = r_rand_img / pixel**2 - r_offD + r_sb_2Mpc


	with h5py.File( rand_path + 'random_i-band_rand-stack_Mean_jack_img_z-ref-aveg.h5', 'r') as f:
		i_rand_img = np.array( f['a'])

	BG_file = out_path + 'photo-z_tot-BCG-star-Mass_i-band_BG-profile_params_diag-fit.csv'
	cat = pds.read_csv( BG_file )
	i_offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	i_sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	off_i_band_rand_img = i_rand_img / pixel**2 - i_offD + i_sb_2Mpc


	r_BG_sub_img = r_band_img / pixel**2 - off_r_band_rand_img
	i_BG_sub_img = i_band_img / pixel**2 - off_i_band_rand_img

	cen_x, cen_y = np.int( r_band_img.shape[1] / 2 ), np.int( r_band_img.shape[0] / 2 )
	weit_img = ( r_BG_sub_img * inves_r_rms2 + i_BG_sub_img * inves_i_rms2 ) / ( inves_r_rms2 + inves_i_rms2 )

	cut_L = np.int( 1.4e3 / L_pix )
	n200 = 200 / L_pix

	#... r image
	r_cut_img = r_BG_sub_img[ cen_y - cut_L: cen_y + cut_L, cen_x - cut_L: cen_x + cut_L ]

	r_filt_img_0 = ndimage.gaussian_filter( r_cut_img, sigma = 3,)
	r_mag_map_0 = 22.5 - 2.5 * np.log10( r_filt_img_0 )

	r_filt_img_1 = ndimage.gaussian_filter( r_cut_img, sigma = 7,)
	r_mag_map_1 = 22.5 - 2.5 * np.log10( r_filt_img_1 )

	r_filt_img_2 = ndimage.gaussian_filter( r_cut_img, sigma = 11,)
	r_mag_map_2 = 22.5 - 2.5 * np.log10( r_filt_img_2 )

	r_filt_img_3 = ndimage.gaussian_filter( r_cut_img, sigma = 17,)
	r_mag_map_3 = 22.5 - 2.5 * np.log10( r_filt_img_3 )

	r_filt_img_4 = ndimage.gaussian_filter( r_cut_img, sigma = 21,)
	r_mag_map_4 = 22.5 - 2.5 * np.log10( r_filt_img_4 )

	#... i image
	i_cut_img = i_BG_sub_img[ cen_y - cut_L: cen_y + cut_L, cen_x - cut_L: cen_x + cut_L ]

	i_filt_img_0 = ndimage.gaussian_filter( i_cut_img, sigma = 3,)
	i_mag_map_0 = 22.5 - 2.5 * np.log10( i_filt_img_0 )

	i_filt_img_1 = ndimage.gaussian_filter( i_cut_img, sigma = 7,)
	i_mag_map_1 = 22.5 - 2.5 * np.log10( i_filt_img_1 )

	i_filt_img_2 = ndimage.gaussian_filter( i_cut_img, sigma = 11,)
	i_mag_map_2 = 22.5 - 2.5 * np.log10( i_filt_img_2 )

	i_filt_img_3 = ndimage.gaussian_filter( i_cut_img, sigma = 17,)
	i_mag_map_3 = 22.5 - 2.5 * np.log10( i_filt_img_3 )

	i_filt_img_4 = ndimage.gaussian_filter( i_cut_img, sigma = 21,)
	i_mag_map_4 = 22.5 - 2.5 * np.log10( i_filt_img_4 )


	#... r+i image
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
	for jj in range( 11 ):
		color_str.append( mpl.cm.rainbow_r(jj / 10) )


	fig = plt.figure( figsize = (14.4, 4.8) )
	ax0 = fig.add_axes( [0.03, 0.09, 0.30, 0.85] )
	ax1 = fig.add_axes( [0.33, 0.09, 0.30, 0.85] )
	ax2 = fig.add_axes( [0.63, 0.09, 0.30, 0.85] )
	sub_ax2 = fig.add_axes( [0.93, 0.09, 0.02, 0.85] )


	ax0.imshow( r_cut_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax0.contour( r_mag_map_0, origin = 'lower', levels = [26, 100], alpha = 0.75, colors = [ color_str[0], 'w'] )

	cs = ax0.contour( r_mag_map_1, origin = 'lower', levels = [27, 100], alpha = 0.75, colors = [ color_str[2], 'w'] )

	cs = ax0.contour( r_mag_map_2, origin = 'lower', levels = [28, 100], alpha = 0.75, colors = [ color_str[3], 'w'] )

	cs = ax0.contour( r_mag_map_3, origin = 'lower', levels = [29, 100], alpha = 0.75, colors = [ color_str[4], 'w'] )

	cs = ax0.contour( r_mag_map_4, origin = 'lower', levels = [30, 31, 32, 100,], alpha = 0.75, 
		colors = [ color_str[6], color_str[8], color_str[10], 'w' ] )

	ax0.set_xlim( 0, cut_L * 2 )
	ax0.set_ylim( 0, cut_L * 2 )

	## # of pixels pre 100kpc
	ax0.set_xticklabels( labels = [] )
	ax0.set_yticklabels( labels = [] )

	ticks_0 = np.arange( cut_L, 0, -1 * n200)
	ticks_1 = np.arange( cut_L, 2 * cut_L, n200)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	n_tick = np.ceil( cut_L / n200)
	tick_R = np.r_[ np.arange(-n_tick * 200, 0, 200), np.arange(0, n_tick * 200, 200) ] / 1000
	tick_lis = [ '%.1f' % ll for ll in tick_R ]

	ax0.set_xticks( ticks, minor = True, )
	ax0.set_xticklabels( labels = tick_lis[1:], minor = True,)

	ax0.set_yticks( ticks, minor = True )
	ax0.set_yticklabels( labels = tick_lis[1:], minor = True,)
	ax0.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False)
	ax0.annotate( text = 'r band', xy = (0.1, 0.9), xycoords = 'axes fraction',)

	ax0.set_xlabel( 'Mpc' )
	ax0.set_ylabel( 'Mpc' )


	ax1.imshow( i_cut_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax1.contour( i_mag_map_0, origin = 'lower', levels = [26, 100], alpha = 0.75, colors = [ color_str[0], 'w'] )

	cs = ax1.contour( i_mag_map_1, origin = 'lower', levels = [27, 100], alpha = 0.75, colors = [ color_str[2], 'w'] )

	cs = ax1.contour( i_mag_map_2, origin = 'lower', levels = [28, 100], alpha = 0.75, colors = [ color_str[3], 'w'] )

	cs = ax1.contour( i_mag_map_3, origin = 'lower', levels = [29, 100], alpha = 0.75, colors = [ color_str[4], 'w'] )

	cs = ax1.contour( i_mag_map_4, origin = 'lower', levels = [30, 31, 32, 100,], alpha = 0.75, 
		colors = [ color_str[6], color_str[8], color_str[10], 'w' ] )

	ax1.set_xlim( 0, cut_L * 2 )
	ax1.set_ylim( 0, cut_L * 2 )

	## # of pixels pre 100kpc
	ax1.set_xticklabels( labels = [] )
	ax1.set_yticklabels( labels = [] )

	ticks_0 = np.arange( cut_L, 0, -1 * n200)
	ticks_1 = np.arange( cut_L, 2 * cut_L, n200)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	n_tick = np.ceil( cut_L / n200)
	tick_R = np.r_[ np.arange(-n_tick * 200, 0, 200), np.arange(0, n_tick * 200, 200) ] / 1000
	tick_lis = [ '%.1f' % ll for ll in tick_R ]

	ax1.set_xticks( ticks, minor = True, )
	ax1.set_xticklabels( labels = tick_lis[1:], minor = True,)

	# ax1.set_yticks( ticks, minor = True )
	# ax1.set_yticklabels( labels = tick_lis, minor = True,)
	ax1.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False)
	ax1.annotate( text = 'i band', xy = (0.1, 0.9), xycoords = 'axes fraction',)

	ax1.set_xlabel( 'Mpc' )


	ax2.imshow( cut_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax2.contour( mag_map_0, origin = 'lower', levels = [26, 100], alpha = 0.75, colors = [ color_str[0], 'w'] )

	cs = ax2.contour( mag_map_1, origin = 'lower', levels = [27, 100], alpha = 0.75, colors = [ color_str[2], 'w'] )

	cs = ax2.contour( mag_map_2, origin = 'lower', levels = [28, 100], alpha = 0.75, colors = [ color_str[3], 'w'] )

	cs = ax2.contour( mag_map_3, origin = 'lower', levels = [29, 100], alpha = 0.75, colors = [ color_str[4], 'w'] )

	cs = ax2.contour( mag_map_4, origin = 'lower', levels = [30, 31, 32, 100,], alpha = 0.75, 
		colors = [ color_str[6], color_str[8], color_str[10], 'w' ] )

	sb_lis = np.arange(26, 33, 1)

	c_bounds = np.r_[ sb_lis[0] - 0.5, sb_lis + 0.5 ]
	me_map = mpl.colors.ListedColormap( [color_str[0], color_str[2], color_str[3], color_str[4], color_str[6], color_str[8], color_str[10] ] )
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = sub_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 31, 32], 
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '31', '32'] )
	cbs.ax.invert_yaxis()

	ax2.set_xlim( 0, cut_L * 2 )
	ax2.set_ylim( 0, cut_L * 2 )

	## # of pixels pre 100kpc
	ax2.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax2.set_yticklabels( labels = [] )

	ticks_0 = np.arange( cut_L, 0, -1 * n200)
	ticks_1 = np.arange( cut_L, 2 * cut_L, n200)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	n_tick = np.ceil( cut_L / n200)
	tick_R = np.r_[ np.arange(-n_tick * 200, 0, 200), np.arange(0, n_tick * 200, 200) ] / 1000
	tick_lis = [ '%.1f' % ll for ll in tick_R ]

	ax2.set_xticks( ticks, minor = True, )
	ax2.set_xticklabels( labels = tick_lis[1:], minor = True,)

	# ax2.set_yticks( ticks, minor = True )
	# ax2.set_yticklabels( labels = tick_lis, minor = True,)
	ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False)
	ax2.annotate( text = 'r band + i band', xy = (0.1, 0.9), xycoords = 'axes fraction',)

	ax2.set_xlabel( 'Mpc' )
	# ax2.set_ylabel( 'Mpc' )

	plt.savefig('/home/xkchen/r+i_band_stacked_img.png', dpi = 300)
	plt.close()

# ri_aveg_img()


for kk in range( 1 ):

	## flux imgs
	with h5py.File( img_path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:
		tmp_img = np.array( f['a'])

	# with h5py.File( '/home/xkchen/figs/extend_bcgM_cat/entire_2D_check/' + 
	# 				'photo-z_match_tot-BCG-star-Mass_%s-band_stack_test_img.h5' % band[kk], 'r') as f:
	# 	tmp_img = np.array( f['a'])

	##.. pre-results
	# with h5py.File( '/home/xkchen/figs/extend_bcgM_cat/entire_2D_check/' + 
	# 				'pre_tot-BCG-star-Mass_%s-band_stack_test_img.h5' % band[kk], 'r') as f:

	# with h5py.File( '/home/xkchen/figs/extend_bcgM_cat/entire_2D_check/' + 
	# 				'pre_gri-common_tot-BCG-star-Mass_%s-band_stack_test_img.h5' % band[kk], 'r') as f:
	#	tmp_img = np.array( f['a'])

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
	BG_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]

	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc

	# BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	cp_rand_img_0 = shift_rand_img[::-1, ::-1]
	cp_rand_img_1 = shift_rand_img[::-1, :]
	cp_rand_img_2 = shift_rand_img[:, ::-1]
	BG_sub_img = tmp_img / pixel**2 - cp_rand_img_2


	idnn = np.isnan( BG_sub_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()


	## 2D signal
	cut_img = tmp_img[ y_lo_lim: y_up_lim + 1, x_lo_lim: x_up_lim + 1 ] / pixel**2
	# id_nan = np.isnan( cut_img )
	# cut_img[id_nan] = 0.

	cut_rand = rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ] / pixel**2
	# id_nan = np.isnan( cut_rand )
	# cut_rand[id_nan] = 0.

	cut_off_rand = shift_rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ]
	# id_nan = np.isnan( cut_off_rand )
	# cut_off_rand[id_nan] = 0.

	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	# id_nan = np.isnan( cut_BG_sub_img )
	# cut_BG_sub_img[id_nan] = 0.


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

	filt_img_4 = ndimage.gaussian_filter( cen_region, sigma = 29,)  # sigma = 21, 22, 27, 25
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )


	filt_BG_sub_img = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 29,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )


	## SB profiles in mag / arcsec^2
	modi_rand_sb = rand_sb[kk] - offD + sb_2Mpc
	modi_rand_mag = 22.5 - 2.5 * np.log10( modi_rand_sb )
	modi_rand_mag_err = 2.5 * rand_err[kk] / ( np.log(10) * modi_rand_sb )

	clus_mag = 22.5 - 2.5 * np.log10( tot_sb[kk] )
	clus_mag_err = 2.5 * tot_err[kk] / ( np.log(10) * tot_sb[kk] )

	nbg_clus_mag = 22.5 - 2.5 * np.log10( nbg_tot_sb[kk] )
	nbg_clus_mag_err = 2.5 * nbg_tot_err[kk] / ( np.log(10) * nbg_tot_sb[kk] )

	idnn = np.isnan( nbg_clus_mag )
	nbg_clus_mag = nbg_clus_mag[ idnn == False ]
	nbg_clus_mag_err = nbg_clus_mag_err[ idnn == False ]


	##... figs
	#...compare the center region only
	color_lis = []

	for jj in np.arange( 11 ):
		color_lis.append( mpl.cm.rainbow_r( jj / 10) )

	dd_lis = np.arange(26, 33, 1)


	fig = plt.figure( figsize = (10.0, 4.8) )
	ax2 = fig.add_axes([0.08, 0.12, 0.39, 0.80])
	ax3 = fig.add_axes([0.59, 0.12, 0.39, 0.80])
	cb_ax2 = fig.add_axes( [0.47, 0.12, 0.02, 0.8] )

	tf = ax2.pcolormesh( cut_BG_sub_img, cmap = 'Greys', vmin = -2e-2, vmax = 2e-2, alpha = 0.75,)

	ax2.contour( mag_map_0, origin = 'lower', levels = [26, 100], colors = [ color_lis[0], 'w'], 
		extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

	ax2.contour( mag_map_1, origin = 'lower', levels = [27, 100], colors = [ color_lis[2], 'w'], 
		extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

	ax2.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis[3], 'w'], 
		extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

	ax2.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis[4], 'w'], 
		extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

	ax2.contour( mag_map_4, origin = 'lower', levels = [30, 100], colors = [ color_lis[6], 'w'], 
		extent = (x_peak - np.int( 3 * n500 ), x_peak + np.int( 3 * n500 ) + 1, y_peak - np.int( 3 * n500 ), y_peak + np.int( 3 * n500 ) + 1), )

	ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = [31, 32, 100], colors = [ color_lis[8], color_lis[10], 'w'], 
		extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut), )


	# dl0 = 250
	# dl1 = 450

	# Box = Rectangle( xy = (x_peak - dl0, y_peak - dl0), width = dl0, height = dl0, fill = False, 
	# 					ec = 'r', ls = '--', linewidth = 1, alpha = 0.75)
	# ax2.add_patch( Box )

	# Box = Rectangle( xy = (x_peak - dl1, y_peak - dl1), width = dl1, height = dl1, fill = False, 
	# 					ec = 'r', ls = '-', linewidth = 1, alpha = 0.75)
	# ax2.add_patch( Box )	


	me_map = mpl.colors.ListedColormap( 
		[ color_lis[0], color_lis[2], color_lis[3], color_lis[4], color_lis[6], color_lis[8], color_lis[10] ] )

	c_bounds = np.r_[ dd_lis[0] - 0.5, dd_lis + 0.5]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 31, 32], 
		spacing = 'proportional', orientation = 'vertical',)

	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '31', '32'], fontsize = 15)
	cbs.ax.invert_yaxis()

	ax2.set_xticklabels( labels = [] )
	ax2.set_yticklabels( labels = [] )

	x_ticks_0 = np.arange( xn - x_lo_cut, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_cut, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( -(len(x_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax2.set_xticks( x_ticks, minor = True, )
	ax2.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
	ax2.xaxis.set_ticks_position('bottom')
	ax2.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

	y_ticks_0 = np.arange( yn - y_lo_cut, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_cut, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( -(len(y_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax2.set_yticks( y_ticks, minor = True )
	ax2.set_yticklabels( labels = tick_lis, minor = True, fontsize = 15,)
	ax2.set_ylabel( '$\\mathrm{Y} \; [\\mathrm{M}pc] $', fontsize = 15,)
	ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)

	xp, yp = np.int( cut_BG_sub_img.shape[1] / 2 ), np.int( cut_BG_sub_img.shape[0] / 2 )
	ax2.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
	ax2.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )


	ax3.plot( tot_r[kk] / 1e3, clus_mag, ls = '--', color = 'b', alpha = 0.5, label = 'BCG + ICL + Background',)
	ax3.fill_between( tot_r[kk] / 1e3, y1 = clus_mag - clus_mag_err, y2 = clus_mag + clus_mag_err, color = 'b', alpha = 0.12,)

	ax3.plot(nbg_tot_r[kk][ idnn == False ] / 1e3, nbg_clus_mag, ls = '-', color = 'r', alpha = 0.5, label = 'BCG + ICL',)
	ax3.fill_between(nbg_tot_r[kk][ idnn == False ] / 1e3, y1 = nbg_clus_mag - nbg_clus_mag_err, 
		y2 = nbg_clus_mag + nbg_clus_mag_err, color = 'r', alpha = 0.12,)

	ax3.plot( rand_r[kk] / 1e3, modi_rand_mag, ls = ':', color = 'k', alpha = 0.5, label = 'Background',)
	ax3.fill_between( rand_r[kk] / 1e3, y1 = modi_rand_mag - modi_rand_mag_err, y2 = modi_rand_mag + modi_rand_mag_err, color = 'k', alpha = 0.12,)

	ax3.legend( loc = 1, fontsize = 14, frameon = False, markerfirst = False,)
	ax3.annotate( s = 'r-band', xy = (0.20, 0.15), xycoords = 'axes fraction', fontsize = 15,)

	ax3.axvline( x = phyR_psf / 1e3, ls = '-.', linewidth = 3.5, color = 'k', alpha = 0.20,)
	ax3.text( 3.1e-3, 27, s = 'PSF', fontsize = 22, rotation = 'vertical', color = 'k', alpha = 0.25, fontstyle = 'italic',)

	ax3.set_xlim( 3e-3, 1e0  )
	ax3.set_ylim( 20.5, 32.5 )

	ax3.invert_yaxis()
	ax3.set_xscale('log')
	ax3.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
	ax3.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)

	x_tick_arr = [ 1e-2, 1e-1, 1e0]
	tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']

	ax3.set_xticks( x_tick_arr )
	ax3.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
	ax3.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	ax3.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/mass_bin_%s_band_BG_sub_2D.png' % band[kk], dpi = 300)
	plt.close()

