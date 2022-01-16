import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
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

#...
from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func

from light_measure import light_measure_weit
from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set

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

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

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


rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

BG_path = '/home/xkchen/figs/extend_bcgM_cat/BGs/'
img_path = '/home/xkchen/figs/extend_bcgM_cat/SBs/'

diff_path = '/home/xkchen/figs/extend_bcgM_cat/BCG_Mstar_diff_img/'


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


	tmp_diff_mag = []
	tmp_diff_R = []

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

		tmp_diff_mag.append( _tt_mu_F( nbg_low_r[kk] ) - nbg_low_sb[kk] )
		tmp_diff_R.append( nbg_low_r[kk] )


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


	idx0 = ( tmp_diff_R[0] >= 0.01 ) & ( tmp_diff_R[0] <= 0.05 )
	d_mag0 = tmp_diff_mag[0][idx0]

	idx1 = ( tmp_diff_R[1] >= 0.01 ) & ( tmp_diff_R[1] <= 0.05 )
	d_mag1 = tmp_diff_mag[1][idx1]

	idx2 = ( tmp_diff_R[2] >= 0.01 ) & ( tmp_diff_R[2] <= 0.05 )
	d_mag2 = tmp_diff_mag[2][idx2]

	raise

	return

# mu_1D_compare()



#.
L_pix = Da_ref * 10**3 * pixel / rad2asec

n500 = 500 / L_pix
n1000 = 1000 / L_pix
n900 = 900 / L_pix
n200 = 200 / L_pix

#. estimate Background subtract image

kk = 0 ## 0, 1, 2 --> r, g, i band

subset_img = []

for ll in ( 1, 0 ):

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

	cut_BG_arr = shift_rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ] / pixel**2
	id_nan = np.isnan( cut_BG_arr )
	cut_BG_arr[id_nan] = 0.

	print( np.std(cut_BG_arr) )

	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_BG_sub_img )
	cut_BG_sub_img[id_nan] = 0.

	y_peak, x_peak = np.where( cut_BG_sub_img == np.nanmax( cut_BG_sub_img ) )
	y_peak, x_peak = y_peak[0], x_peak[0]
	cen_region = cut_BG_sub_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, 
								 x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]

	subset_img.append( cen_region )

	##... save in fits files (SB in each pixel)
	# Nx1, Nx2 = cen_region.shape[0], cen_region.shape[1]

	# keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'Z_OBS', 'P_SCALE']
	# value = ['T', 32, 2, Nx1, Nx2, x_peak, y_peak, z_ref, pixel]
	# ff = dict( zip( keys, value ) )
	# fill = fits.Header( ff )
	# fits.writeto( diff_path + '%s_%s-band_stacked-img_with-BG-sub.fits' % (cat_lis[ ll ], band[kk]), 
	# 				cen_region, header = fill, overwrite = True )


xp, yp = np.int( cen_region.shape[1] / 2 ), np.int( cen_region.shape[0] / 2 )
diffi_img = subset_img[0] - subset_img[1]

##... save in fits files (SB in each pixel)
# Nx1, Nx2 = diffi_img.shape[0], diffi_img.shape[1]
# keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'Z_OBS', 'P_SCALE']
# value = ['T', 32, 2, Nx1, Nx2, x_peak, y_peak, z_ref, pixel]
# ff = dict( zip( keys, value ) )
# fill = fits.Header( ff )
# fits.writeto( diff_path + 'BCG-M_bin_%s-band_stacked-img_diffi_with-BG-sub.fits' % band[kk], diffi_img, header = fill, overwrite = True )


##... SB profile of the diffi_img
# R_bins = np.logspace(0, np.log10(940), 45)

# calib_img = diffi_img * pixel**2
# weit_arr = np.ones( ( diffi_img.shape[0], diffi_img.shape[1] ),)

# Intns, phy_r, Intns_err, npix, nratio = light_measure_weit( calib_img, weit_arr, pixel, xp, yp, z_ref, R_bins )

# id_nul = npix < 1.
# R_obs, mu_obs, mu_err = phy_r[ id_nul == False ], Intns[ id_nul == False ], Intns_err[ id_nul == False ]

# keys = ['r', 'sb', 'sb_err']
# values = [ R_obs, mu_obs, mu_err ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( diff_path + 'BCG-M_bin_%s-band_diffi-img_SB.csv' % band[ kk ] )


# tmp_r, tmp_sb, tmp_err = [], [], []

# for oo in range( 3 ):
	
# 	dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/BCG_Mstar_diffi_img/' + 
# 						'BCG-M_bin_%s-band_diffi-img_SB.csv' % band[oo] )
# 	tt_r = np.array( dat['r'])
# 	tt_sb = np.array( dat['sb'])
# 	tt_err = np.array( dat['sb_err'])

# 	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
# 	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

# 	tmp_r.append( tt_r )
# 	tmp_sb.append( tt_mag )
# 	tmp_err.append( tt_mag_err )

# plt.figure()
# ax = plt.subplot(111)

# for oo in range( 3 ):
# 	ax.plot( tmp_r[oo], tmp_sb[oo], ls = '-', color = color_s[oo], alpha = 0.75, label = '%s band' % band[oo],)

# ax.legend( loc = 1, frameon = False,)
# ax.set_xlim(1e1, 1e3)
# ax.set_xscale('log')
# ax.set_xlabel('R [kpc]')

# ax.set_ylim( 21, 33 )
# ax.invert_yaxis()
# ax.set_ylabel('$\\mu \; [mag \; arcsec^{-2}]$')
# ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
# ax.tick_params( axis = 'both', which = 'both', direction = 'in',)

# plt.savefig('/home/xkchen/BCG-M_bin_diffi_img_SB.png', dpi = 300)
# plt.close()



##. rs estimate
Mh_clus = 10**14.41  # M_sun
c_mass = np.array( [ 6.95, 5.87 ] )

mrho_zref = rhom_set( z_ref )[1]
mrho_zref = mrho_zref * h**2  # M_sun / kpc^3

R200m = ( 3 * Mh_clus / (4 * np.pi * mrho_zref * 200) )**(1 / 3)

R_s = R200m / c_mass
R_s_pix = R_s / L_pix



##... figs
# fig = plt.figure( figsize = (13.80, 4.8 ) )
# ax0 = fig.add_axes( [0.06, 0.12, 0.25, 0.85] )
# sub_ax0 = fig.add_axes( [0.311, 0.18, 0.015, 0.726] )

# ax1 = fig.add_axes( [0.368, 0.12, 0.25, 0.85] )
# sub_ax1 = fig.add_axes( [0.619, 0.18, 0.015, 0.726] )

# ax2 = fig.add_axes( [0.676, 0.12, 0.25, 0.85] )
# sub_ax2 = fig.add_axes( [0.927, 0.18, 0.015, 0.726] )

# axes = [ ax0, ax1, ax2 ]
# sub_axes = [ sub_ax0, sub_ax1, sub_ax2 ]


fig = plt.figure( figsize = (13.20, 4.8 ) )
ax0 = fig.add_axes( [0.07, 0.08, 0.27, 0.90] )
# sub_ax0 = fig.add_axes( [0.075, 0.25, 0.12, 0.025] )

ax1 = fig.add_axes( [0.35, 0.08, 0.27, 0.90] )

ax2 = fig.add_axes( [0.63, 0.08, 0.27, 0.90] )
sub_ax2 = fig.add_axes( [0.905, 0.157, 0.015, 0.745] )

axes = [ ax0, ax1, ax2 ]

N_sigma = 15 # 35, 30, 25, 20, 15, 10
mag_levels = np.array( [25, 26, 27, 28, 29] )

for ll in range( 2 ):

	_ll_img = subset_img[ ll ]

	filt_BG_sub_img = ndimage.gaussian_filter( _ll_img, sigma = N_sigma,)

	ax = axes[ ll ]
	# _c_ax = sub_axes[ ll ]


	##... LogNormal
	# imf = ax.imshow( filt_BG_sub_img, origin = 'lower', cmap = 'rainbow', vmin = -5e-4, vmax = 2e-1,
	# 	norm = mpl.colors.SymLogNorm( linthresh = 3e-4, linscale = 5e-3, base = 10),)

	# cbs = fig.colorbar( imf, cax = _c_ax, orientation = 'vertical', ticks = [ 1e-4, 1e-3, 1e-2, 1e-1],)
	# cbs.ax.tick_params( axis = 'y', which = 'both', direction = 'in',)
	# cbs.ax.set_yticklabels( ['1e-4', '1e-3', '1e-2', '1e-1'] )
	# # cbs.ax.yaxis.set_minor_locator( ticker.LogitLocator( minor = True,) )
	# cbs.ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


	##... linear
	# imf = ax.imshow( filt_BG_sub_img, origin = 'lower', cmap = 'rainbow', vmin = 1e-4, vmax = 1e-2,)
	# cbs = fig.colorbar( imf, cax = _c_ax, orientation = 'vertical', )
	# cbs.ax.tick_params( axis = 'y', which = 'both', direction = 'in',)
	# cbs.formatter.set_powerlimits( (0,-3) )
	# cbs.ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


	imf = ax.imshow( filt_BG_sub_img, origin = 'lower', cmap = 'seismic', vmin = -1e-2, vmax = 1e-2,)

	##... adjust in central region
	# imf.cmap.set_over('w')

	# mag_map = 22.5 - 2.5 * np.log10( filt_BG_sub_img )
	# cc_mag_map = mag_map.copy()

	# id_nul = mag_map >= 29.38
	# cc_mag_map[ id_nul ] = np.nan

	# ax.imshow( cc_mag_map, origin = 'lower', cmap = 'Greys_r', vmin = 23, vmax = 33)
	# ax.contour( mag_map, origin = 'lower', levels = mag_levels, cmap = 'autumn_r', linewidths = 0.75,)

	##... special region
	clust = Circle( xy = (xp, yp), radius = R_s_pix[ll], fill = False, ec = 'k', ls = '-', linewidth = 1,)
	ax.add_patch( clust )

	clust = Circle( xy = (xp, yp), radius = n200, fill = False, ec = 'k', ls = '--', linewidth = 1,)
	ax.add_patch(clust)

ax0.set_xticklabels( labels = [] )
ax0.set_yticklabels( labels = [] )

#. legend for the two circles
ax0.axhline( y = -5 * n1000, ls = '-', label = '$r_{s}$', color = 'k',)
ax0.axhline( y = -6 * n1000, ls = '--', label = '$R_{\\mathrm{SOI}}$', color = 'k',)
ax0.legend( loc = 3, frameon = False, fontsize = 15,) # facecolor = 'w', edgecolor = 'w', framealpha = 0.75)

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
ax0.set_ylabel( '$\\mathrm{Y} \; [\\mathrm{M}pc] $', fontsize = 15,)

ax0.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
ax0.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
ax0.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
ax0.annotate( text = fig_name[1], xy = (0.05, 0.90), fontsize = 18, xycoords = 'axes fraction', color = 'k',
				bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 0.75, 'pad': 2 },)

#. add colorbar for contour
# cen_colors = []
# for oo in range( len(mag_levels) ):
# 	cen_colors.append( mpl.cm.autumn_r( oo / len(mag_levels) ) )

# me_map = mpl.colors.ListedColormap( cen_colors )
# c_bounds = np.r_[ mag_levels[0] - 0.5, mag_levels + 0.5]

# me_norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

# cbs = mpl.colorbar.ColorbarBase( ax = sub_ax0, cmap = me_map, norm = me_norm, extend = 'neither', 
# 	ticks = list( mag_levels ), spacing = 'proportional', orientation = 'horizontal',)

# cbs.ax.set_xlabel( '$mag \; arcsec^{-2}$', fontsize = 12, labelpad = -0.2,)
# cbs.ax.set_xticklabels( ['%d' % pp for pp in mag_levels ], fontsize = 12)
# cbs.ax.tick_params( axis = 'x', which = 'major', direction = 'in')


ax1.set_xticklabels( labels = [] )
ax1.set_yticklabels( labels = [] )

ax1.set_xticks( x_ticks, minor = True, )
ax1.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

ax1.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
ax1.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
ax1.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)
ax1.annotate( text = fig_name[0], xy = (0.05, 0.90), fontsize = 18, xycoords = 'axes fraction', color = 'k',
				bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 0.75, 'pad': 2 },)


#. diffi image
_ll_img = diffi_img
filt_diffi_img = ndimage.gaussian_filter( _ll_img, sigma = N_sigma,)

##... LogNormal
# imf = ax2.imshow( filt_BG_sub_img, origin = 'lower', cmap = 'bwr', vmin = -3e-2, vmax = 3e-2, 
# 			norm = mpl.colors.SymLogNorm( linthresh = 6e-4, linscale = 1e-3, base = 10),)

# cbs = fig.colorbar( imf, cax = sub_ax2, orientation = 'vertical', ticks = [ -1e-2, -1e-3, 0, 1e-3, 1e-2 ],
# 		label = '$\\mu_{%s} \; [nanomaggies \; arcsec^{-2} ]$' % band[ kk ], )
# cbs.ax.tick_params( axis = 'y', which = 'both', direction = 'in',)
# cbs.ax.set_yticklabels( ['-1e-2', '-1e-3', '0', '1e-3', '1e-2'] )
# cbs.ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


##... linear
imf = ax2.imshow( filt_diffi_img, origin = 'lower', cmap = 'seismic', vmin = -1e-2, vmax = 1e-2,)

cbs = fig.colorbar( imf, cax = sub_ax2, orientation = 'vertical',)
cbs.ax.tick_params( axis = 'y', which = 'both', direction = 'in', labelsize = 15,)
cbs.formatter.set_powerlimits( (0,-3) )
cbs.ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
cbs.ax.set_ylabel('$\\mu_{%s} \; [nanomaggies \; arcsec^{-2} ]$' % band[ kk ], fontsize = 15,)

#. adjust in central region
# imf.cmap.set_over('w')
# mag_map = 22.5 - 2.5 * np.log10( filt_diffi_img )
# cc_mag_map = mag_map.copy()

# id_nul = mag_map >= 29.426
# cc_mag_map[ id_nul ] = np.nan

# ax2.imshow( cc_mag_map, origin = 'lower', cmap = 'Greys_r', vmin = 23, vmax = 33)
# ax2.contour( mag_map, origin = 'lower', levels = mag_levels, cmap = 'autumn_r', linewidths = 0.75,)


clust = Circle( xy = (xp, yp), radius = n200, fill = False, ec = 'k', ls = '--', linewidth = 1,)
ax2.add_patch(clust)

ax2.set_xticklabels( labels = [] )
ax2.set_yticklabels( labels = [] )

ax2.set_xticks( x_ticks, minor = True, )
ax2.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
ax2.xaxis.set_ticks_position('bottom')
ax2.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

ax2.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
ax2.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )
ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)

ax2.annotate( text = 'Difference', xy = (0.05, 0.90), fontsize = 18, xycoords = 'axes fraction', color = 'k',
			bbox = {'facecolor': 'w', 'edgecolor':'w', 'alpha': 0.75, 'pad': 2 },)

# plt.savefig('/home/xkchen/BCG-M_bin_%s-band_diffi_img_%d-sigma.png' % (band[ kk ], N_sigma), dpi = 100)
plt.savefig('/home/xkchen/BCG_M_bin_%s_band_diffi_img.pdf' % band[ kk ], dpi = 100)
plt.close()

