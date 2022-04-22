import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

from light_measure import light_measure_weit
from img_sat_fig_out_mode import arr_jack_func
from img_sat_BG_sub_SB import aveg_BG_sub_func, stack_BG_sub_func


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to( U.arcsec )

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
L_pix = Da_ref * pixel * 1e3 / rad2arcsec
n50_kpc = 50 / L_pix
n100_kpc = 100 / L_pix


### === ### data load
##. sample information
bin_rich = [ 20, 30, 50, 210 ]


##. background at symmetry points
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_cat/nself_BG_SBs/'
# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_cat/self_shufl_BG/'
# path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/SBs/'
# BG_name = 'Background (symmetry points within cluster)'


##. shuffle test
BG_path = '/home/xkchen/Downloads/BGs/'          ###. shuffle more than one times
out_path = '/home/xkchen/Downloads/nBG_SBs/'
BG_name = 'Background (shuffle-0)'
path = '/home/xkchen/Downloads/SBs/'


##. Background images with BCG-masked
# BG_path = '/home/xkchen/figs_cp/BG_with_bcg_mask/'
# out_path = '/home/xkchen/figs_cp/nBG_SBs/'
# path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/SBs/'
# BG_name = 'Background (shuffle-0, no-BCG)'


sub_name = ['low-rich', 'medi-rich', 'high-rich']

def img_2D_stack():
	##... 2D stacked image compare
	for ll in range( 3 ):

		for kk in range( 2,3 ):

			band_str = band[ kk ]

			##.
			with h5py.File( path + 
				'Extend_BCGM_gri-common_%s_%s-band_Mean_jack_img_z-ref.h5' % (sub_name[ll], band_str), 'r') as f:
				tmp_sat_img = np.array( f['a'] )

			xn, yn = tmp_sat_img.shape[1] / 2, tmp_sat_img.shape[0] / 2

			id_nn = np.isnan( tmp_sat_img )
			eff_y, eff_x = np.where( id_nn == False )

			##. Background
			# with h5py.File( BG_path + 
			# 	'Extend_BCGM_gri-common_%s_%s-band_shufl-0_BG_Mean_jack_img_z-ref.h5' % (sub_name[ll], band_str), 'r') as f:
			# 	tmp_BG_img = np.array( f['a'] )

			with h5py.File( BG_path + 
				'Extend_BCGM_gri-common_%s_%s-band_shufl-1_BG_Mean_jack_img_z-ref.h5' % (sub_name[ll], band_str), 'r') as f:
				tmp_BG_img = np.array( f['a'] )

			x_ticks_0 = np.arange( xn, eff_x.min(), -1 * n100_kpc )
			x_ticks_1 = np.arange( xn, eff_x.max(), n100_kpc )
			x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]
			tick_R_x = np.r_[ np.arange( -(len(x_ticks_0) - 1 ) * 100, 0, 100), np.arange(0, 100 * ( len(x_ticks_1) ), 100) ]
			tick_lis_x = [ '%.0f' % (ll ) for ll in tick_R_x ]

			y_ticks_0 = np.arange( yn, eff_y.min(), -1 * n100_kpc )
			y_ticks_1 = np.arange( yn, eff_y.max(), n100_kpc )
			y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]
			tick_R_y = np.r_[ np.arange( -(len( y_ticks_0) - 1 ) * 100, 0, 100), np.arange(0, 100 * ( len( y_ticks_1) ), 100) ]
			tick_lis_y = [ '%.0f' % (ll ) for ll in tick_R_y ]


			fig = plt.figure( figsize = (20, 4.8) )
			ax0 = fig.add_axes([0.02, 0.09, 0.27, 0.84])
			ax1 = fig.add_axes([0.34, 0.09, 0.27, 0.84])
			ax2 = fig.add_axes([0.66, 0.09, 0.27, 0.84])

			ax0.set_title( 'satellite + background' )
			tf = ax0.imshow( tmp_sat_img / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = -1e-2, vmax = 1e-2,)
			plt.colorbar( tf, ax = ax0, fraction = 0.038, pad = 0.01, label = '$\\mu \; [nanomaggy \, / \, arcsec^2]$')

			ax0.set_xlim( eff_x.min(), eff_x.max() )
			ax0.set_ylim( eff_y.min(), eff_y.max() )

			ax0.set_xticklabels( labels = [] )
			ax0.set_yticklabels( labels = [] )

			ax0.set_xticks( x_ticks, minor = True, )
			ax0.set_xticklabels( labels = tick_lis_x, minor = True,)

			ax0.set_yticks( y_ticks, minor = True, )
			ax0.set_yticklabels( labels = tick_lis_y, minor = True,)

			ax0.set_ylabel('R [kpc]')
			ax0.set_xlabel('R [kpc]')
			ax0.tick_params( axis = 'both', which = 'both', direction = 'in')


			ax1.set_title( 'background' )
			tf = ax1.imshow( tmp_BG_img / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = -1e-2, vmax = 1e-2,)
			plt.colorbar( tf, ax = ax1, fraction = 0.038, pad = 0.01, label = '$\\mu \; [nanomaggy \, / \, arcsec^2]$')

			ax1.set_xlim( eff_x.min(), eff_x.max() )
			ax1.set_ylim( eff_y.min(), eff_y.max() )

			ax1.set_xticklabels( labels = [] )
			ax1.set_yticklabels( labels = [] )

			ax1.set_xticks( x_ticks, minor = True, )
			ax1.set_xticklabels( labels = tick_lis_x, minor = True,)

			ax1.set_yticks( y_ticks, minor = True, )
			ax1.set_yticklabels( labels = tick_lis_y, minor = True,)

			ax1.set_ylabel('R [kpc]')
			ax1.set_xlabel('R [kpc]')
			ax1.tick_params( axis = 'both', which = 'both', direction = 'in')


			ax2.set_title( 'satellite' )
			tf = ax2.imshow( ( tmp_sat_img - tmp_BG_img ) / pixel**2, origin = 'lower', cmap = 'bwr', vmin = -1e-2, vmax = 1e-2,)
			plt.colorbar( tf, ax = ax2, fraction = 0.038, pad = 0.01, label = '$\\mu \; [nanomaggy \, / \, arcsec^2]$')

			ax2.set_xlim( eff_x.min(), eff_x.max() )
			ax2.set_ylim( eff_y.min(), eff_y.max() )

			ax2.set_xticklabels( labels = [] )
			ax2.set_yticklabels( labels = [] )

			ax2.set_xticks( x_ticks, minor = True, )
			ax2.set_xticklabels( labels = tick_lis_x, minor = True,)

			ax2.set_yticks( y_ticks, minor = True, )
			ax2.set_yticklabels( labels = tick_lis_y, minor = True,)

			ax2.set_ylabel('R [kpc]')
			ax2.set_xlabel('R [kpc]')
			ax2.tick_params( axis = 'both', which = 'both', direction = 'in')

			plt.savefig(
				'/home/xkchen/rich_%d-%d_%s-band_2D_img_compare.png' % (bin_rich[ll], bin_rich[ll+1], band_str), dpi = 300)
			plt.close()

	return

img_2D_stack()


##... BG-sub SB(r) of sat. ( background stacking )
N_sample = 50

##. entire samples
for ll in range( 3 ):

	for kk in range( 2,3 ):

		band_str = band[ kk ]

		sat_sb_file = path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'
		out_file = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str

		# bg_sb_file = BG_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
		# bg_sb_file = BG_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-0_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
		bg_sb_file = BG_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-1_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'

		stack_BG_sub_func( sat_sb_file, bg_sb_file, band[ kk ], N_sample, out_file )        



##. figs comparison
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'g', 'r']
line_s = [ '--', '-', '-.']

line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


##. shuffle within subsample
for ll in range( 3 ):

	tot_R, tot_SB, tot_SB_err = [], [], []

	for kk in range( 2,3 ):

		with h5py.File( path + 
			'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band' % band[kk] + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tot_R.append( tt_r )
		tot_SB.append( tt_sb )
		tot_SB_err.append( tt_err )


	tot_bg_R, tot_bg_SB, tot_bg_err = [], [], []

	for kk in range( 2,3 ):

		# with h5py.File( BG_path + 
		#     'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl_BG' % band[kk] + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

		with h5py.File( BG_path + 
			'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-1_BG' % band[kk] + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tot_bg_R.append( tt_r )
		tot_bg_SB.append( tt_sb )
		tot_bg_err.append( tt_err )


	# tot_nbg_R, tot_nbg_SB, tot_nbg_err = [], [], []

	# for kk in range( 2,3 ):

	# 	dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_aveg-jack_BG-sub_SB.csv' % band[kk],)
	# 	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	# 	tot_nbg_R.append( tt_r )
	# 	tot_nbg_SB.append( tt_sb )
	# 	tot_nbg_err.append( tt_sb_err )


	##.
	for kk in range( 1 ):

		fig = plt.figure()
		ax1 = fig.add_axes([0.15, 0.12, 0.8, 0.8 ])

		ax1.errorbar( tot_R[kk], tot_SB[kk], yerr = tot_SB_err[kk], marker = '.', ls = '-', color = 'r',
			ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Satellite + Background',)

		ax1.plot( tot_bg_R[kk], tot_bg_SB[kk], ls = '--', color = 'k', alpha = 0.75, label = BG_name,)
		ax1.fill_between( tot_bg_R[kk], y1 = tot_bg_SB[kk] - tot_bg_err[kk],
						y2 = tot_bg_SB[kk] + tot_bg_err[kk], color = 'k', alpha = 0.15,)

		ax1.legend( loc = 3, frameon = False,)

		ax1.set_xscale('log')
		ax1.set_xlabel('R [kpc]')

		# ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.75, 0.10), xycoords = 'axes fraction',)
		ax1.annotate( s = line_name[ll] + ', i-band', xy = (0.75, 0.10), xycoords = 'axes fraction',)

		ax1.set_ylim( 2e-3, 2e-2 )
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
		ax1.set_yscale('log')

		# plt.savefig('/home/xkchen/%s_sat_%s-band_BG_compare.png' % (sub_name[ll], band[kk]), dpi = 300)
		plt.savefig('/home/xkchen/%s_sat_i-band_BG_compare.png' % sub_name[ll], dpi = 300)
		plt.close()

