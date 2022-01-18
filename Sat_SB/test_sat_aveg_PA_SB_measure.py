import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds

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

from img_sat_PA_SB_profile import PA_SB_Zx_func
from img_sat_PA_SB_profile import aveg_jack_PA_SB_func
from light_measure import jack_SB_func


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

Da_ref = Test_model.angular_diameter_distance( z_ref ).value

rad2asec = U.rad.to(U.arcsec)


### === light profile along different direction
def bin_divid_test():

	path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/SBs/'
	BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/BGs/'
	cp_BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufle_test/BGs/'


	cat_lis = ['inner', 'middle', 'outer']
	fig_name = ['Inner', 'Middle', 'Outer']

	color_s = [ 'r', 'g', 'darkred' ]

	line_c = [ 'b', 'r', 'm']
	line_s = [ '--', '-', '-.']


	#. satellite image
	tmp_img = []

	for mm in range( 3 ):

		sub_img = []

		for kk in range( 3 ):

			with h5py.File( path + 
				'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_Mean_jack_img_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

				tt_img = np.array( f['a'] )

			sub_img.append( tt_img )

		tmp_img.append( sub_img )


	test_img = tmp_img[0][0]

	Nx, Ny = test_img.shape[1], test_img.shape[0]
	xn, yn = np.int( Nx / 2), np.int( Ny / 2)

	x_pix, y_pix = np.arange( Nx ), np.arange( Ny )
	lx, ly = np.meshgrid( x_pix, y_pix )


	dR = np.sqrt( ( (2 * lx + 1) / 2 - (2 * xn + 1) / 2)**2 + 
				( (2 * ly + 1) / 2 - (2 * yn + 1) / 2)**2 )

	diff_x = (2 * lx + 1) / 2 - (2 * xn + 1) / 2
	diff_y = (2 * ly + 1) / 2 - (2 * yn + 1) / 2


	r_bins = np.logspace( 0, 2.56, 35 )

	weit_arry = np.ones( (Ny, Nx),)

	h_array, v_array = PA_SB_Zx_func( test_img, weit_arry, pixel, xn, yn, z_ref, r_bins )

	SB_h_R, SB_h, SB_h_err, N_pix_h, nsum_ratio_h = h_array[:]
	SB_v_R, SB_v, SB_v_err, N_pix_v, nsum_ratio_v = v_array[:]

	id_hx = N_pix_h > 0.
	SB_h_R, SB_h, SB_h_err = SB_h_R[ id_hx ], SB_h[ id_hx ], SB_h_err[ id_hx ]

	id_vx = N_pix_v > 0.
	SB_v_R, SB_v, SB_v_err = SB_v_R[ id_vx ], SB_v[ id_vx ], SB_v_err[ id_vx ]


	plt.figure()
	ax = plt.subplot(111)

	ax.errorbar( SB_h_R, SB_h, yerr = SB_h_err, marker = '', ls = '-', color = 'r',
				ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, )

	ax.errorbar( SB_v_R, SB_v, yerr = SB_v_err, marker = '', ls = '-', color = 'b',
				ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, )

	ax.set_xscale('log')
	ax.set_yscale('log')

	plt.show()

	raise

	fig = plt.figure( figsize = (19.2, 4.8) )
	ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
	ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
	ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

	ax0.imshow( test_img / pixel**2, origin = 'lower', cmap = 'winter', norm = mpl.colors.LogNorm(),)
	ax1.imshow( test_img / pixel**2, origin = 'lower', cmap = 'winter', norm = mpl.colors.LogNorm(),)
	ax2.imshow( test_img / pixel**2, origin = 'lower', cmap = 'winter', norm = mpl.colors.LogNorm(),)

	for kk in range( 20, len( r_bins ) - 1 ):
		#. points located in radius bin
		id_vx = ( dR >= r_bins[ kk ] ) & ( dR < r_bins[ kk+1 ] )
		set_px, set_py = np.where( id_vx )

		#. points along the row direction
		id_ux_0 = np.abs( diff_x ) < r_bins[ kk ]
		id_uy_0 = ( np.abs( diff_y ) >= r_bins[ kk ] ) & ( np.abs( diff_y ) < r_bins[ kk+1 ] )

		id_lim_0 = ( id_ux_0 & id_uy_0 ) & id_vx

		set_px_0, set_py_0 = np.where( id_lim_0 )


		#. points along the columns direction
		id_ux_1 = np.abs( diff_y ) < r_bins[ kk ]
		id_uy_1 = ( np.abs( diff_x ) >= r_bins[ kk ] ) & ( np.abs( diff_x ) < r_bins[ kk+1 ] )

		id_lim_1 = ( id_ux_1 & id_uy_1 ) & id_vx

		set_px_1, set_py_1 = np.where( id_lim_1 )

		ax0.plot( set_px, set_py, 'o', color = mpl.cm.rainbow( kk / len(r_bins) ), alpha = 0.25, markersize = 4,)
		ax1.plot( set_px_0, set_py_0, 's', color = mpl.cm.rainbow( kk / len(r_bins) ), alpha = 0.25, markersize = 4,)
		ax2.plot( set_px_1, set_py_1, 's', color = mpl.cm.rainbow( kk / len(r_bins) ), alpha = 0.25, markersize = 4,)

		ax0.set_xlim( xn - 400, xn + 400 )
		ax0.set_ylim( xn - 400, xn + 400 )

		ax1.set_xlim( xn - 400, xn + 400 )
		ax1.set_ylim( xn - 400, xn + 400 )

		ax2.set_xlim( xn - 400, xn + 400 )
		ax2.set_ylim( xn - 400, xn + 400 )

	for kk in range( len(r_bins) ):
		clust = Circle( xy = (xn, yn), radius = r_bins[kk], fill = False, ec = 'k', ls = '-', linewidth = 1,)
		ax0.add_patch( clust )

		clust = Circle( xy = (xn, yn), radius = r_bins[kk], fill = False, ec = 'k', ls = '-', linewidth = 1,)
		ax1.add_patch( clust )

		clust = Circle( xy = (xn, yn), radius = r_bins[kk], fill = False, ec = 'k', ls = '-', linewidth = 1,)
		ax2.add_patch( clust )

	plt.savefig('/home/xkchen/PA_binned_test.png', dpi = 300)
	plt.show()

	return

# bin_divid_test()
# raise


### === measure test
"""
img_path = '/home/xkchen/fig_tmp/Extend_Mbcg_sat_stack/'
SB_path = '/home/xkchen/figs/'

N_bin = 100   ## number of jackknife subsample
n_rbins = 35

cat_lis = ['inner', 'middle', 'outer']

for ll in range( 3 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		J_sub_img = img_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = img_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'

		J_sub_sb = SB_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.csv'
		jack_SB_file = SB_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.csv'

		aveg_jack_PA_SB_func( J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, N_bin, n_rbins, pixel, z_ref, band_str)

"""


### === SB profile compare
cat_lis = ['inner', 'middle', 'outer']
fig_name = ['Inner', 'Middle', 'Outer']

tmp_h_R, tmp_h_sb, tmp_h_err = [], [], []
tmp_v_R, tmp_v_sb, tmp_v_err = [], [], []

for mm in range( 3 ):

	sub_R_0, sub_sb_0, sub_err_0 = [], [], []
	sub_R_1, sub_sb_1, sub_err_1 = [], [], []

	for kk in range( 3 ):

		band_str = band[ kk ]

		dat = pds.read_csv('/home/xkchen/figs_cp/aveg_PA_SBs/' + 
						'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_Mean_jack_SB-pro_z-ref.csv' % (cat_lis[mm], band_str),)
		sub_R_h = np.array( dat['r_h'] )
		sub_sb_h = np.array( dat['sb_h'] )
		sb_err_h = np.array( dat['sb_err_h'] )

		sub_R_v = np.array( dat['r_v'] )
		sub_sb_v = np.array( dat['sb_v'] )
		sb_err_v = np.array( dat['sb_err_v'] )

		#. 
		sub_R_0.append( sub_R_h )
		sub_sb_0.append( sub_sb_h )
		sub_err_0.append( sb_err_h )

		sub_R_1.append( sub_R_v )
		sub_sb_1.append( sub_sb_v )
		sub_err_1.append( sb_err_v )

	tmp_h_R.append( sub_R_0 )
	tmp_h_sb.append( sub_sb_0 )
	tmp_h_err.append( sub_err_0 )

	tmp_v_R.append( sub_R_1 )
	tmp_v_sb.append( sub_sb_1 )
	tmp_v_err.append( sub_err_1 )


tmp_R, tmp_sb, tmp_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/SBs/' + 
					'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	tmp_R.append( sub_R )
	tmp_sb.append( sub_sb )
	tmp_err.append( sub_err )


for kk in range( 3 ):

	fig = plt.figure( )
	ax = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	ax.errorbar( tmp_R[0][kk], tmp_sb[0][kk], yerr = tmp_err[0][kk], marker = '', ls = '-', color = 'b',
				ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.5, label = fig_name[0],)

	ax.errorbar( tmp_R[1][kk], tmp_sb[1][kk], yerr = tmp_err[1][kk], marker = '', ls = '-', color = 'g',
				ecolor = 'g', mfc = 'none', mec = 'g', capsize = 1.5, alpha = 0.5, label = fig_name[1],)

	l1 = ax.errorbar( tmp_R[2][kk], tmp_sb[2][kk], yerr = tmp_err[2][kk], marker = '', ls = '-', color = 'r',
				ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.5, label = fig_name[2],)


	ax.errorbar( tmp_h_R[0][kk], tmp_h_sb[0][kk], yerr = tmp_h_err[0][kk], marker = '', ls = '--', color = 'b',
				ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.5,)

	ax.errorbar( tmp_h_R[1][kk], tmp_h_sb[1][kk], yerr = tmp_h_err[1][kk], marker = '', ls = '--', color = 'g',
				ecolor = 'g', mfc = 'none', mec = 'g', capsize = 1.5, alpha = 0.5,)

	l2 = ax.errorbar( tmp_h_R[2][kk], tmp_h_sb[2][kk], yerr = tmp_h_err[2][kk], marker = '', ls = '--', color = 'r',
				ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.5,)


	ax.errorbar( tmp_v_R[0][kk], tmp_v_sb[0][kk], yerr = tmp_v_err[0][kk], marker = '', ls = ':', color = 'b',
				ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.5,)

	ax.errorbar( tmp_v_R[1][kk], tmp_v_sb[1][kk], yerr = tmp_v_err[1][kk], marker = '', ls = ':', color = 'g',
				ecolor = 'g', mfc = 'none', mec = 'g', capsize = 1.5, alpha = 0.5,)

	l3 = ax.errorbar( tmp_v_R[2][kk], tmp_v_sb[2][kk], yerr = tmp_v_err[2][kk], marker = '', ls = ':', color = 'r',
				ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.5,)

	#. ratio
	interp_F0 = interp.interp1d( tmp_R[0][kk], tmp_sb[0][kk], kind = 'cubic', fill_value = 'extrapolate')
	interp_F1 = interp.interp1d( tmp_R[1][kk], tmp_sb[1][kk], kind = 'cubic', fill_value = 'extrapolate')
	interp_F2 = interp.interp1d( tmp_R[2][kk], tmp_sb[2][kk], kind = 'cubic', fill_value = 'extrapolate')

	sub_ax.plot( tmp_h_R[0][kk], tmp_h_sb[0][kk] / interp_F0( tmp_h_R[0][kk] ), ls = '--', color = 'b', alpha = 0.5,)
	sub_ax.plot( tmp_h_R[1][kk], tmp_h_sb[1][kk] / interp_F1( tmp_h_R[1][kk] ), ls = '--', color = 'g', alpha = 0.5,)
	sub_ax.plot( tmp_h_R[2][kk], tmp_h_sb[2][kk] / interp_F2( tmp_h_R[2][kk] ), ls = '--', color = 'r',)

	sub_ax.plot( tmp_v_R[0][kk], tmp_v_sb[0][kk] / interp_F0( tmp_v_R[0][kk] ), ls = ':', color = 'b', alpha = 0.5)
	sub_ax.plot( tmp_v_R[1][kk], tmp_v_sb[1][kk] / interp_F1( tmp_v_R[1][kk] ), ls = ':', color = 'g', alpha = 0.5)
	sub_ax.plot( tmp_v_R[2][kk], tmp_v_sb[2][kk] / interp_F2( tmp_v_R[2][kk] ), ls = ':', color = 'r', alpha = 0.5)

	sub_ax.plot( tmp_R[0][kk], tmp_sb[0][kk] / tmp_sb[0][kk], ls = '-', color = 'b', alpha = 0.5,)
	sub_ax.fill_between( tmp_R[0][kk], y1 = ( tmp_sb[0][kk] - tmp_err[0][kk] ) / tmp_sb[0][kk], 
					y2 = ( tmp_sb[0][kk] + tmp_err[0][kk] ) / tmp_sb[0][kk], color = 'b', alpha = 0.12,)

	sub_ax.plot( tmp_R[1][kk], tmp_sb[1][kk] / tmp_sb[1][kk], ls = '-', color = 'g', alpha = 0.5,)
	sub_ax.fill_between( tmp_R[1][kk], y1 = ( tmp_sb[1][kk] - tmp_err[1][kk] ) / tmp_sb[1][kk], 
					y2 = ( tmp_sb[1][kk] + tmp_err[1][kk] ) / tmp_sb[1][kk], color = 'g', alpha = 0.12,)

	sub_ax.plot( tmp_R[2][kk], tmp_sb[2][kk] / tmp_sb[2][kk], ls = '-', color = 'r', alpha = 0.5,)
	sub_ax.fill_between( tmp_R[2][kk], y1 = ( tmp_sb[2][kk] - tmp_err[2][kk] ) / tmp_sb[2][kk], 
					y2 = ( tmp_sb[2][kk] + tmp_err[2][kk] ) / tmp_sb[2][kk], color = 'r', alpha = 0.12,)


	legend_2 = ax.legend( handles = [l1, l2, l3], 
			labels = ['Azimuth average ($\\mu_{A}$)', 'Horizontal average ($\\mu_{H}$)', 'Vertical average ($\\mu_{V}$)'], loc = 'upper center', frameon = False,)
	ax.legend( loc = 1, frameon = False,)
	ax.add_artist( legend_2 )
	ax.annotate( s = '%s-band' % band[ kk ], xy = (0.85, 0.70), xycoords = 'axes fraction',)

	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')

	ax.set_ylim( 2e-3, 6e0)
	ax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax.set_yscale('log')

	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xscale('log')
	sub_ax.set_xlabel('$R \; [kpc]$')
	sub_ax.set_ylabel('$\\mu \; / \; \\mu_{A}$', labelpad = 8)

	sub_ax.set_ylim( 0.9, 1.55 )

	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_sat_BG_compare.png' % band[kk], dpi = 300)
	plt.close()

