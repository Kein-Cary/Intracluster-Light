import matplotlib as mpl
import matplotlib.pyplot as plt

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

##.
from light_measure import light_measure_weit
from img_sat_resamp import resamp_func
from img_sat_BG_extract import origin_img_cut_func

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to( U.arcsec )
pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === data load
BG_path = '/home/xkchen/figs_cp/T200_test/BGs/'
path = '/home/xkchen/figs_cp/T200_test/SBs/'


def img_2D_stack():

	Da_ref = Test_model.angular_diameter_distance( z_ref ).value
	L_pix = Da_ref * pixel * 1e3 / rad2arcsec
	n50_kpc = 50 / L_pix
	n100_kpc = 100 / L_pix

	##... 2D stacked image compare
	for kk in range( 3 ):

		band_str = band[ kk ]

		##.
		with h5py.File( path + 'T200-test_%s-band_Mean_jack_img_z-ref.h5' % band_str, 'r') as f:
			tmp_sat_img = np.array( f['a'] )

		xn, yn = tmp_sat_img.shape[1] / 2, tmp_sat_img.shape[0] / 2

		id_nn = np.isnan( tmp_sat_img )
		eff_y, eff_x = np.where( id_nn == False )

		##. Background
		with h5py.File( BG_path + 'T200_test_%s-band_BG_Mean_jack_img_z-ref.h5' % band_str, 'r') as f:
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

		ax0.set_title( 'satellite + background, %s-band' % band_str )
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
			'/home/xkchen/T200_%s-band_2D_img_compare.png' % band_str, dpi = 300)
		plt.close()

	return

img_2D_stack()
raise


tot_R, tot_SB, tot_SB_err = [], [], []
tot_bg_R, tot_bg_SB, tot_bg_err = [], [], []

for kk in range( 3 ):

	band_str = band[ kk ]

	##. satellite
	with h5py.File( path + 'T200-test_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tot_R.append( tt_r )
	tot_SB.append( tt_sb )
	tot_SB_err.append( tt_err )

	##. background 
	with h5py.File( BG_path + 'T200_test_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tot_bg_R.append( tt_r )
	tot_bg_SB.append( tt_sb )
	tot_bg_err.append( tt_err )


##... figs
for kk in range( 3 ):

	fig = plt.figure()
	ax1 = fig.add_axes([0.15, 0.12, 0.8, 0.8 ])

	ax1.errorbar( tot_R[kk], tot_SB[kk], yerr = tot_SB_err[kk], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Satellite + Background',)

	ax1.plot( tot_bg_R[kk], tot_bg_SB[kk], ls = '--', color = 'k', alpha = 0.75, label = 'Background',)
	ax1.fill_between( tot_bg_R[kk], y1 = tot_bg_SB[kk] - tot_bg_err[kk],
					y2 = tot_bg_SB[kk] + tot_bg_err[kk], color = 'k', alpha = 0.15,)

	ax1.legend( loc = 3, frameon = False,)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.annotate( s = '%s-band' % band[kk], xy = (0.75, 0.10), xycoords = 'axes fraction',)

	ax1.set_ylim( 2e-3, 2e-2 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	plt.savefig('/home/xkchen/T200_sat_%s-band_BG_compare.png' % band[kk], dpi = 300)
	plt.close()

