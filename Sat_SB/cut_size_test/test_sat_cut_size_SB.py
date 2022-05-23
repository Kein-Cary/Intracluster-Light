import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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

rad2asec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25

band = ['r', 'g', 'i']

Da_ref = Test_model.angular_diameter_distance( z_ref ).value

L_pix = Da_ref * 10**3 * pixel / rad2asec
R400 = 400 / L_pix
pix_area = pixel**2


### === ### data load
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/cutsize_test/SBs/'

pre_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/SBs/'
pre_BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/BGs/'



#. fixed i_Mag10 case ( divided by 0.191 * R200m + fixed i_Mag_10 )
cat_lis = ['inner', 'middle', 'outer']
fig_name = ['Inner', 'Middle', 'Outer']

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r', 'm']
line_s = [ '--', '-', '-.']


id_size = ['small', 'wider']


##... previous cutsize
pre_R, pre_SB, pre_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( pre_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	pre_R.append( sub_R )
	pre_SB.append( sub_sb )
	pre_err.append( sub_err )


tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for mm in range( 3 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for kk in range( 3 ):

		with h5py.File( pre_BG_path + 
			'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		_sub_bg_R.append( tt_r )
		_sub_bg_sb.append( tt_sb )
		_sub_bg_err.append( tt_err )

	tmp_bg_R.append( _sub_bg_R )
	tmp_bg_SB.append( _sub_bg_sb )
	tmp_bg_err.append( _sub_bg_err )



##... sat SBs
sm_tt_R, sm_tt_sb, sm_tt_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( path + 
			'Sat_iMag10-fix_%s_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_size[0], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	sm_tt_R.append( sub_R )
	sm_tt_sb.append( sub_sb )
	sm_tt_err.append( sub_err )


wi_tt_R, wi_tt_sb, wi_tt_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( path + 
			'Sat_iMag10-fix_%s_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_size[1], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	wi_tt_R.append( sub_R )
	wi_tt_sb.append( sub_sb )
	wi_tt_err.append( sub_err )



### === figs

for kk in range( 3 ):

	top_fig = plt.figure( )
	ax2 = top_fig.add_axes( [0.11, 0.10, 0.85, 0.85] )

	for mm in range( 3 ):

		fig = plt.figure( )
		ax1 = fig.add_axes( [0.11, 0.10, 0.85, 0.85] )

		l1, = ax1.plot( sm_tt_R[mm][kk], sm_tt_sb[mm][kk], ls = ':', color = color_s[kk], alpha = 0.75,)
		l2, = ax1.plot( pre_R[mm][kk], pre_SB[mm][kk], ls = '--', color = color_s[kk], alpha = 0.75,)
		l3, = ax1.plot( wi_tt_R[mm][kk], wi_tt_sb[mm][kk], ls = '-', color = color_s[kk], alpha = 0.75, label = fig_name[mm],)

		#. BG_profile
		l4, = ax1.plot( tmp_bg_R[mm][kk], tmp_bg_SB[mm][kk], ls = '-.', color = color_s[kk], alpha = 0.75,)

		legend_2 = ax1.legend( handles = [l1, l2, l3, l4], labels = ['$0.5 \, R_{cut}$', '$R_{cut}\, = \,320$', '$1.5 \, R_{cut}$', 'BG'], 
					loc = 'upper center', frameon = False,)
		ax1.legend( loc = 1, frameon = False,)
		ax1.add_artist( legend_2 )

		ax1.annotate( s = '%s-band' % band[kk], xy = (0.75, 0.50), xycoords = 'axes fraction',)

		ax1.set_xscale('log')
		ax1.set_xlabel('R [kpc]')

		ax1.set_ylim( 1e-3, 6e0)
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
		ax1.set_yscale('log')

		plt.savefig('/home/xkchen/%s_%s_sat_BG_compare.png' % (cat_lis[mm], band[ kk ]), dpi = 300)
		plt.close()


		#. combine
		l1, = ax2.plot( sm_tt_R[mm][kk], sm_tt_sb[mm][kk], ls = ':', color = color_s[mm], alpha = 0.75,)
		l2, = ax2.plot( pre_R[mm][kk], pre_SB[mm][kk], ls = '--', color = color_s[mm], alpha = 0.75,)
		l3, = ax2.plot( wi_tt_R[mm][kk], wi_tt_sb[mm][kk], ls = '-', color = color_s[mm], alpha = 0.75, label = fig_name[mm],)

		#. BG_profile
		l4, = ax2.plot( tmp_bg_R[mm][kk], tmp_bg_SB[mm][kk], ls = '-.', color = color_s[mm], alpha = 0.75,)

	legend_2 = ax2.legend( handles = [l1, l2, l3, l4], labels = ['$0.5 \, R_{cut}$', '$R_{cut}\, = \,320$', '$1.5 \, R_{cut}$', 'BG'], 
				loc = 'upper center', frameon = False,)
	ax2.legend( loc = 1, frameon = False,)
	ax2.add_artist( legend_2 )

	ax2.annotate( s = '%s-band' % band[kk], xy = (0.75, 0.50), xycoords = 'axes fraction',)

	ax2.set_xscale('log')
	ax2.set_xlabel('R [kpc]')

	ax2.set_ylim( 1e-3, 6e0)
	ax2.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax2.set_yscale('log')

	plt.savefig('/home/xkchen/%s_sat_BG_compare.png' % band[ kk ], dpi = 300)
	plt.close()


raise

for kk in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for mm in range( 3 ):

		ax1.errorbar( tmp_R[mm][kk], tmp_sb[mm][kk], yerr = tmp_err[mm][kk], marker = '.', ls = line_s[mm], color = color_s[kk],
			ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = fig_name[mm],)

	ax1.legend( loc = 1, frameon = False,)

	ax1.annotate( s = '%s-band' % band[kk], xy = (0.70, 0.50), xycoords = 'axes fraction',)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.set_ylim( 1e-3, 6e0)
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	plt.savefig('/home/xkchen/%s_sat_BG_compare.png' % band[kk], dpi = 300)
	plt.close()

