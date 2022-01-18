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

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === ### data load

#. fixed i_Mag10 case
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/SBs/'
# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/BGs/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/nBG_SBs/'

#. use stacked cluster image as Background (without Ng weighted + no mask weight)
BG_path = '/home/xkchen/figs_cp/stacked_BG/no_mask_weit/BGs/'
out_path = '/home/xkchen/figs_cp/stacked_BG/no_mask_weit/nBG_SBs/'



#. fixed i_Mag10 case ( divided by 0.191 * R200m + fixed i_Mag_10 )
cat_lis = ['inner', 'middle', 'outer']
fig_name = ['Inner', 'Middle', 'Outer']

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r', 'm']
line_s = [ '--', '-', '-.']


tmp_R, tmp_sb, tmp_err = [], [], []

##... sat SBs
for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	tmp_R.append( sub_R )
	tmp_sb.append( sub_sb )
	tmp_err.append( sub_err )


##... BG profile derived from mock Background stacking
tmp_bR, tmp_BG, tmp_BG_err = [], [], []

for mm in range( 3 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for kk in range( 3 ):

		with h5py.File( BG_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		_sub_bg_R.append( tt_r )
		_sub_bg_sb.append( tt_sb )
		_sub_bg_err.append( tt_err )

	tmp_bR.append( _sub_bg_R )
	tmp_BG.append( _sub_bg_sb )
	tmp_BG_err.append( _sub_bg_err )


##... BG-sub SB(r) of sat. ( background stacking )
N_sample = 100

for mm in range( 3 ):

	for kk in range( 3 ):

		sat_sb_file = path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_' % (cat_lis[mm], band[kk]) + 'jack-sub-%d_SB-pro_z-ref.h5'
		bg_sb_file = BG_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk])
		out_file = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_aveg-jack_BG-sub_SB.csv' % (cat_lis[mm], band[kk])

		stack_BG_sub_func( sat_sb_file, bg_sb_file, band[ kk ], N_sample, out_file )


##.. figs and comparison
nbg_R, nbg_SB, nbg_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s-band_aveg-jack_BG-sub_SB.csv' % (cat_lis[mm], band[kk]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	nbg_R.append( sub_R )
	nbg_SB.append( sub_sb )
	nbg_err.append( sub_err )


### === figs
for mm in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for kk in range( 3 ):

		l1 = ax1.errorbar( tmp_R[mm][kk], tmp_sb[mm][kk], yerr = tmp_err[mm][kk], marker = '.', ls = '-', color = color_s[kk],
			ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

		l2, = ax1.plot( tmp_bR[mm][kk], tmp_BG[mm][kk], ls = '--', color = color_s[kk], alpha = 0.75,)
		ax1.fill_between( tmp_bR[mm][kk], y1 = tmp_BG[mm][kk] - tmp_BG_err[mm][kk],
			y2 = tmp_BG[mm][kk] + tmp_BG_err[mm][kk], color = color_s[kk], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l1, l2], labels = ['Satellite + Background', 'Background'], loc = 'upper center', frameon = False,)
	ax1.legend( loc = 1, frameon = False,)
	ax1.add_artist( legend_2 )
	ax1.annotate( s = fig_name[mm], xy = (0.70, 0.50), xycoords = 'axes fraction',)

	# ax1.set_xlim( 1e0, 4e2 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.set_ylim( 1e-3, 6e0)
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	plt.savefig('/home/xkchen/%s_sat_BG_compare.png' % cat_lis[mm], dpi = 300)
	plt.close()


plt.figure()
ax1 = plt.subplot(111)

for kk in range( 3 ):

	ax1.errorbar( nbg_R[0][kk], nbg_SB[0][kk], yerr = nbg_err[0][kk], marker = '', ls = '--', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.errorbar( nbg_R[1][kk], nbg_SB[1][kk], yerr = nbg_err[1][kk], marker = '', ls = '-', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

	ax1.errorbar( nbg_R[2][kk], nbg_SB[2][kk], yerr = nbg_err[2][kk], marker = '', ls = '-.', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

legend_2 = ax1.legend( [ fig_name[0], fig_name[1], fig_name[2] ], loc = 3, frameon = False,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xlim( 1e0, 4e2 )
ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]')

ax1.set_ylim( 5e-5, 6e0)
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

plt.savefig('/home/xkchen/sat_BG-sub_compare.png', dpi = 300)
plt.close()


fig = plt.figure( )
ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

for kk in range( 3 ):

	ax1.errorbar( nbg_R[0][kk], nbg_SB[0][kk], yerr = nbg_err[0][kk], marker = '', ls = '--', color = color_s[kk], ecolor = color_s[kk], 
		mfc = 'none', mec = color_s[kk], alpha = 0.75, capsize = 1.5,)

	ax1.errorbar( nbg_R[1][kk], nbg_SB[1][kk], yerr = nbg_err[1][kk], marker = '', ls = '-', color = color_s[kk], ecolor = color_s[kk], 
		mfc = 'none', mec = color_s[kk], alpha = 0.75, capsize = 1.5, label = '%s-band' % band[kk])

	ax1.errorbar( nbg_R[2][kk], nbg_SB[2][kk], yerr = nbg_err[2][kk], marker = '', ls = '-.', color = color_s[kk], ecolor = color_s[kk], 
		mfc = 'none', mec = color_s[kk], alpha = 0.75, capsize = 1.5,)

	_kk_tmp_F = interp.interp1d( nbg_R[2][kk], nbg_SB[2][kk], kind = 'linear', fill_value = 'extrapolate')

	sub_ax1.plot( nbg_R[0][kk], nbg_SB[0][kk] / _kk_tmp_F( nbg_R[0][kk] ), ls = '--', color = color_s[kk], alpha = 0.75,)
	sub_ax1.plot( nbg_R[1][kk], nbg_SB[1][kk] / _kk_tmp_F( nbg_R[1][kk] ), ls = '-', color = color_s[kk], alpha = 0.75,)

	sub_ax1.plot( nbg_R[2][kk], nbg_SB[2][kk] / nbg_SB[2][kk], ls = '-.', color = color_s[kk], alpha = 0.75,)
	sub_ax1.fill_between( nbg_R[2][kk], y1 = (nbg_SB[2][kk] - nbg_err[2][kk]) / nbg_SB[2][kk], 
				y2 = (nbg_SB[2][kk] + nbg_err[2][kk]) / nbg_SB[2][kk], color = color_s[kk], alpha = 0.12,)

legend_2 = ax1.legend( [fig_name[0], fig_name[1], fig_name[2] ], loc = 3, frameon = False,)
ax1.legend( loc = 1, frameon = False,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlim( 1e0, 4e2 )

ax1.set_ylim( 2e-5, 1e1 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$')
sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{outer}$', labelpad = 8)

# sub_ax1.set_ylim( 0.1, 1.5 )
sub_ax1.set_ylim( 0.1, 1.75 )

sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/sat_BG-sub_SB_compare.png', dpi = 300)
plt.close()

