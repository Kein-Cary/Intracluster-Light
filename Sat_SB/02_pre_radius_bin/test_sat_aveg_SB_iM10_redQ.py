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
#. fixed i_Mag10 case + redsequence division
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/redQ_divid_test/SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/redQ_divid_test/BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/redQ_divid_test/nBG_SBs/'


#. fixed i_Mag10 case ( divided by 0.191 * R200m + fixed i_Mag_10 )
cat_lis = ['inner', 'middle', 'outer']
fig_name = ['Inner', 'Middle', 'Outer']

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r', 'm']
line_s = [ '--', '-', '-.']


##... BG-sub SB(r) of sat. ( background stacking )
# N_sample = 50

# for tt in range( 2 ):
# 	id_redQ = ['below', 'above'][ tt ]

# 	for mm in range( 3 ):

# 		for kk in range( 3 ):

# 			sat_sb_file = path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_' % (cat_lis[mm], id_redQ, band[kk]) + 'jack-sub-%d_SB-pro_z-ref.h5'
# 			bg_sb_file = BG_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_redQ, band[kk])
# 			out_file = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_aveg-jack_BG-sub_SB.csv' % (cat_lis[mm], id_redQ, band[kk])

# 			stack_BG_sub_func( sat_sb_file, bg_sb_file, band[ kk ], N_sample, out_file )


### === ### figs
id_redQ = ['below', 'above']

bl_R, bl_sb, bl_sb_err = [], [], []

##... sat SBs
for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_redQ[0], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	bl_R.append( sub_R )
	bl_sb.append( sub_sb )
	bl_sb_err.append( sub_err )


bl_nbg_R, bl_nbg_SB, bl_nbg_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		dat = pds.read_csv( out_path + 
			'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_aveg-jack_BG-sub_SB.csv' % (cat_lis[mm], id_redQ[0], band[kk]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	bl_nbg_R.append( sub_R )
	bl_nbg_SB.append( sub_sb )
	bl_nbg_err.append( sub_err )


bl_bg_R, bl_bg_SB, bl_bg_err = [], [], []

for mm in range( 3 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for kk in range( 3 ):

		with h5py.File( BG_path + 
			'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_redQ[0], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		_sub_bg_R.append( tt_r )
		_sub_bg_sb.append( tt_sb )
		_sub_bg_err.append( tt_err )

	bl_bg_R.append( _sub_bg_R )
	bl_bg_SB.append( _sub_bg_sb )
	bl_bg_err.append( _sub_bg_err )



up_nbg_R, up_nbg_SB, up_nbg_err = [], [], []

for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		dat = pds.read_csv( out_path + 
			'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_aveg-jack_BG-sub_SB.csv' % (cat_lis[mm], id_redQ[1], band[kk]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	up_nbg_R.append( sub_R )
	up_nbg_SB.append( sub_sb )
	up_nbg_err.append( sub_err )


##... BG profile derived from mock Background stacking
up_bg_R, up_bg_SB, up_bg_err = [], [], []

for mm in range( 3 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for kk in range( 3 ):

		with h5py.File( BG_path + 
			'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_redQ[1], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		_sub_bg_R.append( tt_r )
		_sub_bg_sb.append( tt_sb )
		_sub_bg_err.append( tt_err )

	up_bg_R.append( _sub_bg_R )
	up_bg_SB.append( _sub_bg_sb )
	up_bg_err.append( _sub_bg_err )


up_R, up_sb, up_sb_err = [], [], []

##... sat SBs
for mm in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], id_redQ[1], band[kk]), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	up_R.append( sub_R )
	up_sb.append( sub_sb )
	up_sb_err.append( sub_err )



##... figs
for mm in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for kk in range( 3 ):

		l1 = ax1.errorbar( bl_R[mm][kk], bl_sb[mm][kk], yerr = bl_sb_err[mm][kk], marker = '.', ls = '-', color = color_s[kk],
			ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

		l2, = ax1.plot( bl_bg_R[mm][kk], bl_bg_SB[mm][kk], ls = '--', color = color_s[kk], alpha = 0.75,)
		ax1.fill_between( bl_bg_R[mm][kk], y1 = bl_bg_SB[mm][kk] - bl_bg_err[mm][kk],
			y2 = bl_bg_SB[mm][kk] + bl_bg_err[mm][kk], color = color_s[kk], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l1, l2], labels = ['Satellite + Background', 'Background'], loc = 'upper center', frameon = False,)
	ax1.legend( loc = 1, frameon = False,)
	ax1.add_artist( legend_2 )
	ax1.annotate( s = fig_name[mm] + ', below RQ', xy = (0.70, 0.50), xycoords = 'axes fraction',)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.set_ylim( 1e-3, 6e0)
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	plt.savefig('/home/xkchen/%s_%s-redQ_sat_BG_compare.png' % (cat_lis[mm], id_redQ[0]), dpi = 300)
	plt.close()


for mm in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for kk in range( 3 ):

		l1 = ax1.errorbar( up_R[mm][kk], up_sb[mm][kk], yerr = up_sb_err[mm][kk], marker = '.', ls = '-', color = color_s[kk],
			ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

		l2, = ax1.plot( up_bg_R[mm][kk], up_bg_SB[mm][kk], ls = '--', color = color_s[kk], alpha = 0.75,)
		ax1.fill_between( up_bg_R[mm][kk], y1 = up_bg_SB[mm][kk] - up_bg_err[mm][kk],
			y2 = up_bg_SB[mm][kk] + up_bg_err[mm][kk], color = color_s[kk], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l1, l2], labels = ['Satellite + Background', 'Background'], loc = 'upper center', frameon = False,)
	ax1.legend( loc = 1, frameon = False,)
	ax1.add_artist( legend_2 )
	ax1.annotate( s = fig_name[mm] + ', above RQ', xy = (0.70, 0.50), xycoords = 'axes fraction',)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.set_ylim( 1e-3, 6e0)
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	plt.savefig('/home/xkchen/%s_%s-redQ_sat_BG_compare.png' % (cat_lis[mm], id_redQ[1]), dpi = 300)
	plt.close()



plt.figure()
ax1 = plt.subplot(111)

for kk in range( 3 ):

	ax1.errorbar( bl_nbg_R[0][kk], bl_nbg_SB[0][kk], yerr = bl_nbg_err[0][kk], marker = '', ls = '--', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.errorbar( bl_nbg_R[1][kk], bl_nbg_SB[1][kk], yerr = bl_nbg_err[1][kk], marker = '', ls = '-', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

	ax1.errorbar( bl_nbg_R[2][kk], bl_nbg_SB[2][kk], yerr = bl_nbg_err[2][kk], marker = '', ls = '-.', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.annotate( s = 'Below RQ', xy = (0.70, 0.50), xycoords = 'axes fraction',)

legend_2 = ax1.legend( [ fig_name[0], fig_name[1], fig_name[2] ], loc = 'upper center', frameon = False,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xlim( 1e0, 4e2 )
ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]')

ax1.set_ylim( 5e-5, 6e0)
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

plt.savefig('/home/xkchen/sat_%s-redQ_BG-sub_compare.png' % id_redQ[0], dpi = 300)
plt.close()


plt.figure()
ax1 = plt.subplot(111)

for kk in range( 3 ):

	ax1.errorbar( up_nbg_R[0][kk], up_nbg_SB[0][kk], yerr = up_nbg_err[0][kk], marker = '', ls = '--', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.errorbar( up_nbg_R[1][kk], up_nbg_SB[1][kk], yerr = up_nbg_err[1][kk], marker = '', ls = '-', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

	ax1.errorbar( up_nbg_R[2][kk], up_nbg_SB[2][kk], yerr = up_nbg_err[2][kk], marker = '', ls = '-.', color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.annotate( s = 'Above RQ', xy = (0.70, 0.50), xycoords = 'axes fraction',)

legend_2 = ax1.legend( [ fig_name[0], fig_name[1], fig_name[2] ], loc = 'upper center', frameon = False,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xlim( 1e0, 4e2 )
ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]')

ax1.set_ylim( 5e-5, 6e0)
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

plt.savefig('/home/xkchen/sat_%s-redQ_BG-sub_compare.png' % id_redQ[1], dpi = 300)
plt.close()



for kk in range( 3 ):
	
	fig = plt.figure( )
	ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	for mm in range( 3 ):

		l1 = ax1.errorbar( bl_R[mm][kk], bl_sb[mm][kk], yerr = bl_sb_err[mm][kk], marker = '.', ls = '--', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5,)

		l2 = ax1.errorbar( up_R[mm][kk], up_sb[mm][kk], yerr = up_sb_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)


		sub_ax1.plot( bl_R[mm][kk], bl_sb[mm][kk] / up_sb[mm][kk], ls = '--', color = color_s[mm], alpha = 0.75,)

		sub_ax1.plot( up_R[mm][kk], up_sb[mm][kk] / up_sb[mm][kk], ls = '-', color = color_s[mm], alpha = 0.75,)

		sub_ax1.fill_between( up_R[mm][kk], y1 = (up_sb[mm][kk] - up_sb_err[mm][kk]) / up_sb[mm][kk], 
					y2 = (up_sb[mm][kk] + up_sb_err[mm][kk]) / up_sb[mm][kk], color = color_s[mm], alpha = 0.12,)

	legend_2 = ax1.legend( handles = [l1, l2], labels = ['Below RQ', 'Above RQ'], loc = 'upper center', frameon = False,)
	ax1.legend( loc = 1, frameon = False,)
	ax1.add_artist( legend_2 )

	ax1.annotate( s = '%s-band' % band[kk], xy = (0.70, 0.50), xycoords = 'axes fraction',)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.set_ylim( 1e-3, 6e0)
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$')
	sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{above}$', labelpad = 8)

	sub_ax1.set_ylim( 0.8, 2.3 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s-band_sat_SB_compare.png' % band[kk], dpi = 300)
	plt.close()


for kk in range( 3 ):
	
	fig = plt.figure( )
	ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	for mm in range( 3 ):

		l1 = ax1.errorbar( bl_nbg_R[mm][kk], bl_nbg_SB[mm][kk], yerr = bl_nbg_err[mm][kk], marker = '.', ls = '--', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5,)

		l2 = ax1.errorbar( up_nbg_R[mm][kk], up_nbg_SB[mm][kk], yerr = up_nbg_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)


		sub_ax1.plot( bl_nbg_R[mm][kk], bl_nbg_SB[mm][kk] / up_nbg_SB[mm][kk], ls = '--', color = color_s[mm], alpha = 0.75,)

		sub_ax1.plot( up_nbg_R[mm][kk], up_nbg_SB[mm][kk] / up_nbg_SB[mm][kk], ls = '-', color = color_s[mm], alpha = 0.75,)

		sub_ax1.fill_between( up_nbg_R[mm][kk], y1 = (up_nbg_SB[mm][kk] - up_nbg_err[mm][kk]) / up_nbg_SB[mm][kk], 
					y2 = (up_nbg_SB[mm][kk] + up_nbg_err[mm][kk]) / up_nbg_SB[mm][kk], color = color_s[mm], alpha = 0.12,)

	legend_2 = ax1.legend( handles = [l1, l2], labels = ['Below RQ', 'Above RQ'], loc = 'upper center', frameon = False,)
	ax1.legend( loc = 1, frameon = False,)
	ax1.add_artist( legend_2 )

	ax1.annotate( s = '%s-band' % band[kk], xy = (0.70, 0.50), xycoords = 'axes fraction',)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]')

	ax1.set_ylim( 1e-5, 6e0)
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$')
	sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{above}$', labelpad = 8)

	sub_ax1.set_ylim( 0.8, 2.5 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s-band_sat_BG-sub_SB_compare.png' % band[kk], dpi = 300)
	plt.close()

