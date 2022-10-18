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

##. sat_img with BCG
# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/BGs/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/noBG_SBs/'
# path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/SBs/'

##. sat_img without BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'


R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

N_sample = 100

##. background shuffle list order
list_order = 13


##. background subtraction
for tt in range( len(R_bins) - 1 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##.
		sat_sb_file = ( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

		bg_sb_file = ( BG_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5',)[0]

		sub_out_file = ( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band' % band_str + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5',)[0]

		out_file = ( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)[0]

		stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = sub_out_file )

raise



### === figs
# color_s = ['b', 'c', 'g', 'r', 'm']
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##... sat SBs
tmp_R, tmp_sb, tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		band_str = band[ kk ]

		with h5py.File( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	tmp_R.append( sub_R )
	tmp_sb.append( sub_sb )
	tmp_err.append( sub_err )


##... BG SBs
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for kk in range( 3 ):

		band_str = band[ kk ]

		with h5py.File( BG_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' % (band_str, list_order), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		_sub_bg_R.append( tt_r )
		_sub_bg_sb.append( tt_sb )
		_sub_bg_err.append( tt_err )

	tmp_bg_R.append( _sub_bg_R )
	tmp_bg_SB.append( _sub_bg_sb )
	tmp_bg_err.append( _sub_bg_err )


##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		band_str = band[ kk ]

		dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	nbg_R.append( sub_R )
	nbg_SB.append( sub_sb )
	nbg_err.append( sub_err )


##.
y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [1e-3, 1e0], [5e-3, 6e0] ]

for kk in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for mm in range( len(R_bins) - 1 ):

		l2 = ax1.errorbar( tmp_R[mm][kk], tmp_sb[mm][kk], yerr = tmp_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)

		l3, = ax1.plot( tmp_bg_R[mm][kk], tmp_bg_SB[mm][kk], ls = '--', color = color_s[mm], alpha = 0.75,)
		ax1.fill_between( tmp_bg_R[mm][kk], y1 = tmp_bg_SB[mm][kk] - tmp_bg_err[mm][kk], 
							y2 = tmp_bg_SB[mm][kk] + tmp_bg_err[mm][kk], color = color_s[mm], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l2, l3], 
				labels = ['Satellite + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)
	ax1.add_artist( legend_2 )

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = '%s-band' % band[kk], xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( y_lim_0[kk][0], y_lim_0[kk][1] )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/sat_%s-band_BG_compare.png' % band[kk], dpi = 300)
	plt.close()


for kk in range( 3 ):

	fig = plt.figure( )
	ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	ax1.errorbar( nbg_R[-1][kk], nbg_SB[-1][kk], yerr = nbg_err[-1][kk], marker = '', ls = '-', color = color_s[-1],
		ecolor = color_s[-1], mfc = 'none', mec = color_s[-1], capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

	_kk_tmp_F = interp.interp1d( nbg_R[-1][kk], nbg_SB[-1][kk], kind = 'cubic', fill_value = 'extrapolate',)

	# sub_ax1.plot( nbg_R[-1][kk], nbg_SB[-1][kk] / _kk_tmp_F( nbg_R[-1][kk] ), ls = '--', color = 'r', alpha = 0.75,)
	# sub_ax1.fill_between( nbg_R[-1][kk], y1 = (nbg_SB[-1][kk] - nbg_err[-1][kk]) / _kk_tmp_F( nbg_R[-1][kk] ), 
	#             y2 = (nbg_SB[-1][kk] + nbg_err[-1][kk]) / _kk_tmp_F( nbg_R[-1][kk] ), color = 'r', alpha = 0.12,)

	for mm in range( len(R_bins) - 2 ):

		ax1.errorbar( nbg_R[mm][kk], nbg_SB[mm][kk], yerr = nbg_err[mm][kk], marker = '', ls = '--', color = color_s[mm], 
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

		sub_ax1.plot( nbg_R[mm][kk], nbg_SB[mm][kk] / _kk_tmp_F( nbg_R[mm][kk] ), ls = '--', color = color_s[mm], alpha = 0.75,)
		sub_ax1.fill_between( nbg_R[mm][kk], y1 = (nbg_SB[mm][kk] - nbg_err[mm][kk]) / _kk_tmp_F( nbg_R[mm][kk] ), 
					y2 = (nbg_SB[mm][kk] + nbg_err[mm][kk]) / _kk_tmp_F( nbg_R[mm][kk] ), color = color_s[mm], alpha = 0.12,)

	daa = nbg_SB[-2][kk] / _kk_tmp_F( nbg_R[-2][kk] )


	ax1.annotate( s = '%s-band' % band[kk], xy = (0.65, 0.85), xycoords = 'axes fraction', fontsize = 12,)
	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xlim( 2e0, 5e1 )
	ax1.set_xscale('log')

	ax1.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
	# ax1.set_ylim( 2e-3, 5e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,$ (%s)' % fig_name[-1], labelpad = 8, fontsize = 12,)
	sub_ax1.set_ylim( 0.40, 1.0 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/sat_%s-band_BG-sub_compare.png' % band[kk], dpi = 300)
	plt.close()
