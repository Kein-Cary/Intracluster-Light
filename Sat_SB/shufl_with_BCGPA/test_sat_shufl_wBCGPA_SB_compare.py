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


### === ### over all compare

band_str = 'r'

R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
list_order = 13

color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

###=== sat_img without BCG
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

##. random located satellite align the major axis of BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/noBG_SBs/'

##. random located satellite align image frame
cp_BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'


##... sat SBs
tmp_R, tmp_sb, tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_Mean_jack_SB-pro_z-ref.h5' 
		% (R_bins[tt], R_bins[tt + 1], band_str), 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_R.append( tt_r )
	tmp_sb.append( tt_sb )
	tmp_err.append( tt_err )


##... BG SBs
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( BG_path + 
		'Sat-all_%.2f-%.2fR200m_%s-band_wBCG-PA_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5'
		% (R_bins[tt], R_bins[tt + 1], band_str, list_order), 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_bg_R.append( tt_r )
	tmp_bg_SB.append( tt_sb )
	tmp_bg_err.append( tt_err )


##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv'
		% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )


##... pre-BG-sub_SB
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv'
		% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	cp_nbg_R.append( tt_r )
	cp_nbg_SB.append( tt_sb )
	cp_nbg_err.append( tt_sb_err )


##... pre-BGs
cp_bg_R, cp_bg_SB, cp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( cp_BG_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5'
		% (R_bins[tt], R_bins[tt + 1], band_str, list_order), 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	cp_bg_R.append( tt_r )
	cp_bg_SB.append( tt_sb )
	cp_bg_err.append( tt_err )



###=== sat_img with BCG
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/SBs/'

##. random located satellite align the major axis of BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/BGs_wBCG/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/noBG_SBs_wBCG/'

##. random located satellite align image frame
cp_BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/BGs/'
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/noBG_SBs/'


##... sat SBs
tmp_R_cc, tmp_sb_cc, tmp_err_cc = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_R_cc.append( tt_r )
	tmp_sb_cc.append( tt_sb )
	tmp_err_cc.append( tt_err )


##... BG SBs
tmp_bg_R_cc, tmp_bg_SB_cc, tmp_bg_err_cc = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( BG_path + 
		'Sat-all_%.2f-%.2fR200m_%s-band_wBCG-PA_shufl-%d_BG-wBCG_Mean_jack_SB-pro_z-ref.h5' 
		% (R_bins[tt], R_bins[tt + 1], band_str, list_order), 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_bg_R_cc.append( tt_r )
	tmp_bg_SB_cc.append( tt_sb )
	tmp_bg_err_cc.append( tt_err )


##... BG-subtracted SB profiles
nbg_R_cc, nbg_SB_cc, nbg_err_cc = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
		% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R_cc.append( tt_r )
	nbg_SB_cc.append( tt_sb )
	nbg_err_cc.append( tt_sb_err )


##... pre-BG-sub_SB
cp_nbg_R_cc, cp_nbg_SB_cc, cp_nbg_err_cc = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv'
		% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	cp_nbg_R_cc.append( tt_r )
	cp_nbg_SB_cc.append( tt_sb )
	cp_nbg_err_cc.append( tt_sb_err )


##... pre-BGs
cp_bg_R_cc, cp_bg_SB_cc, cp_bg_err_cc = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( cp_BG_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
		% (R_bins[tt], R_bins[tt + 1], band_str, list_order), 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	cp_bg_R_cc.append( tt_r )
	cp_bg_SB_cc.append( tt_sb )
	cp_bg_err_cc.append( tt_err )


### === ### figs
"""
for mm in range( len(R_bins) - 1 ):

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.04, 0.10, 0.28, 0.85])
	ax1 = fig.add_axes([0.38, 0.10, 0.28, 0.85])
	ax2 = fig.add_axes([0.70, 0.10, 0.28, 0.85])

	#.
	ax0.errorbar( tmp_R[mm], tmp_sb[mm], yerr = tmp_err[mm], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Applied BCG mask', alpha = 0.65,)

	ax0.errorbar( tmp_R_cc[mm], tmp_sb_cc[mm], yerr = tmp_err_cc[mm], marker = '.', ls = '--', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'No BCG mask', alpha = 0.65,)

	ax0.legend( loc = 3, frameon = False, fontsize = 12,)

	ax0.set_xscale('log')
	ax0.set_xlabel('R [kpc]', fontsize = 12,)

	ax0.annotate( s = '%s-band, ' % band_str + fig_name[mm], xy = (0.50, 0.85), xycoords = 'axes fraction', fontsize = 12,)

	ax0.set_ylim( 3e-3, 5e0 )
	ax0.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax0.set_yscale('log')

	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)


	#.
	l1 = ax1.errorbar( tmp_bg_R[mm], tmp_bg_SB[mm], yerr = tmp_bg_err[mm], marker = '.', ls = '-', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Align with BCG',)

	ax1.errorbar( cp_bg_R[mm], cp_bg_SB[mm], yerr = cp_bg_err[mm], marker = '.', ls = '--', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Align with frame',)

	#.
	l2 = ax1.errorbar( tmp_bg_R_cc[mm], tmp_bg_SB_cc[mm], yerr = tmp_bg_err_cc[mm], marker = '.', ls = '-', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5,)

	ax1.errorbar( cp_bg_R_cc[mm], cp_bg_SB_cc[mm], yerr = cp_bg_err_cc[mm], marker = '.', ls = '--', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5,)

	if mm == 0:
		legend_2 = ax1.legend( handles = [l1, l2], 
					labels = ['Applied BCG mask', 'No BCG mask' ], loc = 1, frameon = False, fontsize = 12,)

		ax1.legend( loc = 2, frameon = False, fontsize = 12,)
		ax1.add_artist( legend_2 )

	else:
		legend_2 = ax1.legend( handles = [l1, l2], 
					labels = ['Applied BCG mask', 'No BCG mask' ], loc = 1, frameon = False, fontsize = 12,)

		ax1.legend( loc = 4, frameon = False, fontsize = 12,)
		ax1.add_artist( legend_2 )

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	if mm == 0:
		ax1.set_ylim( 4e-3, 3e-2 )
	else:
		ax1.set_ylim( 1e-3, 1e-2 )

	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)


	#.
	ax2.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, 
		label = 'Applied BCG mask + Align with BCG')

	ax2.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '--', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75,
		label = 'Applied BCG mask + Align with frame')

	ax2.errorbar( nbg_R_cc[mm], nbg_SB_cc[mm], yerr = nbg_err_cc[mm], marker = '.', ls = '-', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75,
		label = 'No BCG mask + Align with BCG')

	ax2.errorbar( cp_nbg_R_cc[mm], cp_nbg_SB_cc[mm], yerr = cp_nbg_err_cc[mm], marker = '.', ls = '--', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, 
		label = 'No BCG mask + Align with frame')

	ax2.legend( loc = 3, frameon = False, fontsize = 12,)

	ax2.set_xlim( 1e0, 1e2 )
	ax2.set_xscale('log')
	ax2.set_xlabel('R [kpc]', fontsize = 12,)

	ax2.set_ylim( 1e-3, 5e0 )
	ax2.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax2.set_yscale('log')

	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/sat_%.2f-%.2fR200m_%s-band_SB_compare.png'
		% (R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
	plt.close()

"""


"""
for mm in range( len(R_bins) - 1 ):

	##.
	fig = plt.figure( figsize = (12.8, 5.4) )
	ax1 = fig.add_axes([0.08, 0.32, 0.42, 0.63])
	sub_ax1 = fig.add_axes([0.08, 0.11, 0.42, 0.21])
	ax2 = fig.add_axes([0.57, 0.11, 0.42, 0.84])

	#.
	l1 = ax1.errorbar( tmp_bg_R[mm], tmp_bg_SB[mm], yerr = tmp_bg_err[mm], marker = '.', ls = '-', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Align with BCG',)

	ax1.errorbar( cp_bg_R[mm], cp_bg_SB[mm], yerr = cp_bg_err[mm], marker = '.', ls = '--', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Align with frame',)

	_kk_tmp_F = interp.interp1d( tmp_bg_R[mm], tmp_bg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( cp_bg_R[mm] )

	sub_ax1.plot( cp_bg_R[mm], ( cp_bg_SB[mm] - _tt_SBs ) / _tt_SBs, ls = '--', color = 'r',)
	sub_ax1.fill_between( cp_bg_R[mm], y1 = (cp_bg_SB[mm] - _tt_SBs - cp_bg_err[mm]) / _tt_SBs, 
					y2 = (cp_bg_SB[mm] - _tt_SBs + cp_bg_err[mm]) / _tt_SBs, color = 'r', alpha = 0.12,)

	#.
	l2 = ax1.errorbar( tmp_bg_R_cc[mm], tmp_bg_SB_cc[mm], yerr = tmp_bg_err_cc[mm], marker = '.', ls = '-', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5,)

	ax1.errorbar( cp_bg_R_cc[mm], cp_bg_SB_cc[mm], yerr = cp_bg_err_cc[mm], marker = '.', ls = '--', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5,)

	_kk_tmp_F = interp.interp1d( tmp_bg_R_cc[mm], tmp_bg_SB_cc[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( cp_bg_R_cc[mm] )

	sub_ax1.plot( cp_bg_R_cc[mm], ( cp_bg_SB_cc[mm] - _tt_SBs ) / _tt_SBs, ls = '--', color = 'b',)
	sub_ax1.fill_between( cp_bg_R_cc[mm], y1 = (cp_bg_SB_cc[mm] - _tt_SBs - cp_bg_err_cc[mm]) / _tt_SBs, 
					y2 = (cp_bg_SB_cc[mm] - _tt_SBs + cp_bg_err_cc[mm]) / _tt_SBs, color = 'b', alpha = 0.12,)

	if mm == 0:
		legend_2 = ax1.legend( handles = [l1, l2], 
					labels = ['Applied BCG mask', 'No BCG mask' ], loc = 1, frameon = False, fontsize = 12,)

		ax1.legend( loc = 2, frameon = False, fontsize = 12,)
		ax1.add_artist( legend_2 )

	else:
		legend_2 = ax1.legend( handles = [l1, l2], 
					labels = ['Applied BCG mask', 'No BCG mask' ], loc = 1, frameon = False, fontsize = 12,)

		ax1.legend( loc = 4, frameon = False, fontsize = 12,)
		ax1.add_artist( legend_2 )

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	if mm == 0:
		ax1.set_ylim( 4e-3, 3e-2 )
	else:
		ax1.set_ylim( 1e-3, 1e-2 )

	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.annotate( s = '$(\\mu - \\mu_{Align \, with \, BCG}) \; / \; \\mu_{Align \, with \, BCG}$', 
					xy = (0.50, 0.75), xycoords = 'axes fraction', fontsize = 12,)

	sub_ax1.set_ylim( -0.7, 0.7)
	ax1.set_xticklabels( labels = [] )

	#.
	_kk_tmp_F = interp.interp1d( tmp_bg_R[mm], tmp_bg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( tmp_bg_R_cc[mm] )

	ax2.plot( tmp_bg_R_cc[mm], (tmp_bg_SB_cc[mm] - _tt_SBs ) / _tt_SBs, ls = '--', color = 'r',)
	ax2.fill_between( tmp_bg_R_cc[mm], y1 = (tmp_bg_SB_cc[mm] - _tt_SBs - tmp_bg_err_cc[mm]) / _tt_SBs, 
					y2 = (tmp_bg_SB_cc[mm] - _tt_SBs + tmp_bg_err_cc[mm]) / _tt_SBs, color = 'r', alpha = 0.12,)

	#.
	_kk_tmp_F = interp.interp1d( cp_bg_R[mm], cp_bg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( cp_bg_R_cc[mm] )

	ax2.plot( cp_bg_R_cc[mm], ( cp_bg_SB_cc[mm] - _tt_SBs ) / _tt_SBs, ls = '-', color = 'b',)
	ax2.fill_between( cp_bg_R_cc[mm], y1 = (cp_bg_SB_cc[mm] - _tt_SBs - cp_bg_err_cc[mm]) / _tt_SBs, 
					y2 = (cp_bg_SB_cc[mm] - _tt_SBs + cp_bg_err_cc[mm]) / _tt_SBs, color = 'b', alpha = 0.12,)

	ax2.set_xlim( ax1.get_xlim() )
	ax2.set_xscale('log')
	ax2.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	# ax2.set_ylabel('$(\\mu - \\mu_{Applied \, BCG \, mask}) \; / \; \\mu_{Applied \, BCG \, mask}$', 
	# 				labelpad = 10, fontsize = 12,)

	ax2.annotate( s = '$(\\mu - \\mu_{Applied \, BCG \, mask}) \; / \; \\mu_{Applied \, BCG \, mask}$', 
					xy = (0.25, 0.10), xycoords = 'axes fraction', fontsize = 14,)

	ax2.annotate( s = '%s-band, ' % band_str + fig_name[mm], xy = (0.45, 0.9), xycoords = 'axes fraction', fontsize = 12,)

	if mm == 0:
		ax2.set_ylim( -0.1, 1.2 )

	else:
		ax2.set_ylim( -0.5, 0.5 )

	ax2.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/sat_%.2f-%.2fR200m_%s-band_BG-pros_compare.png'
		% (R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
	plt.close()

"""



for mm in range( len(R_bins) - 1 ):

	##.
	fig = plt.figure( figsize = (12.8, 5.4) )
	ax1 = fig.add_axes([0.08, 0.32, 0.42, 0.63])
	sub_ax1 = fig.add_axes([0.08, 0.11, 0.42, 0.21])
	ax2 = fig.add_axes([0.57, 0.11, 0.42, 0.84])

	#.
	l1 = ax1.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '.', ls = '-', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Align with BCG',)

	ax1.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '--', 
		color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Align with frame',)

	_kk_tmp_F = interp.interp1d( nbg_R[mm], nbg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( cp_nbg_R[mm] )

	sub_ax1.plot( cp_nbg_R[mm], ( cp_nbg_SB[mm] - _tt_SBs ) / _tt_SBs, ls = '--', color = 'r',)
	sub_ax1.fill_between( cp_nbg_R[mm], y1 = (cp_nbg_SB[mm] - _tt_SBs - cp_nbg_err[mm]) / _tt_SBs, 
					y2 = (cp_nbg_SB[mm] - _tt_SBs + cp_nbg_err[mm]) / _tt_SBs, color = 'r', alpha = 0.12,)

	#.
	l2 = ax1.errorbar( nbg_R_cc[mm], nbg_SB_cc[mm], yerr = nbg_err_cc[mm], marker = '.', ls = '-', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5,)

	ax1.errorbar( cp_nbg_R_cc[mm], cp_nbg_SB_cc[mm], yerr = cp_nbg_err_cc[mm], marker = '.', ls = '--', 
		color = 'b', ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5,)

	_kk_tmp_F = interp.interp1d( nbg_R_cc[mm], nbg_SB_cc[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( cp_nbg_R_cc[mm] )

	sub_ax1.plot( cp_nbg_R_cc[mm], ( cp_nbg_SB_cc[mm] - _tt_SBs ) / _tt_SBs, ls = '--', color = 'b',)
	sub_ax1.fill_between( cp_nbg_R_cc[mm], y1 = (cp_nbg_SB_cc[mm] - _tt_SBs - cp_nbg_err_cc[mm]) / _tt_SBs, 
					y2 = (cp_nbg_SB_cc[mm] - _tt_SBs + cp_nbg_err_cc[mm]) / _tt_SBs, color = 'b', alpha = 0.12,)

	legend_2 = ax1.legend( handles = [l1, l2], 
				labels = ['Applied BCG mask', 'No BCG mask' ], loc = 1, frameon = False, fontsize = 12,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)
	ax1.add_artist( legend_2 )

	ax1.set_xlim( 1e0, 1e2 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 1e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.annotate( s = '$(\\mu - \\mu_{Align \, with \, BCG}) \; / \; \\mu_{Align \, with \, BCG}$', 
					xy = (0.03, 0.07), xycoords = 'axes fraction', fontsize = 12,)

	sub_ax1.set_ylim( -0.15, 0.15 )
	ax1.set_xticklabels( labels = [] )

	#.
	_kk_tmp_F = interp.interp1d( nbg_R[mm], nbg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( nbg_R_cc[mm] )

	ax2.plot( nbg_R_cc[mm], (nbg_SB_cc[mm] - _tt_SBs ) / _tt_SBs, ls = '--', color = 'r',)
	ax2.fill_between( nbg_R_cc[mm], y1 = (nbg_SB_cc[mm] - _tt_SBs - nbg_err_cc[mm]) / _tt_SBs, 
					y2 = (nbg_SB_cc[mm] - _tt_SBs + nbg_err_cc[mm]) / _tt_SBs, color = 'r', alpha = 0.12,)

	#.
	_kk_tmp_F = interp.interp1d( cp_nbg_R[mm], cp_nbg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)
	_tt_SBs = _kk_tmp_F( cp_nbg_R_cc[mm] )

	ax2.plot( cp_nbg_R_cc[mm], ( cp_nbg_SB_cc[mm] - _tt_SBs ) / _tt_SBs, ls = '-', color = 'b',)
	ax2.fill_between( cp_nbg_R_cc[mm], y1 = (cp_nbg_SB_cc[mm] - _tt_SBs - cp_nbg_err_cc[mm]) / _tt_SBs, 
					y2 = (cp_nbg_SB_cc[mm] - _tt_SBs + cp_nbg_err_cc[mm]) / _tt_SBs, color = 'b', alpha = 0.12,)

	ax2.set_xlim( ax1.get_xlim() )
	ax2.set_xscale('log')
	ax2.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	# ax2.set_ylabel('$(\\mu - \\mu_{Applied \, BCG \, mask}) \; / \; \\mu_{Applied \, BCG \, mask}$', 
	# 				labelpad = 10, fontsize = 12,)

	ax2.annotate( s = '$(\\mu - \\mu_{Applied \, BCG \, mask}) \; / \; \\mu_{Applied \, BCG \, mask}$', 
					xy = (0.03, 0.25), xycoords = 'axes fraction', fontsize = 14,)

	ax2.annotate( s = '%s-band, ' % band_str + fig_name[mm], xy = (0.05, 0.9), xycoords = 'axes fraction', fontsize = 12,)

	if mm == 0:
		ax2.set_ylim( -0.05, 0.95 )
	else:
		ax2.set_ylim( -0.1, 0.1 )

	ax2.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/sat_%.2f-%.2fR200m_%s-band_BG-sub-SB_compare.png'
		% (R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
	plt.close()

