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
##. sample information
bin_rich = [ 20, 30, 50, 210 ]

##. sat_img without bcg
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'

#.
sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

##... BG-sub SB(r) of sat. ( background stacking )
N_sample = 100

##.. shuffle order list
list_order = 13

R_bins = np.array( [0, 300, 2000] )

### === BG-sub SB profiles
"""
##. subsamples
for dd in range( 2 ):

	for ll in range( len(R_bins) - 1 ):

		for kk in range( 1 ):

			band_str = band[ kk ]

			##.
			sat_sb_file = ( path + '%s_clust_%d-%dkpc_%s-band' % (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str) 
						+ '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			bg_sb_file = ( BG_path + '%s_clust_%d-%dkpc_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
						% (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str, list_order),)[0]

			sub_out_file = ( out_path + '%s_clust_%d-%dkpc_%s-band' % (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str)  
						+ '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5',)[0]

			out_file = ( out_path + '%s_clust_%d-%dkpc_%s-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str),)[0]

			stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = sub_out_file )

raise
"""


### === figs and comparison
sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

#.
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']
samp_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

#.
color_s = ['r', 'b', 'g', 'm', 'k']

#.
fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

	else:
		fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)


### === results comparison
band_str = 'r'

dpt_R, dpt_SB, dpt_err = [], [], []
dpt_bg_R, dpt_bg_SB, dpt_bg_err = [], [], []
dpt_nbg_R, dpt_nbg_SB, dpt_nbg_err = [], [], []

#.
for qq in range( 2 ):

	##... sat SBs
	tmp_R, tmp_sb, tmp_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		with h5py.File( path + '%s_clust_%d-%dkpc_%s-band_Mean_jack_SB-pro_z-ref.h5' 
			% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tmp_R.append( tt_r )
		tmp_sb.append( tt_sb )
		tmp_err.append( tt_err )

	dpt_R.append( tmp_R )
	dpt_SB.append( tmp_sb )
	dpt_err.append( tmp_err )


	##... BG_SBs
	tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		with h5py.File( BG_path + '%s_clust_%d-%dkpc_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' % 
			(cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str, list_order), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tmp_bg_R.append( tt_r )
		tmp_bg_SB.append( tt_sb )
		tmp_bg_err.append( tt_err )

	dpt_bg_R.append( tmp_bg_R )
	dpt_bg_SB.append( tmp_bg_SB )
	dpt_bg_err.append( tmp_bg_err )


	##... BG-subtracted SB profiles
	nbg_R, nbg_SB, nbg_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%d-%dkpc_%s-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_R.append( tt_r )
		nbg_SB.append( tt_sb )
		nbg_err.append( tt_sb_err )

	dpt_nbg_R.append( nbg_R )
	dpt_nbg_SB.append( nbg_SB )
	dpt_nbg_err.append( nbg_err )


	##.
	plt.figure()
	ax1 = plt.subplot(111)

	for mm in range( len(R_bins) - 1 ):

		l2 = ax1.errorbar( tmp_R[mm], tmp_sb[mm], yerr = tmp_err[mm], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)

		l3, = ax1.plot( tmp_bg_R[mm], tmp_bg_SB[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
		ax1.fill_between( tmp_bg_R[mm], y1 = tmp_bg_SB[mm] - tmp_bg_err[mm], 
							y2 = tmp_bg_SB[mm] + tmp_bg_err[mm], color = color_s[mm], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l2, l3], 
				labels = ['Satellite + Background', 'Background' ], loc = 1, frameon = False, fontsize = 12,)

	ax1.legend( loc = 5, frameon = False, fontsize = 12,)
	ax1.add_artist( legend_2 )

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = samp_name[ qq ] + ', %s-band' % band_str, xy = (0.55, 0.03), xycoords = 'axes fraction', fontsize = 12,)
	ax1.set_ylim( 1e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_clust_sat_%s-band_BG_compare.png' % (cat_lis[qq], band_str), dpi = 300)
	plt.close()


##.
for mm in range( len(R_bins) - 1 ):

	#.
	fig = plt.figure( )
	ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	#.
	ax1.errorbar( dpt_nbg_R[1][mm], dpt_nbg_SB[1][mm], yerr = dpt_nbg_err[1][mm], marker = '', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = samp_name[1],)

	ax1.errorbar( dpt_nbg_R[0][mm], dpt_nbg_SB[0][mm], yerr = dpt_nbg_err[0][mm], marker = '', ls = '--', color = 'b', 
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = samp_name[0],)

	_kk_tmp_F = interp.interp1d( dpt_nbg_R[1][mm], dpt_nbg_SB[1][mm], kind = 'cubic', fill_value = 'extrapolate')
	_mm_SB = _kk_tmp_F( dpt_nbg_R[0][mm] )

	sub_ax1.plot( dpt_nbg_R[0][mm], dpt_nbg_SB[0][mm] / _mm_SB, ls = '--', color = 'b', alpha = 0.75,)
	sub_ax1.fill_between( dpt_nbg_R[0][mm], y1 = (dpt_nbg_SB[0][mm] - dpt_nbg_err[0][mm]) / _mm_SB, 
				y2 = (dpt_nbg_SB[0][mm] + dpt_nbg_err[0][mm]) / _mm_SB, color = 'b', alpha = 0.15,)

	ax1.annotate( s = fig_name[mm] + ', %s-band' % band_str, xy = (0.03, 0.35), xycoords = 'axes fraction', fontsize = 12,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xlim( 1e0, 1e2 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 1e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	# sub_ax1.set_ylabel('$\\mu \, $' + '(Low $ M_{\\ast}^{\\mathrm{BCG} }$)' + 
	# 				'$\; / \;$' + '$\\mu \,$' + '(High $ M_{\\ast}^{\\mathrm{BCG} }$)', labelpad = 8, fontsize = 12,)

	sub_ax1.annotate( s = '$\\mu \, $' + '(Low $ M_{\\ast}^{\\mathrm{BCG} }$)' + 
					'$\; / \;$' + '$\\mu \,$' + '(High $ M_{\\ast}^{\\mathrm{BCG} }$)', xy = (0.03, 0.75), xycoords = 'axes fraction',)
	sub_ax1.set_ylim( 0.85, 1.3 )
	sub_ax1.axhline( y = 1, ls = ':', color = 'k', alpha = 0.15,)

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_clust_sat_%d-%dkpc_%s-band_BG-sub_compare.png' 
				% (cat_lis[qq], R_bins[mm], R_bins[mm+1], band_str), dpi = 300)
	plt.close()

