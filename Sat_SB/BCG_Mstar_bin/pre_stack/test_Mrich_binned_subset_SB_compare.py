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

##. sat_img with bcg
# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/BGs/'
# path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SBs/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/noBG_SBs/'

##. sat_img without bcg
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'


sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']


##... BG-sub SB(r) of sat. ( background stacking )
N_sample = 100


##.. shuffle order list
list_order = 13

R_bins = np.array( [0, 300, 2000] )

"""
for dd in range( 2 ):

	for tt in range( len(R_bins) - 1 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			##.
			sat_sb_file = ( path + '%s_clust_%d-%dkpc_%s-band' % (cat_lis[dd], R_bins[tt], R_bins[tt+1], band_str) 
							+ '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			bg_sb_file = ( BG_path + '%s_clust_%d-%dkpc_%s-band' % (cat_lis[dd], R_bins[tt], R_bins[tt+1], band_str) 
							+ '_shufl-%d_BG' % list_order + '_Mean_jack_SB-pro_z-ref.h5',)[0]

			out_file = ( out_path + '%s_clust_%d-%dkpc' % (cat_lis[dd], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)[0]

			stack_BG_sub_func( sat_sb_file, bg_sb_file, band[ kk ], N_sample, out_file )

raise
"""


### === ### figs and comparison
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

samp_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']


color_s = []
for dd in range( len(R_bins) ):

	color_s.append( mpl.cm.plasma( dd / len(R_bins) ) )

fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

	else:
		fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)


for kk in range( 3 ):

	band_str = band[ kk ]

	tmp_R, tmp_sb, tmp_err = [], [], []

	##... sat SBs
	for tt in range( len(R_bins) - 1 ):

		sub_R, sub_sb, sub_err = [], [], []

		for dd in range( 2 ):

			with h5py.File( path + '%s_clust_%d-%dkpc_%s-band' % (cat_lis[dd], R_bins[tt], R_bins[tt+1], band_str) 
				+ '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

				tt_r = np.array(f['r'])
				tt_sb = np.array(f['sb'])
				tt_err = np.array(f['sb_err'])

			sub_R.append( tt_r )
			sub_sb.append( tt_sb )
			sub_err.append( tt_err )

		tmp_R.append( sub_R )
		tmp_sb.append( sub_sb )
		tmp_err.append( sub_err )


	tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

		for dd in range( 2 ):

			with h5py.File( BG_path + '%s_clust_%d-%dkpc_%s-band' % (cat_lis[dd], R_bins[tt], R_bins[tt+1], band_str) 
							+ '_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' % list_order, 'r') as f:

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

		for dd in range( 2 ):

			band_str = band[ kk ]

			dat = pds.read_csv( out_path + '%s_clust_%d-%dkpc' % (cat_lis[dd], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

			tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

			sub_R.append( tt_r )
			sub_sb.append( tt_sb )
			sub_err.append( tt_sb_err )

		nbg_R.append( sub_R )
		nbg_SB.append( sub_sb )
		nbg_err.append( sub_err )


	##. figs
	y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
	y_lim_1 = [ [2e-3, 4e0], [2e-3, 1e0], [5e-3, 6e0] ]


	for mm in range( len(R_bins) - 1 ):

		fig = plt.figure()
		ax1 = fig.add_axes( [0.12, 0.11, 0.80, 0.85] )

		l1 = ax1.errorbar( tmp_R[mm][0], tmp_sb[mm][0], yerr = tmp_err[mm][0], marker = '.', ls = '-', color = 'b',
					ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = samp_name[0],)

		l2, = ax1.plot( tmp_bg_R[mm][0], tmp_bg_SB[mm][0], ls = '--', color = 'b', alpha = 0.75,)
		ax1.fill_between( tmp_bg_R[mm][0], y1 = tmp_bg_SB[mm][0] - tmp_bg_err[mm][0], 
					y2 = tmp_bg_SB[mm][0] + tmp_bg_err[mm][0], color = 'b', alpha = 0.12)


		ax1.errorbar( tmp_R[mm][1], tmp_sb[mm][1], yerr = tmp_err[mm][1], marker = '.', ls = '-', color = 'r',
					ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = samp_name[1],)

		ax1.plot( tmp_bg_R[mm][1], tmp_bg_SB[mm][1], ls = '--', color = 'r', alpha = 0.75,)
		ax1.fill_between( tmp_bg_R[mm][1], y1 = tmp_bg_SB[mm][1] - tmp_bg_err[mm][1], 
					y2 = tmp_bg_SB[mm][1] + tmp_bg_err[mm][1], color = 'r', alpha = 0.12)

		legend_2 = ax1.legend( handles = [l1, l2], 
					labels = ['Satellite + Background', 'Background' ], loc = 4, frameon = False,)

		ax1.legend( loc = 1, frameon = False,)
		ax1.add_artist( legend_2 )

		ax1.set_xscale('log')
		ax1.set_xlabel('R [kpc]')

		ax1.annotate( s = fig_name[mm] + ', %s-band' % band[kk], xy = (0.25, 0.85), xycoords = 'axes fraction',)

		ax1.set_ylim( y_lim_0[kk][0], y_lim_0[kk][1] )
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
		ax1.set_yscale('log')

		plt.savefig('/home/xkchen/clust_%d-%dkpc_sat_%s-band_BG_compare.png' 
					% ( R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
		plt.close()



	for mm in range( len(R_bins) - 1 ):

		fig = plt.figure()
		ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
		sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

		ax1.errorbar( nbg_R[mm][0], nbg_SB[mm][0], yerr = nbg_err[mm][0], marker = '.', ls = '-', color = 'b',
					ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = samp_name[0],)

		ax1.errorbar( nbg_R[mm][1], nbg_SB[mm][1], yerr = nbg_err[mm][1], marker = '.', ls = '-', color = 'r',
					ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = samp_name[1],)

		ax1.legend( loc = 1, frameon = False,)

		ax1.set_xlim( 2e0, 4e1 )
		ax1.set_xscale('log')
		ax1.set_xlabel('R [kpc]')

		ax1.annotate( s = fig_name[mm] + ', %s-band' % band[kk], xy = (0.25, 0.15), xycoords = 'axes fraction',)

		ax1.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
		ax1.set_yscale('log')


		_kk_tmp_F = interp.interp1d( nbg_R[mm][1], nbg_SB[mm][1], kind = 'cubic', fill_value = 'extrapolate')

		sub_ax1.plot( nbg_R[mm][0], nbg_SB[mm][0] / _kk_tmp_F( nbg_R[mm][1] ), ls = '-', color = 'b',)
		sub_ax1.fill_between( nbg_R[mm][0], y1 = ( nbg_SB[mm][0] - nbg_err[mm][0] ) / _kk_tmp_F( nbg_R[mm][1] ),
							y2 = ( nbg_SB[mm][0] + nbg_err[mm][0] ) / _kk_tmp_F( nbg_R[mm][1] ), color = 'b', alpha = 0.15,)
		
		sub_ax1.set_xlim( ax1.get_xlim() )
		sub_ax1.set_xscale('log')
		sub_ax1.set_xlabel('$R \; [kpc]$')

		sub_ax1.set_ylabel('$\\mu \; / \; \\mu \, ( High \, M_{\\ast}^{\\mathrm{BCG}} )$', labelpad = 8)
		sub_ax1.set_ylim( 0.9, 1.1 )

		sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		ax1.set_xticklabels( labels = [] )

		plt.savefig('/home/xkchen/clust_%d-%dkpc_sat_%s-band_BG-sub_SB_compare.png' 
					% ( R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
		plt.close()

