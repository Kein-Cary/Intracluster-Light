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

import scipy.signal as signal
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
a_ref = 1 / (z_ref + 1)

band = ['r', 'g', 'i']


##. sat_img without BCGs
path = '/home/xkchen/figs_cp/cc_rich_rebin/nobcg_SBs/'
BG_path = '/home/xkchen/figs_cp/cc_rich_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs_cp/cc_rich_rebin/nobcg_BGsub_SBs/'
cat_path = '/home/xkchen/figs_cp/cc_rich_rebin/cat/'


bin_rich = [ 20, 30, 50, 210 ]

sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


##. compare the SB profile of entire satellite sample
list_order = 13

y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [7e-4, 1e0], [5e-3, 6e0] ]


###. ratio and slope profile
"""
for kk in range( 3 ):

	band_str = band[ kk ]

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.05, 0.32, 0.28, 0.63])
	sub_ax0 = fig.add_axes([0.05, 0.11, 0.28, 0.21])

	ax1 = fig.add_axes([0.38, 0.32, 0.28, 0.63])
	sub_ax1 = fig.add_axes([0.38, 0.11, 0.28, 0.21])

	ax2 = fig.add_axes([0.71, 0.32, 0.28, 0.63])
	sub_ax2 = fig.add_axes([0.71, 0.11, 0.28, 0.21])

	axes = [ ax0, ax1, ax2 ]
	sub_axes = [ sub_ax0, sub_ax1, sub_ax2 ]


	#. data load
	for qq in range( 3 ):

		##. rich < 30
		if qq == 0:
			R_bins = np.array([0, 150, 300, 500, 2000])

		##. rich (30, 50)
		if qq == 1:
			R_bins = np.array( [ 0, 300, 400, 550, 5000] )

		##. rich > 50
		if qq == 2:
			R_bins = np.array([0, 400, 600, 750, 2000])

		color_s = ['b', 'g', 'r', 'k', 'm']

		fig_name = []
		for dd in range( len(R_bins) - 1 ):

			if dd == 0:
				fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

			elif dd == len(R_bins) - 2:
				fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

			else:
				fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

		#. SB profile
		nbg_R, nbg_SB, nbg_err = [], [], []

		for tt in range( len(R_bins) - 1 ):

			dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

			tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

			nbg_R.append( tt_r )
			nbg_SB.append( tt_sb )
			nbg_err.append( tt_sb_err )

		#.
		sub_R, sub_ratio, sub_ratio_err = [], [], []
		sub_R1, sub_slope, sub_slope_err, sub_R_crit, sub_R_crit_std = [], [], [], [], []

		for tt in range( len(R_bins) - 2 ):

			cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB_ratio.csv' % band_str,)

			tt_r, tt_eta, tt_eta_err = np.array( cat['R'] ), np.array( cat['ratio'] ), np.array( cat['ratio_err'] )

			sub_R.append( tt_r )
			sub_ratio.append( tt_eta )
			sub_ratio_err.append( tt_eta_err )


			cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB_ratio_slope.csv' % band_str,)

			tt_r, tt_slope, tt_slope_err = np.array( cat['R'] ), np.array( cat['slope'] ), np.array( cat['slope_err'] )
			tt_Rc, tt_Rc_std = np.array( cat['R_crit'] )[0], np.array( cat['std_R_crit'] )[0]

			sub_R1.append( tt_r )
			sub_slope.append( tt_slope )
			sub_slope_err.append( tt_slope_err )

			sub_R_crit.append( tt_Rc )
			sub_R_crit_std.append( tt_Rc_std )

		##.
		gax, sub_gax = axes[qq], sub_axes[qq]

		gax.errorbar( nbg_R[-1], nbg_SB[-1], yerr = nbg_err[-1], marker = '', ls = '-', color = 'k', ecolor = 'k',
					mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

		for mm in range( len(R_bins) -2 ):

			gax.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '', ls = '--', color = color_s[mm], 
						ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

			sub_gax.plot( sub_R[mm], sub_ratio[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
			sub_gax.fill_between( sub_R[mm], y1 = sub_ratio[mm] - sub_ratio_err[mm], 
								y2 = sub_ratio[mm] + sub_ratio_err[mm], color = color_s[mm], alpha = 0.12,)

			gax.axvline( x = sub_R_crit[mm], ls = ':', color = color_s[mm], alpha = 0.75,)
			sub_gax.axvline( x = sub_R_crit[mm], ls = ':', color = color_s[mm], alpha = 0.75,)

			ty = np.linspace(0, 100, 1000)
			tx0, tx1 = np.ones( 1000, ) * sub_R_crit[mm], np.ones( 1000, ) * sub_R_crit_std[mm]

			gax.fill_betweenx( y = ty, x1 = tx0 - tx1, x2 = tx0 + tx1, color = color_s[mm], alpha = 0.12,)
			sub_gax.fill_betweenx( y = ty, x1 = tx0 - tx1, x2 = tx0 + tx1, color = color_s[mm], alpha = 0.12,)


		gax.annotate( s = line_name[qq] + ', %s-band' % band_str, xy = (0.45, 0.05), xycoords = 'axes fraction', fontsize = 14,)
		gax.legend( loc = 3, frameon = False, fontsize = 13,)

		gax.set_xlim( 2e0, 5e1 )
		gax.set_xscale('log')

		gax.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
		gax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 14)
		gax.set_yscale('log')

		sub_gax.set_xlim( gax.get_xlim() )
		sub_gax.set_xscale('log')
		sub_gax.set_xlabel('$R \; [kpc]$', fontsize = 14)

		sub_gax.set_ylabel('$\\mu \; / \; \\mu \, (R_{largest})$', labelpad = 8)
		sub_gax.set_ylim( 0.45, 1.05 )

		gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
		sub_gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
		sub_gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		gax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s-band_BG-sub_sat-SB.png' % band_str, dpi = 300)
	plt.close()


for kk in range( 3 ):

	band_str = band[ kk ]

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.05, 0.53, 0.28, 0.42])
	sub_ax0 = fig.add_axes([0.05, 0.11, 0.28, 0.42])

	ax1 = fig.add_axes([0.38, 0.53, 0.28, 0.42])
	sub_ax1 = fig.add_axes([0.38, 0.11, 0.28, 0.42])

	ax2 = fig.add_axes([0.71, 0.53, 0.28, 0.42])
	sub_ax2 = fig.add_axes([0.71, 0.11, 0.28, 0.42])

	axes = [ ax0, ax1, ax2 ]
	sub_axes = [ sub_ax0, sub_ax1, sub_ax2 ]


	#. data load
	for qq in range( 3 ):

		##. rich < 30
		if qq == 0:
			R_bins = np.array([0, 150, 300, 500, 2000])

		##. rich (30, 50)
		if qq == 1:
			R_bins = np.array( [ 0, 300, 400, 550, 5000] )

		##. rich > 50
		if qq == 2:
			R_bins = np.array([0, 400, 600, 750, 2000])

		color_s = ['b', 'g', 'r', 'k', 'm']

		fig_name = []
		for dd in range( len(R_bins) - 1 ):

			if dd == 0:
				fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

			elif dd == len(R_bins) - 2:
				fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

			else:
				fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

		#.
		sub_R, sub_ratio, sub_ratio_err = [], [], []
		sub_R1, sub_slope, sub_slope_err, sub_R_crit, sub_R_crit_std = [], [], [], [], []

		for tt in range( len(R_bins) - 2 ):

			cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB_ratio.csv' % band_str,)

			tt_r, tt_eta, tt_eta_err = np.array( cat['R'] ), np.array( cat['ratio'] ), np.array( cat['ratio_err'] )

			sub_R.append( tt_r )
			sub_ratio.append( tt_eta )
			sub_ratio_err.append( tt_eta_err )


			cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB_ratio_slope.csv' % band_str,)

			tt_r, tt_slope, tt_slope_err = np.array( cat['R'] ), np.array( cat['slope'] ), np.array( cat['slope_err'] )
			tt_Rc, tt_Rc_std = np.array( cat['R_crit'] )[0], np.array( cat['std_R_crit'] )[0]

			sub_R1.append( tt_r )
			sub_slope.append( tt_slope )
			sub_slope_err.append( tt_slope_err )

			sub_R_crit.append( tt_Rc )
			sub_R_crit_std.append( tt_Rc_std )

		##.
		gax, sub_gax = axes[qq], sub_axes[qq]

		for mm in range( len(R_bins) -2 ):

			gax.plot( sub_R[mm], sub_ratio[mm], ls = '--', color = color_s[mm], alpha = 0.75, label = fig_name[mm],)
			gax.fill_between( sub_R[mm], y1 = sub_ratio[mm] - sub_ratio_err[mm], y2 = sub_ratio[mm] + sub_ratio_err[mm],
						color = color_s[mm], alpha = 0.12,)

			sub_gax.plot( sub_R1[mm], sub_slope[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
			sub_gax.fill_between( sub_R1[mm], y1 = sub_slope[mm] - sub_slope_err[mm], 
								y2 = sub_slope[mm] + sub_slope_err[mm], color = color_s[mm], alpha = 0.12,)

			gax.axvline( x = sub_R_crit[mm], ls = ':', color = color_s[mm], alpha = 0.75,)
			sub_gax.axvline( x = sub_R_crit[mm], ls = ':', color = color_s[mm], alpha = 0.75,)

			ty = np.linspace(-100, 100, 1000)
			tx0, tx1 = np.ones( 1000, ) * sub_R_crit[mm], np.ones( 1000, ) * sub_R_crit_std[mm]

			gax.fill_betweenx( y = ty, x1 = tx0 - tx1, x2 = tx0 + tx1, color = color_s[mm], alpha = 0.12,)
			sub_gax.fill_betweenx( y = ty, x1 = tx0 - tx1, x2 = tx0 + tx1, color = color_s[mm], alpha = 0.12,)


		gax.annotate( s = line_name[qq] + ', %s-band' % band_str, xy = (0.45, 0.05), xycoords = 'axes fraction', fontsize = 14,)
		gax.legend( loc = 3, frameon = False, fontsize = 13,)

		gax.set_xlim( 2e0, 5e1 )
		gax.set_xscale('log')

		gax.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
		gax.set_ylabel('$\\mu \; / \; \\mu \, (R_{largest})$', labelpad = 8)
		gax.set_ylim( 0.45, 1.05 )


		sub_gax.set_xlim( gax.get_xlim() )
		sub_gax.set_xscale('log')
		sub_gax.set_xlabel('$R \; [kpc]$', fontsize = 14)

		sub_gax.set_ylabel('ratio slope', labelpad = 8)
		sub_gax.set_ylim( -0.04, 0.04 )

		gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
		sub_gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
		sub_gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		gax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s-band_BG-sub_sat-SB_ratio_slope.png' % band_str, dpi = 300)
	plt.close()

"""

###. figs of R_crit
color_s = ['b', 'g', 'r']
line_s = [':', '--', '-']
marks = ['s', '>', 'o']
mark_size = [10, 25, 35]

fig = plt.figure()
ax1 = fig.add_axes( [0.12, 0.11, 0.80, 0.85] )

for qq in range( 3 ):

	##. rich < 30
	if qq == 0:
		R_bins = np.array([0, 150, 300, 500, 2000])

	##. rich (30, 50)
	if qq == 1:
		R_bins = np.array( [ 0, 300, 400, 550, 5000] )

	##. rich > 50
	if qq == 2:
		R_bins = np.array([0, 400, 600, 750, 2000])


	fig_name = []
	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

		else:
			fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

	##. Rt estimate
	Rc, Rc_std = [], []

	for dd in range( len(R_bins) - 2 ):

		dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[dd], R_bins[dd + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB_ratio_slope.csv' % band[0],)

		tt_Rc, tt_Rc_std = np.array( dat['R_crit'] )[0], np.array( dat['std_R_crit'] )[0]		

		Rc.append( tt_Rc )
		Rc_std.append( tt_Rc_std )

	##. read sample catalog and get the average centric distance
	R_aveg = []
	R_sat_arr = []
	for dd in range( len( R_bins ) - 1 ):
		cat = pds.read_csv( cat_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv'
					% ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

		x_Rc = np.array( cat['R_sat'] )   ## Mpc / h
		cp_x_Rc = x_Rc * 1e3 * a_ref / h  ## kpc

		R_aveg.append( np.mean( cp_x_Rc) )
		R_sat_arr.append( cp_x_Rc )

	ax1.errorbar( R_aveg[:-1], Rc, yerr = Rc_std, marker = marks[qq], ls = '-', color = color_s[qq],
            ecolor = color_s[qq], mfc = 'none', mec = color_s[qq], capsize = 1.5, label = line_name[qq],)

ax1.legend( loc = 2, frameon = False, fontsize = 12)

ax1.set_ylabel('$ R_{t} \; [kpc]$')
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.set_xlabel('$\\bar{R}_{sat} \; [kpc]$')
ax1.set_xscale('log')
# ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in',)

plt.savefig('/home/xkchen/Rt_compare.png', dpi = 300)
plt.close()

