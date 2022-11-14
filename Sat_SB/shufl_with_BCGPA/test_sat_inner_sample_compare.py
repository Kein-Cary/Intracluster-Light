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


### === data load
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/BGs/'

out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/noBG_SBs/'
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'

#.
# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m


##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 1 ):

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


##... pre-BG-sub_SB
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 1 ):

		band_str = band[ kk ]

		dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	cp_nbg_R.append( sub_R )
	cp_nbg_SB.append( sub_sb )
	cp_nbg_err.append( sub_err )


##... aveg Sat_SB for subsample in ( 0~0.24 * R200m )
band_str = band[0]

dat = pds.read_csv( out_path + 
	'Extend_BCGM_gri-common_all_0.00-0.24R200m_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

cen_R, cen_SB, cen_SBerr = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )



### === figs
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

##.
y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [1e-3, 1e0], [5e-3, 6e0] ]


##.
fig = plt.figure( figsize = (10.8, 4.8) )
ax0 = fig.add_axes([0.08, 0.32, 0.42, 0.63])
sub_ax0 = fig.add_axes([0.08, 0.11, 0.42, 0.21])
ax1 = fig.add_axes([0.57, 0.32, 0.42, 0.63])
sub_ax1 = fig.add_axes([0.57, 0.11, 0.42, 0.21])

ax0.errorbar( cen_R, cen_SB, yerr = cen_SBerr, marker = '', ls = '-', color = 'k', ecolor = 'k', 
			mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.5, 
			label = '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[0], R_bins[2]),)

ax1.plot( cen_R, cen_SBerr, ls = '-', color = 'k', alpha = 0.5,)

_kk_F0 = interp.interp1d( cen_R, cen_SB, kind = 'linear', fill_value = 'extrapolate',)
_kk_F1 = interp.interp1d( cen_R, cen_SBerr, kind = 'linear', fill_value = 'extrapolate',)

for mm in range( 2 ):

	ax0.errorbar( nbg_R[mm][0], nbg_SB[mm][0], yerr = nbg_err[mm][0], marker = '', ls = '--', color = color_s[mm], 
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5, label = fig_name[mm],)

	# sub_ax0.plot( nbg_R[mm][0], nbg_SB[mm][0] - _kk_F0( nbg_R[mm][0] ), ls = '--', color = color_s[mm], alpha = 0.5,)
	# sub_ax0.fill_between( nbg_R[mm][0], y1 = nbg_SB[mm][0] - _kk_F0( nbg_R[mm][0] ) - nbg_err[mm][0], 
	# 				y2 = nbg_SB[mm][0] - _kk_F0( nbg_R[mm][0] ) + nbg_err[mm][0], ls = '--', color = color_s[mm], alpha = 0.15,)

	# ax1.plot( nbg_R[mm][0], nbg_err[mm][0], ls = '--', color = color_s[mm], alpha = 0.5,)
	# sub_ax1.plot( nbg_R[mm][0], nbg_err[mm][0] - _kk_F1( nbg_R[mm][0] ), ls = '--', color = color_s[mm], alpha = 0.5,)

	sub_ax0.plot( nbg_R[mm][0], nbg_SB[mm][0] / _kk_F0( nbg_R[mm][0] ), ls = '--', color = color_s[mm], alpha = 0.5,)
	sub_ax0.fill_between( nbg_R[mm][0], y1 = ( nbg_SB[mm][0]  - nbg_err[mm][0] ) / _kk_F0( nbg_R[mm][0] ), 
					y2 = ( nbg_SB[mm][0] + nbg_err[mm][0] ) / _kk_F0( nbg_R[mm][0] ), ls = '--', color = color_s[mm], alpha = 0.15,)

	ax1.plot( nbg_R[mm][0], nbg_err[mm][0], ls = '--', color = color_s[mm], alpha = 0.5,)
	sub_ax1.plot( nbg_R[mm][0], nbg_err[mm][0] / _kk_F1( nbg_R[mm][0] ), ls = '--', color = color_s[mm], alpha = 0.5,)

#.
ax0.legend( loc = 3, frameon = False, fontsize = 12,)
ax0.annotate( s = 'r-band', xy = (0.03, 0.35), xycoords = 'axes fraction', fontsize = 12,)

ax0.set_xlim( 1e0, 5e1 )
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)

ax0.set_ylim( 2e-3, 5e0 )
ax0.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax0.set_yscale('log')


sub_ax0.set_xlim( ax0.get_xlim() )
sub_ax0.set_xscale('log')
sub_ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)
# sub_ax0.set_ylabel('$\\mu - \\mu\,(%.2f \\leq R \\leq %.2f \, R_{200m})$' % (R_bins[0], R_bins[2]),)
sub_ax0.annotate( s = '$\\mu / \\mu\,(%.2f \\leq R \\leq %.2f \, R_{200m})$' % (R_bins[0], R_bins[2]),
				xy = (0.03, 0.65), xycoords = 'axes fraction', fontsize = 12,)
sub_ax0.set_ylim( 0.7, 1.4 )

ax1.set_xlim( ax0.get_xlim() )
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

ax1.set_ylim( 7e-5, 2e-2 )
ax1.set_ylabel('$\\sigma_{\\mu} \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)
# sub_ax1.set_ylabel('$\\sigma_{\\mu} - \\sigma_{\\mu}\,(%.2f \\leq R \\leq %.2f \, R_{200m})$' % (R_bins[0], R_bins[2]),)
sub_ax1.annotate(s = '$\\sigma_{\\mu} / \\sigma_{\\mu}\,(%.2f \\leq R \\leq %.2f \, R_{200m})$' % (R_bins[0], R_bins[2]),
				xy = (0.03, 0.55), xycoords = 'axes fraction', fontsize = 12,)
sub_ax1.set_ylim( 1.1, 1.55 )

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
sub_ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax0.set_xticklabels( [] )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.set_xticklabels( [] )

plt.savefig('/home/xkchen/cen_sR_bin_SB_compare.png', dpi = 300)
plt.close()


raise

##.
for kk in range( 1 ):

	##.
	fig = plt.figure( figsize = (10.8, 4.8) )
	ax1 = fig.add_axes([0.08, 0.32, 0.42, 0.63])
	sub_ax1 = fig.add_axes([0.08, 0.11, 0.42, 0.21])
	ax2 = fig.add_axes([0.57, 0.11, 0.42, 0.84])

	ax1.errorbar( cp_nbg_R[-1][kk], cp_nbg_SB[-1][kk], yerr = cp_nbg_err[-1][kk], marker = '', ls = '-', color = color_s[-1],
		ecolor = color_s[-1], mfc = 'none', mec = color_s[-1], capsize = 1.5, alpha = 0.5, label = fig_name[-1],)

	_kk_tmp_F = interp.interp1d(cp_nbg_R[-1][kk], cp_nbg_SB[-1][kk], kind = 'cubic', fill_value = 'extrapolate',)

	#.
	for mm in range( len(R_bins) - 2 ):

		if mm == 1:
			l1 = ax1.errorbar( cp_nbg_R[mm][kk], cp_nbg_SB[mm][kk], yerr = cp_nbg_err[mm][kk], marker = '', ls = '-', color = color_s[mm], 
				ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5, label = fig_name[mm],)

		else:
			ax1.errorbar( cp_nbg_R[mm][kk], cp_nbg_SB[mm][kk], yerr = cp_nbg_err[mm][kk], marker = '', ls = '-', color = color_s[mm], 
				ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5, label = fig_name[mm],)

		#.
		cc_inerp = _kk_tmp_F( cp_nbg_R[mm][kk] )
		ax2.plot( cp_nbg_R[mm][kk], cp_nbg_SB[mm][kk] / cc_inerp, ls = '-', color = color_s[mm], alpha = 0.5,)
		ax2.fill_between( cp_nbg_R[mm][kk], y1 = (cp_nbg_SB[mm][kk] - cp_nbg_err[mm][kk]) / cc_inerp, 
					y2 = (cp_nbg_SB[mm][kk] + cp_nbg_err[mm][kk]) / cc_inerp, color = color_s[mm], alpha = 0.12,)

	#.
	for mm in range( 3 ):

		l2 = ax1.errorbar( nbg_R[mm][kk], nbg_SB[mm][kk], yerr = nbg_err[mm][kk], marker = '', ls = '--', color = color_s[mm], 
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5, lw = 2.5,)

		#.
		cc_inerp = _kk_tmp_F( nbg_R[mm][kk] )
		ax2.plot( nbg_R[mm][kk], nbg_SB[mm][kk] / cc_inerp, ls = '--', color = color_s[mm], alpha = 0.5,)
		# ax2.fill_between( nbg_R[mm][kk], y1 = (nbg_SB[mm][kk] - nbg_err[mm][kk]) / cc_inerp, 
		# 			y2 = (nbg_SB[mm][kk] + nbg_err[mm][kk]) / cc_inerp, color = color_s[mm], alpha = 0.12,)

	#.
	for mm in range( 3 ):

		_cc_tmp_F = interp.interp1d( nbg_R[mm][kk], nbg_SB[mm][kk], kind = 'cubic', fill_value = 'extrapolate',)

		cc_inerp = _cc_tmp_F( cp_nbg_R[mm][kk] )
		sub_ax1.plot( cp_nbg_R[mm][kk], cp_nbg_SB[mm][kk] / cc_inerp, ls = '--', color = color_s[mm], alpha = 0.5,)
		sub_ax1.fill_between( cp_nbg_R[mm][kk], y1 = (cp_nbg_SB[mm][kk] - cp_nbg_err[mm][kk]) / cc_inerp, 
					y2 = (cp_nbg_SB[mm][kk] + cp_nbg_err[mm][kk]) / cc_inerp, color = color_s[mm], alpha = 0.12,)

	#.
	legend_2 = ax1.legend( handles = [ l1, l2], labels = [ 'Align with Frame', 'Align with BCG' ], 
				loc = 1, frameon = False, fontsize = 12,)
	ax1.legend( loc = 3, frameon = False, fontsize = 12,)
	ax1.add_artist( legend_2 )

	ax1.annotate( s = '%s-band' % band[kk], xy = (0.03, 0.55), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_xlim( 1e0, 5e1 )
	ax1.set_xscale('log')
	# ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	ax1.set_ylim( 2e-3, 5e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	# sub_ax1.set_ylabel('$\\mu_{Align \, with \, frame} \; / \; \\mu_{Align \, with \, BCG}$', labelpad = 10, fontsize = 12,)
	sub_ax1.annotate( s = '$\\mu_{Align \, with \, frame} \; / \; \\mu_{Align \, with \, BCG}$', 
					xy = (0.03, 0.75), xycoords = 'axes fraction', fontsize = 15,)

	sub_ax1.set_ylim( 0.95, 1.05 )
	sub_ax1.axhline( y = 1, ls = ':', color = 'gray', alpha = 0.25,)
	ax1.set_xticklabels( labels = [] )

	ax2.set_xlim( ax1.get_xlim() )
	ax2.set_xscale('log')
	ax2.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	ax2.set_ylabel('$\\mu \; / \; \\mu \,$ (%s)' % fig_name[-1], fontsize = 12,)
	ax2.set_ylim( 0.40, 0.85 )

	ax2.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/sat_%s-band_BG-sub_compare.png' % band[kk], dpi = 300)
	plt.close()


### === 
