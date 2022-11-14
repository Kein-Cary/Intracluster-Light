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


### === ### cosmology model
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

##. sat_img without BCG
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_fixRsat_only/BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_fixRsat_only/noBG_SBs/'

#.
sub_name = ['low-rich', 'medi-rich', 'high-rich']

##. sample information
bin_rich = [ 20, 30, 50, 210 ]

R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

N_sample = 100

##. background shuffle list order
list_order = 13


# ##. background subtraction
# for tt in range( len(R_bins) - 1 ):

# 	for kk in range( 1 ):

# 		band_str = band[ kk ]

# 		##.
# 		sat_sb_file = ( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
# 						'_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

# 		bg_sb_file = ( BG_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
# 						'_%s-band_fixRsat-shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5',)[0]

# 		sub_out_file = ( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
# 						'_%s-band' % band_str + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5',)[0]

# 		out_file = ( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
# 						'_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)[0]

# 		stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = sub_out_file )

# raise


### === ### figs
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

#.
band_str = 'r'

##... sat SBs
tmp_R, tmp_sb, tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_R.append( tt_r )
	tmp_sb.append( tt_sb )
	tmp_err.append( tt_err )

##... BG_SBs
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( BG_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_fixRsat-shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
		% (R_bins[tt], R_bins[tt + 1], band_str, list_order), 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_bg_R.append( tt_r )
	tmp_bg_SB.append( tt_sb )
	tmp_bg_err.append( tt_err )

##... BG-sub_SBs
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )


##... previous case
##. random located satellite align image frame
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'

alinF_nbg_R, alinF_nbg_SB, alinF_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv'
		% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	alinF_nbg_R.append( tt_r )
	alinF_nbg_SB.append( tt_sb )
	alinF_nbg_err.append( tt_sb_err )

	
##. random located satellite align the major axis of BCG
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/noBG_SBs/'

alinB_nbg_R, alinB_nbg_SB, alinB_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 
		'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv'
		% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	alinB_nbg_R.append( tt_r )
	alinB_nbg_SB.append( tt_sb )
	alinB_nbg_err.append( tt_sb_err )


##... figs
plt.figure()
ax1 = plt.subplot(111)

for mm in range( len(R_bins) - 1 ):

	l2 = ax1.errorbar( tmp_R[mm], tmp_sb[mm], yerr = tmp_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)

	l3, = ax1.plot( tmp_bg_R[mm], tmp_bg_SB[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
	ax1.fill_between( tmp_bg_R[mm], y1 = tmp_bg_SB[mm] - tmp_bg_err[mm], 
						y2 = tmp_bg_SB[mm] + tmp_bg_err[mm], color = color_s[mm], alpha = 0.12)

legend_2 = ax1.legend( handles = [l2, l3], 
			labels = ['Satellite + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( s = '%s-band' % band_str, xy = (0.25, 0.90), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 1e-3, 5e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/sat_%s-band_BG_compare.png' % band_str, dpi = 300)
plt.close()


##.
for mm in range( len(R_bins) - 1 ):

	#.
	fig = plt.figure( )
	ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	#.
	ax1.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'No Alignment',)

	_kk_tmp_F = interp.interp1d( nbg_R[mm], nbg_SB[mm], kind = 'cubic', fill_value = 'extrapolate')

	ax1.errorbar( alinF_nbg_R[mm], alinF_nbg_SB[mm], yerr = alinF_nbg_err[mm], marker = '', ls = '-', color = 'g',
		ecolor = 'g', mfc = 'none', mec = 'g', capsize = 1.5, alpha = 0.75, label = 'Align with Frame',)

	ax1.errorbar( alinB_nbg_R[mm], alinB_nbg_SB[mm], yerr = alinB_nbg_err[mm], marker = '', ls = '-', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Align with BCG',)

	##.
	_mm_SB = _kk_tmp_F( alinF_nbg_R[mm] )

	sub_ax1.plot( alinF_nbg_R[mm], alinF_nbg_SB[mm] / _mm_SB, ls = '--', color = 'g', alpha = 0.75,)
	sub_ax1.fill_between( alinF_nbg_R[mm], y1 = (alinF_nbg_SB[mm] - alinF_nbg_err[mm]) / _mm_SB, 
				y2 = (alinF_nbg_SB[mm] + alinF_nbg_err[mm]) / _mm_SB, color = 'g', alpha = 0.15,)

	_mm_SB = _kk_tmp_F( alinB_nbg_R[mm] )

	sub_ax1.plot( alinB_nbg_R[mm], alinB_nbg_SB[mm] / _mm_SB, ls = '--', color = 'b', alpha = 0.75,)
	sub_ax1.fill_between( alinB_nbg_R[mm], y1 = (alinB_nbg_SB[mm] - alinB_nbg_err[mm]) / _mm_SB, 
				y2 = (alinB_nbg_SB[mm] + alinB_nbg_err[mm]) / _mm_SB, color = 'b', alpha = 0.15,)

	ax1.annotate( s = fig_name[mm] + ', %s-band' % band_str, xy = (0.03, 0.15), xycoords = 'axes fraction', fontsize = 12,)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)

	ax1.set_xlim( 1e0, 5e1 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 2e-3, 5e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,$(No Alignment)', labelpad = 10, fontsize = 12,)

	sub_ax1.set_ylim( 0.85, 1.05 )
	sub_ax1.axhline( y = 1, ls = ':', color = 'k', alpha = 0.15,)

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/sat_%s-band_%.2f-%.2fR200m_BG-sub-SB_compare.png' 
				% (band_str, R_bins[mm], R_bins[mm + 1]), dpi = 300)
	plt.close()

