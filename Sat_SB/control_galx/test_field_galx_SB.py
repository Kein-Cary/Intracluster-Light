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

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#.
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

bin_rich = [ 20, 30, 50, 210 ]

sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


##. R_limmits
R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m


### === Background subtraction
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/BGs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/noBG_SBs/'

band_str = 'r'

N_sample = 100

"""
##. sub-sample matched
for ll in range( 3 ):

	##. subsamples BG_sub profiles
	for tt in range( len(R_bins) - 1 ):

		##.
		sat_sb_file = ( path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

		bg_sb_file = ( BG_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_BG__Mean_jack_SB-pro_z-ref.h5' % band_str,)[0]

		sub_out_file = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band' % band_str + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5',)[0]

		out_file = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)[0]

		stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = sub_out_file )

raise
"""


### === SBs compare
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
cp_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

band_str = 'r'

##. over all field galaxy 
with h5py.File( path + 
	'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

	all_R = np.array(f['r'])
	all_SB = np.array(f['sb'])
	all_err = np.array(f['sb_err'])


with h5py.File( BG_path + 
	'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

	all_bg_R = np.array(f['r'])
	all_bg_SB = np.array(f['sb'])
	all_bg_err = np.array(f['sb_err'])


dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_aveg-jack_BG-sub_SB.csv' % band_str )

all_nbg_R = np.array( dat['r'] )
all_nbg_SB = np.array( dat['sb'] )
all_nbg_err = np.array( dat['sb_err'] )


##.
tmp_R, tmp_sb, tmp_err = [], [], []

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		with h5py.File( path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_Mean_jack_SB-pro_z-ref.h5' % 
			(sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	tmp_R.append( sub_R )
	tmp_sb.append( sub_sb )
	tmp_err.append( sub_err )


##.
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for ll in range( 3 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		with h5py.File( BG_path + 
			'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
			'_%s-band_BG__Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

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

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	nbg_R.append( sub_R )
	nbg_SB.append( sub_sb )
	nbg_err.append( sub_err )


##... sat SBs
cp_R, cp_sb, cp_err = [], [], []

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		with h5py.File( cp_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_Mean_jack_SB-pro_z-ref.h5' 
			% (sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	cp_R.append( sub_R )
	cp_sb.append( sub_sb )
	cp_err.append( sub_err )


##.
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []
	
	for tt in range( len(R_bins) - 1 ):

		dat = pds.read_csv( cp_out_path + 
						'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' % 
						(sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	cp_nbg_R.append( sub_R )
	cp_nbg_SB.append( sub_sb )
	cp_nbg_err.append( sub_err )



##.. figs
color_s = ['b', 'g', 'c', 'r', 'm']

line_s = [ ':', '--', '-' ]

##.
if R_str == 'scale':

	fig_name = []
	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

		else:
			fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##.
for ll in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	# ax1.errorbar( all_R, all_SB, yerr = all_err, marker = '.', ls = '-', lw = 3, color = 'gray', 
	# 		ecolor = 'gray', mfc = 'none', mec = 'gray', capsize = 1.5, label = 'All galaxies', )

	# ax1.errorbar( all_bg_R, all_bg_SB, yerr = all_bg_err, marker = '.', lw = 3, ls = '--', color = 'gray', 
	# 		ecolor = 'gray', mfc = 'none', mec = 'gray', capsize = 1.5,)

	ax1.plot( all_R, all_SB, ls = '-', lw = 3, color = 'gray', label = 'All galaxies', alpha = 0.75,)
	ax1.plot( all_bg_R, all_bg_SB, ls = '--', lw = 3, color = 'gray', alpha = 0.75,)

	for mm in range( len(R_bins) - 1 ):

		l2 = ax1.errorbar( tmp_R[ll][mm], tmp_sb[ll][mm], yerr = tmp_err[ll][mm], marker = '.', ls = '-', lw = 0.8, color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm], alpha = 0.75,)

		l3, = ax1.plot( tmp_bg_R[ll][mm], tmp_bg_SB[ll][mm], ls = '--', color = color_s[mm], alpha = 0.75, lw = 0.8,)
		ax1.fill_between( tmp_bg_R[ll][mm], y1 = tmp_bg_SB[ll][mm] - tmp_bg_err[ll][mm], 
							y2 = tmp_bg_SB[ll][mm] + tmp_bg_err[ll][mm], color = color_s[mm], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l2, l3], 
				labels = ['Galaxy + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)
	ax1.add_artist( legend_2 )

	ax1.set_xlim( 1e0, 5e2 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = line_name[ll] + ', %s-band' % band_str, xy = (0.55, 0.35), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 2e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_sat_%s-band_BG_compare.png' % (sub_name[ll], band_str), dpi = 300)
	plt.close()


##.
"""
for ll in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	# ax1.errorbar( all_nbg_R, all_nbg_SB, yerr = all_nbg_err, marker = '.', lw = 3.0, ls = '--', color = 'gray', 
	# 		ecolor = 'gray', mfc = 'none', mec = 'gray', capsize = 1.5, label = 'All galaxies', alpha = 0.75,)

	ax1.plot( all_nbg_R, all_nbg_SB, lw = 3.0, ls = '--', color = 'gray', alpha = 0.75, label = 'All galaxies', )

	for mm in range( len(R_bins) - 1 ):

		ax1.errorbar( nbg_R[ll][mm], nbg_SB[ll][mm], yerr = nbg_err[ll][mm], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm], alpha = 0.75,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 1e0, 5e1 )
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = line_name[ll] + ', %s-band' % band_str, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 5e-4, 5e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_sat_%s-band_BG-sub_SB.png' % (sub_name[ll], band_str), dpi = 300)
	plt.close()

"""

##.
for ll in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	# ax1.errorbar( all_nbg_R, all_nbg_R * all_nbg_SB, yerr = all_nbg_R * all_nbg_err, marker = '.', lw = 3.0, ls = '--', color = 'gray', 
	# 		ecolor = 'gray', mfc = 'none', mec = 'gray', capsize = 1.5, label = 'All galaxies', alpha = 0.75,)

	ax1.plot( all_nbg_R, all_nbg_R * all_nbg_SB, lw = 3.0, ls = '--', color = 'gray', alpha = 0.75, label = 'All galaxies', )

	for mm in range( len(R_bins) - 1 ):

		ax1.errorbar( nbg_R[ll][mm], nbg_R[ll][mm] * nbg_SB[ll][mm], yerr = nbg_R[ll][mm] * nbg_err[ll][mm], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm], alpha = 0.75,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 1e0, 5e1 )
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = line_name[ll] + ', %s-band' % band_str, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 1e-1, 8e0 )
	ax1.set_ylabel('$ F = R \\times \\mu \; [kpc \\times nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_sat_%s-band_BG-sub_SB.png' % (sub_name[ll], band_str), dpi = 300)
	plt.close()


raise


##.
"""
for mm in range( len(R_bins) - 1 ):

	for ll in range( 3 ):

		plt.figure()
		ax1 = plt.subplot(111)

		# ax1.errorbar( all_R, all_SB, yerr = all_err, marker = '.', ls = '-', color = 'k', 
		# 	ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Control, All galaxies',)
		ax1.plot( all_R, all_SB, ls = '-', color = 'gray', lw = 3, alpha = 0.75, label = 'Control, All galaxies',)

		ax1.errorbar( tmp_R[ll][mm], tmp_sb[ll][mm], yerr = tmp_err[ll][mm], marker = '.', ls = '-', color = 'b',
			ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'Control, subsample-mapped',)

		ax1.errorbar( cp_R[ll][mm], cp_sb[ll][mm], yerr = cp_err[ll][mm], marker = '.', ls = '-', color = 'r',
			ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Member',)

		ax1.errorbar( cp_nbg_R[ll][mm], cp_nbg_SB[ll][mm], yerr = cp_nbg_err[ll][mm], marker = '.', ls = '--', color = 'c',
			ecolor = 'c', mfc = 'none', mec = 'c', capsize = 1.5, label = 'Member,BG-subtracted',)

		ax1.legend( loc = 1, frameon = False, fontsize = 12,)

		ax1.set_xscale('log')
		ax1.set_xlabel('R [kpc]', fontsize = 12,)

		ax1.annotate( s = fig_name[mm] + '\n' + line_name[ll] + ', %s-band' % band_str, 
					xy = (0.45, 0.45), xycoords = 'axes fraction', fontsize = 12,)

		ax1.set_ylim( 1e-3, 4e0 )
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
		ax1.set_yscale('log')

		ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

		plt.savefig('/home/xkchen/%s_contrl-galx_%.2f-%.2fR200m_%s-band_SB_compare.png' % (sub_name[ ll ], R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
		plt.close()

"""

##.
for mm in range( len(R_bins) - 1 ):

	for ll in range( 3 ):

		fig = plt.figure( )
		ax1 = fig.add_axes([0.12, 0.11, 0.80, 0.85])

		# ax1.errorbar( all_nbg_R, all_nbg_R * all_nbg_SB, yerr = all_nbg_R * all_nbg_err, marker = '.', lw = 2.5, ls = '-', color = 'k', 
		# 		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Control, All galaxies',)

		ax1.plot( all_nbg_R, all_nbg_R * all_nbg_SB, lw = 3, ls = '-', color = 'gray', alpha = 0.75, label = 'Control, All galaxies',)

		ax1.errorbar( nbg_R[ll][mm], nbg_R[ll][mm] * nbg_SB[ll][mm], yerr = nbg_R[ll][mm] * nbg_err[ll][mm], marker = '.', ls = '-', color = 'b',
			ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'Control, subsample-mapped',)

		ax1.errorbar( cp_nbg_R[ll][mm], cp_nbg_R[ll][mm] * cp_nbg_SB[ll][mm], yerr = cp_nbg_R[ll][mm] * cp_nbg_err[ll][mm], marker = '.', 
			ls = '--', color = 'r', ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Member',)

		ax1.legend( loc = 3, frameon = False, fontsize = 12,)

		ax1.set_xscale('log')
		ax1.set_xlim( 1e0, 1e2 )
		ax1.set_xlabel('R [kpc]', fontsize = 12,)

		ax1.annotate( s = fig_name[mm] + '\n' + line_name[ll] + ', %s-band' % band_str, 
					xy = (0.03, 0.35), xycoords = 'axes fraction', fontsize = 12,)

		ax1.set_ylim( 6e-2, 8e0 )
		ax1.set_ylabel('$ F = R \\times \\mu \; [kpc \\times nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
		ax1.set_yscale('log')

		ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

		plt.savefig('/home/xkchen/%s_contrl-galx_%.2f-%.2fR200m_%s-band_BG-sub_SB_compare.png' % 
					(sub_name[ ll ], R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
		plt.close()

