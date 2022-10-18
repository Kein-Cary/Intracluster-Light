import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import scipy.signal as signal

import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

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
# R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m
R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m


### === Background subtraction
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/BGs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/noBG_SBs/'

band_str = 'r'

N_sample = 100

"""
##. subsamples BG_sub profiles
for tt in range( len(R_bins) - 1 ):

	##.
	sat_sb_file = ( path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
					(R_bins[tt], R_bins[tt + 1], band_str) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

	bg_sb_file = ( BG_path + 
		'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % (R_bins[tt], R_bins[tt + 1], band_str),)[0]

	sub_out_file = ( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
					(R_bins[tt], R_bins[tt + 1], band_str) + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5',)[0]

	out_file = ( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
					(R_bins[tt], R_bins[tt + 1], band_str) + '_aveg-jack_BG-sub_SB.csv',)[0]

	stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = sub_out_file )


##. all field galaxy
sat_sb_file = path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'
bg_sb_file = BG_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
sub_out_file = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5'
out_file = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_aveg-jack_BG-sub_SB.csv' % band_str

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

#.
with h5py.File( BG_path + 
	'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

	all_bg_R = np.array(f['r'])
	all_bg_SB = np.array(f['sb'])
	all_bg_err = np.array(f['sb_err'])

#.
dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_aveg-jack_BG-sub_SB.csv' % band_str )

all_nbg_R = np.array( dat['r'] )
all_nbg_SB = np.array( dat['sb'] )
all_nbg_err = np.array( dat['sb_err'] )


##.
tmp_R, tmp_sb, tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
		(R_bins[tt], R_bins[tt + 1], band_str) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_R.append( tt_r )
	tmp_sb.append( tt_sb )
	tmp_err.append( tt_err )


##.
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( BG_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % 
		(R_bins[tt], R_bins[tt + 1], band_str) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_bg_R.append( tt_r )
	tmp_bg_SB.append( tt_sb )
	tmp_bg_err.append( tt_err )


##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
						(R_bins[tt], R_bins[tt + 1], band_str) + '_aveg-jack_BG-sub_SB.csv',)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )


##... sat SBs
cp_R, cp_sb, cp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( cp_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
		+ '_%s-band_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	cp_R.append( tt_r )
	cp_sb.append( tt_sb )
	cp_err.append( tt_err )


##.
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	cp_nbg_R.append( tt_r )
	cp_nbg_SB.append( tt_sb )
	cp_nbg_err.append( tt_sb_err )


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
plt.figure()
ax1 = plt.subplot(111)

# ax1.errorbar( all_R, all_SB, yerr = all_err, marker = '.', ls = '-', lw = 3.0, color = 'k', 
# 		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'All galaxies',)

# ax1.errorbar( all_bg_R, all_bg_SB, yerr = all_bg_err, marker = '.', lw = 3.0, ls = '--', color = 'k', 
# 		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5,)

ax1.plot( all_R, all_SB, ls = '-', lw = 3.0, color = 'gray', label = 'All galaxies',)
ax1.plot( all_bg_R, all_bg_SB, lw = 3.0, ls = '--', color = 'gray',)

for mm in range( len(R_bins) - 1 ):

	l2 = ax1.errorbar( tmp_R[mm], tmp_sb[mm], yerr = tmp_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm], alpha = 0.75,)

	l3, = ax1.plot( tmp_bg_R[mm], tmp_bg_SB[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
	ax1.fill_between( tmp_bg_R[mm], y1 = tmp_bg_SB[mm] - tmp_bg_err[mm], 
						y2 = tmp_bg_SB[mm] + tmp_bg_err[mm], color = color_s[mm], alpha = 0.12)

legend_2 = ax1.legend( handles = [l2, l3], 
			labels = ['Galaxy + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xlim( 1e0, 5e2 )
ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.set_ylim( 2e-3, 4e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/ctrl-galx_%s-band_BG_compare.png' % band_str, dpi = 300)
plt.close()


##.
fig = plt.figure( )
ax1 = fig.add_axes([0.12, 0.11, 0.80, 0.85])

# ax1.errorbar( all_nbg_R, all_nbg_SB, yerr = all_nbg_err, marker = '.', lw = 2.5, ls = '-', color = 'k', 
# 		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'All galaxies')
ax1.plot( all_nbg_R, all_nbg_SB, lw = 3.5, ls = '--', color = 'gray', label = 'All galaxies',)


for mm in range( len(R_bins) - 1 ):

	ax1.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm], alpha = 0.75,)

ax1.legend( loc = 3, frameon = False, fontsize = 12,)

ax1.set_xscale('log')
ax1.set_xlim( 1e0, 5e1 )
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.set_ylim( 2e-3, 5e0 )
ax1.set_ylabel('$ \\mu \; [ nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/ctrl-galx_%s-band_BG-sub_SB.png' % band_str, dpi = 300)
plt.close()


fig = plt.figure( )
ax1 = fig.add_axes([0.12, 0.11, 0.80, 0.85])

# ax1.errorbar( all_nbg_R, all_nbg_R * all_nbg_SB, yerr = all_nbg_R * all_nbg_err, marker = '.', lw = 2.5, ls = '-', color = 'k', 
# 		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'All galaxies')
ax1.plot( all_nbg_R, all_nbg_R * all_nbg_SB, lw = 3.5, ls = '--', color = 'gray', label = 'All galaxies',)


for mm in range( len(R_bins) - 1 ):

	ax1.errorbar( nbg_R[mm], nbg_R[mm] * nbg_SB[mm], yerr = nbg_R[mm] * nbg_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm], alpha = 0.75,)

ax1.legend( loc = 3, frameon = False, fontsize = 12,)

ax1.set_xscale('log')
ax1.set_xlim( 1e0, 5e1 )
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.set_ylim( 1e-1, 8e0 )
ax1.set_ylabel('$ F = R \\times \\mu \; [kpc \\times nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/ctrl-galx_%s-band_Rweit_BG-sub_SB.png' % band_str, dpi = 300)
plt.close()


raise

##.
"""
for mm in range( len(R_bins) - 1 ):

	plt.figure()
	ax1 = plt.subplot(111)

	# ax1.errorbar( all_R, all_SB, yerr = all_err, marker = '.', ls = '-', color = 'k', 
	# 	ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Control, All galaxies',)
	ax1.plot( all_R, all_SB, ls = '--', lw = 3, color = 'gray', label = 'Control, All galaxies',)

	ax1.errorbar( tmp_R[mm], tmp_sb[mm], yerr = tmp_err[mm], marker = '.', ls = '-', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'Control, subsample-mapped',)

	ax1.errorbar( cp_R[mm], cp_sb[mm], yerr = cp_err[mm], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Member',)

	ax1.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '--', color = 'c',
		ecolor = 'c', mfc = 'none', mec = 'c', capsize = 1.5, label = 'Member,BG-subtracted',)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)

	ax1.set_xlim( 1e0, 5e2 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = fig_name[mm] + ', %s-band' % band_str, 
				xy = (0.05, 0.05), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 2e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/contrl-galx_%.2f-%.2fR200m_%s-band_SB_compare.png' % (R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
	plt.close()

"""

##.
for mm in range( len(R_bins) - 1 ):

	fig = plt.figure( )
	ax1 = fig.add_axes([0.12, 0.11, 0.80, 0.85])
	# ax2 = fig.add_axes([0.12, 0.10, 0.80, 0.21])

	# ax1.errorbar( all_nbg_R, all_nbg_SB, yerr = all_nbg_err, marker = '.', lw = 2.5, ls = '-', color = 'k', 
	# 		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Control, All galaxies',)
	ax1.plot( all_nbg_R, all_nbg_R * all_nbg_SB, lw = 2.5, ls = '-', color = 'gray', label = 'Control, All galaxies',)

	ax1.errorbar( nbg_R[mm], nbg_R[mm] * nbg_SB[mm], yerr = nbg_err[mm], marker = '.', ls = '-', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'Control, subsample-mapped',)

	ax1.errorbar( cp_nbg_R[mm], cp_nbg_R[mm] * cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '--', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Member',)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 1e0, 1e2 )
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = fig_name[mm] + ', %s-band' % band_str, 
				xy = (0.03, 0.35), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 6e-2, 9e0 )
	ax1.set_ylabel('$ F = R \\times \\mu \; [kpc \\times nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/contrl-galx_%.2f-%.2fR200m_%s-band_BG-sub_SB_compare.png' % (R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
	plt.close()

