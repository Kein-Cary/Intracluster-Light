"""
compare the cumulative Luminosity of satellite and control galaxy
"""
import sys 
sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')

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

from pynverse import inversefunc
from scipy import interpolate as interp
from scipy import integrate as integ
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

rad2arcsec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

Da_ref = Test_model.angular_diameter_distance( z_ref ).value  ## Mpc.



### === data load
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/BGs/'

cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/map_cat/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/noBG_SBs/'

cp_cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
cp_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

#.
bin_rich = [ 20, 30, 50, 210 ]

R_str = 'scale'
# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

band_str = 'r'

##.
color_s = ['b', 'g', 'c', 'r', 'm']
line_s = [ ':', '--', '-' ]

fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##... sample magnitude
tmp_r_cMag = []
tmp_r_mag = []

cp_r_cMag = []
cp_r_mag = []

for tt in range( len(R_bins) - 1 ):

	tt_cmag = np.array([])
	tt_mag = np.array([])

	dt_cmag = np.array([])
	dt_mag = np.array([])

	for ll in range( 3 ):

		#. member
		dat = fits.open( cp_cat_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_params.fits' % 
						(bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1]),)

		dat_arr = dat[1].data

		tt_cmag = np.r_[ tt_cmag, dat_arr['cModelMag_r'] ]
		tt_mag = np.r_[ tt_mag, dat_arr['modelMag_r'] ]

		#. control
		dat = fits.open( cat_path + 
						'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%.2f-%.2fR200m_r-band_cat.fits' 
						% (R_bins[tt], R_bins[tt + 1]),)

		dat_arr = dat[1].data

		dt_cmag = np.r_[ dt_cmag, dat_arr['cModelMag_r'] ]
		dt_mag = np.r_[ dt_mag, dat_arr['modelMag_r'] ]

	#.
	tmp_r_cMag.append( dt_cmag )
	tmp_r_mag.append( dt_mag )

	cp_r_cMag.append( tt_cmag )
	cp_r_mag.append( tt_mag )


##... cumu_L profiles ~ (BG-subtracted SB profiles)
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
						(R_bins[tt], R_bins[tt + 1], band_str) + '_aveg-jack_BG-sub_SB.csv',)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )

##.
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	cp_nbg_R.append( tt_r )
	cp_nbg_SB.append( tt_sb )
	cp_nbg_err.append( tt_sb_err )


##... cumulative Luminosity
cumu_R, cumu_L, cumu_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 
		'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg-cumuL.csv' % 
		(R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cumu_R.append( tt_r )
	cumu_L.append( tt_L )
	cumu_Lerr.append( tt_L_err )


##.
cp_cumu_R, cp_cumu_L, cp_cumu_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
						+ '_%s-band_BG-sub-SB_aveg-cumuL.csv' % band_str)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cp_cumu_R.append( tt_r )
	cp_cumu_L.append( tt_L )
	cp_cumu_Lerr.append( tt_L_err )


###... match cumu_L within 50kpc
off_L = []
off_frac = []

cp_off_L = []
cp_off_frac = []

R_fix = 30   ## 20, 30, 50 .kpc

for tt in range( len(R_bins) - 1 ):

	# ##. sample magnitude check
	# mean_cMag_f = np.mean( 10**( 0.4 * (22.5 - tmp_r_cMag[ tt ] ) ) )
	# cc_mean_cMag_f = np.mean( 10**( 0.4 * (22.5 - cp_r_cMag[ tt ] ) ) )

	# medi_cMag_f = np.median( 10**( 0.4 * (22.5 - tmp_r_cMag[ tt ] ) ) )
	# cc_medi_cMag_f = np.median( 10**( 0.4 * (22.5 - cp_r_cMag[ tt ] ) ) )

	# print( mean_cMag_f / cc_mean_cMag_f )
	# print( medi_cMag_f / cc_medi_cMag_f )

	tag_mag = np.mean( 10**( 0.4 * (22.5 - tmp_r_cMag[ tt ] ) ) )

	##. control galaxy
	tcf = interp.interp1d(cumu_R[tt], cumu_L[tt], kind = 'cubic', fill_value = 'extrapolate',)
	dd_mag = tcf( R_fix )

	dd_eta = tag_mag / dd_mag
	dd_L = cumu_L[ tt ] * dd_eta

	off_frac.append( dd_eta )
	off_L.append( dd_L )

	##. member galaxy
	tcf = interp.interp1d( cp_cumu_R[tt], cp_cumu_L[tt], kind = 'cubic', fill_value = 'extrapolate',)
	dd_mag = tcf( R_fix )

	dd_eta = tag_mag / dd_mag
	dd_L = cp_cumu_L[ tt ] * dd_eta

	cp_off_frac.append( dd_eta )
	cp_off_L.append( dd_L )


##. figs SB pros compare
fig = plt.figure( )
ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

for mm in range( len(R_bins) - 1 ):

	l2 = ax1.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '.', ls = '--', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75,)

	l3 = ax1.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

	##.
	_kk_tmp_F = interp.interp1d( nbg_R[mm], nbg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax1.plot( cp_nbg_R[mm], cp_nbg_SB[mm] / _kk_tmp_F( cp_nbg_R[mm] ), ls = '--', color = color_s[mm], alpha = 0.75,)

	sub_ax1.fill_between( cp_nbg_R[mm], y1 = (cp_nbg_SB[mm] - cp_nbg_err[mm]) / _kk_tmp_F( cp_nbg_R[mm]), 
			y2 = (cp_nbg_SB[mm] + cp_nbg_err[mm]) / _kk_tmp_F( cp_nbg_R[mm]), color = color_s[mm], alpha = 0.12,)

legend_2 = ax1.legend( handles = [ l2, l3], labels = [ 'Control, subsample-mapped', 'RedMaPPer Member' ], 
			loc = 1, frameon = False, fontsize = 12,)

ax1.legend( loc = 3, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlim( 2e0, 5e1 )
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( s = '%s-band' % band_str, xy = (0.03, 0.45), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 2e-3, 5e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

sub_ax1.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 10, fontsize = 12,)
sub_ax1.set_ylim( 0.9, 1.1 )

sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/contrl-galx_%s-band_BG-sub_SB_compare.png' % band_str, dpi = 300)
plt.close()


fig = plt.figure( )
ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

for mm in range( len(R_bins) - 1 ):

	l2 = ax1.errorbar( nbg_R[mm], nbg_SB[mm] * off_frac[ mm ], yerr = nbg_err[mm], marker = '.', ls = '--', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75,)

	l3 = ax1.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm] * cp_off_frac[ mm ], yerr = cp_nbg_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

	##.
	_kk_tmp_F = interp.interp1d( nbg_R[mm], nbg_SB[mm] * off_frac[ mm ], kind = 'cubic', fill_value = 'extrapolate',)

	_cc_tmp_sb = _kk_tmp_F( cp_nbg_R[mm] )

	sub_ax1.plot( cp_nbg_R[mm], cp_nbg_SB[mm] * cp_off_frac[ mm ] / _cc_tmp_sb, ls = '--', color = color_s[mm], alpha = 0.75,)
	sub_ax1.fill_between( cp_nbg_R[mm], y1 = (cp_nbg_SB[mm] * cp_off_frac[ mm ] - cp_nbg_err[mm]) / _cc_tmp_sb, 
			y2 = (cp_nbg_SB[mm] * cp_off_frac[ mm ] + cp_nbg_err[mm]) / _cc_tmp_sb, color = color_s[mm], alpha = 0.12,)

legend_2 = ax1.legend( handles = [ l2, l3], labels = [ 'Control, subsample-mapped', 'RedMaPPer Member' ], 
			loc = 1, frameon = False, fontsize = 12,)

ax1.legend( loc = 3, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlim( 2e0, 5e1 )
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( s = '%s-band' % band_str, xy = (0.03, 0.45), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 2e-3, 5e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

sub_ax1.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 10, fontsize = 12,)
sub_ax1.set_ylim( 0.9, 1.1 )

sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/contrl-galx_%s-band_BG-sub_SB_offset_compare.png' % band_str, dpi = 300)
plt.close()


##. figs
for tt in range( len(R_bins) - 1 ):

	#.
	medi_cMag_f = np.mean( 10**( 0.4 * (22.5 - tmp_r_cMag[ tt ] ) ) )
	cc_medi_cMag_f = np.mean( 10**( 0.4 * (22.5 - cp_r_cMag[ tt ] ) ) )


	fig = plt.figure( figsize = (10.8, 4.8) )

	ax0 = fig.add_axes([0.08, 0.31, 0.40, 0.63])
	sub_ax0 = fig.add_axes([0.08, 0.10, 0.40, 0.21])

	ax1 = fig.add_axes([0.58, 0.31, 0.40, 0.63])
	sub_ax1 = fig.add_axes([0.58, 0.10, 0.40, 0.21])


	ax0.errorbar( nbg_R[tt], nbg_SB[tt] * off_frac[ tt ], yerr = nbg_err[tt], marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Control',)

	ax0.errorbar( cp_nbg_R[tt], cp_nbg_SB[tt] * cp_off_frac[ tt ], yerr = cp_nbg_err[tt], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'Member',)

	##.
	_kk_tmp_F = interp.interp1d( nbg_R[tt], nbg_SB[tt] * off_frac[ tt ], kind = 'cubic', fill_value = 'extrapolate',)
	_cc_tmp_sb = _kk_tmp_F( cp_nbg_R[tt] )

	sub_ax0.plot( cp_nbg_R[tt], cp_nbg_SB[tt] * cp_off_frac[ tt ] / _cc_tmp_sb, ls = '--', color = 'r', alpha = 0.75,)

	sub_ax0.fill_between( cp_nbg_R[tt], y1 = (cp_nbg_SB[tt] * cp_off_frac[ tt ] - cp_nbg_err[tt]) / _cc_tmp_sb, 
			y2 = (cp_nbg_SB[tt] * cp_off_frac[ tt ] + cp_nbg_err[tt]) / _cc_tmp_sb, color = 'r', alpha = 0.12,)

	sub_ax0.axhline( y = 1, ls = ':', color = 'k',)

	ax0.legend( loc = 3, frameon = False, fontsize = 12,)

	ax0.set_xscale('log')
	ax0.set_xlim( 1e0, 1e2 )

	ax0.annotate( s = '%s, %s-band' % (fig_name[tt], band_str), xy = (0.40, 0.85), xycoords = 'axes fraction', fontsize = 12,)

	ax0.set_ylim( 1e-3, 5e0 )
	ax0.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax0.set_yscale('log')

	sub_ax0.set_xlim( ax0.get_xlim() )
	sub_ax0.set_xscale('log')
	sub_ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax0.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 8, fontsize = 12,)
	sub_ax0.set_ylim( 0.9, 1.1 )

	sub_ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.set_xticklabels( labels = [] )


	ax1.errorbar( cumu_R[tt], cumu_L[tt] * off_frac[ tt ], yerr = cumu_Lerr[tt], marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Control',)

	ax1.errorbar( cp_cumu_R[tt], cp_cumu_L[tt] * cp_off_frac[ tt ], yerr = cp_cumu_Lerr[tt], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'Member',)

	ax1.axhline( y = medi_cMag_f, ls = ':', color = 'b', lw = 2.5,)
	ax1.axhline( y = cc_medi_cMag_f, ls = ':', color = 'r',)

	##.
	_kk_tmp_F = interp.interp1d( cumu_R[tt], cumu_L[tt] * off_frac[ tt ], kind = 'cubic', fill_value = 'extrapolate',)
	_cc_tmp_sb = _kk_tmp_F( cp_cumu_R[tt] )

	sub_ax1.plot( cp_cumu_R[tt], cp_cumu_L[tt] * cp_off_frac[ tt ] / _cc_tmp_sb, ls = '--', color = 'r', alpha = 0.75,)

	sub_ax1.fill_between( cp_cumu_R[tt], y1 = (cp_cumu_L[tt] * cp_off_frac[ tt ] - cp_cumu_Lerr[tt]) / _cc_tmp_sb, 
			y2 = (cp_cumu_L[tt] * cp_off_frac[ tt ] + cp_cumu_Lerr[tt]) / _cc_tmp_sb, color = 'r', alpha = 0.12,)

	sub_ax1.axhline( y = 1, ls = ':', color = 'k',)

	ax1.legend( loc = 4, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 1e0, 3e2 )
	# ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 8e-1, 5e1 )
	ax1.set_ylabel('F [ nanomaggy ]', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$F_{Member} \; / \; F_{Control}$', labelpad = 8, fontsize = 12,)
	sub_ax1.set_ylim( 0.9, 1.1 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig(
		'/home/xkchen/contrl-galx_%s-band_%.2f-%.2fR200m_BG-sub_cumuL_off_compare.png' % 
		(band_str, R_bins[tt], R_bins[tt + 1]), dpi = 300,)
	plt.close()

