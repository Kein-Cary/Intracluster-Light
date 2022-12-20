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


### ============ randomly binned case
def sR_cut_test():

	BG_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/sR_cut_test/BGs/'
	out_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/sR_cut_test/noBG_SBs/'
	path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/sR_cut_test/SBs/'

	#.
	N_sample = 50

	list_order = 13

	band_str = 'r'

	# #.
	# for id_part in range( 2 ):

	# 	##.
	# 	sat_sb_file = ( path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half'
	# 			% (band_str, id_part) + '_jack-sub-%d_SB-pro_z-ref.h5', )[0]

	# 	sub_out_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half'
	# 			% (band_str, id_part) + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5', )[0]

	# 	out_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half' 
	# 			% (band_str, id_part) + '_aveg-jack_BG-sub_SB.csv', )[0]

	# 	#.
	# 	bg_sb_file = ( BG_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
	# 			'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5', )[0]

	# 	stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, 
	# 						sub_out_file = sub_out_file, is_subBG = True)

	### === figs
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	fig_name = []

	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

		else:
			fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

	#.
	tmp_R, tmp_SB, tmp_SB_err = [], [], []

	for id_part in range( 2 ):

		with h5py.File( path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half'
				% (band_str, id_part) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tmp_R.append( tt_r )
		tmp_SB.append( tt_sb )
		tmp_SB_err.append( tt_err )

	#.
	tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

	for id_part in range( 2 ):

		with h5py.File( BG_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half_shufl-%d_BG'
			% (band_str, id_part, list_order) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tmp_bg_R.append( tt_r )
		tmp_bg_SB.append( tt_sb )
		tmp_bg_err.append( tt_err )

	#.
	nbg_R, nbg_SB, nbg_err = [], [], []

	for id_part in range( 2 ):

		dat = pds.read_csv( out_path + 
			'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half' % (band_str, id_part) + '_aveg-jack_BG-sub_SB.csv',)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_R.append( tt_r )
		nbg_SB.append( tt_sb )
		nbg_err.append( tt_sb_err )


	##...
	cp_out_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/noBG_SBs/'

	cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

	for tt in range( 4,5 ):

		dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' 
				% (R_bins[tt], R_bins[tt + 1]) + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		cp_nbg_R = np.array( dat['r'] )
		cp_nbg_SB = np.array( dat['sb'] )
		cp_nbg_err = np.array( dat['sb_err'] )


	### === figs
	line_name = ['Inner', 'Outer']
	color_s = ['b', 'r', 'k']


	fig = plt.figure()
	ax1 = fig.add_axes( [0.12, 0.11, 0.80, 0.85] )

	for dd in range( 2 ):

		l2 = ax1.errorbar( tmp_R[dd], tmp_SB[dd], yerr = tmp_SB_err[dd], marker = '.', ls = '-', color = color_s[dd],
			ecolor = color_s[dd], mfc = 'none', mec = color_s[dd], capsize = 1.5, alpha = 0.75, label = line_name[dd],)

		l3, = ax1.plot( tmp_bg_R[dd], tmp_bg_SB[dd], ls = '--', color = color_s[dd], alpha = 0.75,)
		ax1.fill_between( tmp_bg_R[dd], y1 = tmp_bg_SB[dd] - tmp_bg_err[dd], 
							y2 = tmp_bg_SB[dd] + tmp_bg_err[dd], color = color_s[dd], alpha = 0.12)

	legend_2 = ax1.legend( handles = [l2, l3], 
				labels = ['Satellite + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)
	ax1.add_artist( legend_2 )

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 1e-3, 6e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/Outer_bin_sat_SB_BG_compare.png', dpi = 300)
	plt.close()


	fig = plt.figure( )
	ax1 = fig.add_axes( [0.12, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.12, 0.11, 0.85, 0.21] )

	ax1.errorbar( cp_nbg_R, cp_nbg_SB, yerr = cp_nbg_err, marker = '', ls = '-', color = color_s[-1],
		ecolor = color_s[-1], mfc = 'none', mec = color_s[-1], capsize = 1.5, alpha = 0.75, label = 'Total',)

	_kk_tmp_F = interp.interp1d( cp_nbg_R, cp_nbg_SB, kind = 'cubic', fill_value = 'extrapolate',)

	for dd in range( 2 ):

		ax1.plot( nbg_R[dd], nbg_SB[dd], ls = '--', color = color_s[dd], alpha = 0.75, label = line_name[dd],)
		ax1.fill_between( nbg_R[dd], y1 = nbg_SB[dd] - nbg_err[dd], y2 = nbg_SB[dd] + nbg_err[dd], color = color_s[dd], alpha = 0.12,)

		sub_ax1.plot( nbg_R[dd], nbg_SB[dd] / _kk_tmp_F( nbg_R[dd] ), ls = '--', color = color_s[dd], alpha = 0.75,)
		sub_ax1.fill_between( nbg_R[dd], y1 = ( nbg_SB[dd] - nbg_err[dd] ) / _kk_tmp_F( nbg_R[dd] ), 
				y2 = ( nbg_SB[dd] + nbg_err[dd] ) / _kk_tmp_F( nbg_R[dd] ), color = color_s[dd], alpha = 0.12,)

	sub_ax1.fill_between( cp_nbg_R, y1 = (cp_nbg_SB - cp_nbg_err) / cp_nbg_SB, 
			y2 = (cp_nbg_SB + cp_nbg_err) / cp_nbg_SB, color = 'k', alpha = 0.12,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xlim( 1e0, 1e2 )
	ax1.set_xscale('log')

	ax1.set_ylim( 8e-4, 5e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,$(Total)', labelpad = 8, fontsize = 12,)
	sub_ax1.set_ylim( 0.75, 1.25 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax1.set_xticklabels( [] )

	plt.savefig('/home/xkchen/Outer_bin_sat_SB_compare.png', dpi = 300)
	plt.close()

	return

sR_cut_test()


##. update the ratio profiles among different radiusbin
BG_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/sR_cut_test/BGs/'
out_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/sR_cut_test/noBG_SBs/'
path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/sR_cut_test/SBs/'

#.
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#.
N_sample = 50

list_order = 13

band_str = 'r'


##...
nbg_R, nbg_SB, nbg_err = [], [], []

for id_part in range( 2 ):

	dat = pds.read_csv( out_path + 
		'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_%d-half' % (band_str, id_part) + '_aveg-jack_BG-sub_SB.csv',)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )


##...
cp_out_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/noBG_SBs/'

cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' 
			% (R_bins[tt], R_bins[tt + 1]) + '_r-band_aveg-jack_BG-sub_SB.csv',)

	tt_r = np.array( dat['r'] )
	tt_sb = np.array( dat['sb'] )
	tt_sb_err = np.array( dat['sb_err'] )

	cp_nbg_R.append( tt_r )
	cp_nbg_SB.append( tt_sb )
	cp_nbg_err.append( tt_sb_err )


##...
cp_eta_R, cp_eta, cp_eta_err = [], [], []

for tt in range( len(R_bins) - 2 ):

	dat = pds.read_csv( cp_out_path + 
			'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_r-band_aveg-jack_BG-sub_SB_ratio.csv'
			% (R_bins[tt], R_bins[tt+1]),)

	tt_R = np.array( dat['R'] )
	tt_eta = np.array( dat['ratio'] )
	tt_eta_err = np.array( dat['ratio_err'] )

	cp_eta_R.append( tt_R )
	cp_eta.append( tt_eta )
	cp_eta_err.append( tt_eta_err )


##...
cp_Rbins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 0.614, 1] )

fig_name = []

for dd in range( len(cp_Rbins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % cp_Rbins[dd + 1] )

	elif dd == len(cp_Rbins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % cp_Rbins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (cp_Rbins[dd], cp_Rbins[dd + 1]),)

color_s = ['b', 'c', 'g', 'm', 'r', 'k']


##.
fig = plt.figure()
ax = fig.add_axes( [0.12, 0.32, 0.85, 0.63] )
sub_ax = fig.add_axes( [0.12, 0.11, 0.85, 0.21] )

ax.errorbar( nbg_R[1], nbg_SB[1], yerr = nbg_err[1], marker = '', ls = '-', color = 'k',
	ecolor = color_s[-1], mfc = 'none', mec = color_s[-1], capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

_kk_tmp_F = interp.interp1d( nbg_R[1], nbg_SB[1], kind = 'cubic', fill_value = 'extrapolate',)
_kk_tmp_eF = interp.interp1d( nbg_R[1], nbg_err[1], kind = 'cubic', fill_value = 'extrapolate',)

#.
for dd in range( len(cp_Rbins) - 2 ):

	if dd <= 4:

		ax.errorbar( cp_nbg_R[dd], cp_nbg_SB[dd], yerr = cp_nbg_err[dd], marker = '', ls = '--', 
			color = color_s[dd], ecolor = color_s[dd], mfc = 'none', mec = color_s[dd], capsize = 1.5, alpha = 0.75, 
			label = fig_name[dd],)

		_cc_SB = _kk_tmp_F( cp_nbg_R[dd] )
		_cc_err = _kk_tmp_eF( cp_nbg_R[dd] )

		#.
		p_err1 = ( cp_nbg_err[dd] / _cc_SB )**2
		p_err2 = ( _cc_err * cp_nbg_SB[dd] / _cc_SB**2 )**2

		tmp_eta_err = np.sqrt( p_err1 + p_err2 )

		sub_ax.plot( cp_nbg_R[dd], cp_nbg_SB[dd] / _cc_SB, ls = '--', color = color_s[dd], alpha = 0.75,)
		sub_ax.fill_between( cp_nbg_R[dd], y1 = cp_nbg_SB[dd] / _cc_SB - tmp_eta_err, 
					y2 = cp_nbg_SB[dd] / _cc_SB + tmp_eta_err, color = color_s[dd], alpha = 0.12,)

	else:

		ax.errorbar( nbg_R[0], nbg_SB[0], yerr = nbg_err[0], marker = '', ls = '-', 
			color = color_s[dd], ecolor = color_s[dd], mfc = 'none', mec = color_s[dd], capsize = 1.5, alpha = 0.75, 
			label = fig_name[dd],)

		_cc_SB = _kk_tmp_F( nbg_R[0] )
		_cc_err = _kk_tmp_eF( nbg_R[0] )

		#.
		p_err1 = ( nbg_err[0] / _cc_SB )**2
		p_err2 = ( _cc_err * nbg_SB[0] / _cc_SB**2 )**2

		tmp_eta_err = np.sqrt( p_err1 + p_err2 )

		sub_ax.plot( nbg_R[0], nbg_SB[0] / _cc_SB, ls = '--', color = color_s[dd], alpha = 0.75,)
		sub_ax.fill_between( nbg_R[0], y1 = nbg_SB[0] / _cc_SB - tmp_eta_err, 
					y2 = nbg_SB[0] / _cc_SB + tmp_eta_err, color = color_s[dd], alpha = 0.12,)

#.
ax.legend( loc = 3, frameon = False, fontsize = 12,)

ax.set_xlim( 2e0, 1e2 )
ax.set_xscale('log')

ax.set_ylim( 1e-3, 5e0 )
ax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax.set_yscale('log')

sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xscale('log')
sub_ax.set_xlabel('$R \; [kpc]$', fontsize = 12,)

sub_ax.annotate( s = '$\\mu \; / \; \\mu \,$(%s)' % fig_name[-1], xy = (0.03, 0.05), 
				xycoords = 'axes fraction', fontsize = 12,)
sub_ax.set_ylim( 0.25, 1.10 )

sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/new_R_bin_ratio.png', dpi = 300)
plt.close()


##.
fig = plt.figure()
sub_ax = fig.add_axes( [0.12, 0.11, 0.85, 0.80] )

for dd in range( len(R_bins) - 2 ):

	sub_ax.plot( cp_eta_R[dd], cp_eta[dd], ls = '-', color = color_s[dd], alpha = 0.75, label = fig_name[dd],)
	sub_ax.fill_between( cp_eta_R[dd], y1 = cp_eta[dd] - cp_eta_err[dd], 
			y2 = cp_eta[dd] + cp_eta_err[dd], color = color_s[dd], alpha = 0.12,)

#.
sub_ax.legend( loc = 3, frameon = False, fontsize = 12,)

sub_ax.set_xlim( 2e0, 5e1 )
sub_ax.set_xscale('log')
sub_ax.set_xlabel('$R \; [kpc]$', fontsize = 12,)

_pre_name = '$R \\geq %.2f \, R_{200m}$' % R_bins[-2]

sub_ax.set_ylabel( '$\\mu \; / \; \\mu \,$(%s)' % _pre_name, fontsize = 12,)

sub_ax.set_ylim( 0.3, 1.02 )

sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/old_R_bin_ratio_cc.png', dpi = 300)
plt.close()



fig = plt.figure()
sub_ax = fig.add_axes( [0.12, 0.11, 0.85, 0.80] )

_kk_tmp_F = interp.interp1d( nbg_R[1], nbg_SB[1], kind = 'cubic', fill_value = 'extrapolate',)
_kk_tmp_eF = interp.interp1d( nbg_R[1], nbg_err[1], kind = 'cubic', fill_value = 'extrapolate',)

#.
for dd in range( len(cp_Rbins) - 2 ):

	if dd <= 4:
		_cc_SB = _kk_tmp_F( cp_nbg_R[dd] )
		_cc_err = _kk_tmp_eF( cp_nbg_R[dd] )

		#.
		p_err1 = ( cp_nbg_err[dd] / _cc_SB )**2
		p_err2 = ( _cc_err * cp_nbg_SB[dd] / _cc_SB**2 )**2

		tmp_eta_err = np.sqrt( p_err1 + p_err2 )

		sub_ax.plot( cp_nbg_R[dd], cp_nbg_SB[dd] / _cc_SB, ls = '-', color = color_s[dd], alpha = 0.75,
					label = fig_name[dd],)
		sub_ax.fill_between( cp_nbg_R[dd], y1 = cp_nbg_SB[dd] / _cc_SB - tmp_eta_err, 
					y2 = cp_nbg_SB[dd] / _cc_SB + tmp_eta_err, color = color_s[dd], alpha = 0.12,)

	else:
		_cc_SB = _kk_tmp_F( nbg_R[0] )
		_cc_err = _kk_tmp_eF( nbg_R[0] )

		#.
		p_err1 = ( nbg_err[0] / _cc_SB )**2
		p_err2 = ( _cc_err * nbg_SB[0] / _cc_SB**2 )**2

		tmp_eta_err = np.sqrt( p_err1 + p_err2 )

		sub_ax.plot( nbg_R[0], nbg_SB[0] / _cc_SB, ls = '-', color = color_s[dd], alpha = 0.75,)
		sub_ax.fill_between( nbg_R[0], y1 = nbg_SB[0] / _cc_SB - tmp_eta_err, 
					y2 = nbg_SB[0] / _cc_SB + tmp_eta_err, color = color_s[dd], alpha = 0.12,)

#.
sub_ax.legend( loc = 3, frameon = False, fontsize = 12,)

sub_ax.set_xlim( 2e0, 5e1 )
sub_ax.set_xscale('log')
sub_ax.set_xlabel('$R \; [kpc]$', fontsize = 12,)

sub_ax.set_ylabel( '$\\mu \; / \; \\mu \,$(%s)' % fig_name[-1], fontsize = 12,)
sub_ax.set_ylim( 0.3, 1.02 )

sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/new_R_bin_ratio_cc.png', dpi = 300)
plt.close()

raise


### ============ randomly binned case
def randomly_cut_test():

	BG_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/rand_cut_test/BGs/'
	out_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/rand_cut_test/noBG_SBs/'
	path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/rand_cut_test/SBs/'

	#.
	N_sample = 50

	list_order = 13

	band_str = 'r'

	N_rnd = 10

	##.
	"""
	for tt in range( N_rnd ):

		for id_part in range( 2 ):

			##.
			sat_sb_file = ( path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d'
				% (band_str, tt, id_part) + '_jack-sub-%d_SB-pro_z-ref.h5', )[0]

			sub_out_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d'
				% (band_str, tt, id_part) + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5', )[0]

			out_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d'
				% (band_str, tt, id_part) + '_aveg-jack_BG-sub_SB.csv', )[0]

			#.
			bg_sb_file = ( BG_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d_shufl-%d_BG'
				% (band_str, tt, id_part, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5', )[0]

			stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, 
								sub_out_file = sub_out_file, is_subBG = True)

	raise
	"""


	### === figs
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	fig_name = []

	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

		else:
			fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

	#.
	tmp_R, tmp_SB, tmp_SB_err = [], [], []

	for tt in range( N_rnd ):

		dd_R, dd_SB, dd_SB_err = [], [], []

		for id_part in range( 2 ):

			with h5py.File( path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d'
				% (band_str, tt, id_part) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

				tt_r = np.array(f['r'])
				tt_sb = np.array(f['sb'])
				tt_err = np.array(f['sb_err'])

			dd_R.append( tt_r )
			dd_SB.append( tt_sb )
			dd_SB_err.append( tt_err )

		tmp_R.append( dd_R )
		tmp_SB.append( dd_SB )
		tmp_SB_err.append( dd_SB_err )

	#.
	tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

	for tt in range( N_rnd ):

		dd_R, dd_SB, dd_SB_err = [], [], []

		for id_part in range( 2 ):

			with h5py.File( BG_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d_shufl-%d_BG'
				% (band_str, tt, id_part, list_order) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

				tt_r = np.array(f['r'])
				tt_sb = np.array(f['sb'])
				tt_err = np.array(f['sb_err'])

			dd_R.append( tt_r )
			dd_SB.append( tt_sb )
			dd_SB_err.append( tt_err )

		tmp_bg_R.append( dd_R )
		tmp_bg_SB.append( dd_SB )
		tmp_bg_err.append( dd_SB_err )

	#.
	nbg_R, nbg_SB, nbg_err = [], [], []

	for tt in range( N_rnd ):

		dd_R, dd_SB, dd_SB_err = [], [], []

		for id_part in range( 2 ):

			dat = pds.read_csv( out_path + 
				'Extend_BCGM_gri-common_all_0.56-1.00R200m_%s-band_rand-%d_half-%d' % (band_str, tt, id_part) + '_aveg-jack_BG-sub_SB.csv',)

			tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

			dd_R.append( tt_r )
			dd_SB.append( tt_sb )
			dd_SB_err.append( tt_err )

		nbg_R.append( dd_R )
		nbg_SB.append( dd_SB )
		nbg_err.append( dd_SB_err )


	##...
	cp_out_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/noBG_SBs/'

	cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

	for tt in range( 4,5 ):

		dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' 
				% (R_bins[tt], R_bins[tt + 1]) + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		cp_nbg_R = np.array( dat['r'] )
		cp_nbg_SB = np.array( dat['sb'] )
		cp_nbg_err = np.array( dat['sb_err'] )


	### === figs
	line_name = ['Part-0', 'Part-1']
	color_s = ['b', 'r', 'k']

	# for tt in range( N_rnd ):

	# 	fig = plt.figure()
	# 	ax1 = fig.add_axes( [0.12, 0.11, 0.80, 0.85] )

	# 	for dd in range( 2 ):

	# 		l2 = ax1.errorbar( tmp_R[tt][dd], tmp_SB[tt][dd], yerr = tmp_SB_err[tt][dd], marker = '.', ls = '-', color = color_s[dd],
	# 			ecolor = color_s[dd], mfc = 'none', mec = color_s[dd], capsize = 1.5, alpha = 0.75, label = line_name[dd],)

	# 		l3, = ax1.plot( tmp_bg_R[tt][dd], tmp_bg_SB[tt][dd], ls = '--', color = color_s[dd], alpha = 0.75,)
	# 		ax1.fill_between( tmp_bg_R[tt][dd], y1 = tmp_bg_SB[tt][dd] - tmp_bg_err[tt][dd], 
	# 							y2 = tmp_bg_SB[tt][dd] + tmp_bg_err[tt][dd], color = color_s[dd], alpha = 0.12)

	# 	legend_2 = ax1.legend( handles = [l2, l3], 
	# 				labels = ['Satellite + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

	# 	ax1.legend( loc = 1, frameon = False, fontsize = 12,)
	# 	ax1.add_artist( legend_2 )

	# 	ax1.set_xscale('log')
	# 	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	# 	ax1.set_ylim( 1e-3, 6e0 )
	# 	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	# 	ax1.set_yscale('log')

	# 	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	# 	plt.savefig('/home/xkchen/Outer_bin_sat_rand_cut_stack_SB_BG_test_%d.png' % tt, dpi = 300)
	# 	plt.close()


	for tt in range( N_rnd ):

		fig = plt.figure( )
		ax1 = fig.add_axes( [0.12, 0.32, 0.85, 0.63] )
		sub_ax1 = fig.add_axes( [0.12, 0.11, 0.85, 0.21] )

		ax1.errorbar( cp_nbg_R, cp_nbg_SB, yerr = cp_nbg_err, marker = '', ls = '-', color = 'k',
			ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Total',)

		sub_ax1.fill_between( cp_nbg_R, y1 = (cp_nbg_SB - cp_nbg_err) / cp_nbg_SB, 
				y2 = (cp_nbg_SB + cp_nbg_err) / cp_nbg_SB, color = 'k', alpha = 0.25,)

		_kk_tmp_F = interp.interp1d( cp_nbg_R, cp_nbg_SB, kind = 'cubic', fill_value = 'extrapolate',)


		#.
		ax1.plot( nbg_R[tt][0], nbg_SB[tt][0], ls = '--', color = 'b', alpha = 0.65, label = line_name[0],)
		sub_ax1.plot( nbg_R[tt][0], nbg_SB[tt][0] / _kk_tmp_F( nbg_R[tt][0] ), ls = '--', color = 'b', alpha = 0.65,)

		ax1.fill_between( nbg_R[tt][0], y1 = nbg_SB[tt][0] - nbg_err[tt][0], y2 = nbg_SB[tt][0] + nbg_err[tt][0], color = 'b', alpha = 0.12,)
		sub_ax1.fill_between( nbg_R[tt][0], y1 = ( nbg_SB[tt][0] - nbg_err[tt][0] ) / _kk_tmp_F( nbg_R[tt][0] ), 
				y2 = ( nbg_SB[tt][0] + nbg_err[tt][0] ) / _kk_tmp_F( nbg_R[tt][0] ), color = 'b', alpha = 0.12,)

		#.
		ax1.plot( nbg_R[tt][1], nbg_SB[tt][1], ls = '--', color = 'r', alpha = 0.65, label = line_name[1],)
		sub_ax1.plot( nbg_R[tt][1], nbg_SB[tt][1] / _kk_tmp_F( nbg_R[tt][1] ), ls = '--', color = 'r', alpha = 0.65,)

		ax1.fill_between( nbg_R[tt][1], y1 = nbg_SB[tt][1] - nbg_err[tt][1], y2 = nbg_SB[tt][1] + nbg_err[tt][1], color = 'r', alpha = 0.12,)
		sub_ax1.fill_between( nbg_R[tt][1], y1 = ( nbg_SB[tt][1] - nbg_err[tt][1] ) / _kk_tmp_F( nbg_R[tt][1] ), 
				y2 = ( nbg_SB[tt][1] + nbg_err[tt][1] ) / _kk_tmp_F( nbg_R[tt][1] ), color = 'r', alpha = 0.12,)

		ax1.legend( loc = 3, frameon = False, fontsize = 12,)

		ax1.set_xlim( 1e0, 1e2 )
		ax1.set_xscale('log')

		ax1.set_ylim( 8e-4, 5e0 )
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
		ax1.set_yscale('log')

		sub_ax1.set_xlim( ax1.get_xlim() )
		sub_ax1.set_xscale('log')
		sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

		sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,$(Total)', labelpad = 8, fontsize = 12,)
		sub_ax1.set_ylim( 0.75, 1.25 )

		sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
		ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

		ax1.set_xticklabels( [] )

		plt.savefig('/home/xkchen/Outer_bin_sat_rand_cut_stack_BG-sub-SB_test_%d.png' % tt, dpi = 300)
		plt.close()

	raise


	#.
	fig = plt.figure( )
	ax1 = fig.add_axes( [0.12, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.12, 0.11, 0.85, 0.21] )

	ax1.errorbar( cp_nbg_R, cp_nbg_SB, yerr = cp_nbg_err, marker = '', ls = '-', color = 'k',
		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Total',)

	sub_ax1.fill_between( cp_nbg_R, y1 = (cp_nbg_SB - cp_nbg_err) / cp_nbg_SB, 
			y2 = (cp_nbg_SB + cp_nbg_err) / cp_nbg_SB, color = 'k', alpha = 0.25,)

	_kk_tmp_F = interp.interp1d( cp_nbg_R, cp_nbg_SB, kind = 'linear', fill_value = 'extrapolate',)

	for tt in range( N_rnd ):

		#.
		ax1.plot( nbg_R[tt][0], nbg_SB[tt][0], ls = '--', color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.65, )
		sub_ax1.plot( nbg_R[tt][0], nbg_SB[tt][0] / _kk_tmp_F( nbg_R[tt][0] ), ls = '--', color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.65,)

		# if tt == 0:
		# 	ax1.fill_between( nbg_R[tt][0], y1 = nbg_SB[tt][0] - nbg_err[tt][0], y2 = nbg_SB[tt][0] + nbg_err[tt][0], color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.12,)
		# 	sub_ax1.fill_between( nbg_R[tt][0], y1 = ( nbg_SB[tt][0] - nbg_err[tt][0] ) / _kk_tmp_F( nbg_R[tt][0] ), 
		# 			y2 = ( nbg_SB[tt][0] + nbg_err[tt][0] ) / _kk_tmp_F( nbg_R[tt][0] ), color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.12,)

		#.
		ax1.plot( nbg_R[tt][1], nbg_SB[tt][1], ls = ':', color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.65, )
		sub_ax1.plot( nbg_R[tt][1], nbg_SB[tt][1] / _kk_tmp_F( nbg_R[tt][1] ), ls = ':', color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.65,)

		# if tt == 0:
		# 	ax1.fill_between( nbg_R[tt][1], y1 = nbg_SB[tt][1] - nbg_err[tt][1], y2 = nbg_SB[tt][1] + nbg_err[tt][1], color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.12,)
		# 	sub_ax1.fill_between( nbg_R[tt][1], y1 = ( nbg_SB[tt][1] - nbg_err[tt][1] ) / _kk_tmp_F( nbg_R[tt][1] ), 
		# 			y2 = ( nbg_SB[tt][1] + nbg_err[tt][1] ) / _kk_tmp_F( nbg_R[tt][1] ), color = mpl.cm.rainbow( tt / N_rnd ), alpha = 0.12,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xlim( 1e0, 1e2 )
	ax1.set_xscale('log')

	ax1.set_ylim( 8e-4, 5e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,$(Total)', labelpad = 8, fontsize = 12,)
	sub_ax1.set_ylim( 0.75, 1.25 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax1.set_xticklabels( [] )

	plt.savefig('/home/xkchen/Outer_bin_sat_rand_cut_stack_SB_compare.png', dpi = 300)
	plt.close()

	return

#.
randomly_cut_test()

