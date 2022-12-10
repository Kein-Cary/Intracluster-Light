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

#.
from light_measure import light_measure_weit
from img_sat_BG_sub_SB import stack_BG_sub_func
from light_measure import cov_MX_func
from img_sat_fig_out_mode import arr_jack_func


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

##. fixed R for all richness subsample
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m


### === BG-sub SB profiles
"""
for dd in range( 2 ):

	for ll in range( len(R_bins) - 1 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			##.
			sat_sb_file = ( path + '%s_clust_%.2f-%.2fR200m_%s-band' % (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str) 
						+ '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			sub_out_file = ( out_path + '%s_clust_%.2f-%.2fR200m_%s-band' % (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str)  
						+ '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5',)[0]

			out_file = ( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str),)[0]

			#.
			# bg_sb_file = ( BG_path + '%s_clust_%.2f-%.2fR200m_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
			# 			% (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str, list_order),)[0]

			# stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = sub_out_file )

			#.
			bg_sb_file = ( BG_path + 
				'%s_clust_%.2f-%.2fR200m_%s-band_shufl-%d_BG' % (cat_lis[dd], R_bins[ll], R_bins[ll+1], band_str, list_order) 
				+ '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, 
							sub_out_file = sub_out_file, is_subBG = True)

raise
"""

### === covMatrix and corMatrix
"""
##. cov-Matrix of BG-sub SBs
for ll in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		for tt in range( len(R_bins) - 1 ):

			tmp_R, tmp_SB = [], []

			for dd in range( N_sample ):

				dat = pds.read_csv(out_path + 
							'%s_clust_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
							% (cat_lis[ll], R_bins[tt], R_bins[tt+1], band_str, dd),)

				tt_r, tt_sb = np.array( dat['r'] ), np.array( dat['sb'] )

				tmp_R.append( tt_r )
				tmp_SB.append( tt_sb )

			##.
			R_mean, cov_MX, cor_MX = cov_MX_func( tmp_R, tmp_SB, id_jack = True )

			#.
			with h5py.File( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_BG-sub-SB_cov-arr.h5'
				% (cat_lis[ll], R_bins[ll], R_bins[ll+1], band_str), 'w') as f:

				f['R_kpc'] = np.array( R_mean )
				f['cov_MX'] = np.array( cov_MX )
				f['cor_MX'] = np.array( cor_MX )

			#.
			fig = plt.figure()
			ax = fig.add_axes([0.12, 0.11, 0.85, 0.80])

			ax.imshow( cor_MX, origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1,)

			plt.savefig('/home/xkchen/%s_clust_Sat_all_%.2f-%.2fR200m_%s-band_cormax.png'
						% (cat_lis[ll], R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)
			plt.close()


##. aveg ratio profile
for ll in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		for tt in range( len(R_bins) - 2 ):

			#.
			tmp_R, tmp_ratio = [], []

			for dd in range( N_sample ):

				#.
				cc_dat = pds.read_csv( out_path + 
						'%s_clust_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5'
						% (cat_lis[ll], R_bins[-2], R_bins[-1], band_str, dd),)

				cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )

				id_nan = np.isnan( cc_tt_sb )
				id_px = id_nan == False

				_tt_tmp_F = interp.interp1d( cc_tt_r[ id_px ], cc_tt_sb[ id_px ], kind = 'cubic', fill_value = 'extrapolate',)


				#.
				dat = pds.read_csv( out_path + 
						'%s_clust_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5'
						% (cat_lis[ll], R_bins[tt], R_bins[tt + 1], band_str, dd),)

				tt_r, tt_sb = np.array( dat['r'] ), np.array( dat['sb'] )
				tt_eta = tt_sb / _tt_tmp_F( tt_r )

				tmp_R.append( tt_r )
				tmp_ratio.append( tt_eta )

			#.
			aveg_R_0, aveg_ratio, aveg_eta_err = arr_jack_func( tmp_ratio, tmp_R, N_sample )[:3]

			keys = [ 'R', 'ratio', 'ratio_err' ]
			values = [ aveg_R_0, aveg_ratio, aveg_eta_err ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB_ratio.csv'
						% (cat_lis[ll], R_bins[tt], R_bins[tt + 1], band_str),)

			#.
			R_mean, cov_MX, cor_MX = cov_MX_func( tmp_R, tmp_ratio, id_jack = True )

			#.
			with h5py.File( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_BG-sub-SB_ratio_cov-arr.h5'
				 % (cat_lis[ll], R_bins[tt], R_bins[tt + 1], band_str), 'w') as f:
				f['R_kpc'] = np.array( R_mean )
				f['cov_MX'] = np.array( cov_MX )
				f['cor_MX'] = np.array( cor_MX )

			plt.figure()
			plt.imshow( cor_MX, origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1)
			plt.savefig('/home/xkchen/%s_clust_Sat_all_%.2f-%.2fR200m_%s-band_ratio_cov_arr.png' 
						% (cat_lis[ll], R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)
			plt.close()

raise
"""


### === figs and comparison
sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

#.
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']
samp_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

#.
color_s = ['b', 'g', 'r', 'm', 'k']
line_s = ['--','-']
line_c = ['b', 'r']

#.
fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


### === results comparison
band_str = 'i'

dpt_R, dpt_SB, dpt_err = [], [], []
dpt_bg_R, dpt_bg_SB, dpt_bg_err = [], [], []
dpt_nbg_R, dpt_nbg_SB, dpt_nbg_err = [], [], []

##.
for qq in range( 2 ):

	##... sat SBs
	tmp_R, tmp_sb, tmp_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		with h5py.File( path + '%s_clust_%.2f-%.2fR200m_%s-band_Mean_jack_SB-pro_z-ref.h5' 
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

		with h5py.File( BG_path + '%s_clust_%.2f-%.2fR200m_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
			% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str, list_order), 'r') as f:

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
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
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

	if band_str == 'i':
		ax1.set_ylim( 3e-3, 1e1 )

	else:
		ax1.set_ylim( 1e-3, 5e0 )

	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_clust_sat_%s-band_BG_compare.png' % (cat_lis[qq], band_str), dpi = 300)
	plt.close()


##.
dpt_eta_R, dpt_eta_V, dpt_eta_err = [], [], []

for qq in range( 2 ):

	##... sat SBs
	tmp_R, tmp_eta, tmp_eta_err = [], [], []

	for ll in range( len(R_bins) - 2 ):

		dat = pds.read_csv( out_path + 
					'%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB_ratio.csv'
					% (cat_lis[qq], R_bins[ll], R_bins[ll + 1], band_str),)

		tt_r = np.array( dat['R'] )
		tt_eta = np.array( dat['ratio'] )
		tt_err = np.array( dat['ratio_err'] )

		#.
		tmp_R.append( tt_r )
		tmp_eta.append( tt_eta )
		tmp_eta_err.append( tt_err )

	#.
	dpt_eta_R.append( tmp_R )
	dpt_eta_V.append( tmp_eta )
	dpt_eta_err.append( tmp_eta_err )

	#.
	fig = plt.figure( )
	ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
	sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

	#.
	ax1.errorbar( dpt_nbg_R[qq][-1], dpt_nbg_SB[qq][-1], yerr = dpt_nbg_err[qq][-1], marker = '', ls = '-', color = 'k',
		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

	#.
	for mm in range( len(R_bins) - 2 ):

		ax1.errorbar( dpt_nbg_R[qq][mm], dpt_nbg_SB[qq][mm], yerr = dpt_nbg_err[qq][mm], marker = '', ls = '--', color = color_s[mm], 
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

		sub_ax1.plot( tmp_R[mm], tmp_eta[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
		sub_ax1.fill_between( tmp_R[mm], y1 = tmp_eta[mm] - tmp_eta_err[mm], 
					y2 = tmp_eta[mm] + tmp_eta_err[mm], color = color_s[mm], alpha = 0.15,)

	ax1.annotate( s = samp_name[qq] + ', %s-band' % band_str, xy = (0.65, 0.85), xycoords = 'axes fraction', fontsize = 12,)

	ax1.legend( loc = 3, frameon = False, fontsize = 12,)

	ax1.set_xlim( 1e0, 5e1 )
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	if band_str == 'i':
		ax1.set_ylim( 3e-3, 1e1 )

	else:
		ax1.set_ylim( 1e-3, 5e0 )

	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,$ (%s)' % fig_name[-1], labelpad = 10, fontsize = 12,)

	sub_ax1.set_ylim( 0.2, 0.98 )

	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax1.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_clust_sat_%s-band_BG-sub_compare.png' % (cat_lis[qq], band_str), dpi = 300)
	plt.close()


##. ratio profiles difference
fig = plt.figure( )
ax1 = fig.add_axes([0.12, 0.12, 0.85, 0.80])

for qq in range( 2 ):

	for mm in range( len(R_bins) - 2 ):

		#.
		if qq == 0:
			l1, = ax1.plot( dpt_eta_R[qq][mm], dpt_eta_V[qq][mm], ls = line_s[qq], color = color_s[mm], alpha = 0.75, label = fig_name[mm],)
			ax1.fill_between( dpt_eta_R[qq][mm], y1 = (dpt_eta_V[qq][mm] - dpt_eta_err[qq][mm]), 
						y2 = (dpt_eta_V[qq][mm] + dpt_eta_err[qq][mm]), color = color_s[mm], alpha = 0.15,)

		else:
			l2, = ax1.plot( dpt_eta_R[qq][mm], dpt_eta_V[qq][mm], ls = line_s[qq], color = color_s[mm], alpha = 0.75,)
			ax1.fill_between( dpt_eta_R[qq][mm], y1 = (dpt_eta_V[qq][mm] - dpt_eta_err[qq][mm]), 
						y2 = (dpt_eta_V[qq][mm] + dpt_eta_err[qq][mm]), color = color_s[mm], alpha = 0.15,)

#.
legend_0 = plt.legend( handles = [l1, l2], labels = [ samp_name[0], samp_name[1] ],
			loc = 4, frameon = False, fontsize = 13, markerfirst = False,)

ax1.legend( loc = 3, frameon = False, fontsize = 13, markerfirst = False,)
ax1.add_artist( legend_0 )

ax1.set_xlim( 1e0, 5e1 )
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

ax1.set_ylabel('$\\mu \; / \; \\mu \,$ (%s)' % fig_name[-1], labelpad = 10, fontsize = 12,)

ax1.set_ylim( 0.2, 0.98 )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

plt.savefig('/home/xkchen/Sat_BCG_Mstar_%s-band_bin_ratio_compare.png' % band_str, dpi = 300)
plt.close()

