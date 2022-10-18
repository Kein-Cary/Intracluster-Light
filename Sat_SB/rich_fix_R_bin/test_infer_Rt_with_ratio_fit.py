"""
try to infer critical Radius of R-only binned subsamples
by:
1) SB profile ratio
2) slope of ratio profile between satellite and control galaxies
"""

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

from pynverse import inversefunc
from astropy.table import Table, QTable
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



### === data load

##. sat_img without BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'


##. shuffle list
list_order = 13

sub_name = ['low-rich', 'medi-rich', 'high-rich']

bin_rich = [ 20, 30, 50, 210 ]
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']

line_s = [':', '--', '-']
color_s = ['b', 'g', 'm', 'r', 'k']


##. R_limmits
# R_str = 'phy'
# R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ### kpc

R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m


##. line name
if R_str == 'phy':

	fig_name = []
	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

		else:
			fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

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
for kk in range( 1 ):

	band_str = band[ kk ]

	for qq in range( 3 ):

		#. data load
		crit_eta = [ 0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95 ]
		crit_R = []

		#.
		fit_param = []

		out_R_lim = 15   ##. kpc

		for tt in range( len(R_bins) - 1 ):

			#.
			if R_str == 'phy':

				dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

				tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

				##. use for ratio compare
				cc_dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[-2], R_bins[-1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

				cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )

			#.
			if R_str == 'scale':

				dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[tt], R_bins[tt + 1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

				tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

				##. use for ratio compare
				cc_dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[-2], R_bins[-1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

				cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )

			#.
			id_rx = cc_tt_r < 80 ##.kpc
			_tt_tmp_F = interp.interp1d( cc_tt_r[ id_rx ], cc_tt_sb[ id_rx ], kind = 'cubic', fill_value = 'extrapolate',)

			#.
			if tt == len(R_bins) - 2:

				##.
				cen_R_lim_0 = np.ones( len( crit_eta ), ) * np.nan
				pre_R_lim = np.nan

				fit_param.append([ np.nan, np.nan ])

			#.
			else:

				##. critical radius
				id_rx = tt_r <= out_R_lim

				sm_r = tt_r[ id_rx ]

				tt_eta = tt_sb[ id_rx ] / _tt_tmp_F( sm_r )
				tt_var = tt_sb_err[ id_rx ] / _tt_tmp_F( sm_r )

				tck = np.polyfit( sm_r, tt_eta, 1, w = 1 / tt_var )
				tcf = np.poly1d( tck )

				#.
				fit_param.append([ tcf[1], tcf[0] ])

				x_new = np.logspace( -3, 2.4, 500 )
				y_new = tcf( x_new )

				##.
				cen_R_lim_0 = np.zeros( len(crit_eta), )

				for oo in range( len( crit_eta) ):

					tag_eta = y_new[0] * ( 1 - crit_eta[ oo ] )
					cen_R_lim_0[ oo ] = inversefunc( tcf, tag_eta )

				##.
				fig = plt.figure()
				ax = fig.add_axes([0.12, 0.11, 0.80, 0.85 ])

				ax.plot( sm_r, tt_eta, 'r-', )
				ax.fill_between( sm_r, y1 = tt_eta - tt_var, y2 = tt_eta + tt_var, color = 'r',)

				ax.plot( x_new, y_new, 'g:',)

				#. 
				for oo in range( len( crit_eta) ):

					ax.axvline( x = cen_R_lim_0[ oo ], ls = ':', color = 'k',)

				ax.set_xlim( 2e0, 2e2 )
				ax.set_xscale('log')
				ax.set_ylim( 0.0, 1.0 )

				ax.tick_params( axis = 'both', which = 'both', direction = 'in',)
				ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

				plt.savefig('/home/xkchen/Sat_%s_%s_%s-band_%d_interp_test.png' % 
							(sub_name[qq], R_str, band_str, tt), dpi = 300)
				plt.close()

			##.
			crit_R.append( cen_R_lim_0 )

		##. save the crit_R
		values = crit_R

		tab_file = Table( values, names = fig_name )
		tab_file.write( out_path + 'Extend_BCGM_gri-common_%s_%s_%s-band_polyfit_Rt_test.fits' % 
						(sub_name[qq], R_str, band_str), overwrite = True)

		##. save the params of fitting
		##. a * x + b
		keys = ['a', 'b']
		values = [ np.array( fit_param )[:,0], np.array( fit_param )[:,1] ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 
					'Extend_BCGM_gri-common_%s_%s_%s-band_polyfit_params.csv' % (sub_name[qq], R_str, band_str),)


##... 
y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [7e-4, 1e0], [5e-3, 6e0] ]

for kk in range( 1 ):

	band_str = band[ kk ]

	##. ratio thresh for Rt estimation
	crit_eta = [ 0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95 ]

	##.
	for oo in range( len(crit_eta) ):

		id_set = oo

		##. 
		fig = plt.figure( figsize = (19.84, 4.8) )
		ax0 = fig.add_axes([0.05, 0.11, 0.28, 0.84])
		ax1 = fig.add_axes([0.38, 0.11, 0.28, 0.84])
		ax2 = fig.add_axes([0.71, 0.11, 0.28, 0.84])

		axes = [ ax0, ax1, ax2 ]


		#. data load
		for qq in range( 3 ):

			nbg_R, nbg_SB, nbg_err = [], [], []

			for tt in range( len(R_bins) - 1 ):

				if R_str == 'phy':

					dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc_%s-band_aveg-jack_BG-sub_SB.csv' % 
										(sub_name[qq], R_bins[tt], R_bins[tt + 1], band_str), )

					tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

				if R_str == 'scale':

					dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' % 
										(sub_name[qq], R_bins[tt], R_bins[tt + 1], band_str), )

					tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

				nbg_R.append( tt_r )
				nbg_SB.append( tt_sb )
				nbg_err.append( tt_sb_err )

			##.
			pat = fits.open( out_path + 'Extend_BCGM_gri-common_%s_%s_%s-band_polyfit_Rt_test.fits' % (sub_name[qq], R_str, band_str),)
			p_table = pat[1].data

			crit_R = []

			for tt in range( len(R_bins) - 1 ):

				Rt_arr = np.array( p_table[ fig_name[tt] ] )

				crit_R.append( Rt_arr[ id_set ] )

			##.
			pat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%s_%s-band_polyfit_params.csv' % (sub_name[qq], R_str, band_str),)
			fit_a = np.array( pat['a'] )
			fit_b = np.array( pat['b'] )


			#.
			gax = axes[qq]

			_kk_tmp_F = interp.interp1d( nbg_R[-1], nbg_SB[-1], kind = 'cubic', fill_value = 'extrapolate',)

			for mm in range( len(R_bins) -2 ):

				##.
				_dp_eta = nbg_SB[mm] / _kk_tmp_F( nbg_R[mm] )

				##. extent the ratio profile
				x_new = np.logspace( -1, 2.5, 500 )
				tck_F = np.poly1d( [ fit_a[ mm ], fit_b[ mm ] ] )
				y_new = tck_F( x_new )

				gax.plot( nbg_R[mm], _dp_eta, ls = '-', color = color_s[mm], alpha = 0.75, label = fig_name[mm],)

				gax.fill_between( nbg_R[mm], y1 = _dp_eta - nbg_err[mm] / _kk_tmp_F( nbg_R[mm] ), 
							y2 = _dp_eta + nbg_err[mm] / _kk_tmp_F( nbg_R[mm] ), color = color_s[mm], alpha = 0.12,)

				gax.axvline( x = crit_R[mm], ls = '--', color = color_s[mm], alpha = 0.75,)

				gax.plot( x_new, y_new, ls = ':', color = color_s[mm],)

			gax.annotate( s = line_name[qq] + ', %s-band' % band_str, xy = (0.35, 0.87), 
							xycoords = 'axes fraction', fontsize = 14, backgroundcolor = 'w',)
			gax.legend( loc = 3, frameon = True, fontsize = 13,)

			gax.set_xlim( 1e0, 1e2 )
			gax.set_xscale('log')
			gax.set_xlabel('$R \; [kpc]$', fontsize = 14)

			gax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
								fontsize = 11, labelpad = 8)
			gax.set_ylim( 0, 1 )

			gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
			gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

		plt.savefig('/home/xkchen/%s_%s-band_sat-BG-sub_SB-ratio_%.2feta.png' % 
					( R_str, band_str, crit_eta[ id_set ] ), dpi = 300)
		plt.close()

