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


R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


N_sample = 100

##. background shuffle list order
list_order = 13


##.
for kk in range( 1 ):

	band_str = band[ kk ]

	#. data load
	crit_eta_0 = 0.15  # 0.15
	crit_R_0 = []

	crit_eta = [ 0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95 ]
	crit_R = []

	out_R_lim = 30  ##. kpc

	for tt in range( len(R_bins) - 1 ):
	# for tt in range( 1 ):

		dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		##. use for ratio compare
		cc_dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[-2], R_bins[-1]) 
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )


		id_rx = cc_tt_r < 80 ##.kpc
		_tt_tmp_F = interp.interp1d( cc_tt_r[ id_rx ], cc_tt_sb[ id_rx ], kind = 'cubic', fill_value = 'extrapolate',)


		if tt == len(R_bins) - 2:

			##.
			cen_R_lim_0 = np.ones( len( crit_eta ), ) * np.nan
			pre_R_lim = np.nan

		else:

			#. critical radius
			id_rx = tt_r <= out_R_lim

			sm_r = tt_r[ id_rx ]

			tt_eta = tt_sb[ id_rx ] / _tt_tmp_F( sm_r )
			sm_eta = signal.savgol_filter( tt_eta, 5, 1 )  ##. 7, 2

			_tmp_interp_F = interp.interp1d( sm_r, sm_eta, kind = 'linear', fill_value = 'extrapolate',)

			new_R = np.logspace( np.log10( sm_r[0] ), np.log10( sm_r[-1] ), 500)
			new_eta = _tmp_interp_F( new_R )

			#. use crit_eta_0 to find the R_lim for interpolation
			tag_eta = new_eta[0] * ( 1 - crit_eta_0 )
			pre_R_lim = inversefunc( _tmp_interp_F, tag_eta )


			##. extent the ratio profile
			id_vx = new_R <= pre_R_lim

			lim_R = new_R[ id_vx ]
			lim_eta = new_eta[ id_vx ]

			# x_new = np.logspace( -3, 2.4, 200 )
			x_new = np.linspace( 0.01, 200, 700)
			tck_F = interp.interp1d( lim_R, lim_eta, kind = 'linear', fill_value = 'extrapolate',)
			y_new = tck_F( x_new )
			
			id_nul = y_new < 0.
			y_new[ id_nul ] = 0.

			#.
			cen_R_lim_0 = np.zeros( len(crit_eta), )

			for oo in range( len( crit_eta) ):

				_pp_F = interp.interp1d( x_new[ id_nul == False], y_new[ id_nul == False], kind = 'linear', fill_value = 'extrapolate',)

				tag_eta = y_new[0] * ( 1 - crit_eta[ oo ] )
				cen_R_lim_0[ oo ] = inversefunc( tck_F, tag_eta )

		crit_R.append( cen_R_lim_0 )
		crit_R_0.append( pre_R_lim )

		# plt.figure()
		# plt.plot( new_R, new_eta, 'r-',)
		# plt.plot( x_new, y_new, 'b--', )
		# plt.xlim( 2e0, 2e2 )
		# plt.xscale('log')
		# plt.ylim( -0.05, 1.05 )
		# plt.axhline( y = 0,)
		# plt.savefig('/home/xkchen/%d_interp_test.png' % tt, dpi = 300)
		# plt.close()

		# raise

	##. save the crit_R
	L0 = len( crit_R[0] )
	L1 = len( crit_R_0 )

	dcp_R = np.zeros( np.max([ L0, L1]), )
	dcp_R[:L1] = crit_R_0
	dcp_R[L1:L0] = np.nan

	values = [ dcp_R ] + crit_R

	tab_file = Table( values, names = ['pre_R_lim'] + fig_name )
	tab_file.write( out_path + 'Extend_BCGM_gri-common_Rs-bin_over-rich_%s-band_smooth-exten_Rt_test.fits' % band_str, overwrite = True)

# raise


##... figs
line_s = [':', '--', '-']
color_s = ['b', 'g', 'm', 'r', 'k']

y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [7e-4, 1e0], [5e-3, 6e0] ]

for kk in range( 1 ):

	band_str = band[ kk ]

	##. ratio thresh for Rt estimation
	crit_eta = [0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95]

	for oo in range( len(crit_eta) ):

		id_set = oo

		##.
		nbg_R, nbg_SB, nbg_err = [], [], []

		for tt in range( len(R_bins) - 1 ):

			dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
										+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

			tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

			nbg_R.append( tt_r )
			nbg_SB.append( tt_sb )
			nbg_err.append( tt_sb_err )

		#.
		pat = fits.open( out_path + 'Extend_BCGM_gri-common_Rs-bin_over-rich_%s-band_smooth-exten_Rt_test.fits' % band_str,)
		p_table = pat[1].data

		crit_R = []
		crit_R0 = p_table['pre_R_lim']

		for tt in range( len(R_bins) - 1 ):

			Rt_arr = np.array( p_table[ fig_name[tt] ] )

			crit_R.append( Rt_arr[ id_set ] )

		##. 
		fig = plt.figure( )
		gax = fig.add_axes([0.11, 0.11, 0.80, 0.85])

		_kk_tmp_F = interp.interp1d( nbg_R[-1], nbg_SB[-1], kind = 'cubic', fill_value = 'extrapolate',)

		for mm in range( len(R_bins) -2 ):

			##.
			_dd_eta = nbg_SB[mm] / _kk_tmp_F( nbg_R[mm] )
			_dp_eta = signal.savgol_filter( _dd_eta, 5, 1 )


			##. extent the ratio profile
			id_vx = nbg_R[ mm ] <= crit_R0[mm]

			lim_R = nbg_R[mm][ id_vx ]
			lim_eta = _dp_eta[ id_vx ]

			x_new = np.logspace( -1, 2.5, 500 )

			tck_F = interp.interp1d( lim_R, lim_eta, kind = 'linear', fill_value = 'extrapolate',)
			y_new = tck_F( x_new )

			##.
			gax.plot( nbg_R[mm], _dp_eta, ls = '-', color = color_s[mm], alpha = 0.75, label = fig_name[mm],)

			gax.fill_between( nbg_R[mm], y1 = _dp_eta - nbg_err[mm] / _kk_tmp_F( nbg_R[mm] ), 
						y2 = _dp_eta + nbg_err[mm] / _kk_tmp_F( nbg_R[mm] ), color = color_s[mm], alpha = 0.12,)

			gax.axvline( x = crit_R[mm], ls = '--', color = color_s[mm], alpha = 0.75,)

			gax.plot( x_new, y_new, ls = ':', color = color_s[mm],)

		#.
		gax.annotate( s = '%s-band' % band_str, xy = (0.03, 0.83), xycoords = 'axes fraction', fontsize = 14, backgroundcolor = 'w',)
		gax.legend( loc = 3, frameon = True, fontsize = 13,)

		gax.set_xlim( 1e0, 2e2 )
		# gax.set_xlim( 2e0, 5e1 )

		gax.set_xscale('log')
		gax.set_xlabel('$R \; [kpc]$', fontsize = 14)

		gax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
							fontsize = 11, labelpad = 8)
		gax.set_ylim( -0.05, 1.05 )
		gax.axhline( y = 0, ls = ':', color = 'gray',)

		gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
		gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

		plt.savefig('/home/xkchen/Rs-bin_over-rich_%s-band_sat-BG-sub_SB-ratio_%.2feta.png' % (band_str, crit_eta[ id_set ] ), dpi = 300)
		plt.close()

