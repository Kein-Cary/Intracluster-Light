import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy import modeling as Model
from astropy.table import Table, QTable
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord
from pynverse import inversefunc
from scipy import optimize

#.
from light_measure import cov_MX_func
from img_sat_fig_out_mode import arr_jack_func
from img_sat_BG_sub_SB import stack_BG_sub_func


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
##. sat_img without BCG
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'


##.
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#. for figs
color_s = ['b', 'g', 'r', 'm', 'k']

#.
fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \, / \, R_{200m}\\leq %.2f$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \, / \, R_{200m}\\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$R \, / \, R_{200m}{=}[%.2f, \; %.2f]$' % (R_bins[dd], R_bins[dd + 1]),)

##.
band_str = 'r'

N_sample = 100

##... Pre-calculation of BG-sub SBs
pre_nbg_R, pre_nbg_sb, pre_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

		##.
		sat_sb_file = ( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

		#.
		bg_sb_file = ( BG_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band_shufl-13_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5',)[0]

		tt_jk_R, tt_jk_SB, tt_jk_err = stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample )

		pre_nbg_R.append( tt_jk_R )
		pre_nbg_sb.append( tt_jk_SB )
		pre_nbg_err.append( tt_jk_err )


##... BG SBs
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( BG_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_shufl-13_BG_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_bg_R.append( tt_r )
	tmp_bg_SB.append( tt_sb )
	tmp_bg_err.append( tt_err )


##... Pre-calculation of ratio error
cc_tt_r, cc_tt_sb = pre_nbg_R[-1], pre_nbg_sb[-1]

id_nan = np.isnan( cc_tt_sb )
id_px = id_nan == False

_tt_tmp_F = interp.interp1d( cc_tt_r[ id_px ], cc_tt_sb[ id_px ], kind = 'cubic', fill_value = 'extrapolate',)

#.
pre_eta_R, pre_eta, pre_eta_err = [], [], []

for tt in range( len(R_bins) - 2 ):

	tmp_R, tmp_ratio = [], []

	for dd in range( N_sample ):

		#.
		with h5py.File( path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
			+ '_%s-band_jack-sub-%d_SB-pro_z-ref.h5' % (band_str, dd), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])
			tt_npix = np.array(f['npix'])

		#.
		id_Nul = tt_npix < 1
		tt_r[ id_Nul ] = np.nan
		tt_sb[ id_Nul ] = np.nan
		tt_err[ id_Nul ] = np.nan

		interp_mu_F = interp.interp1d( tmp_bg_R[tt], tmp_bg_SB[tt], kind = 'linear', fill_value = 'extrapolate')

		_kk_sb = tt_sb - interp_mu_F( tt_r )		
		_kk_eta = _kk_sb / _tt_tmp_F( tt_r ) 

		tmp_R.append( tt_r )
		tmp_ratio.append( _kk_eta )

	#.
	aveg_R_0, aveg_ratio, aveg_eta_err = arr_jack_func( tmp_ratio, tmp_R, N_sample )[:3]

	pre_eta_R.append( aveg_R_0 )
	pre_eta.append( aveg_ratio )
	pre_eta_err.append( aveg_eta_err )


##... BG-subtracted SBs
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	pt_r = np.array( dat['r'] )
	pt_sb = np.array( dat['sb'] )
	pt_err = np.array( dat['sb_err'] )

	#.
	nbg_R.append( pt_r )
	nbg_SB.append( pt_sb )
	nbg_err.append( pt_err )


### === 
fig = plt.figure()
ax = fig.add_axes([0.12, 0.32, 0.80, 0.63])
sub_ax = fig.add_axes([0.12, 0.11, 0.80, 0.21])

for tt in range( len(R_bins) - 1 ):

	l1, = ax.plot( nbg_R[tt], nbg_err[tt], ls = '-', color = color_s[tt], alpha = 0.65, label = fig_name[tt],)
	l2, = ax.plot( pre_nbg_R[tt], pre_nbg_err[tt], ls = '--', color = color_s[tt], alpha = 0.65,)

	_kk_F = interp.interp1d( nbg_R[tt], nbg_err[tt], kind = 'cubic', fill_value = 'extrapolate')
	_kk_err = _kk_F( pre_nbg_R[tt] )

	sub_ax.plot( pre_nbg_R[tt], pre_nbg_err[tt] / _kk_err, ls = '--', color = color_s[tt], alpha = 0.65,)

#.
legend_0 = ax.legend( handles = [l1, l2], labels = ['Accounting for $\\sigma_{BG}$', 'w/o $\\sigma_{BG}$'],
			loc = 1, frameon = False, fontsize = 12, markerfirst = False,)

ax.legend( loc = 3, frameon = False, fontsize = 12, markerfirst = False,)
ax.add_artist( legend_0 )

ax.set_xlim( 1e0, 7e1 )
ax.set_xscale('log')
ax.set_xlabel('$R \; [kpc]$', fontsize = 12,)

ax.set_ylim( 1e-4, 1e-1 )
ax.set_yscale('log')
ax.set_ylabel('$\\sigma_{ \\mu } \; [nanomaggies \, / \, arcsec^{2}]$', fontsize = 12,)

sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xscale('log')
sub_ax.set_xlabel('$R \; [kpc]$', fontsize = 12,)

sub_ax.annotate( s = '$\\sigma$(w/o $\\sigma_{BG}$) $\, / \,\\sigma$(Accounting for $\\sigma_{BG}$)', 
					xy = (0.03, 0.10), xycoords = 'axes fraction', fontsize = 12,)
sub_ax.set_ylim( 0.55, 1.05 )

sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

ax.set_xticklabels( [] )

plt.savefig('/home/xkchen/%s-band_SB_pros_err_compare.png' % band_str, dpi = 300)
plt.close()


#..
cp_k_r = nbg_R[ -1 ]
cp_k_sb = nbg_SB[ -1 ]
cp_k_err = nbg_err[ -1 ]

tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)
tmp_eF = interp.splrep( cp_k_r, cp_k_err, s = 0)

#.
for tt in range( len(R_bins) - 2 ):

	tt_r = nbg_R[ tt ]
	tt_sb = nbg_SB[ tt ]
	tt_err = nbg_err[ tt ]

	##. randomly points
	n_r = len( tt_r )
	n_N = 10000

	rnd_eta = np.zeros( n_r,)
	rnd_eta_err = np.zeros( n_r,)

	##.
	for dd in range( n_r ):

		cen = tt_sb[ dd ]
		scal = tt_err[ dd ]

		rand_arr_0 = np.random.normal( loc = cen, scale = scal, size = n_N )
		rand_arr_1 = np.random.normal( loc = interp.splev( tt_r[ dd ], tmp_F, der = 0), 
										scale = interp.splev( tt_r[ dd ], tmp_eF, der = 0), size = n_N )

		ratio_arr = rand_arr_0 / rand_arr_1

		rnd_eta[ dd ] = np.mean( ratio_arr )
		rnd_eta_err[ dd ] = np.std( ratio_arr )


	##.
	tmp_eta = tt_sb / interp.splev( tt_r, tmp_F, der = 0)

	_cc_err = interp.splev( tt_r, tmp_eF, der = 0)
	_cc_SB = interp.splev( tt_r, tmp_F, der = 0)

	p_err1 = ( tt_err / _cc_SB )**2
	p_err2 = ( _cc_err * tt_sb / _cc_SB**2 )**2

	tmp_eta_err = np.sqrt( p_err1 + p_err2 )


	##.
	dat = pds.read_csv( out_path + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB_ratio.csv'
					 % (R_bins[tt], R_bins[tt+1], band_str),)

	kk_r = np.array( dat['R'] )
	kk_eta = np.array( dat['ratio'] )
	kk_eta_err = np.array( dat['ratio_err'] )


	# ##... Sun's estimate
	# tt_arr = np.load('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
	# 				'sat_all_%.2f-%.2fR200m_%s-band_Sun2022_ratio.npy' 
	# 				% (R_bins[tt], R_bins[tt + 1], band_str),)

	# t_eta = tt_arr[0]
	# t_eta_err0 = tt_arr[1]
	# t_eta_err1 = tt_arr[2]


	#.
	fig = plt.figure( figsize = (10.8, 4.8) )
	ax0 = fig.add_axes([0.07, 0.31, 0.42, 0.63])
	sub_ax0 = fig.add_axes([0.07, 0.10, 0.42, 0.21])
	ax1 = fig.add_axes([0.57, 0.10, 0.42, 0.84])

	#.
	ax0.plot( kk_r, kk_eta, ls = '-', color = 'r', label = 'Accounting for $\\sigma_{BG}$')
	ax0.fill_between( kk_r, y1 = kk_eta - kk_eta_err, y2 = kk_eta + kk_eta_err, color = 'r', alpha = 0.15,)

	ax0.plot( tt_r, rnd_eta, ls = '--', color = 'b', label = 'Gaussian Sampled')
	ax0.fill_between( tt_r, y1 = rnd_eta - rnd_eta_err, y2 = rnd_eta + rnd_eta_err, color = 'b', ls = '--', alpha = 0.15,)

	# ax0.plot( pre_eta_R[tt], pre_eta[tt], ls = ':', color = 'k', label = 'Before correction')
	ax0.errorbar( pre_eta_R[tt], pre_eta[tt], yerr = pre_eta_err[tt], marker = '', ls = ':', color = 'k', 
					ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'w/o $\\sigma_{BG}$')

	sub_ax0.plot( tt_r, kk_eta / rnd_eta, ls = '--', color = 'b',)
	sub_ax0.plot( pre_eta_R[tt], pre_eta[tt] / rnd_eta, ls = ':', color = 'k',)

	ax1.plot( kk_r, kk_eta_err, ls = '-', color = 'r', label = 'Accounting for $\\sigma_{BG}$')
	ax1.plot( tt_r, tmp_eta_err, ls = ':', color = 'g', label = 'Error propagation')
	ax1.plot( tt_r, rnd_eta_err, ls = '--', color = 'b', label = 'Gaussian Sampled')
	ax1.plot( pre_eta_R[tt], pre_eta_err[tt], ls = ':', color = 'k', label = 'w/o $\\sigma_{BG}$')

	#. Sun+2022
	# ax0.plot( tt_r[tt_r < 100], t_eta, ls = '--', color = 'g', label = 'Sun+2022')
	# sub_ax0.plot( tt_r[tt_r < 100], t_eta / kk_eta[tt_r < 100], ls = '--', color = 'g',)
	# ax1.plot( tt_r[ tt_r < 100 ], t_eta_err0, ls = '--', color = 'g', alpha = 0.5, label = 'Sun+2022, lower')
	# ax1.plot( tt_r[ tt_r < 100 ], t_eta_err1, ls = ':', lw = 3, color = 'g', alpha = 0.5, label = 'Sun+2022, upper')

	#.
	ax0.legend( loc = 3, frameon = False, fontsize = 11,)
	ax0.set_xlim( 1e0, 7e1 )
	ax0.set_xscale('log')
	ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	ax0.set_ylim( 0.38, 0.88 )
	ax0.set_ylabel('$\\eta{=}\\mu / \\mu\,$(%s)' % fig_name[-1], fontsize = 12,)

	sub_ax0.set_xlim( ax0.get_xlim() )
	sub_ax0.set_xscale('log')
	sub_ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)
	sub_ax0.annotate( s = '$\\eta \; / \; \\eta\,$(Gaussian Sampled)', 
						xy = (0.03, 0.65), xycoords = 'axes fraction', fontsize = 12,)
	sub_ax0.set_ylim( 0.97, 1.03 )
	sub_ax0.axhline( y = 1, ls = ':', color = 'Gray', alpha = 0.12,)

	ax0.set_xticklabels( [] )

	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	sub_ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


	ax1.legend( loc = 2, frameon = False, fontsize = 12,)
	ax1.annotate(s = fig_name[tt], xy = (0.03, 0.55), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_xlim( ax0.get_xlim() )
	ax1.set_xscale('log')
	ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	ax1.set_ylim( 3e-3, 3e-1 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\sigma_{ \\eta }$', fontsize = 12,)

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	# ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	plt.savefig('/home/xkchen/sat_SB_ratio_err_%.2f-%.2fR200m_%s-band.png' 
				% (R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)
	plt.close()

