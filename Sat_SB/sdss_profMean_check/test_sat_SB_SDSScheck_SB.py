"""
take 2000 satellite image to test
SB(r), compare with SDSS profMean
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import mechanize
from io import StringIO

import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

from img_sat_pros_stack import single_img_SB_func
from img_sat_pros_stack import aveg_SB_func
from img_sat_pros_stack import jack_aveg_SB_func
from img_sat_fig_out_mode import arr_jack_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25


### === average SB profile derived from SDSS profMean
s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat.csv')
cc_ra, cc_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

cc_coord = SkyCoord( ra = cc_ra * U.deg, dec = cc_dec * U.deg )

N_samples = 30

for kk in range( 3 ):

	band_str = band[ kk ]

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/' + 
						'SDSS_profMean_check_cat_%s-band_pos_z-ref.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	kk_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	idx, sep, d3d = kk_coord.match_to_catalog_sky( cc_coord )
	id_lim = sep.value < 2.7e-4

	mp_s_ra, mp_s_dec = cc_ra[ idx[ id_lim ] ], cc_dec[ idx[ id_lim ] ]
	mp_s_z = bcg_z[ id_lim ]

	r_bins = np.logspace( 0, 2.2, 25)


	pros_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean/ra%.4f_dec%.4f_SDSS_prof.txt'
	sub_out_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/stack_SBs/sat_SB_check_%s-band_' % band_str + 'jack-sub-%d_photo-SB_pros.csv'
	aveg_out_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/stack_SBs/SDSS_profMean_check_cat_%s-band_aveg-SDSS-prof.csv' % band_str

	jack_aveg_SB_func( N_samples, mp_s_ra, mp_s_dec, band_str, mp_s_z, pros_file, r_bins, sub_out_file, aveg_out_file, z_ref = z_ref )


### === SB(r) compare
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat.csv')
R_sat = np.array( dat['R_sat'] )
R_sat = R_sat * 1e3 / h


tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for kk in range( 3 ):

	with h5py.File( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/stack_SBs/' + 
					'sat_SB_check_%s-band_BG_Mean_jack_SB-pro_z-ref.h5' % band[kk], 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_bg_R.append( tt_r )
	tmp_bg_SB.append( tt_sb )
	tmp_bg_err.append( tt_err )


tmp_R, tmp_sb, tmp_err = [], [], []
tmp_nbg_sb, tmp_BG = [], []

for kk in range( 3 ):

	#. 1D profiles
	with h5py.File( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/stack_SBs/' + 
					'sat_SB_check_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	interp_BG_F = interp.interp1d( tmp_bg_R[kk], tmp_bg_SB[kk], kind = 'linear', fill_value = 'extrapolate')

	# _kk_BG = np.sum( interp_BG_F( R_sat ) ) / len( R_sat )
	_kk_BG = interp_BG_F( tt_r )

	tmp_R.append( tt_r )
	tmp_sb.append( tt_sb )
	tmp_err.append( tt_err )

	tmp_nbg_sb.append( tt_sb - _kk_BG )
	tmp_BG.append( _kk_BG )


tmp_sdss_R, tmp_sdss_sb, tmp_sdss_err = [], [], []

for kk in range( 3 ):

	cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/stack_SBs/' + 
						'SDSS_profMean_check_cat_%s-band_aveg-SDSS-prof.csv' % band[ kk ])

	tt_r, tt_sb, tt_err = np.array( cat['R'] ), np.array( cat['aveg_sb'] ), np.array( cat['aveg_sb_err'] )

	tmp_sdss_R.append( tt_r )
	tmp_sdss_sb.append( tt_sb )
	tmp_sdss_err.append( tt_err )


## figs
color_s = ['r', 'g', 'darkred']

plt.figure()
ax = plt.subplot(111)
for kk in range( 3 ):

	ax.plot( tmp_R[kk], tmp_sb[kk], ls = '-', color = color_s[kk], alpha = 0.65, label = '%s-band' % band[kk])
	ax.fill_between( tmp_R[kk], y1 = tmp_sb[kk], y2 = tmp_sb[kk], color = color_s[kk], alpha = 0.15,)

	ax.plot( tmp_bg_R[kk], tmp_bg_SB[kk], ls = '--', color = color_s[kk], alpha = 0.65,)

legend_2 = ax.legend( ['Satellite + Background', 'Background'], loc = 'upper center', frameon = False,)
ax.legend( loc = 1, frameon = False,)
ax.add_artist( legend_2 )

ax.set_xscale('log')
ax.set_xlabel('$R \; [kpc]$')

ax.set_ylim( 1e-3, 8e0)
ax.set_yscale('log')
ax.set_ylabel('$\\mu \; [nanomaggies \, / \, arcsec^{2} ]$')

plt.savefig('/home/xkchen/sat_SB_check.png', dpi = 300)
plt.close()


plt.figure()
ax = plt.subplot(111)
for kk in range( 3 ):

	ax.plot( tmp_R[kk], tmp_sb[kk], ls = '-', color = color_s[kk], alpha = 0.65, label = '%s-band' % band[kk])

	ax.plot( tmp_sdss_R[kk], tmp_sdss_sb[kk], ls = '--', color = color_s[kk], alpha = 0.65,)

	ax.plot( tmp_R[kk], tmp_nbg_sb[kk], ls = '-.', color = color_s[kk], alpha = 0.65,)

	ax.fill_between( tmp_R[kk], y1 = tmp_sb[kk] - tmp_err[kk], y2 = tmp_sb[kk] + tmp_err[kk], color = color_s[kk], alpha = 0.15,)
	ax.fill_between( tmp_R[kk], y1 = tmp_nbg_sb[kk] - tmp_err[kk], y2 = tmp_nbg_sb[kk] + tmp_err[kk], color = color_s[kk], alpha = 0.15,)

legend_2 = ax.legend( [ 'image stacking', 'SDSS profMean', 'Background subtracted', 'Background'], loc = 3, frameon = False,)
ax.legend( loc = 1, frameon = False,)
ax.add_artist( legend_2 )

ax.set_xlim(1, 200)
ax.set_xscale('log')
ax.set_xlabel('$R \; [kpc]$')

ax.set_ylim( 1e-4, 8e0)
ax.set_yscale('log')
ax.set_ylabel('$\\mu \; [nanomaggies \, / \, arcsec^{2} ]$')

plt.savefig('/home/xkchen/sat_BG-sub_SB_check.png', dpi = 300)
plt.close()

