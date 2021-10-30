import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.signal as signal

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov

from scipy.interpolate import splev, splrep
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

from color_2_mass import jk_sub_SB_func
from img_BG_sub_SB_measure import BG_sub_sb_func, sub_color_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

band = ['r', 'g', 'i']
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === ###
def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

## === ## fixed rich bin
#... combine masking
# cat_lis = [ 'younger', 'older' ]
# fig_name = [ 'younger', 'older' ]
# file_s = 'age-bin'
# pre_BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/BGs/'
# pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/SBs/'


cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast} ^{BCG}$', 'high $M_{\\ast} ^{BCG}$']
file_s = 'BCG_M-bin'
pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/SBs/'
pre_BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'


BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
path = '/home/xkchen/figs/re_measure_SBs/SBs/'

### === mass profile
N_bin = 30
'''
#. color profiles
for mm in range( 2 ):

	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % (cat_lis[ mm ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref_pk-off.h5'
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
		out_sub_sb = BG_path + '%s_%s-band_' % (cat_lis[mm], band[kk]) + 'jack-sub-%d_BG-sub_SB.csv'

		jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, out_sub_sb )

for mm in range( 2 ):

	sub_r_file = BG_path + '%s_r-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv'
	sub_g_file = BG_path + '%s_g-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv'
	sub_i_file = BG_path + '%s_i-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv'

	sub_c_file = BG_path + '%s_jack-sub-' % cat_lis[mm] + '%d_color_profile.csv'
	aveg_c_file = BG_path + '%s_color_profile.csv' % cat_lis[mm]

	sub_color_func( N_bin, sub_r_file, sub_g_file, sub_i_file, sub_c_file, aveg_c_file )
'''

raise

###############################
### === after correction
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( tt_sb )
	nbg_low_err.append( tt_err )

nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( tt_sb )
	nbg_hi_err.append( tt_err )


#..figs
fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

for kk in range( 3 ):

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75,)
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
		y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s band' % band[kk])
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
		y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.15,)

legend_2 = plt.legend( [ fig_name[0], fig_name[1]], 
	loc = 3, frameon = False, fontsize = 15,)

ax.legend( loc = 1, frameon = False, fontsize = 15,)
ax.add_artist( legend_2 )

ax.set_xlim( 1e0, 1e3)
ax.set_ylim( 5e-5, 2e1)
ax.set_yscale('log')

ax.set_xscale('log')
ax.set_xlabel('R [kpc]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 15,)

plt.savefig('/home/xkchen/%s_BG-sub-SB_compare.png' % file_s, dpi = 300)
plt.close()


mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
hi_gr = signal.savgol_filter( hi_gr, 7, 3)

mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
lo_gr = signal.savgol_filter( lo_gr, 7, 3)


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( hi_c_r, hi_gr, ls = '-', color = 'r', alpha = 0.75, linewidth = 1, label = fig_name[1],)
ax.fill_between( hi_c_r, y1 = hi_gr - hi_gr_err, y2 = hi_gr + hi_gr_err, color = 'r', alpha = 0.15,)

ax.plot( lo_c_r, lo_gr, ls = '-', color = 'b', alpha = 0.75, linewidth = 1, label = fig_name[0],)
ax.fill_between( lo_c_r, y1 = lo_gr - lo_gr_err, y2 = lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 0.5, 1.7 )

ax.set_xlim(1e0, 1e3)
ax.set_xscale('log')
ax.set_ylabel('g - r', fontsize = 12,)
ax.set_xlabel('R [kpc]', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/%s_g2r_compare.png' % file_s, dpi = 300)
plt.close()
