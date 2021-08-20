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
from color_2_mass import get_c2mass_func
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

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

### === 
def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === ### data load

rand_path = '/home/xkchen/tmp_run/data_files/jupyter/random_ref_SB/'

rand_r, rand_sb, rand_err = [], [], []

for ii in range( 3 ):
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_SB-pro_z-ref-aveg.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	rand_r.append( tt_r )
	rand_sb.append( tt_sb )
	rand_err.append( tt_err )

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

##...BCG location adjust test

# sample_lis = 'BCG_M'
sample_lis = 'rich'

if sample_lis == 'BCG_M':

	# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'
	# path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/SBs/'
	# cat_lis = [ 'low-age', 'hi-age' ]
	# fig_name = [ 'younger', 'older' ]
	# file_s = 'age-bin'

	BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'
	path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/SBs/'
	cat_lis = [ 'low-rich', 'hi-rich' ]
	fig_name = [ 'low $\\lambda$', 'high $\\lambda$' ]
	file_s = 'rich-bin'

if sample_lis == 'rich':

	# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/BGs/'
	# path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/SBs/'
	# cat_lis = [ 'younger', 'older' ]
	# fig_name = [ 'younger', 'older' ]
	# file_s = 'age-bin'

	BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'
	path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/SBs/'
	cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
	fig_name = ['low $M_{\\ast} ^{BCG}$', 'high $M_{\\ast} ^{BCG}$']
	file_s = 'BCG_M-bin'

### === ### BG estimate and BG-subtraction (based on diag-err only)
"""
for mm in range( 2 ):

	for kk in range( 3 ):

		with h5py.File( 
			path + 'photo-z_match_gri-common_%s_%s-band_Mean_jack_SB-pro_z-ref_pk-off.h5' % (cat_lis[mm], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

		p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
		bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

		R_psf = 10

		lo_R_lim = 500

		hi_R_lim = 1.4e3
		trunk_R = 2e3

		out_params_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
		out_pros_file = BG_path + 'photo-z_%s_%s-band_BG-profile_diag-fit.csv' % (cat_lis[ mm ], band[kk])

		clust_SB_fit_func(
			tt_r, tt_sb, tt_err, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, out_params_file, out_pros_file, trunk_R = trunk_R,)

		## fig
		p_dat = pds.read_csv( out_params_file )
		( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD, I_e, R_e) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0],
																np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0], 
																np.array(p_dat['offD'])[0], np.array(p_dat['I_e'])[0], np.array(p_dat['R_e'])[0] )

		fit_rnd_sb = cc_rand_sb_func( tt_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)  
		sign_fit = sersic_func( tt_r, I_e, R_e, 2.1)
		BG_pros = fit_rnd_sb - offD
		comb_F = BG_pros + sign_fit

		sb_2Mpc = sersic_func( trunk_R, I_e, R_e, 2.1)
		norm_sign = sign_fit - sb_2Mpc
		norm_BG = comb_F - norm_sign

		c_dat = pds.read_csv( out_pros_file )
		chi_ov_nu = np.array( c_dat['chi2nu'] )[0]
		chi_inner_m = np.array( c_dat['chi2nu_inner'] )[0]

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[ mm ] + ', %s band' % band[kk] )

		ax.plot( tt_r, tt_sb, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
		ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.12)

		ax.plot(tt_r, comb_F, ls = '-', color = 'k', alpha = 0.5, label = 'Best fitting',)
		ax.plot(tt_r, norm_sign, ls = '-.', color = 'k', alpha = 0.5, label = 'signal (model)',)
		ax.plot(tt_r, norm_BG, ls = '--', color = 'k', alpha = 0.5, label = 'BackGround')

		ax.axvline(x = lo_R_lim, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

		ax.annotate(text = '$\\chi^2 / \\nu = %.5f$' % chi_ov_nu, xy = (0.60, 0.60), xycoords = 'axes fraction', color = 'k',)

		ax.set_xlim(1e2, 4e3)
		ax.set_xscale('log')

		if kk == 1:
			ax.set_ylim( 2e-3, 5e-3)
		else:
			ax.set_ylim(2e-3, 7e-3)

		ax.set_xlabel('R [kpc]')
		ax.set_ylabel('SB [nanomaggies / arcsec^2]')
		ax.legend( loc = 1,)
		ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
		ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

		plt.subplots_adjust(left = 0.15, right = 0.9,)
		plt.savefig('/home/xkchen/%s_%s-band_SB_n=2.1-sersic.png' % (cat_lis[ mm ], band[kk]), dpi = 300)
		plt.close()

N_bin = 30
for mm in range( 2 ):

	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % (cat_lis[ mm ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref_pk-off.h5'

		sb_out_put = BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[ mm ], band[kk])
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
		BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)
"""


### === ### BG-sub SB and color
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


Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

for kk in range( 3 ):

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45,)
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
		y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = '%s band' % band[kk])
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
		y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.7, ymax = 1.0, linewidth = 1.5,)

if sample_lis == 'BCG_M':
	legend_2 = plt.legend( [ fig_name[0] + ' at fixed $M_{\\ast}^{BCG}$', fig_name[1] + ' at fixed $ M_{\\ast}^{BCG} $', 'PSF scale'], 
		loc = 3, frameon = False, fontsize = 15,)

if sample_lis == 'rich':
	legend_2 = plt.legend( [ fig_name[0] + ' at fixed $ \\lambda $', fig_name[1] + ' at fixed $ \\lambda $', 'PSF scale'], 
		loc = 3, frameon = False, fontsize = 15,)

legend_20 = ax.legend( loc = 1, frameon = False, fontsize = 15,)
ax.add_artist( legend_2 )

ax.set_xlim( 1e0, 1e3)
ax.set_ylim( 5e-5, 2e1)
ax.set_yscale('log')

ax.set_xscale('log')
ax.set_xlabel('R [kpc]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 15,)

plt.savefig('/home/xkchen/%s_sample_BG-sub-SB_compare.png' % file_s, dpi = 300)
plt.close()


## SB profile
fig = plt.figure( figsize = (19.90, 5.0) )
ax0 = fig.add_axes([0.05, 0.13, 0.28, 0.80])
ax1 = fig.add_axes([0.38, 0.13, 0.28, 0.80])
ax2 = fig.add_axes([0.71, 0.13, 0.28, 0.80])

ylims = [ [7e-5, 7e0], [5e-5, 2e0], [1e-4, 1.5e1] ]

for kk in ( 1, 0, 2 ):

	if kk == 0:
		ax = ax1
	if kk == 1:
		ax = ax0
	if kk == 2:
		ax = ax2

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, label = fig_name[0])
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
		y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = fig_name[1])
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
		y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.annotate(text = '%s band' % band[kk], xy = (0.10, 0.10), xycoords = 'axes fraction', color = 'k', fontsize = 15,)
	ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.0, ymax = 0.45,linewidth = 1.5, label = 'PSF scale')

	ax.set_xlim(1e0, 1e3)
	ax.set_ylim( ylims[kk][0], ylims[kk][1] )

	ax.set_yscale('log')

	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]', fontsize = 15,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 15, )
	ax.legend( loc = 1, fontsize = 15,)

plt.savefig('/home/xkchen/%s_sample_BG-sub-SB.png' % file_s, dpi = 300)
plt.close()


## g-r color
hi_g2r, hi_g2r_err = color_func( nbg_hi_sb[1], nbg_hi_err[1], nbg_hi_sb[0], nbg_hi_err[0] )
low_g2r, low_g2r_err = color_func( nbg_low_sb[1], nbg_low_err[1], nbg_low_sb[0], nbg_low_err[0] )

### smooth color profile
idnan = np.isnan( hi_g2r )
idx_lim = nbg_hi_r[0] < nbg_hi_r[0][idnan][0]
sm_hi_g2r = signal.savgol_filter( hi_g2r[idx_lim], 7, 3)

sm_hi_r = nbg_hi_r[0][idx_lim]
sm_hi_g2r_err = hi_g2r_err[idx_lim]


idnan = np.isnan( low_g2r )
idx_lim = nbg_low_r[0] < nbg_low_r[0][idnan][0]
sm_low_g2r = signal.savgol_filter( low_g2r[idx_lim], 7, 3)

sm_low_r = nbg_low_r[0][idx_lim]
sm_low_g2r_err = low_g2r_err[idx_lim]


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

if sample_lis == 'BCG_M':
	ax.plot( sm_hi_r, sm_hi_g2r, ls = '-', color = 'r', alpha = 0.5, linewidth = 1, label = fig_name[1] + ' at fixed $M_{\\ast}^{BCG}$')
	ax.fill_between( sm_hi_r, y1 = sm_hi_g2r - sm_hi_g2r_err, y2 = sm_hi_g2r + sm_hi_g2r_err, color = 'r', alpha = 0.12,)

	ax.plot( sm_low_r, sm_low_g2r, ls = '-', color = 'b', alpha = 0.5, linewidth = 1, label = fig_name[0] + ' at fixed $M_{\\ast}^{BCG}$')
	ax.fill_between( sm_low_r, y1 = sm_low_g2r - sm_low_g2r_err, y2 = sm_low_g2r + sm_low_g2r_err, color = 'b', alpha = 0.12,)

if sample_lis == 'rich':
	ax.plot( sm_hi_r, sm_hi_g2r, ls = '-', color = 'r', alpha = 0.5, linewidth = 1, label = fig_name[1] + ' at fixed $ \\lambda $')
	ax.fill_between( sm_hi_r, y1 = sm_hi_g2r - sm_hi_g2r_err, y2 = sm_hi_g2r + sm_hi_g2r_err, color = 'r', alpha = 0.12,)

	ax.plot( sm_low_r, sm_low_g2r, ls = '-', color = 'b', alpha = 0.5, linewidth = 1, label = fig_name[0] + ' at fixed $ \\lambda $')
	ax.fill_between( sm_low_r, y1 = sm_low_g2r - sm_low_g2r_err, y2 = sm_low_g2r + sm_low_g2r_err, color = 'b', alpha = 0.12,)

ax.axvline( x = phyR_psf[1], ls = ':', color = 'k', alpha = 0.5, ymin = 0.7, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')

ax.legend( loc = 3, frameon = False, fontsize = 15,)
ax.set_ylim( 0.6, 1.65 )

ax.set_xlim(1e0, 1e3)
ax.set_xscale('log')
ax.set_ylabel('g - r', fontsize = 15,)
ax.set_xlabel('R [kpc]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/%s_g2r_color_compare.png' % file_s, dpi = 300)
plt.close()


## g-i color
hi_g2i, hi_g2i_err = color_func( nbg_hi_sb[1], nbg_hi_err[1], nbg_hi_sb[2], nbg_hi_err[2] )
low_g2i, low_g2i_err = color_func( nbg_low_sb[1], nbg_low_err[1], nbg_low_sb[2], nbg_low_err[2] )

idnan = np.isnan( hi_g2i )
idx_lim = nbg_hi_r[0] < nbg_hi_r[0][idnan][0]
sm_hi_g2i = signal.savgol_filter( hi_g2i[idx_lim], 7, 3)

sm_hi_r = nbg_hi_r[0][idx_lim]
sm_hi_g2i_err = hi_g2i_err[idx_lim]


idnan = np.isnan( low_g2i )
idx_lim = nbg_low_r[0] < nbg_low_r[0][idnan][0]
sm_low_g2i = signal.savgol_filter( low_g2i[idx_lim], 7, 3)

sm_low_r = nbg_low_r[0][idx_lim]
sm_low_g2i_err = low_g2i_err[idx_lim]


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

if sample_lis == 'BCG_M':
	ax.plot( sm_hi_r, sm_hi_g2i, ls = '-', color = 'r', alpha = 0.5, linewidth = 1, label = fig_name[1] + ' at fixed $M_{\\ast}^{BCG}$')
	ax.fill_between( sm_hi_r, y1 = sm_hi_g2i - sm_hi_g2i_err, y2 = sm_hi_g2i + sm_hi_g2i_err, color = 'r', alpha = 0.12,)

	ax.plot( sm_low_r, sm_low_g2i, ls = '-', color = 'b', alpha = 0.5, linewidth = 1, label = fig_name[0] + ' at fixed $M_{\\ast}^{BCG}$')
	ax.fill_between( sm_low_r, y1 = sm_low_g2i - sm_low_g2i_err, y2 = sm_low_g2i + sm_low_g2i_err, color = 'b', alpha = 0.12,)

if sample_lis == 'rich':
	ax.plot( sm_hi_r, sm_hi_g2i, ls = '-', color = 'r', alpha = 0.5, linewidth = 1, label = fig_name[1] + ' at fixed $ \\lambda $')
	ax.fill_between( sm_hi_r, y1 = sm_hi_g2i - sm_hi_g2i_err, y2 = sm_hi_g2i + sm_hi_g2i_err, color = 'r', alpha = 0.12,)

	ax.plot( sm_low_r, sm_low_g2i, ls = '-', color = 'b', alpha = 0.5, linewidth = 1, label = fig_name[0] + ' at fixed $ \\lambda $')
	ax.fill_between( sm_low_r, y1 = sm_low_g2i - sm_low_g2i_err, y2 = sm_low_g2i + sm_low_g2i_err, color = 'b', alpha = 0.12,)

ax.axvline( x = phyR_psf[1], ls = ':', color = 'k', alpha = 0.5, ymin = 0.7, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')

ax.legend( loc = 3, frameon = False, fontsize = 15,)
ax.set_ylim( 1.2, 2.3 )

ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')
ax.set_ylabel('g - i', fontsize = 15,)
ax.set_xlabel('R [kpc]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/%s_g2i_color_compare.png' % file_s, dpi = 300)
plt.close()

