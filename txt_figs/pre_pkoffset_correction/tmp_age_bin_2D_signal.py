import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

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

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit

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
band = ['r', 'g', 'i']

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

path = '/home/xkchen/mywork/ICL/code/z_form_age_SB/SB_pros/'
out_path = '/home/xkchen/mywork/ICL/code/z_form_age_SB/BG_estimate/'

color_s = ['r', 'g', 'b']

fig_name = [ 'younger', 'older' ]
cat_lis = [ 'younger', 'older' ]

### === ### SB profile
# younger age bin
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []
for kk in range( 3 ):

	with h5py.File( out_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( mag_sb )
	nbg_low_err.append( mag_err )

low_r, low_sb, low_err = [], [], [] 
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_younger_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	low_r.append( tt_r )
	low_sb.append( mag_sb )
	low_err.append( mag_err )

# older age bin
nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( out_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( mag_sb )
	nbg_hi_err.append( mag_err )

hi_r, hi_sb, hi_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_older_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mag_sb = 22.5 - 2.5 * np.log10( tt_sb )
	mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	hi_r.append( tt_r )
	hi_sb.append( mag_sb )
	hi_err.append( mag_err )
"""
plt.figure()
ax = plt.subplot(111)

for kk in range( 3 ):

	if kk == 0:
		ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, label = fig_name[0] + ',' + band[kk],)
		ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

		ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = fig_name[1] + ',' + band[kk],)
		ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)
	else:
		ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, )
		ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

		ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = band[kk],)
		ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)
	'''
	if kk == 0:
		ax.plot(low_r[kk], low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, label = fig_name[0] + ',' + band[kk],)
		ax.fill_between(low_r[kk], y1 = low_sb[kk] - low_err[kk], y2 = low_sb[kk] + low_err[kk], color = color_s[kk], alpha = 0.12,)

		ax.plot(hi_r[kk], hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = fig_name[1] + ',' + band[kk],)
		ax.fill_between(hi_r[kk], y1 = hi_sb[kk] - hi_err[kk], y2 = hi_sb[kk] + hi_err[kk], color = color_s[kk], alpha = 0.12,)
	else:
		ax.plot(low_r[kk], low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, )
		ax.fill_between(low_r[kk], y1 = low_sb[kk] - low_err[kk], y2 = low_sb[kk] + low_err[kk], color = color_s[kk], alpha = 0.12,)

		ax.plot(hi_r[kk], hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = band[kk],)
		ax.fill_between(hi_r[kk], y1 = hi_sb[kk] - hi_err[kk], y2 = hi_sb[kk] + hi_err[kk], color = color_s[kk], alpha = 0.12,)
	'''
ax.set_ylim( 22, 34,)
# ax.set_ylim( 22, 30,)

ax.set_xlim( 1e1, 2e3)
ax.invert_yaxis()

ax.legend( loc = 1)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [mag / $arcsec^2$]')
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

plt.subplots_adjust( left = 0.15 )
#plt.savefig('/home/xkchen/figs/rgi-band_age-bin_sample_SB_compare.jpg', dpi = 300)
plt.savefig('/home/xkchen/figs/rgi-band_age-bin_sample_BG-sub-SB_compare.jpg', dpi = 300)
plt.show()
"""

### === ### 2D signal
from fig_out_module import ri_2D_signal, BG_sub_2D_signal

rand_path = '/home/xkchen/mywork/ICL/code/ref_BG_profile/'
img_path = '/home/xkchen/mywork/ICL/code/z_form_age_SB/SB_pros/'

'''
## r+i image case
for ll in range( 2 ):

	r_img_file = img_path + 'photo-z_match_%s_r-band_Mean_jack_img_z-ref.h5' % cat_lis[ll]
	r_rms_file = '/home/xkchen/mywork/ICL/code/z_form_age_SB/%s_r-band_stack_test_rms.h5' % cat_lis[ll]

	i_img_file = img_path + 'photo-z_match_%s_i-band_Mean_jack_img_z-ref.h5' % cat_lis[ll]
	i_rms_file = '/home/xkchen/mywork/ICL/code/z_form_age_SB/%s_i-band_stack_test_rms.h5' % cat_lis[ll]

	rand_r_img_file = rand_path + 'random_r-band_rand-stack_Mean_jack_img_z-ref-aveg.h5'
	rand_i_img_file = rand_path + 'random_i-band_rand-stack_Mean_jack_img_z-ref-aveg.h5'

	r_BG_file = out_path + 'photo-z_%s_r-band_BG-profile_params_diag-fit.csv' % cat_lis[ll]
	i_BG_file = out_path + 'photo-z_%s_i-band_BG-profile_params_diag-fit.csv' % cat_lis[ll]

	fig_title = '%s, r+i stacking image' % fig_name[ll]
	out_fig_file = '/home/xkchen/figs/%s_r+i_stacking-img.png' % cat_lis[ll]

	ri_2D_signal( r_img_file, r_rms_file, i_img_file, i_rms_file, rand_r_img_file, rand_i_img_file, 
		r_BG_file, i_BG_file, fig_title, out_fig_file, z_ref, pixel)
'''

for kk in range( 3 ):

	for ll in range( 2 ):
		img_file = img_path + 'photo-z_match_%s_%s-band_Mean_jack_img_z-ref.h5' % (cat_lis[ll], band[kk])
		random_img_file = rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[kk]
		BG_file = out_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ll], band[kk])
		band_str = band[kk]
		out_fig_name = '/home/xkchen/figs/%s_%s-band_2D_flux_compare.png' % (cat_lis[ll], band[kk])

		BG_sub_2D_signal( img_file, random_img_file, BG_file, z_ref, pixel, band_str, out_fig_name)

