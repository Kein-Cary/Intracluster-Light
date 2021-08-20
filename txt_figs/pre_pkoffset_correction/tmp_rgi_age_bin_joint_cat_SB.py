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

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov

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
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

color_s = ['r', 'g', 'b']

path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin/'

### === ### Mean SB profile load
low_age_r, low_age_sb, low_age_err = [], [], []
for ii in range( 3 ):

	with h5py.File( path + 'photo-z_match_gri-common_younger_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	low_age_r.append( tt_r )
	low_age_sb.append( tt_sb )
	low_age_err.append( tt_err )

hi_age_r, hi_age_sb, hi_age_err = [], [], []
for ii in range( 3 ):

	with h5py.File( path + 'photo-z_match_gri-common_older_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])
	hi_age_r.append( tt_r )
	hi_age_sb.append( tt_sb )
	hi_age_err.append( tt_err )

## SB profile of random
rand_path = '/home/xkchen/mywork/ICL/code/ref_BG_profile/'

rand_r, rand_sb, rand_err = [], [], []
for ii in range( 3 ):
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_SB-pro_z-ref-aveg.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])
	rand_r.append( tt_r )
	rand_sb.append( tt_sb )
	rand_err.append( tt_err )

### === ### BG estimate (based on diag-err only)
BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin_BG/'

fig_name = [ 'younger', 'older' ]
cat_lis = [ 'younger', 'older' ]
'''
for formz_dex in (0, 1):

	for kk in range( 3 ):

		if formz_dex == 0:
			tt_r, tt_sb, tt_err = low_age_r[kk], low_age_sb[kk], low_age_err[kk]
		else:
			tt_r, tt_sb, tt_err = hi_age_r[kk], hi_age_sb[kk], hi_age_err[kk]

		params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

		p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
		bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

		R_psf = 10
		lo_R_lim = 500
		hi_R_lim = 1.4e3
		trunk_R = 2e3

		out_params_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ formz_dex ], band[kk])
		out_pros_file = BG_path + 'photo-z_%s_%s-band_BG-profile_diag-fit.csv' % (cat_lis[ formz_dex ], band[kk])

		clust_SB_fit_func( tt_r, tt_sb, tt_err, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, out_params_file, out_pros_file, trunk_R = trunk_R,)

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
		ax.set_title( fig_name[ formz_dex ] + ', %s band' % band[kk] )

		ax.plot( tt_r, tt_sb, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
		ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.12)

		ax.plot(tt_r, comb_F, ls = '-', color = 'k', alpha = 0.5, label = 'fitting',)
		ax.plot(tt_r, norm_sign, ls = '-.', color = 'k', alpha = 0.5, label = 'signal (model)',)
		ax.plot(tt_r, norm_BG, ls = '--', color = 'k', alpha = 0.5, label = 'BackGround')

		ax.axvline(x = lo_R_lim, ls = ':', color = 'r', alpha = 0.5,)
		ax.axvline(x = hi_R_lim, ls = ':', color = 'g', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

		ax.annotate(text = '$\\chi^2 / \\nu = %.5f$' % chi_ov_nu, xy = (0.55, 0.60), xycoords = 'axes fraction', color = 'k',)
		ax.annotate(text = '$\\chi^2(r \\leq 1.4Mpc) / \\nu = %.5f$' % chi_inner_m, xy = (0.55, 0.55), xycoords = 'axes fraction', color = 'g',)

		ax.set_xlim(1e2, 3e3)
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
		plt.savefig('/home/xkchen/figs/%s_%s-band_SB_n=2.1-sersic.png' % (cat_lis[ formz_dex ], band[kk]), dpi = 300)
		plt.close()
'''

'''
N_bin = 30
for formz_dex in (0, 1):
	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % (cat_lis[ formz_dex ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref.h5'
		sb_out_put = BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[ formz_dex ], band[kk])
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ formz_dex ], band[kk])
		BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)

print( 'done!' )
'''
### === ### covariance before and after BG-subtraction
N_bin = 30
'''
cov_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/cov_arr/'

R_low = 10

for mm in range( 2 ):

	for ll in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % ( cat_lis[mm], band[ll] ) + 'jack-sub-%d_SB-pro_z-ref.h5'
		out_file = cov_path + 'photo-z_%s_%s-band_cov-cor_arr.h5' % ( cat_lis[mm], band[ll] )

		BG_pro_cov( jk_sub_sb, N_bin, out_file, R_low)

		with h5py.File( out_file, 'r') as f:
			cov_MX = np.array( f['cov_Mx'])
			cor_MX = np.array( f['cor_Mx'])
			R_mean = np.array( f['R_kpc'])

		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.45, 0.80])
		ax1 = fig.add_axes([0.50, 0.10, 0.45, 0.80])

		ax0.set_title( fig_name[ mm ] + ', %s band, coV_arr / 1e-8' % band[ll] )
		tf = ax0.imshow(cov_MX / 1e-8, origin = 'lower', cmap = 'coolwarm', 
			norm = mpl.colors.SymLogNorm(linthresh = 1e0, linscale = 1e0, vmin = -3e0, vmax = 3e0, base = 5),)
		plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = '$ SB^2 $')

		ax0.set_xticklabels( labels = [] )

		ax0.set_ylim(0, len(R_mean) - 1 )
		yticks = ax0.get_yticks( )
		tik_lis = ['%.1f' % ll for ll in R_mean[ yticks[:-1].astype( np.int ) ] ]
		ax0.set_yticks( yticks[:-1] )
		ax0.set_yticklabels( labels = tik_lis, )
		ax0.set_ylim(-0.5, len(R_mean) - 0.5 )

		ax1.set_title( fig_name[ mm ] + ', %s band, coR_arr' % band[ll] )
		tf = ax1.imshow(cor_MX, origin = 'lower', cmap = 'seismic', vmin = -1, vmax = 1,)

		ax1.set_xticklabels( labels = [] )

		ax1.set_ylim(0, len(R_mean) - 1 )
		yticks = ax1.get_yticks( )
		tik_lis = ['%.1f' % ll for ll in R_mean[ yticks[:-1].astype( np.int ) ] ]
		ax1.set_yticks( yticks[:-1] )
		ax1.set_yticklabels( labels = tik_lis, )
		ax1.set_ylim(-0.5, len(R_mean) - 0.5 )

		plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01,)
		plt.savefig('/home/xkchen/figs/%s_%s-band_pre-BG-sub_%d-kpc-out_coV-coR_arr.png' % (cat_lis[mm], band[ll], R_low), dpi = 300)
		plt.close()

print( 'done!' )


for mm in range( 2 ):

	for kk in range( 3 ):

		sub_sb_file = path + 'photo-z_match_gri-common_%s_%s-band_' % ( cat_lis[mm], band[kk] ) + 'jack-sub-%d_SB-pro_z-ref.h5'
		BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[mm], band[kk])
		out_file = cov_path + '%s_%s-band_BG-sub_cov-cor_arr.h5' % (cat_lis[mm], band[kk])

		R_lim0 = 10
		R_lim1 = 1e3

		BG_sub_cov_func( sub_sb_file, N_bin, BG_file, out_file, R_lim0, R_lim1)

		with h5py.File( out_file, 'r') as f:
			cov_MX = np.array( f['cov_MX'])
			cor_MX = np.array( f['cor_MX'])

		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.45, 0.80])
		ax1 = fig.add_axes([0.50, 0.10, 0.45, 0.80])

		if kk == 2:
			ax0.set_title( fig_name[ mm ] + ', %s band' % band[kk] + ', coV_arr / 1e-7')
			tf = ax0.imshow(cov_MX / 1e-7, origin = 'lower', cmap = 'coolwarm',
					norm = mpl.colors.SymLogNorm(linthresh = 1e0, linscale = 1e0, vmin = -3e0, vmax = 3e0, base = 5),)
		else:
			ax0.set_title( fig_name[ mm ] + ', %s band' % band[kk] + ', coV_arr / 1e-8')
			tf = ax0.imshow(cov_MX / 1e-8, origin = 'lower', cmap = 'coolwarm', 
				norm = mpl.colors.SymLogNorm(linthresh = 1e0, linscale = 1e0, vmin = -3e0, vmax = 3e0, base = 5),)

		plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = '$ SB^2 $')

		ax1.set_title( fig_name[ mm ] + ', %s band' % band[kk] + ', coR_arr')
		tf = ax1.imshow(cor_MX, origin = 'lower', cmap = 'seismic', vmin = -1, vmax = 1,)
		plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01,)

		plt.savefig('/home/xkchen/figs/%s_%s-band_pos-BG-sub_coV-coR_arr.png' % (cat_lis[mm], band[kk]), dpi = 300)
		plt.close()

print( 'done!' )
'''

### === ### figs
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

print( [len(ll) for ll in nbg_low_r] )
print( [len(ll) for ll in nbg_hi_r] )

## compare to Z05
Z05_r, Z05_sb = [], []
for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	Z05_r.append( R_obs )
	Z05_sb.append( flux_obs )

last_Z05_r, last_Z05_sb = [], []
for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_sub_unmask.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	last_Z05_r.append( R_obs )
	last_Z05_sb.append( flux_obs )

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

for kk in range( 3 ):

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, label = '%s band' % band[kk],)
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, )
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.70, ymax = 1.0,)

legend_2 = plt.legend( [ fig_name[0] + ' at fixed richness', fig_name[1] + ' at fixed richness', 'PSF scale'], loc = 3, frameon = False, fontsize = 13.5,)
legend_20 = ax.legend( loc = 1, frameon = False, fontsize = 15,)
plt.gca().add_artist( legend_2 )

ax.set_xlim(1e0, 1e3)
ax.set_ylim(5e-5, 2e1)
ax.set_yscale('log')

ax.set_xscale('log')
ax.set_xlabel('R [kpc]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

ax.legend( loc = 1, frameon = False, fontsize = 15,)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 15,)

plt.savefig('/home/xkchen/figs/Age-bin_sample_BG-sub-SB_compare.png', dpi = 300)
plt.close()

raise

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

for kk in range( 3 ):

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, label = fig_name[0] + ',%s band' % band[kk],)
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = fig_name[1] + ',%s band' % band[kk],)
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)

	if kk == 0:
		ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.75, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')
	else:
		ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.75, ymax = 1.0, linewidth = 1.5,)

	intep_F = interp.interp1d(nbg_hi_r[kk], nbg_hi_sb[kk], kind = 'cubic')
	id_Rx = ( nbg_low_r[kk] >= nbg_hi_r[kk].min() ) & ( nbg_low_r[kk] <= nbg_hi_r[kk].max() )
	intep_sb = intep_F( nbg_low_r[kk][id_Rx] )

	bx.plot( nbg_low_r[kk][id_Rx], intep_sb / nbg_low_sb[kk][id_Rx], ls = '-', color = color_s[kk], alpha = 0.5,)
	bx.plot( nbg_low_r[kk], nbg_low_sb[kk] / nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.5,)
	bx.fill_between( nbg_low_r[kk], y1 = (nbg_low_sb[kk] - nbg_low_err[kk]) / nbg_low_sb[kk], 
		y2 = (nbg_low_sb[kk] + nbg_low_err[kk]) / nbg_low_sb[kk], color = color_s[kk], alpha = 0.12,)
	bx.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, linewidth = 1.5,)

ax.set_xlim(1e0, 1e3)
ax.set_ylim(5e-5, 2e1)
ax.set_yscale('log')

ax.legend( loc = 3, frameon = False, ncol = 2)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

bx.set_ylim( 0.5, 1.5)
#bx.set_yscale('log')
bx.set_xlim( ax.get_xlim() )
bx.set_xscale('log')
bx.set_xlabel('R [kpc]')
bx.grid(which = 'both', axis = 'both', alpha = 0.25,)
bx.set_ylabel('$\\frac { SB } { SB_{younger} } $', fontsize = 12)
bx.tick_params(axis = 'both', which = 'both', direction = 'out',)
bx.set_yticklabels( labels = [], minor = True)
ax.set_xticklabels( labels = [],)

plt.subplots_adjust( left = 0.15, hspace = 0.0,)
plt.savefig('/home/xkchen/figs/rgi-band_age_sample_BG-sub-SB_compare.png', dpi = 300)
plt.close()

raise

hi_g2r, hi_g2r_err = color_func( nbg_hi_sb[1], nbg_hi_err[1], nbg_hi_sb[0], nbg_hi_err[0] )
hi_r2i, hi_r2i_err = color_func( nbg_hi_sb[0], nbg_hi_err[0], nbg_hi_sb[2], nbg_hi_err[2] )

low_g2r, low_g2r_err = color_func( nbg_low_sb[1], nbg_low_err[1], nbg_low_sb[0], nbg_low_err[0] )
low_r2i, low_r2i_err = color_func( nbg_low_sb[0], nbg_low_err[0], nbg_low_sb[2], nbg_low_err[2] )


keys = ['R_kpc', 'g2r', 'g2r_err', 'r2i', 'r2i_err']
values = [ nbg_hi_r[0], hi_g2r, hi_g2r_err, hi_r2i, hi_r2i_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( BG_path + 'high_BCG_star-Mass_color_profile.csv')

keys = ['R_kpc', 'g2r', 'g2r_err', 'r2i', 'r2i_err']
values = [ nbg_low_r[0], low_g2r, low_g2r_err, low_r2i, low_r2i_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( BG_path + 'low_BCG_star-Mass_color_profile.csv')


plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

ax.plot(nbg_hi_r[0], hi_g2r, ls = '--', color = 'r', alpha = 0.5, lw = 1, label = fig_name[1])
ax.fill_between( nbg_hi_r[0], y1 = hi_g2r - hi_g2r_err, y2 = hi_g2r + hi_g2r_err, color = 'r', alpha = 0.12,)
ax.plot(nbg_low_r[0], low_g2r, ls = '--', color = 'g', alpha = 0.5, lw = 1, label = fig_name[0])
ax.fill_between(nbg_low_r[0], y1 = low_g2r - low_g2r_err, y2 = low_g2r + low_g2r_err, color = 'g', alpha = 0.12,)

bx.plot(nbg_hi_r[0], hi_r2i, ls = '--', color = 'r', lw = 1, alpha = 0.5,)
bx.fill_between(nbg_hi_r[0], y1 = hi_r2i - hi_r2i_err, y2 = hi_r2i + hi_r2i_err, color = 'r', alpha = 0.12,)
bx.plot(nbg_low_r[0], low_r2i, ls = '--', color = 'g', lw = 1, alpha = 0.5,)
bx.fill_between(nbg_low_r[0], y1 = low_r2i - low_r2i_err, y2 = low_r2i + low_r2i_err, color = 'g', alpha = 0.12,)

ax.legend( loc = 3, frameon = False, fontsize = 8,)
ax.set_ylim(0.2, 1.8)
ax.set_xlim(1e0, 2e3)
ax.set_xscale('log')
ax.set_ylabel('g - r')
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

bx.set_ylim(0.4, 1.0)
bx.set_xlim( ax.get_xlim() )
bx.set_xscale('log')
bx.set_ylabel('r - i')
bx.set_xlabel('R [kpc]')
bx.grid(which = 'both', axis = 'both', alpha = 0.25,)
ax.set_xticklabels( labels = [], )

plt.subplots_adjust( hspace = 0.)
plt.savefig('/home/xkchen/figs/age_color_compare.png', dpi = 300)
plt.close()


### === ### compare to mass-bin
mass_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'

mass_low_r, mass_low_sb, mass_low_err = [], [], []
for kk in range( 3 ):
	with h5py.File( mass_path + 'photo-z_low_BCG_star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mass_low_r.append( tt_r )
	mass_low_sb.append( tt_sb )
	mass_low_err.append( tt_err )

# higher mass sample SB profiles
mass_hi_r, mass_hi_sb, mass_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( mass_path + 'photo-z_high_BCG_star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	mass_hi_r.append( tt_r )
	mass_hi_sb.append( tt_sb )
	mass_hi_err.append( tt_err )

plt.figure( figsize = (19.2, 4.8) )
gs = gridspec.GridSpec(1,3, width_ratios = [1,1,1],)

for kk in range( 3 ):

	ax = plt.subplot( gs[kk] )
	ax.set_title('%s band' % band[kk])

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = 'g', alpha = 0.45, label = 'younger')
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = 'g', alpha = 0.12,)
	ax.plot(mass_low_r[kk], mass_low_sb[kk], ls = '--', color = 'r', alpha = 0.45, label = 'low $M_{\\ast}$',)
	#ax.fill_between(mass_low_r[kk], y1 = mass_low_sb[kk] - mass_low_err[kk], y2 = mass_low_sb[kk] + mass_low_err[kk], color = 'r', alpha = 0.12,) 

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = 'g', alpha = 0.45, label = 'older')
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = 'g', alpha = 0.12,)
	ax.plot(mass_hi_r[kk], mass_hi_sb[kk], ls = '-', color = 'r', alpha = 0.45, label = 'high $M_{\\ast}$',)
	#ax.fill_between(mass_hi_r[kk], y1 = mass_hi_sb[kk] - mass_hi_err[kk], y2 = mass_hi_sb[kk] + mass_hi_err[kk], color = 'r', alpha = 0.12,)

	ax.set_xlim(1e1, 2e3)
	ax.set_ylim(5e-5, 1e0)
	ax.set_yscale('log')

	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	if kk == 0:
		ax.legend( loc = 1)
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	if kk != 0:
		ax.set_yticklabels( labels = [],)

plt.subplots_adjust( left = 0.05, bottom = 0.10, right = 0.95, wspace = 0)
plt.savefig('/home/xkchen/figs/Mass-and-age_BG-sub-SB_compare.png', dpi = 300)
plt.close()

raise

### === ### compare to previous result
pre_path = '/home/xkchen/mywork/ICL/code/z_form_age_SB/BG_estimate/'

# younger bin
pre_low_r, pre_low_sb, pre_low_err = [], [], []
for kk in range( 3 ):
	with h5py.File( pre_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	pre_low_r.append( tt_r )
	pre_low_sb.append( tt_sb )
	pre_low_err.append( tt_err )

# older sample
pre_hi_r, pre_hi_sb, pre_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( pre_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	pre_hi_r.append( tt_r )
	pre_hi_sb.append( tt_sb )
	pre_hi_err.append( tt_err )

# color
pre_hi_c = pds.read_csv( pre_path + 'older-sample_color_profile.csv' )
pre_hi_g2r, pre_hi_g2r_err = np.array(pre_hi_c['g2r']), np.array(pre_hi_c['g2r_err'])
pre_hi_r2i, pre_hi_r2i_err = np.array(pre_hi_c['r2i']), np.array(pre_hi_c['r2i_err'])

pre_low_c = pds.read_csv( pre_path + 'younger-sample_color_profile.csv' )
pre_low_g2r, pre_low_g2r_err = np.array(pre_low_c['g2r']), np.array(pre_low_c['g2r_err'])
pre_low_r2i, pre_low_r2i_err = np.array(pre_low_c['r2i']), np.array(pre_low_c['r2i_err'])


for ll in range( 2 ):

	for kk in range( 3 ):

		plt.figure()
		ax = plt.subplot( 111 )
		if ll == 0:
			ax.set_title( fig_name[0] )
			ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = band[kk] + ', gri matched',)
			ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)
			ax.plot(pre_low_r[kk], pre_low_sb[kk], ls = '--', color = 'k', alpha = 0.45, label = 'each band match independently',)

		if ll ==1:
			ax.set_title( fig_name[1] )
			ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = band[kk] + ', gri matched',)
			ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)
			ax.plot(pre_hi_r[kk], pre_hi_sb[kk], ls = '--', color = 'k', alpha = 0.45, label = 'match each band independently',)

		ax.set_xlim(1e1, 2e3)
		ax.set_ylim(5e-5, 1e0)
		ax.set_yscale('log')

		ax.legend( loc = 1)
		ax.set_xscale('log')
		ax.set_xlabel('R [kpc]')
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

		plt.subplots_adjust( left = 0.15 )
		plt.savefig('/home/xkchen/%s_rgi-band_BG-sub-SB_%s-band_SB-check.png' % (cat_lis[ll], band[kk]), dpi = 300)
		plt.close()


plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

ax.plot(nbg_hi_r[0], hi_g2r, ls = '-', color = 'r', alpha = 0.45, lw = 1, label = fig_name[1] + ',gri matched')
ax.fill_between( nbg_hi_r[0], y1 = hi_g2r - hi_g2r_err, y2 = hi_g2r + hi_g2r_err, color = 'r', alpha = 0.12,)
ax.plot(nbg_low_r[0], low_g2r, ls = '-', color = 'g', alpha = 0.45, lw = 1, label = fig_name[0])
ax.fill_between(nbg_low_r[0], y1 = low_g2r - low_g2r_err, y2 = low_g2r + low_g2r_err, color = 'g', alpha = 0.12,)

ax.plot( pre_low_r[0], pre_low_g2r, ls = '--', color = 'g', alpha = 0.5, lw = 1, label = 'match each band independently')
ax.plot( pre_hi_r[0], pre_hi_g2r, ls = '--', color = 'r', alpha = 0.5, lw = 1,)

bx.plot(nbg_hi_r[0], hi_r2i, ls = '-', color = 'r', lw = 1, alpha = 0.5,)
bx.fill_between(nbg_hi_r[0], y1 = hi_r2i - hi_r2i_err, y2 = hi_r2i + hi_r2i_err, color = 'r', alpha = 0.12,)
bx.plot(nbg_low_r[0], low_r2i, ls = '-', color = 'g', lw = 1, alpha = 0.5,)
bx.fill_between(nbg_low_r[0], y1 = low_r2i - low_r2i_err, y2 = low_r2i + low_r2i_err, color = 'g', alpha = 0.12,)

bx.plot( pre_low_r[0], pre_low_r2i, ls = '--', color = 'g', alpha = 0.5, lw = 1,)
bx.plot( pre_hi_r[0], pre_hi_r2i, ls = '--', color = 'r', alpha = 0.5, lw = 1,)


ax.legend( loc = 3, frameon = False, fontsize = 8,)
ax.set_ylim(0.2, 1.8)
ax.set_xlim(1e0, 2e3)
ax.set_xscale('log')
ax.set_ylabel('g - r')
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

bx.set_ylim(0.4, 1.0)
bx.set_xlim( ax.get_xlim() )
bx.set_xscale('log')
bx.set_ylabel('r - i')
bx.set_xlabel('R [kpc]')
bx.grid(which = 'both', axis = 'both', alpha = 0.25,)
ax.set_xticklabels( labels = [], )

plt.subplots_adjust( hspace = 0.)
plt.savefig('/home/xkchen/figs/Age_color_check.png', dpi = 300)
plt.close()

