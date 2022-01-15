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

#path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_pros/'
## 30 sub-samples case
path = '/home/xkchen/mywork/ICL/code/photo_z_match_SB/'

color_s = ['r', 'g', 'b']

cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass', 'tot-BCG-star-Mass' ]

fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$', 'low $M_{\\ast}$ + high $M_{\\ast}$']

low_r, low_sb, low_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_low_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	low_r.append( tt_r )
	low_sb.append( tt_sb )
	low_err.append( tt_err )

hi_r, hi_sb, hi_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_high_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	hi_r.append( tt_r )
	hi_sb.append( tt_sb )
	hi_err.append( tt_err )

tot_r, tot_sb, tot_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tot_r.append( tt_r )
	tot_sb.append( tt_sb )
	tot_err.append( tt_err )

### === ### BG estimate (based on diag-err only)
#out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/BG_estimate/'
## 30 sub-samples case
out_path = '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/'

rand_path = '/home/xkchen/mywork/ICL/code/ref_BG_profile/'

"""
for mass_dex in (0, 1, 2):

	for kk in range( 3 ):

		if mass_dex == 0:
			tt_r, tt_sb, tt_err = low_r[kk], low_sb[kk], low_err[kk]
		elif mass_dex == 1:
			tt_r, tt_sb, tt_err = hi_r[kk], hi_sb[kk], hi_err[kk]
		else:
			tt_r, tt_sb, tt_err = tot_r[kk], tot_sb[kk], tot_err[kk]

		params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[kk]

		p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
		bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

		R_psf = 10
		lo_R_lim = 500
		hi_R_lim = 1.4e3
		trunk_R = 2e3

		out_params_file = out_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mass_dex ], band[kk])
		out_pros_file = out_path + 'photo-z_%s_%s-band_BG-profile_diag-fit.csv' % (cat_lis[ mass_dex ], band[kk])

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
		ax.set_title( fig_name[ mass_dex ] + ', %s band' % band[kk] )

		ax.plot( tt_r, tt_sb, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
		ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.12)

		ax.plot(tt_r, comb_F, ls = '-', color = 'k', alpha = 0.5, label = 'fitting',)
		ax.plot(tt_r, norm_sign, ls = '-.', color = 'k', alpha = 0.5, label = 'signal (model)',)
		ax.plot(tt_r, norm_BG, ls = '--', color = 'k', alpha = 0.5, label = 'BackGround')

		ax.axvline(x = lo_R_lim, ls = ':', color = 'r', alpha = 0.5,)
		ax.axvline(x = hi_R_lim, ls = ':', color = 'g', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

		ax.annotate(text = '$\\chi^2 / \\nu = %.5f$' % chi_ov_nu, xy = (0.1, 0.2), xycoords = 'axes fraction', color = 'k',)
		ax.annotate(text = '$\\chi^2(r \\leq 1.4Mpc) / \\nu = %.5f$' % chi_inner_m, xy = (0.1, 0.1), xycoords = 'axes fraction', color = 'g',)

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
		plt.savefig('/home/xkchen/figs/%s_%s-band_SB_n=2.1-sersic.png' % (cat_lis[ mass_dex ], band[kk]), dpi = 300)
		plt.close()
"""

### === ### BG-sub SB 
# N_bin = 80

# for mass_dex in (0, 1, 2):

# 	for kk in range( 3 ):

# 		jk_sub_sb = path + 'photo-z_match_%s_%s-band_' % (cat_lis[ mass_dex ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref.h5'
# 		sb_out_put = out_path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[ mass_dex ], band[kk])
# 		BG_file = out_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mass_dex ], band[kk])
# 		BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)

### === ### figs and compare
# low mass bin
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []
for kk in range( 3 ):

	#with h5py.File( out_path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
	with h5py.File( out_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( tt_sb )
	nbg_low_err.append( tt_err )

# higher mass sample SB profiles
nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []
for kk in range( 3 ):

	#with h5py.File( out_path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
	with h5py.File( out_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( tt_sb )
	nbg_hi_err.append( tt_err )

# low + high mass bin
nbg_tot_r, nbg_tot_sb, nbg_tot_err = [], [], []
for kk in range( 3 ):

	#with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band_diag-fit-BG-sub_SB.h5' % band[kk], 'r') as f:
	with h5py.File( out_path + 'photo-z_total_star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_tot_r.append( tt_r )
	nbg_tot_sb.append( tt_sb )
	nbg_tot_err.append( tt_err )

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

plt.figure( figsize = (19.2, 4.8) )
gs = gridspec.GridSpec(1,3, width_ratios = [1,1,1],)

for kk in range( 3 ):

	# plt.figure()
	# ax = plt.subplot(111)
	# ax.set_title('low $M_{\\ast}$ , high $M_{\\ast}$ SB profile')

	ax = plt.subplot( gs[kk] )
	ax.set_title( '%s band ' % band[kk] )

	ax.plot(Z05_r[kk], Z05_sb[kk], ls = '-', color = 'k', alpha = 0.5, label = 'Z05',)
	ax.plot( last_Z05_r[kk], last_Z05_sb[kk], ls = ':', color = 'k', alpha = 0.5, label = 'Z05, final version',)

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.45, label = fig_name[0])# + ',' + band[kk],)
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.45, label = fig_name[1])# + ',' + band[kk],)
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.set_xlim(1e1, 2e3)
	ax.set_ylim(5e-5, 1e0)
	ax.set_yscale('log')

	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	if kk == 0:
		ax.legend( loc = 1)
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	if kk != 0:
		ax.set_yticklabels( labels = [],)

	# plt.subplots_adjust( left = 0.15,)
	# plt.savefig('/home/xkchen/figs/%s-band_M-star_sample_BG-sub-SB_compare.png' % band[kk], dpi = 300)
	# plt.close()

plt.subplots_adjust( left = 0.05, bottom = 0.10, right = 0.95, wspace = 0)
plt.savefig('/home/xkchen/figs/M-star_sample_BG-sub-SB_compare.png', dpi = 300)
plt.close()

raise

## low high mass bin comparison
plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

ax.set_title('low $M_{\\ast}$ , high $M_{\\ast}$ SB profile comparison')
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

    intep_F = interp.interp1d(nbg_hi_r[kk], nbg_hi_sb[kk], kind = 'cubic')
    id_Rx = ( nbg_low_r[kk] >= nbg_hi_r[kk].min() ) & ( nbg_low_r[kk] <= nbg_hi_r[kk].max() )
    intep_sb = intep_F( low_r[kk][id_Rx] )
    
    bx.plot( nbg_low_r[kk][id_Rx], intep_sb / nbg_low_sb[kk][id_Rx], ls = '-', color = color_s[kk], alpha = 0.5,)
    bx.plot( nbg_low_r[kk], nbg_low_sb[kk] / nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.5,)
    bx.fill_between( nbg_low_r[kk], y1 = (nbg_low_sb[kk] - nbg_low_err[kk]) / nbg_low_sb[kk], 
                    y2 = (nbg_low_sb[kk] + nbg_low_err[kk]) / nbg_low_sb[kk], color = color_s[kk], alpha = 0.12,)
    
ax.set_xlim(1e1, 2e3)
ax.set_ylim(5e-5, 2e0)
ax.set_yscale('log')

ax.legend( loc = 1)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

bx.set_ylim(0.5, 2.5)

bx.set_xlim( ax.get_xlim() )
bx.set_xscale('log')
bx.set_xlabel('R [kpc]')
bx.grid(which = 'both', axis = 'both', alpha = 0.25,)    
bx.set_ylabel('$\\frac { SB } { SB_{low \, M_{\\ast} } } }$',) #labelpad = 6)
bx.tick_params(axis = 'both', which = 'both', direction = 'out',)
bx.set_yticklabels( labels = [], minor = True)
ax.set_xticklabels( labels = [],)

plt.subplots_adjust( left = 0.15, hspace = 0.05,)
plt.savefig('/home/xkchen/figs/rgi-band_M-star_sample_BG-sub-SB_compare.png', dpi = 300)
plt.close()


### === ### color profile
def color_func( flux_arr_0, flux_err_0, flux_arr_1, flux_err_1):

	mag_arr_0 = 22.5 - 2.5 * np.log10( flux_arr_0 )
	mag_arr_1 = 22.5 - 2.5 * np.log10( flux_arr_1 )
	
	color_pros = mag_arr_0 - mag_arr_1

	sigma_0 = 2.5 * flux_err_0 / (np.log(10) * flux_arr_0 )
	sigma_1 = 2.5 * flux_err_1 / (np.log(10) * flux_arr_1 )

	color_err = np.sqrt( sigma_0**2 + sigma_1**2 )

	return color_pros, color_err

pdat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_g-r.csv')
r_05_0, g2r_05 = np.array(pdat_0['(R)^(1/4)']), np.array(pdat_0['g-r'])
r_05_0 = r_05_0**4

pdat_2 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_r-i.csv')
r_05_1, r2i_05 = np.array(pdat_2['(R)^(1/4)']), np.array(pdat_2['r-i'])
r_05_1 = r_05_1**4

pdat_1 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zhang_SB/Z19_color.csv')
r_19, g2r_19 = np.array(pdat_1['R_kpc']), np.array(pdat_1['g-r'])

hi_g2r, hi_g2r_err = color_func( nbg_hi_sb[1], nbg_hi_err[1], nbg_hi_sb[0], nbg_hi_err[0] )
hi_r2i, hi_r2i_err = color_func( nbg_hi_sb[0], nbg_hi_err[0], nbg_hi_sb[2], nbg_hi_err[2] )

low_g2r, low_g2r_err = color_func( nbg_low_sb[1], nbg_low_err[1], nbg_low_sb[0], nbg_low_err[0] )
low_r2i, low_r2i_err = color_func( nbg_low_sb[0], nbg_low_err[0], nbg_low_sb[2], nbg_low_err[2] )

tot_g2r, tot_g2r_err = color_func( nbg_tot_sb[1], nbg_tot_err[1], nbg_tot_sb[0], nbg_tot_err[0] )
tot_r2i, tot_r2i_err = color_func( nbg_tot_sb[0], nbg_tot_err[0], nbg_tot_sb[2], nbg_tot_err[2] )

## save the color profile
keys = ['R_kpc', 'g2r', 'g2r_err', 'r2i', 'r2i_err']
values = [ hi_r[0], hi_g2r, hi_g2r_err, hi_r2i, hi_r2i_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( out_path + 'high_BCG_star-Mass_color_profile.csv')

keys = ['R_kpc', 'g2r', 'g2r_err', 'r2i', 'r2i_err']
values = [ low_r[0], low_g2r, low_g2r_err, low_r2i, low_r2i_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( out_path + 'low_BCG_star-Mass_color_profile.csv')

keys = ['R_kpc', 'g2r', 'g2r_err', 'r2i', 'r2i_err']
values = [ tot_r[0], tot_g2r, tot_g2r_err, tot_r2i, tot_r2i_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( out_path + 'tot_BCG_star-Mass_color_profile.csv')


plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

ax.plot(hi_r[0], hi_g2r, ls = '--', color = 'r', alpha = 0.5, lw = 1, label = 'high $M_{\\ast}$')
ax.fill_between( hi_r[0], y1 = hi_g2r - hi_g2r_err, y2 = hi_g2r + hi_g2r_err, color = 'r', alpha = 0.12,)
ax.plot(low_r[0], low_g2r, ls = '--', color = 'g', alpha = 0.5, lw = 1, label = 'low $M_{\\ast}$')
ax.fill_between(low_r[0], y1 = low_g2r - low_g2r_err, y2 = low_g2r + low_g2r_err, color = 'g', alpha = 0.12,)

ax.plot(r_05_0, g2r_05, ls = '-.', color = 'k', lw = 1, alpha = 0.5, label = 'Z05')
ax.plot(r_19, g2r_19, ls = '-', color = 'k', lw = 1, alpha = 0.5, label = 'Z19')

bx.plot(hi_r[0], hi_r2i, ls = '--', color = 'r', lw = 1, alpha = 0.5,)
bx.fill_between(hi_r[0], y1 = hi_r2i - hi_r2i_err, y2 = hi_r2i + hi_r2i_err, color = 'r', alpha = 0.12,)
bx.plot(low_r[0], low_r2i, ls = '--', color = 'g', lw = 1, alpha = 0.5,)
bx.fill_between(low_r[0], y1 = low_r2i - low_r2i_err, y2 = low_r2i + low_r2i_err, color = 'g', alpha = 0.12,)
bx.plot(r_05_1, r2i_05, ls = '-.', color = 'k', lw = 1, alpha = 0.5, label = 'Z05')

ax.legend( loc = 3, frameon = False, fontsize = 8,)
ax.set_ylim(0.2, 1.8)
ax.set_xlim(1e0, 1e3)
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
plt.savefig('/home/xkchen/figs/color_compare.png', dpi = 300)
plt.close()


## average of satellite galaxies color (Zhiwei measured)
a_ref = 1 / (1 + z_ref)
cov_low_r = nbg_low_r[0] / a_ref * h
cov_hi_r = nbg_hi_r[0] / a_ref * h

#pdat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/zhiwei_data/absg-absr.csv')
pdat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/zhiwei_data/g-r.csv')
R = np.array( pdat['rbins'] ) * 1e3
hi_g2r_arr0 = np.array( pdat['high, z=0.225'] )
hi_g2r_arr1 = np.array( pdat['high, z=0.275'] )

low_g2r_arr0 = np.array( pdat['low, z=0.225'] )
low_g2r_arr1 = np.array( pdat['low, z=0.275'] )


plt.figure()
ax = plt.subplot(111)

ax.plot(cov_hi_r, hi_g2r, ls = '-', color = 'r', alpha = 0.5, lw = 1,)
ax.fill_between(cov_hi_r, y1 = hi_g2r - hi_g2r_err, y2 = hi_g2r + hi_g2r_err, color = 'r', alpha = 0.12, label = 'high $M_{\\ast}$',)

ax.plot(cov_low_r, low_g2r, ls = '-', color = 'b', alpha = 0.5, lw = 1,)
ax.fill_between(cov_low_r, y1 = low_g2r - low_g2r_err, y2 = low_g2r + low_g2r_err, color = 'b', alpha = 0.12, label = 'low $M_{\\ast}$',)

ax.plot(R, hi_g2r_arr0, ls = '--', color = 'r', alpha = 0.5, label = 'mean color of satellite galaxies')
ax.plot(R, hi_g2r_arr1, ls = ':', color = 'r', alpha = 0.5, )

ax.plot(R, low_g2r_arr0, ls = '--', color = 'b', alpha = 0.5, label = 'z=0.225')
ax.plot(R, low_g2r_arr1, ls = ':', color = 'b', alpha = 0.5, label = 'z=0.275')

ax.legend( loc = 3, frameon = False, fontsize = 8,)
ax.set_ylim(0.4, 1.8)
ax.set_xlim(1e0, 2e3)
ax.set_xscale('log')
ax.set_xlabel('R [kpc / h]')
ax.set_ylabel('g - r')
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

plt.subplots_adjust()
plt.savefig('/home/xkchen/figs/g-r_compare_to_satellite-mean_color.png', dpi = 300)
plt.close()

