"""
this file use for figure adjust
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import ndimage

from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

#.
import sys 
sys.path.append('/home/xkchen/mywork/ICL/code')

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
from img_cat_param_match import match_func
from img_pre_selection import extra_match_func
from color_2_mass import jk_sub_SB_func


#. cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

psf_FWHM = 1.32 # arcsec
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec


### === func. s
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir


### === data load
"""
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/noBG_SBs/'

##. background
band_str = 'r'

#.
with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_SB-pro_z-ref-aveg.h5' % band_str, 'r') as f:
	rand_r = np.array(f['r'])
	rand_sb = np.array(f['sb'])
	rand_err = np.array(f['sb_err'])


##. Background estimate

# with h5py.File( path + 'photo-z_match_clust_%s-band_full-BCG_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
# with h5py.File( path + 'photo-z_match_clust_%s-band_BCG-part-cut_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
# with h5py.File( path + 'photo-z_match_clust_%s-band_BCG-half-cut_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
with h5py.File( path + 'photo-z_match_clust_%s-band_BCG-half-V-cut_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

	tt_r = np.array(f['r'])
	tt_sb = np.array(f['sb'])
	tt_err = np.array(f['sb_err'])

params_file = rand_path + '%s-band_random_SB_fit_params.csv' % band_str

p0 = [ 2e-4, 4.8e-4, 6.8e2 ]
bounds = [ [0, 1e-3], [0, 1e2], [2e2, 3e3] ]

R_psf = 10

lo_R_lim = 500 # 450, 500, 400

hi_R_lim = 1.4e3
trunk_R = 2e3

##.
# out_params_file = BG_path + 'photo-z_%s-band_BG-profile_params_diag-fit.csv' % band_str
# out_pros_file = BG_path + 'photo-z_%s-band_BG-profile_diag-fit.csv' % band_str

# out_params_file = BG_path + 'photo-z_%s-band_part-cut_BG-profile_params_diag-fit.csv' % band_str
# out_pros_file = BG_path + 'photo-z_%s-band_part-cut_BG-profile_diag-fit.csv' % band_str

# out_params_file = BG_path + 'photo-z_%s-band_half-cut_BG-profile_params_diag-fit.csv' % band_str
# out_pros_file = BG_path + 'photo-z_%s-band_half-cut_BG-profile_diag-fit.csv' % band_str

out_params_file = BG_path + 'photo-z_%s-band_half-V-cut_BG-profile_params_diag-fit.csv' % band_str
out_pros_file = BG_path + 'photo-z_%s-band_half-V-cut_BG-profile_diag-fit.csv' % band_str

clust_SB_fit_func( tt_r, tt_sb, tt_err, params_file, R_psf, lo_R_lim, hi_R_lim, p0, bounds, out_params_file, 
					out_pros_file, trunk_R = trunk_R,)

## fig
p_dat = pds.read_csv( out_params_file )
( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD, I_e, R_e) = ( 
														np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], 
														np.array(p_dat['e_x0'])[0], np.array(p_dat['e_A'])[0], 
														np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0], 
														np.array(p_dat['offD'])[0], np.array(p_dat['I_e'])[0], 
														np.array(p_dat['R_e'])[0] )

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

ax.plot( tt_r, tt_sb, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.12)

ax.plot(tt_r, comb_F, ls = '-', color = 'k', alpha = 0.5, label = 'Best fitting',)
ax.plot(tt_r, norm_sign, ls = '-.', color = 'k', alpha = 0.5, label = 'signal (model)',)
ax.plot(tt_r, norm_BG, ls = '--', color = 'k', alpha = 0.5, label = 'BackGround')

ax.axvline(x = lo_R_lim, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

ax.annotate( s = '$\\chi^2 / \\nu = %.5f$' % chi_ov_nu, xy = (0.60, 0.60), xycoords = 'axes fraction', color = 'k',)

ax.set_xlim(1e2, 4e3)
ax.set_xscale('log')

ax.set_ylim(2e-3, 7e-3)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [nanomaggies / arcsec^2]')
ax.legend( loc = 1,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

plt.subplots_adjust(left = 0.15, right = 0.9,)
# plt.savefig('/home/xkchen/norm_stack_%s-band_SB_n=2.1-sersic.png' % band_str, dpi = 300)
# plt.savefig('/home/xkchen/part-cut_stack_%s-band_SB_n=2.1-sersic.png' % band_str, dpi = 300)
# plt.savefig('/home/xkchen/half-cut_stack_%s-band_SB_n=2.1-sersic.png' % band_str, dpi = 300)
plt.savefig('/home/xkchen/half-V-cut_stack_%s-band_SB_n=2.1-sersic.png' % band_str, dpi = 300)
plt.close()

raise
"""


### === BG-sub SB profiles
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/noBG_SBs/'

band_str = 'r'

N_bin = 30


# ##. normal stacking
# jk_sub_sb = path + 'photo-z_match_clust_%s-band_full-BCG_' % band_str + 'jack-sub-%d_SB-pro_z-ref.h5'
# BG_file = BG_path + 'photo-z_%s-band_BG-profile_params_diag-fit.csv' % band_str
# sb_out_put = out_path + 'photo-z_match_clust_%s-band_full-BCG_BG-sub_SB.h5' % band_str
# jk_sub_sb_out = out_path + 'photo-z_match_clust_%s-band_full-BCG_' % band_str + 'jack-sub-%d_BG-sub-SB.h5'

# BG_sub_sb_func( N_bin, jk_sub_sb, sb_out_put, band_str, BG_file )
# jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, jk_sub_sb_out )


# ##. part-cut BCG stacking
# jk_sub_sb = path + 'photo-z_match_clust_%s-band_BCG-part-cut_' % band_str + 'jack-sub-%d_SB-pro_z-ref.h5'
# BG_file = BG_path + 'photo-z_%s-band_part-cut_BG-profile_params_diag-fit.csv' % band_str
# sb_out_put = out_path + 'photo-z_match_clust_%s-band_BCG-part-cut_BG-sub_SB.h5' % band_str
# jk_sub_sb_out = out_path + 'photo-z_match_clust_%s-band_BCG-part-cut_' % band_str + 'jack-sub-%d_BG-sub-SB.h5'

# BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band_str, BG_file )
# jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, jk_sub_sb_out )


# ##. half-cut BCG stacking
# jk_sub_sb = path + 'photo-z_match_clust_%s-band_BCG-half-cut_' % band_str + 'jack-sub-%d_SB-pro_z-ref.h5'
# BG_file = BG_path + 'photo-z_%s-band_half-cut_BG-profile_params_diag-fit.csv' % band_str
# sb_out_put = out_path + 'photo-z_match_clust_%s-band_BCG-half-cut_BG-sub_SB.h5' % band_str
# jk_sub_sb_out = out_path + 'photo-z_match_clust_%s-band_BCG-half-cut_' % band_str + 'jack-sub-%d_BG-sub-SB.h5'

# BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band_str, BG_file )
# jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, jk_sub_sb_out )


##. half-cut BCG stacking
jk_sub_sb = path + 'photo-z_match_clust_%s-band_BCG-half-V-cut_' % band_str + 'jack-sub-%d_SB-pro_z-ref.h5'
BG_file = BG_path + 'photo-z_%s-band_half-V-cut_BG-profile_params_diag-fit.csv' % band_str
sb_out_put = out_path + 'photo-z_match_clust_%s-band_BCG-half-V-cut_BG-sub_SB.h5' % band_str
jk_sub_sb_out = out_path + 'photo-z_match_clust_%s-band_BCG-half-V-cut_' % band_str + 'jack-sub-%d_BG-sub-SB.h5'

BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band_str, BG_file )
jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, jk_sub_sb_out )


##. closed galaxy catalog
dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/closed_sat_cat/' + 
		'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_r-band_closed-mem_position.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
clus_ID = np.array( dat['clus_ID'] )

#.
min_Rpix_R_sat = np.array( dat['min_Rpix_Rsat'] )
min_Rpix_sR_sat = np.array( dat['min_Rpix_sR_sat'] )

#.
a_zx = 1 / ( 1 + bcg_z )
R_sat = min_Rpix_R_sat * a_zx * 1e3 / h 


##. normal stacking
with h5py.File( out_path + 'photo-z_match_clust_%s-band_full-BCG_BG-sub_SB.h5' % band_str, 'r') as f:
	ful_nbg_R = np.array(f['r'])
	ful_nbg_SB = np.array(f['sb'])
	ful_nbg_err = np.array(f['sb_err'])

#.
with h5py.File( path + 'photo-z_match_clust_%s-band_full-BCG_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
	ful_R = np.array(f['r'])
	ful_SB = np.array(f['sb'])
	ful_err = np.array(f['sb_err'])


##. part-cut BCG stacking
with h5py.File( out_path + 'photo-z_match_clust_%s-band_BCG-part-cut_BG-sub_SB.h5' % band_str, 'r') as f:
	part_nbg_R = np.array(f['r'])
	part_nbg_SB = np.array(f['sb'])
	part_nbg_err = np.array(f['sb_err'])

#.
with h5py.File( path + 'photo-z_match_clust_%s-band_BCG-part-cut_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
	part_R = np.array(f['r'])
	part_SB = np.array(f['sb'])
	part_err = np.array(f['sb_err'])


##. half-cut BCG stacking
with h5py.File( out_path + 'photo-z_match_clust_%s-band_BCG-half-cut_BG-sub_SB.h5' % band_str, 'r') as f:
	half_nbg_R = np.array(f['r'])
	half_nbg_SB = np.array(f['sb'])
	half_nbg_err = np.array(f['sb_err'])

#.
with h5py.File( path + 'photo-z_match_clust_%s-band_BCG-half-cut_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
	half_R = np.array(f['r'])
	half_SB = np.array(f['sb'])
	half_err = np.array(f['sb_err'])


##. half-V-cut BCG stacking
with h5py.File( out_path + 'photo-z_match_clust_%s-band_BCG-half-V-cut_BG-sub_SB.h5' % band_str, 'r') as f:
	half_V_nbg_R = np.array(f['r'])
	half_V_nbg_SB = np.array(f['sb'])
	half_V_nbg_err = np.array(f['sb_err'])

#.
with h5py.File( path + 'photo-z_match_clust_%s-band_BCG-half-V-cut_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:
	half_V_R = np.array(f['r'])
	half_V_SB = np.array(f['sb'])
	half_V_err = np.array(f['sb_err'])


##.. figs
fig = plt.figure( figsize = (10.8, 4.8) )
ax = fig.add_axes([0.08, 0.10, 0.40, 0.80])
sub_ax = fig.add_axes([0.56, 0.10, 0.40, 0.80])

ax.plot( ful_nbg_R, ful_nbg_SB, 'r-', alpha = 0.75, label = 'Full BCG',)
ax.fill_between( ful_nbg_R, y1 = ful_nbg_SB - ful_nbg_err, y2 = ful_nbg_SB + ful_nbg_err, color = 'r', alpha = 0.12,)

# ax.plot( part_nbg_R, part_nbg_SB, 'b--', alpha = 0.75, label = 'Part-cut BCG',)
# ax.fill_between( part_nbg_R, y1 = part_nbg_SB - part_nbg_err, y2 = part_nbg_SB + part_nbg_err, color = 'b', alpha = 0.12,)

ax.plot( half_nbg_R, half_nbg_SB, 'b--', alpha = 0.75, label = 'Half-cut BCG',)
ax.fill_between( half_nbg_R, y1 = half_nbg_SB - half_nbg_err, y2 = half_nbg_SB + half_nbg_err, color = 'b', alpha = 0.12,)

ax.plot( half_V_nbg_R, half_V_nbg_SB, 'g:', alpha = 0.75, label = 'Half-V-cut BCG',)
ax.fill_between( half_V_nbg_R, y1 = half_V_nbg_SB - half_V_nbg_err, y2 = half_V_nbg_SB + half_V_nbg_err, color = 'g', alpha = 0.12,)

ax.axvline( x = np.median(R_sat), ls = ':', color = 'k', alpha = 0.25,)

#.
tmp_F = interp.interp1d( ful_nbg_R, ful_nbg_SB, kind = 'linear', fill_value = 'extrapolate',)

# sub_ax.plot( part_nbg_R, tmp_F( part_nbg_R) - part_nbg_SB, ls = '--', color = 'b', alpha = 0.75,)
# sub_ax.fill_between( part_nbg_R, y1 = tmp_F( part_nbg_R) - part_nbg_SB - part_nbg_err , 
# 					y2 = tmp_F( part_nbg_R) - part_nbg_SB + part_nbg_err, color = 'b', alpha = 0.12,)

sub_ax.plot( half_nbg_R, tmp_F( half_nbg_R) - half_nbg_SB, ls = '--', color = 'b', alpha = 0.75,)
sub_ax.fill_between( half_nbg_R, y1 = tmp_F( half_nbg_R) - half_nbg_SB - half_nbg_err, 
					y2 = tmp_F( half_nbg_R) - half_nbg_SB + half_nbg_err, color = 'b', alpha = 0.12,)

sub_ax.plot( half_V_nbg_R, tmp_F( half_V_nbg_R) - half_V_nbg_SB, ls = ':', color = 'g', alpha = 0.75,)
sub_ax.fill_between( half_V_nbg_R, y1 = tmp_F( half_V_nbg_R) - half_V_nbg_SB - half_V_nbg_err, 
					y2 = tmp_F( half_V_nbg_R) - half_V_nbg_SB + half_V_nbg_err, color = 'g', alpha = 0.12,)

sub_ax.axvline( x = np.median(R_sat), ls = ':', color = 'k', alpha = 0.5,)

ax.legend( loc = 1, frameon = False, fontsize = 12,)

ax.set_xlim( 1e0, 5e2 )
ax.set_xlabel('$ R \; [kpc]$', fontsize = 12,)
ax.set_xscale('log')

ax.set_ylim( 8e-4, 8e0 )
ax.set_ylabel('$\\mu \; [nanomaggies \, / \, arcsec^{2}]$', fontsize = 12,)
ax.set_yscale('log')
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)


sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xlabel('$ R \; [kpc]$', fontsize = 12,)
sub_ax.set_xscale('log')

sub_ax.set_ylim( 1e-5, 1e-2 )
sub_ax.set_yscale('log')
sub_ax.set_ylabel('$\\mu(Full \; BCG) - \\mu$')
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/BCG_%s-band_SB_compare.png' % band_str, dpi = 300)
plt.close()



fig = plt.figure()
ax = fig.add_axes([0.13, 0.32, 0.80, 0.63])
sub_ax = fig.add_axes( [0.13, 0.11, 0.80, 0.21] )
sub_gax = fig.add_axes( [0.20, 0.43, 0.25, 0.25] )

ax.plot( ful_nbg_R, ful_nbg_SB, 'r-', alpha = 0.75, label = 'Full BCG',)
ax.fill_between( ful_nbg_R, y1 = ful_nbg_SB - ful_nbg_err, y2 = ful_nbg_SB + ful_nbg_err, color = 'r', alpha = 0.12,)

# ax.plot( part_nbg_R, part_nbg_SB, 'b--', alpha = 0.75, label = 'Part-cut BCG',)
# ax.fill_between( part_nbg_R, y1 = part_nbg_SB - part_nbg_err, y2 = part_nbg_SB + part_nbg_err, color = 'b', alpha = 0.12,)

ax.plot( half_nbg_R, half_nbg_SB, 'b--', alpha = 0.75, label = 'Half-cut BCG',)
ax.fill_between( half_nbg_R, y1 = half_nbg_SB - half_nbg_err, y2 = half_nbg_SB + half_nbg_err, color = 'b', alpha = 0.12,)

ax.plot( half_V_nbg_R, half_V_nbg_SB, 'g:', alpha = 0.75, label = 'Half-V-cut BCG',)
ax.fill_between( half_V_nbg_R, y1 = half_V_nbg_SB - half_V_nbg_err, y2 = half_V_nbg_SB + half_V_nbg_err, color = 'g', alpha = 0.12,)

ax.axvline( x = np.median(R_sat), ls = ':', color = 'k', alpha = 0.25,)


sub_gax.hist( R_sat, bins = 51, density = True, histtype = 'step', color = 'k',)
sub_gax.set_xlabel('$R_{sat} \; [kpc]$', fontsize = 11,)
sub_gax.set_xscale('log')
sub_gax.set_xlim( 5e0, 2e2 )
sub_gax.axvline( x = np.median(R_sat), ls = ':', color = 'k', alpha = 0.5,)
sub_gax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,-4),)
sub_gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 11,)
sub_gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


#.
tmp_F = interp.interp1d( ful_nbg_R, ful_nbg_SB, kind = 'linear', fill_value = 'extrapolate',)

# sub_ax.plot( part_nbg_R, part_nbg_SB / tmp_F( part_nbg_R), ls = '--', color = 'b', alpha = 0.75,)
# sub_ax.fill_between( part_nbg_R, y1 = ( part_nbg_SB - part_nbg_err ) / tmp_F( part_nbg_R), 
# 					y2 = ( part_nbg_SB + part_nbg_err ) / tmp_F( part_nbg_R), color = 'b', alpha = 0.12,)

sub_ax.plot( half_nbg_R, half_nbg_SB / tmp_F( half_nbg_R), ls = '--', color = 'b', alpha = 0.75,)
sub_ax.fill_between( half_nbg_R, y1 = ( half_nbg_SB - half_nbg_err ) / tmp_F( half_nbg_R), 
					y2 = ( half_nbg_SB + half_nbg_err ) / tmp_F( half_nbg_R), color = 'b', alpha = 0.12,)

sub_ax.plot( half_V_nbg_R, half_V_nbg_SB / tmp_F( half_V_nbg_R), ls = ':', color = 'g', alpha = 0.75,)
sub_ax.fill_between( half_V_nbg_R, y1 = ( half_V_nbg_SB - half_V_nbg_err ) / tmp_F( half_V_nbg_R), 
					y2 = ( half_V_nbg_SB + half_V_nbg_err ) / tmp_F( half_V_nbg_R), color = 'g', alpha = 0.12,)

sub_ax.axvline( x = np.median(R_sat), ls = ':', color = 'k', alpha = 0.5,)

ax.legend( loc = 1, frameon = False, fontsize = 12,)

ax.set_xlim( 1e0, 5e2 )
ax.set_xlabel('$ R \; [kpc]$', fontsize = 12,)
ax.set_xscale('log')

ax.set_ylim( 8e-4, 8e0 )
ax.set_ylabel('$\\mu \; [nanomaggies \, / \, arcsec^{2}]$', fontsize = 12,)
ax.set_yscale('log')
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)


sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xlabel('$ R \; [kpc]$', fontsize = 12,)
sub_ax.set_xscale('log')

sub_ax.axhline( y = 1, ls = ':', color = 'k', alpha = 0.5,)
sub_ax.set_ylim( 0.95, 1.05 )
sub_ax.set_ylabel('$\\mu \, / \, \\mu(Full \; BCG)$')
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax.set_xticklabels( [] )

plt.savefig('/home/xkchen/BCG_%s-band_BG-sub-SB_compare.png' % band_str, dpi = 300)
plt.close()

