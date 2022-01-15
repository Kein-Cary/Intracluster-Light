import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import scipy.signal as signal

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
from img_cat_param_match import match_func
from img_pre_selection import extra_match_func

from fig_out_module import arr_jack_func
from Mass_rich_radius import rich2R_Simet

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

### === ### data load and figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

## fixed BCG Mstar samples
cat_lis = [ 'low-rich', 'hi-rich' ]
fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
file_s = 'rich_bin_fixed_BCG_M'

cat_path = '/home/xkchen/figs/'
BG_path = '/home/xkchen/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'
out_path = '/home/xkchen/figs/M2L_fit_test_M/'

#... color profile
c_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
hi_ri, hi_ri_err = np.array( c_dat['r-i'] ), np.array( c_dat['r-i_err'] )

hi_gr = signal.savgol_filter( hi_gr, 7, 3)
hi_ri = signal.savgol_filter( hi_ri, 7, 3)

c_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
lo_ri, lo_ri_err = np.array( c_dat['r-i'] ), np.array( c_dat['r-i_err'] )

lo_gr = signal.savgol_filter( lo_gr, 7, 3)
lo_ri = signal.savgol_filter( lo_ri, 7, 3)


#... sample properties
hi_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )
hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['BCG_age'] )

lo_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )
lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['BCG_age'] )


#... mass, Luminosity profile
dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_R, lo_Li, lo_Li_err = np.array( dat['R'] ), np.array( dat['lumi'] ), np.array( dat['lumi_err'] )

dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_R, hi_Li, hi_Li_err = np.array( dat['R'] ), np.array( dat['lumi'] ), np.array( dat['lumi_err'] )


# Da_ref = Test_model.angular_diameter_distance( z_ref ).value
# phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

fig = plt.figure( figsize = (19.84, 4.8) )
ax0 = fig.add_axes( [0.05, 0.13, 0.28, 0.80] )
ax1 = fig.add_axes( [0.38, 0.13, 0.28, 0.80] )
ax2 = fig.add_axes( [0.71, 0.13, 0.28, 0.80] )

ax0.plot( lo_c_r, lo_gr, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax0.fill_between( lo_c_r, y1 = lo_gr - lo_gr_err, y2 = lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)
ax0.plot( hi_c_r, hi_gr, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax0.fill_between( hi_c_r, y1 = hi_gr - hi_gr_err, y2 = hi_gr + hi_gr_err, color = 'r', alpha = 0.15,)

ax0.legend( loc = 3, frameon = False, fontsize = 18,)
ax0.set_ylim( 0.8, 1.55 )
ax0.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax0.set_xlim( 3, 1.1e3)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [kpc]$', fontsize = 18,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)


ax1.plot( lo_c_r, lo_ri, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax1.fill_between( lo_c_r, y1 = lo_ri - lo_ri_err, y2 = lo_ri + lo_ri_err, color = 'b', alpha = 0.15,)
ax1.plot( hi_c_r, hi_ri, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax1.fill_between( hi_c_r, y1 = hi_ri - hi_ri_err, y2 = hi_ri + hi_ri_err, color = 'r', alpha = 0.15,)

ax1.legend( loc = 2, frameon = False, fontsize = 18,)
ax1.set_ylim( 0.5, 0.8 )
ax1.set_ylabel('$ r \; - \; i $', fontsize = 20,)
ax1.set_xlim( 3, 1.1e3)
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [kpc]$', fontsize = 18,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)


ax2.plot( lo_R, lo_Li, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax2.fill_between( lo_R, y1 = lo_Li - lo_Li_err, y2 = lo_Li + lo_Li_err, color = 'b', alpha = 0.15,)
ax2.plot( hi_R, hi_Li, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax2.fill_between( hi_R, y1 = hi_Li - hi_Li_err, y2 = hi_Li + hi_Li_err, color = 'r', alpha = 0.15,)

ax2.legend( loc = 3, frameon = False, fontsize = 18,)
ax2.set_ylim(1e4, 6e8)
ax2.set_yscale('log')
ax2.set_ylabel('$L_{i} \; [L_{\odot} / kpc^{2}]$', fontsize = 20,)
ax2.set_xlim( 3, 1.1e3)
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [kpc]$', fontsize = 18,)
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/color_test.png', dpi = 300)
plt.close()


#... color profile adjust
p_dat = pds.read_csv( '/home/xkchen/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv' )
a_fit = np.array( p_dat['a'] )[0]
b_fit = np.array( p_dat['b'] )[0]
c_fit = np.array( p_dat['c'] )[0]
d_fit = np.array( p_dat['d'] )[0]

dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_surf_M, hi_surf_M_err = np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

cc_hi_gr = ( np.log10( hi_surf_M ) - d_fit - b_fit * hi_ri - c_fit * np.log10( hi_Li ) ) / a_fit
cc_hi_gr = signal.savgol_filter( cc_hi_gr, 7, 3)

dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_surf_M, lo_surf_M_err = np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

cc_lo_gr = ( np.log10( lo_surf_M ) - d_fit - b_fit * lo_ri - c_fit * np.log10( lo_Li ) ) / a_fit
cc_lo_gr = signal.savgol_filter( cc_lo_gr, 7, 3)

fig = plt.figure()
ax0 = fig.add_axes( [0.12, 0.12, 0.80, 0.80] )

ax0.plot( lo_c_r, cc_lo_gr, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax0.fill_between( lo_c_r, y1 = cc_lo_gr - lo_gr_err, y2 = cc_lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)

ax0.plot( hi_c_r, cc_hi_gr, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax0.fill_between( hi_c_r, y1 = cc_hi_gr - hi_gr_err, y2 = cc_hi_gr + hi_gr_err, color = 'r', alpha = 0.15,)

ax0.legend( loc = 3, frameon = False, fontsize = 18,)
ax0.set_ylim( 0.8, 1.55 )
ax0.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax0.set_xlim( 3, 1.1e3)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [kpc]$', fontsize = 18,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/adjust_g-r_profile.png', dpi = 300)
plt.close()
