import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]
z_ref = 0.25


### ... ### BCG+ICL
cat_lis = [ 'low-age', 'hi-age' ]
fig_name = ['Low $t_{ \\mathrm{age} } \\mid M_{\\ast}^{\\mathrm{BCG}}$', 
			'High $t_{ \\mathrm{age} } \\mid M_{\\ast}^{\\mathrm{BCG}}$']
file_s = 'age_bin_fixed_BCG_M'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'

# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 
# 			'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# file_s = 'BCG_Mstar_bin'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'

# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{ \\mathrm{BCG} } $', 
# 			'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'rich_bin_fixed_BCG_M'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'


#. SB re-measurement results
BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'


mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
hi_gr = signal.savgol_filter( hi_gr, 7, 3)

mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
lo_gr = signal.savgol_filter( lo_gr, 7, 3)

#. extinction correction for BCG+ICL color
if file_s == 'age_bin_fixed_BCG_M':
	gE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-age_bin_gri-common-cat_g-band_dust_value.csv')

if file_s == 'BCG_Mstar_bin':
	gE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_gri-common-cat_g-band_dust_value.csv')

if file_s == 'rich_bin_fixed_BCG_M':
	gE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/clust-rich_gri-common-cat_g-band_dust_value.csv')

samp_dex = np.array( gE_dat['orin_dex'] )
A_g = np.array( gE_dat['A_l'] )

idv = samp_dex == 1

A_g_lo = A_g[ idv ]
mA_g_lo = np.median( A_g[ idv ] )

A_g_hi = A_g[ idv == False ]
mA_g_hi = np.median( A_g[ idv == False ] )


if file_s == 'age_bin_fixed_BCG_M':
	rE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-age_bin_gri-common-cat_r-band_dust_value.csv')

if file_s == 'BCG_Mstar_bin':
	rE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_gri-common-cat_r-band_dust_value.csv')

if file_s == 'rich_bin_fixed_BCG_M':
	rE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/clust-rich_gri-common-cat_r-band_dust_value.csv')

samp_dex = np.array( rE_dat['orin_dex'] )
A_r = np.array( rE_dat['A_l'] )

idv = samp_dex == 1

A_r_lo = A_r[ idv ]
mA_r_lo = np.median( A_r[ idv ] )

A_r_hi = A_r[ idv == False ]
mA_r_hi = np.median( A_r[ idv == False ] )


#. member in ZLWen catalog
lo_dat = pds.read_csv('/home/xkchen/figs/sat_color_weit/ZLW_cat_%s_r-band_Mean-jack_member_color.csv' % cat_lis[0] )
lo_R, lo_mem_gr, lo_mem_gr_err = np.array(lo_dat['R(cMpc/h)']), np.array(lo_dat['g2r']), np.array(lo_dat['g2r_err'])
lo_mem_dered_gr, lo_mem_dered_gr_err = np.array(lo_dat['dered_g2r']), np.array(lo_dat['dered_g2r_err'])

lo_lwt_gr, lo_lwt_gr_err = np.array(lo_dat['Lwt_g2r']), np.array(lo_dat['Lwt_g2r_err'])
lo_lwt_dered_gr, lo_lwt_dered_gr_err = np.array(lo_dat['Lwt_dered_g2r']), np.array(lo_dat['Lwt_dered_g2r_err'])


hi_dat = pds.read_csv('/home/xkchen/figs/sat_color_weit/ZLW_cat_%s_r-band_Mean-jack_member_color.csv' % cat_lis[1] )
hi_R, hi_mem_gr, hi_mem_gr_err = np.array(hi_dat['R(cMpc/h)']), np.array(hi_dat['g2r']), np.array(hi_dat['g2r_err'])
hi_mem_dered_gr, hi_mem_dered_gr_err = np.array(hi_dat['dered_g2r']), np.array(hi_dat['dered_g2r_err'])

hi_lwt_gr, hi_lwt_gr_err = np.array(hi_dat['Lwt_g2r']), np.array(hi_dat['Lwt_g2r_err'])
hi_lwt_dered_gr, hi_lwt_dered_gr_err = np.array(hi_dat['Lwt_dered_g2r']), np.array(hi_dat['Lwt_dered_g2r_err'])


#. member in ZLWen catalog without redshift cut
lo_dat = pds.read_csv('/home/xkchen/figs/sat_color/ZLW_cat_%s_r-band_Mean-jack_member_color.csv' % cat_lis[0] )
lo_nocut_R, lo_nocut_gr, lo_nocut_gr_err = np.array(lo_dat['R(cMpc/h)']), np.array(lo_dat['g2r']), np.array(lo_dat['g2r_err'])
lo_nocut_dered_gr, lo_nocut_dered_gr_err = np.array(lo_dat['dered_g2r']), np.array(lo_dat['dered_g2r_err'])

hi_dat = pds.read_csv('/home/xkchen/figs/sat_color/ZLW_cat_%s_r-band_Mean-jack_member_color.csv' % cat_lis[1] )
hi_nocut_R, hi_nocut_gr, hi_nocut_gr_err = np.array(hi_dat['R(cMpc/h)']), np.array(hi_dat['g2r']), np.array(hi_dat['g2r_err'])
hi_nocut_dered_gr, hi_nocut_dered_gr_err = np.array(hi_dat['dered_g2r']), np.array(hi_dat['dered_g2r_err'])


#. member in SDSS redMapper
red_lo_dat = pds.read_csv('/home/xkchen/figs/sat_color/%s_r-band_Mean-jack_member_color.csv' % cat_lis[0] )
red_lo_R, red_lo_mem_gr, red_lo_mem_gr_err = [ np.array(red_lo_dat['R(cMpc/h)']), np.array(red_lo_dat['g2r']), 
												np.array(red_lo_dat['g2r_err']) ]
red_lo_mem_dered_gr, red_lo_mem_dered_gr_err = np.array(red_lo_dat['dered_g2r']), np.array(red_lo_dat['dered_g2r_err'])
red_lo_R = red_lo_R * 1e3 / h / ( 1 + z_ref )

red_hi_dat = pds.read_csv('/home/xkchen/figs/sat_color/%s_r-band_Mean-jack_member_color.csv' % cat_lis[1] )
red_hi_R, red_hi_mem_gr, red_hi_mem_gr_err = [ np.array(red_hi_dat['R(cMpc/h)']), np.array(red_hi_dat['g2r']), 
												np.array(red_hi_dat['g2r_err']) ]
red_hi_mem_dered_gr, red_hi_mem_dered_gr_err = np.array(red_hi_dat['dered_g2r']), np.array(red_hi_dat['dered_g2r_err'])
red_hi_R = red_hi_R * 1e3 / h / ( 1 + z_ref )


#.. measured from Sigma_g with deredden correction
sig_path_0 = '/home/xkchen/mywork/ICL/data/data_Zhiwei/g2r_all_sample/data/g-r_deext/'

if file_s == 'age_bin_fixed_BCG_M':
	sig_lo_dat = pds.read_csv( sig_path_0 + 'low-age_g-r_deext_allinfo.csv')
	sig_hi_dat = pds.read_csv( sig_path_0 + 'hi-age_g-r_deext_allinfo.csv')

if file_s == 'BCG_Mstar_bin':
	sig_lo_dat = pds.read_csv( sig_path_0 + 'low-BCG_g-r_deext_allinfo.csv')
	sig_hi_dat = pds.read_csv( sig_path_0 + 'hi-BCG_g-r_deext_allinfo.csv')

if file_s == 'rich_bin_fixed_BCG_M':
	sig_lo_dat = pds.read_csv( sig_path_0 + 'low-rich_g-r_deext_allinfo.csv')
	sig_hi_dat = pds.read_csv( sig_path_0 + 'hi-rich_g-r_deext_allinfo.csv')

sig_lo_dered_Rc, sig_lo_mem_dered_gr, sig_lo_mem_dered_gr_err = [ np.array(sig_lo_dat['rbins']), np.array(sig_lo_dat['mcolor']), 
																	np.array(sig_lo_dat['mcolor_err'])]
sig_lo_dered_R = sig_lo_dered_Rc * 1e3 / h / ( 1 + z_ref )

sig_hi_dered_Rc, sig_hi_mem_dered_gr, sig_hi_mem_dered_gr_err = [ np.array(sig_hi_dat['rbins']), np.array(sig_hi_dat['mcolor']), 
																	np.array(sig_hi_dat['mcolor_err']) ]
sig_hi_dered_R = sig_hi_dered_Rc * 1e3 / h / ( 1 + z_ref )


#.. measured from Sigma_g without deredden correction
sig_path_1 = '/home/xkchen/mywork/ICL/data/data_Zhiwei/g2r_all_sample/data/g-r/'

if file_s == 'age_bin_fixed_BCG_M':
	sig_lo_dat = pds.read_csv( sig_path_1 + 'low-age_g-r_allinfo.csv')
	sig_hi_dat = pds.read_csv( sig_path_1 + 'hi-age_g-r_allinfo.csv')

if file_s == 'BCG_Mstar_bin':
	sig_lo_dat = pds.read_csv( sig_path_1 + 'low-BCG_g-r_allinfo.csv')
	sig_hi_dat = pds.read_csv( sig_path_1 + 'hi-BCG_g-r_allinfo.csv')

if file_s == 'rich_bin_fixed_BCG_M':
	sig_lo_dat = pds.read_csv( sig_path_1 + 'low-rich_g-r_allinfo.csv')
	sig_hi_dat = pds.read_csv( sig_path_1 + 'hi-rich_g-r_allinfo.csv')

sig_lo_Rc, sig_lo_mem_gr, sig_lo_mem_gr_err = [ np.array(sig_lo_dat['rbins']), np.array(sig_lo_dat['mcolor']), 
												np.array(sig_lo_dat['mcolor_err']) ]

lo_c_all, lo_c_all_err = np.array( sig_lo_dat['c_all'] ), np.array( sig_lo_dat['c_all_err'] )
lo_bg_c, lo_bg_c_err = np.array( sig_lo_dat['c_bg'] ), np.array( sig_lo_dat['c_bg_err'] )
lo_DcDg, lo_DcDg_err = np.array( sig_lo_dat['DcDg'] ), np.array( sig_lo_dat['DcDg_err'] )
lo_DcDm, lo_DcDm_err = np.array( sig_lo_dat['DcDm'] ), np.array( sig_lo_dat['DcDm_err'] )
lo_sigm_g, lo_sigm_g_err = np.array( sig_lo_dat['sigma'] ), np.array( sig_lo_dat['sigma_err'] )

sig_lo_R = sig_lo_Rc * 1e3 / h / ( 1 + z_ref ) # comoving --> physical radius


sig_hi_Rc, sig_hi_mem_gr, sig_hi_mem_gr_err = [ np.array(sig_hi_dat['rbins']), np.array(sig_hi_dat['mcolor']), 
												np.array(sig_hi_dat['mcolor_err']) ]

hi_c_all, hi_c_all_err = np.array( sig_hi_dat['c_all'] ), np.array( sig_hi_dat['c_all_err'] )
hi_bg_c, hi_bg_c_err = np.array( sig_hi_dat['c_bg'] ), np.array( sig_hi_dat['c_bg_err'] )
hi_DcDg, hi_DcDg_err = np.array( sig_hi_dat['DcDg'] ), np.array( sig_hi_dat['DcDg_err'] )
hi_DcDm, hi_DcDm_err = np.array( sig_hi_dat['DcDm'] ), np.array( sig_hi_dat['DcDm_err'] )
hi_sigm_g, hi_sigm_g_err = np.array( sig_hi_dat['sigma'] ), np.array( sig_hi_dat['sigma_err'] )

sig_hi_R = sig_hi_Rc * 1e3 / h / ( 1 + z_ref ) # comoving --> physical radius


'''
#. fig : satellite color estimation process
fig = plt.figure( figsize = (19.84, 4.8) )
ax0 = fig.add_axes([0.05, 0.14, 0.28, 0.82])
ax1 = fig.add_axes([0.38, 0.14, 0.28, 0.82])
ax2 = fig.add_axes([0.71, 0.14, 0.28, 0.82])

ax0.errorbar(sig_lo_Rc, lo_sigm_g, yerr = lo_sigm_g_err, ls = '--', marker = None, color = 'b', alpha = 0.85, 
	capsize = 3, label = fig_name[0] )
ax0.errorbar(sig_hi_Rc, hi_sigm_g, yerr = hi_sigm_g_err, ls = '-', marker = None, color = 'r', alpha = 0.85, 
	capsize = 3, label = fig_name[1] )

ax0.legend( loc = 1, frameon = False, fontsize = 15,)
ax0.set_xlim( 1e-2, 2.2 )
ax0.set_ylim( 1.8e0, 3e2 )
ax0.set_ylabel('$\\Sigma_{g} \; [\\# \; h^{2} \, \\mathrm{M}pc^{-2}]$', fontsize = 18,)
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [h^{-1} \, \\mathrm{M}pc] $', fontsize = 18,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

ax1.errorbar(sig_lo_Rc, lo_DcDg, yerr = lo_DcDg_err, ls = '--', marker = 'o', color = 'b', alpha = 0.85, 
	capsize = 3, mec = 'b', mfc = 'none', label = fig_name[0] )
l1 = ax1.errorbar(sig_hi_Rc, hi_DcDg, yerr = hi_DcDg_err, ls = '-', marker = 'o', color = 'r', alpha = 0.85, 
	capsize = 3, mec = 'r', mfc = 'none', label = fig_name[1] )

ax1.errorbar(sig_lo_Rc, lo_DcDm, yerr = lo_DcDm_err, ls = '--', marker = 's', color = 'b', alpha = 0.85, 
	capsize = 3, mec = 'b', mfc = 'none', )
l2 = ax1.errorbar(sig_hi_Rc, hi_DcDm, yerr = hi_DcDm_err, ls = '-', marker = 's', color = 'r', alpha = 0.85, 
	capsize = 3, mec = 'r', mfc = 'none', )

legend_0 = ax1.legend( handles = [l1, l2], labels = ['$D_{c}D_{g}$', '$D_{c}D_{m}$'], loc = 4, frameon = False, fontsize = 15,)
ax1.legend( loc = 2, frameon = False, fontsize = 15,)
ax1.add_artist( legend_0 )

ax1.set_xlim( 1e-2, 2.2 )
ax1.set_ylim( 8e-1, 5e4 )
ax1.set_ylabel('# of pairs', fontsize = 18,)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [h^{-1} \, \\mathrm{M}pc] $', fontsize = 18,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

ax2.errorbar(sig_lo_Rc, lo_bg_c, yerr = lo_bg_c_err, ls = ':', marker = None, color = 'b', alpha = 0.85, 
	capsize = 3, )
ax2.errorbar(sig_hi_Rc, hi_bg_c, yerr = hi_bg_c_err, ls = ':', marker = None, color = 'r', alpha = 0.85, 
	capsize = 3, label = 'Background galaxies')

ax2.errorbar(sig_lo_Rc, lo_c_all, yerr = lo_c_all_err, ls = '--', marker = None, color = 'b', alpha = 0.85, 
	capsize = 3, )
ax2.errorbar(sig_hi_Rc, hi_c_all, yerr = hi_c_all_err, ls = '--', marker = None, color = 'r', alpha = 0.85, 
	capsize = 3, label = 'All galaxies')

l1 = ax2.errorbar(sig_lo_Rc, sig_lo_mem_gr, yerr = sig_lo_mem_gr_err, ls = '-', marker = None, color = 'b', alpha = 0.85, 
	capsize = 3, )
l2 = ax2.errorbar(sig_hi_Rc, sig_hi_mem_gr, yerr = sig_hi_mem_gr_err, ls = '-', marker = None, color = 'r', alpha = 0.85, 
	capsize = 3, label = 'Member galaxies')

legend_0 = ax2.legend( handles = [l1, l2], labels = [ fig_name[0], fig_name[1] ], loc = 2, frameon = False, fontsize = 15,)
ax2.legend( loc = 1, frameon = False, fontsize = 15,)
ax2.add_artist( legend_0 )

ax2.set_xlim( 1e-2, 2.2 )
ax2.set_ylim( 1.13, 1.7 )
ax2.set_ylabel('$\\langle g -r \\rangle$', fontsize = 18,)
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [h^{-1} \, \\mathrm{M}pc] $', fontsize = 18,)
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/sat_mean-gr_process.png', dpi = 300)
# plt.savefig('/home/xkchen/sat_mean-gr_process.pdf', dpi = 300)
plt.close()
'''

'''
fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.09, 0.13, 0.40, 0.82])
ax1 = fig.add_axes([0.57, 0.13, 0.40, 0.82])

ax0.errorbar(lo_R * 1e3, lo_mem_dered_gr, ls = '--', color = 'b',)
ax0.fill_between( lo_R * 1e3, y1 = lo_mem_dered_gr - lo_mem_dered_gr_err, y2 = lo_mem_dered_gr + lo_mem_dered_gr_err, color = 'b',
				alpha = 0.15)

ax0.errorbar(hi_R * 1e3, hi_mem_dered_gr, ls = '-', color = 'r', label = 'with redshift cut',)
ax0.fill_between( hi_R * 1e3, y1 = hi_mem_dered_gr - hi_mem_dered_gr_err, y2 = hi_mem_dered_gr + hi_mem_dered_gr_err, color = 'r',
				alpha = 0.15)

ax0.plot(lo_nocut_R * 1e3, lo_nocut_dered_gr, ls = '--', color = 'b', linewidth = 3,)
ax0.plot(hi_nocut_R * 1e3, hi_nocut_dered_gr, ls = '-', color = 'r', label = 'w/o redshift cut', linewidth = 3,)

ax0.annotate( text = '$m_{sat}$ with deredden correction', xy = (0.25, 0.95), xycoords = 'axes fraction', fontsize = 13,)
ax0.legend( loc = 3, frameon = False, fontsize = 13,)

ax0.set_ylim( 1.23, 1.5 )
ax0.set_ylabel('$ g \; - \; r $', fontsize = 13,)
ax0.set_xlim( 9e0, 1e3)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 13,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)


ax1.errorbar(lo_R * 1e3, lo_mem_gr, ls = '--', color = 'b',)
ax1.fill_between( lo_R * 1e3, y1 = lo_mem_gr - lo_mem_gr_err, y2 = lo_mem_gr + lo_mem_gr_err, color = 'b', alpha = 0.15)
ax1.errorbar(hi_R * 1e3, hi_mem_gr, ls = '-', color = 'r', label = 'with redshift cut')
ax1.fill_between( hi_R * 1e3, y1 = hi_mem_gr - hi_mem_gr_err, y2 = hi_mem_gr + hi_mem_gr_err, color = 'r', alpha = 0.15)

ax1.plot(lo_nocut_R * 1e3, lo_nocut_gr, ls = '--', color = 'b', linewidth = 3, )
ax1.plot(hi_nocut_R * 1e3, hi_nocut_gr, ls = '-', color = 'r', label = 'w/o redshift cut', linewidth = 3,)

ax1.legend( loc = 3, frameon = False, fontsize = 15,)
ax1.annotate( text = '$m_{sat}$ without deredden correction', xy = (0.20, 0.95), xycoords = 'axes fraction', fontsize = 13,)

ax1.set_ylim( 1.23, 1.5 )
ax1.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax1.set_xlim( 9e0, 1e3)
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 13,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

plt.savefig('/home/xkchen/%s_sat_color_compare.png' % file_s, dpi = 300)
plt.close()
'''

fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.09, 0.13, 0.40, 0.82])
ax1 = fig.add_axes([0.57, 0.13, 0.40, 0.82])

ax0.errorbar(sig_lo_R / 1e3, lo_sigm_g * h**2 * (1 + z_ref)**2, yerr = lo_sigm_g_err * h**2 * (1 + z_ref)**2, ls = '--', 
	marker = None, color = 'b', alpha = 0.85, capsize = 3, label = fig_name[0] )
ax0.errorbar(sig_hi_R / 1e3, hi_sigm_g * h**2 * (1 + z_ref)**2, yerr = hi_sigm_g_err * h**2 * (1 + z_ref)**2, ls = '-', 
	marker = None, color = 'r', alpha = 0.85, capsize = 3, label = fig_name[1] )

ax0.legend( loc = 1, frameon = False, fontsize = 15,)
ax0.set_xlim( 1e-2, 1.1 )
ax0.set_ylim( 4e0, 3e2 )
ax0.set_ylabel('$\\Sigma_{g} \; [\\# \, \\mathrm{M}pc^{-2}]$', fontsize = 15,)
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax0.set_xticks([ 1e-2, 1e-1, 1e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )


ax1.plot( lo_c_r / 1e3, lo_gr + (mA_r_lo - mA_g_lo), ls = '--', color = 'b', alpha = 0.45, linewidth = 3, 
		label = '$\\mathrm{BCG} {+} \\mathrm{ICL}$, ' + fig_name[0])
ax1.fill_between( lo_c_r / 1e3, y1 = lo_gr + (mA_r_lo - mA_g_lo) - lo_gr_err, y2 = lo_gr + (mA_r_lo - mA_g_lo) + lo_gr_err, 
				color = 'b', alpha = 0.15,)

ax1.plot( hi_c_r / 1e3, hi_gr + (mA_r_hi - mA_g_hi), ls = '-', color = 'r', alpha = 0.45, linewidth = 3, 
		label = '$\\mathrm{BCG} {+} \\mathrm{ICL}$, ' + fig_name[1])
ax1.fill_between( hi_c_r / 1e3, y1 = hi_gr + (mA_r_hi - mA_g_hi) - hi_gr_err, y2 = hi_gr + (mA_r_hi - mA_g_hi) + hi_gr_err, 
				color = 'r', alpha = 0.15,)

ax1.plot( sig_lo_R / 1e3, sig_lo_mem_dered_gr, ls = '--', color = 'b', alpha = 0.75, label = '$\\xi_{cg}$ estimator, ' + fig_name[0])
ax1.fill_between( sig_lo_R / 1e3, y1 = sig_lo_mem_dered_gr - sig_lo_mem_dered_gr_err, 
	y2 = sig_lo_mem_dered_gr + sig_lo_mem_dered_gr_err, color = 'b', alpha = 0.25,)

ax1.plot( sig_hi_R / 1e3, sig_hi_mem_dered_gr, ls = '-', color = 'r', alpha = 0.75, label = '$\\xi_{cg}$ estimator, ' + fig_name[1],)
ax1.fill_between( sig_hi_R / 1e3, y1 = sig_hi_mem_dered_gr - sig_hi_mem_dered_gr_err, 
	y2 = sig_hi_mem_dered_gr + sig_hi_mem_dered_gr_err, color = 'r', alpha = 0.25,)

h1 = ax1.errorbar( lo_R, lo_mem_dered_gr, yerr = lo_mem_dered_gr_err, ls = 'none', marker = 's', ms = 5, color = 'b', capsize = 3,
	mec = 'b', mfc = 'b',)
h2 = ax1.errorbar( hi_R, hi_mem_dered_gr, yerr = hi_mem_dered_gr_err, ls = 'none', marker = 'o', ms = 5, color = 'r', capsize = 3,
	mec = 'r', mfc = 'r',)

legend_0 = ax1.legend( handles = [h1, h2], labels = [ '$\\mathrm{Wen} {+} \\mathrm{2015}$' + ' member, ' + fig_name[0], 
													  '$\\mathrm{Wen} {+} \\mathrm{2015}$' + ' member, ' + fig_name[1] ], 
						loc = 2, frameon = False, fontsize = 12,)

ax1.legend( loc = 3, frameon = False, fontsize = 11,)
ax1.add_artist( legend_0 )

ax1.set_ylim( 0.78, 1.70 )
ax1.set_ylabel('$ g \; - \; r $', fontsize = 17,)
ax1.set_xlim( 3e-3, 1.1 )
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
ax1.set_xticks( x_tick_arr )
ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

plt.savefig('/home/xkchen/%s_ZLW_gr_compare.png' % file_s, dpi = 300)
# plt.savefig('/home/xkchen/%s_ZLW_gr_compare.pdf' % file_s, dpi = 300)
plt.close()

raise

fig = plt.figure( figsize = (12.8, 4.8) )
ax0 = fig.add_axes([0.07, 0.13, 0.40, 0.80])
ax1 = fig.add_axes([0.55, 0.13, 0.40, 0.80])

# ax0.errorbar(red_lo_R, red_lo_mem_dered_gr, yerr = red_lo_mem_dered_gr_err, ls = '--', marker = None, ms = 5, 
# 	color = 'b', alpha = 0.50,)
# ax0.errorbar(red_hi_R, red_hi_mem_dered_gr, yerr = red_hi_mem_dered_gr_err, ls = '-', marker = None, ms = 5, color = 'r', 
# 	alpha = 0.50, label = 'Satellites of SDSS redMaPPer',)

ax0.errorbar(lo_R * 1e3, lo_mem_dered_gr, yerr = lo_mem_dered_gr_err, ls = 'none', marker = 'o', ms = 6, color = 'b', 
	alpha = 0.85, capsize = 3, mec = 'b', mfc = 'none',)
ax0.errorbar(hi_R * 1e3, hi_mem_dered_gr, yerr = hi_mem_dered_gr_err, ls = 'none', marker = 'o', ms = 6, color = 'r', 
	alpha = 0.85, mec = 'r', mfc = 'none', label = '$ \\mathrm{Satellites} \, \\mathrm{Wen} {+} \\mathrm{2015}$', capsize = 3,)

# ax0.errorbar( sig_lo_dered_R, sig_lo_mem_dered_gr, yerr = sig_lo_mem_dered_gr_err, ls = '--', marker = None, ms = 5, 
# 	color = 'b', alpha = 0.75,)
# ax0.errorbar( sig_hi_dered_R, sig_hi_mem_dered_gr, yerr = sig_hi_mem_dered_gr_err, ls = '-', marker = None, ms = 5, color = 'r', 
# 	alpha = 0.75, label = '$\\xi_{cg}$ estimator',)

# ax0.plot( lo_R * 1e3, lo_mem_dered_gr, ls = '--', linewidth = 5, color = 'b', alpha = 0.5,)
# ax0.plot( hi_R * 1e3, hi_mem_dered_gr, ls = '-', linewidth = 5, color = 'r', alpha = 0.5, 
# 	label = 'Satellites of $\\mathrm{Wen} {+} \\mathrm{2015}$')

ax0.plot( sig_lo_dered_R, sig_lo_mem_dered_gr, ls = '--', linewidth = 2, color = 'b', alpha = 0.5,)
ax0.plot( sig_hi_dered_R, sig_hi_mem_dered_gr, ls = '-', linewidth = 2, color = 'r', alpha = 0.5, 
	label = '$\\xi_{cg}$ estimator',)

ax0.plot( hi_R * 1e3, hi_lwt_dered_gr, ls = '-', linewidth = 5, color = 'r', alpha = 0.5,)
ax0.plot( lo_R * 1e3, lo_lwt_dered_gr, ls = '--', linewidth = 5, color = 'b', alpha = 0.5, 
	label = '$ \\mathrm{Satellites} \, \\mathrm{Wen} {+} \\mathrm{2015} \,,L_{r} \, weighted$')

l1, = ax0.plot( lo_c_r, lo_gr + (mA_r_lo - mA_g_lo), ls = '--', color = 'b', alpha = 0.75, )
ax0.fill_between( lo_c_r, y1 = lo_gr + (mA_r_lo - mA_g_lo) - lo_gr_err, 
					y2 = lo_gr + (mA_r_lo - mA_g_lo) + lo_gr_err, color = 'b', alpha = 0.15,)

l2, = ax0.plot( hi_c_r, hi_gr + (mA_r_hi - mA_g_hi), ls = '-', color = 'r', alpha = 0.75, )
ax0.fill_between( hi_c_r, y1 = hi_gr + (mA_r_hi - mA_g_hi) - hi_gr_err, y2 = hi_gr + (mA_r_hi - mA_g_hi) + hi_gr_err, 
					color = 'r', alpha = 0.15, label = '$\\mathrm{BCG} {+} \\mathrm{ICL}$')

ax0.annotate( text = '$m_{sat}$ with deredden correction', xy = (0.30, 0.95), xycoords = 'axes fraction', fontsize = 15,)

legend_0 = ax0.legend( handles = [l1, l2], labels = [ fig_name[0], fig_name[1] ], loc = 6, frameon = False, fontsize = 15,)
ax0.legend( loc = 3, frameon = False, fontsize = 15,)
ax0.add_artist( legend_0 )

ax0.set_ylim( 0.78, 1.60 )
ax0.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax0.set_xlim( 1e0, 1e3)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)


# ax1.errorbar(red_lo_R, red_lo_mem_gr, yerr = red_lo_mem_gr_err, ls = '--', marker = None, ms = 5, color = 'b', 
# 	alpha = 0.50,)
# ax1.errorbar(red_hi_R, red_hi_mem_gr, yerr = red_hi_mem_gr_err, ls = '-', marker = None, ms = 5, color = 'r', 
# 	alpha = 0.50, label = 'Satellites of SDSS redMaPPer',)

ax1.errorbar(lo_R * 1e3, lo_mem_gr, yerr = lo_mem_gr_err, ls = 'none', marker = 'o', ms = 6, color = 'b', alpha = 0.75, capsize = 3,
	mec = 'b', mfc = 'none',)
ax1.errorbar(hi_R * 1e3, hi_mem_gr, yerr = hi_mem_gr_err, ls = 'none', marker = 'o', ms = 6, color = 'r', alpha = 0.75, 
	mec = 'r', mfc = 'none', label = '$ \\mathrm{Satellites} \, \\mathrm{Wen} {+} \\mathrm{2015} $', capsize = 3,)

# ax1.errorbar( sig_lo_R, sig_lo_mem_gr, yerr = sig_lo_mem_gr_err, ls = '--', marker = None, ms = 5, color = 'b', 
# 	alpha = 0.75,)
# ax1.errorbar( sig_hi_R, sig_hi_mem_gr, yerr = sig_hi_mem_gr_err, ls = '-', marker = None, ms = 5, color = 'r', 
# 	alpha = 0.75, label = '$\\xi_{cg}$ estimator',)

# ax1.plot( lo_R * 1e3, lo_mem_gr, ls = '--', linewidth = 5, color = 'b', alpha = 0.5,)
# ax1.plot( hi_R * 1e3, hi_mem_gr, ls = '-', linewidth = 5, color = 'r', alpha = 0.5, 
# 	label = 'Satellites of $\\mathrm{Wen} {+} \\mathrm{2015}$',)

ax1.plot( sig_lo_R, sig_lo_mem_gr, ls = '--', linewidth = 2, color = 'b', alpha = 0.5,)
ax1.plot( sig_hi_R, sig_hi_mem_gr, ls = '-', linewidth = 2, color = 'r', alpha = 0.5, label = '$\\xi_{cg}$ estimator',)

ax1.plot( hi_R * 1e3, hi_lwt_gr, ls = '-', linewidth = 5, color = 'r', alpha = 0.5, 
		label = '$ \\mathrm{Satellites} \, \\mathrm{Wen} {+} \\mathrm{2015} \,,L_{r} \, weighted$')
ax1.plot( lo_R * 1e3, lo_lwt_gr, ls = '--', linewidth = 5, color = 'b', alpha = 0.5,)

l1, = ax1.plot( lo_c_r, lo_gr, ls = '--', color = 'b', alpha = 0.75, )
ax1.fill_between( lo_c_r, y1 = lo_gr - lo_gr_err, y2 = lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)

l2, = ax1.plot( hi_c_r, hi_gr, ls = '-', color = 'r', alpha = 0.75, )
ax1.fill_between( hi_c_r, y1 = hi_gr - hi_gr_err, y2 = hi_gr + hi_gr_err, color = 'r', alpha = 0.15, 
	label = '$\\mathrm{BCG} {+} \\mathrm{ICL}$')

legend_0 = ax1.legend( handles = [l1, l2], labels = [ fig_name[0], fig_name[1] ], loc = 6, frameon = False, fontsize = 15,)
ax1.legend( loc = 3, frameon = False, fontsize = 15,)
ax1.add_artist( legend_0 )

ax1.annotate( text = '$m_{sat}$ without deredden correction', xy = (0.28, 0.95), xycoords = 'axes fraction', fontsize = 15,)

ax1.set_ylim( 0.78, 1.60 )
ax1.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax1.set_xlim( 1e0, 1e3)
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/%s_deredden_gr_compare.png' % file_s, dpi = 300)
plt.close()
