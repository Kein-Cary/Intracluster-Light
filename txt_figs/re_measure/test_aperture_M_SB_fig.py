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
from scipy.interpolate import splev, splrep

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
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

psf_FWHM = 1.32 # arcsec


### === ### data load
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

cat_lis = ['low-lgM20', 'hi-lgM20']
fig_name = ['Low $\; M_{\\ast, \, 20}$', 'High $\; M_{\\ast, \, 20}$']
file_s = 'M20_binned'

# cat_lis = ['low-lgM10', 'hi-lgM10']
# fig_name = ['Low $\; M_{\\ast, \, 10}$', 'High $\; M_{\\ast, \, 10}$']
# file_s = 'M10_binned'

BG_path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/BGs/'
path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/SBs/'

out_path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/surface_M/'

#. surface brightness profiles
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( tt_mag )
	nbg_low_err.append( tt_mag_err )

nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( tt_mag )
	nbg_hi_err.append( tt_mag_err )


#. total BCG stellar mass binned sample
cc_path = '/home/xkchen/figs/re_measure_SBs/BGs/'

cc_fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 
				'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

cc_low_r, cc_low_sb, cc_low_err = [], [], []

for kk in range( 3 ):
	with h5py.File( cc_path + 'photo-z_low_BCG_star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	cc_low_r.append( tt_r )
	cc_low_sb.append( tt_mag )
	cc_low_err.append( tt_mag_err )

cc_hi_r, cc_hi_sb, cc_hi_err = [], [], []

for kk in range( 3 ):
	with h5py.File( cc_path + 'photo-z_high_BCG_star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	cc_hi_r.append( tt_r )
	cc_hi_sb.append( tt_mag )
	cc_hi_err.append( tt_mag_err )


#. color profile
mu_dat = pds.read_csv( cc_path + 'high_BCG_star-Mass_color_profile.csv' )
cc_hi_c_r, cc_hi_gr, cc_hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
cc_hi_gr = signal.savgol_filter( cc_hi_gr, 7, 3)

mu_dat = pds.read_csv( cc_path + 'low_BCG_star-Mass_color_profile.csv' )
cc_lo_c_r, cc_lo_gr, cc_lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
cc_lo_gr = signal.savgol_filter( cc_lo_gr, 7, 3)


mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
hi_gr = signal.savgol_filter( hi_gr, 7, 3)

mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
lo_gr = signal.savgol_filter( lo_gr, 7, 3)


#. surface mass profile
surf_M_dat = pds.read_csv(out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0],)
lo_sm_R = np.array( surf_M_dat['R'] )
lo_sm_M, lo_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )

surf_M_dat = pds.read_csv(out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1],)
hi_sm_R = np.array( surf_M_dat['R'] )
hi_sm_M, hi_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )


dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
					'low_BCG_star-Mass_gri-band-based_corrected_aveg-jack_mass-Lumi.csv')
cc_lo_R, cc_lo_surf_M, cc_lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
					'high_BCG_star-Mass_gri-band-based_corrected_aveg-jack_mass-Lumi.csv')
cc_hi_R, cc_hi_surf_M, cc_hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )


dat = pds.read_csv('/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
					'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv')
aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])



#. mass profile ratios (to the integral sample)
eta_dat = pds.read_csv(out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[0],)
lo_eta_R, lo_eta_M, lo_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )

eta_dat = pds.read_csv(out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[1],)
hi_eta_R, hi_eta_M, hi_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )


eta_dat = pds.read_csv('/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'low_BCG_star-Mass_gri-band_aveg_M-ratio_to_total-sample.csv')
cc_lo_eta_R, cc_lo_eta_M, cc_lo_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )

eta_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'high_BCG_star-Mass_gri-band_aveg_M-ratio_to_total-sample.csv')
cc_hi_eta_R, cc_hi_eta_M, cc_hi_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )



#...figs
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3


fig = plt.figure( figsize = (5.4, 5.4) )
ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

ax1.plot( lo_sm_R / 1e3, lo_sm_M, ls = '--', color = line_c[0], alpha = 0.45, linewidth = 3.5, label = fig_name[0] + '$ \\mid \; \\lambda$',)
# ax1.fill_between( lo_sm_R / 1e3, y1 = lo_sm_M - lo_sm_err, y2 = lo_sm_M + lo_sm_err, color = line_c[0], alpha = 0.15,)

ax1.plot( cc_lo_R / 1e3, cc_lo_surf_M, ls = '--', color = 'b', label = cc_fig_name[0],)
ax1.fill_between( cc_lo_R / 1e3, y1 = cc_lo_surf_M - cc_lo_surf_M_err, y2 = cc_lo_surf_M + cc_lo_surf_M_err, color = line_c[0], alpha = 0.15,)

ax1.plot( hi_sm_R / 1e3, hi_sm_M, ls = '-', color = line_c[1], alpha = 0.45, linewidth = 3.5, label = fig_name[1] + '$ \\mid \; \\lambda$',)
# ax1.fill_between( hi_sm_R / 1e3, y1 = hi_sm_M - hi_sm_err, y2 = hi_sm_M + hi_sm_err, color = line_c[1], alpha = 0.15,)

ax1.plot( cc_hi_R / 1e3, cc_hi_surf_M, ls = '-', color = 'r', label = cc_fig_name[1],)
ax1.fill_between( cc_hi_R / 1e3, y1 = cc_hi_surf_M - cc_hi_surf_M_err, y2 = cc_hi_surf_M + cc_hi_surf_M_err, color = line_c[1], alpha = 0.15,)

ax1.plot( aveg_R / 1e3, aveg_surf_m, ls = '-.', color = 'k', alpha = 0.85, label = 'All clusters',)
# ax1.fill_between( aveg_R / 1e3, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'k', alpha = 0.15,)

ax1.fill_betweenx( y = np.logspace(3, 10, 250), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax1.text( 3.5e-3, 1e7, s = 'PSF', fontsize = 13,)

ax1.set_xlim( 3e-3, 1e0 )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim( 5e3, 2e9 )
ax1.legend( loc = 3, frameon = False, fontsize = 13,)
ax1.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 15,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


sub_ax1.plot( lo_eta_R / 1e3, lo_eta_M, ls = '--', color = line_c[0], alpha = 0.45, linewidth = 3.5,)
# sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta_M - lo_eta_M_err, y2 = lo_eta_M + lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta_M, ls = '-', color = line_c[1], alpha = 0.45, linewidth = 3.5,)
# sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta_M - hi_eta_M_err, y2 = hi_eta_M + hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( cc_lo_eta_R / 1e3, cc_lo_eta_M, ls = '--', color = line_c[0],)
sub_ax1.fill_between( cc_lo_eta_R / 1e3, y1 = cc_lo_eta_M - cc_lo_eta_M_err, 
		y2 = cc_lo_eta_M + cc_lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( cc_hi_eta_R / 1e3, cc_hi_eta_M, ls = '-', color = line_c[1],)
sub_ax1.fill_between( cc_hi_eta_R / 1e3, y1 = cc_hi_eta_M - cc_hi_eta_M_err, 
		y2 = cc_hi_eta_M + cc_hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.fill_betweenx( y = np.logspace(-5, 5), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
sub_ax1.axhline( y = 1, ls = '-.', color = 'k', alpha = 0.85,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
sub_ax1.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax1.set_ylabel('$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{\\mathrm{All \; clusters} }$', fontsize = 15,)
sub_ax1.set_ylim( 0.55, 1.55 )
sub_ax1.set_yticks([0.5, 1, 1.5 ])
sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.5}$','$\\mathrm{1.0}$', '$\\mathrm{1.5}$'] )

sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.set_xticklabels( labels = [],)

# plt.savefig('/home/xkchen/%s_SB_SM_compare.png' % file_s, dpi = 300)
plt.savefig('/home/xkchen/%s_SB_SM_compare.pdf' % file_s, dpi = 300)
plt.close()


raise

fig = plt.figure( figsize = (10.2, 4.8) )
ax0 = fig.add_axes([0.07, 0.13, 0.40, 0.84])
# ax0 = fig.add_axes([0.10, 0.34, 0.40, 0.63])
# sub_ax0 = fig.add_axes([0.10, 0.13, 0.40, 0.21])

ax1 = fig.add_axes([0.58, 0.34, 0.40, 0.63])
sub_ax1 = fig.add_axes([0.58, 0.13, 0.40, 0.21])

for kk in range( 3 ):

	ax0.plot( cc_low_r[kk] / 1e3, cc_low_sb[kk] - 3.5, ls = ':', color = color_s[kk], alpha = 0.75,)
	ax0.fill_between( cc_low_r[kk] / 1e3, y1 = cc_low_sb[kk] - 3.5 - cc_low_err[kk], 
		y2 = cc_low_sb[kk] - 3.5 + cc_low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( cc_hi_r[kk] / 1e3, cc_hi_sb[kk] - 3.5, ls = '-.', color = color_s[kk], alpha = 0.75,)
	ax0.fill_between( cc_hi_r[kk] / 1e3, y1 = cc_hi_sb[kk] - 3.5 - cc_hi_err[kk], 
		y2 = cc_hi_sb[kk] - 3.5 + cc_hi_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( nbg_low_r[kk] / 1e3, nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75,)
	ax0.fill_between( nbg_low_r[kk] / 1e3, y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
		y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( nbg_hi_r[kk] / 1e3, nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s band' % band[kk],)
	ax0.fill_between( nbg_hi_r[kk] / 1e3, y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
		y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.15,)

legend_1 = ax0.legend( [ cc_fig_name[0], cc_fig_name[1], 
						fig_name[0] + '$ \\mid \; \\lambda$', fig_name[1] + '$ \\mid \; \\lambda$'], 
						loc = 3, frameon = False, fontsize = 13,)
legend_0 = ax0.legend( loc = 1, frameon = False, fontsize = 13,)
ax0.add_artist( legend_1 )

ax0.fill_betweenx( y = np.linspace( 10, 36, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax0.text( 3e-3, 27, s = 'PSF', fontsize = 16,)

ax0.set_xlim( 3e-3, 1e0)
ax0.set_ylim( 17, 33.5 )
ax0.invert_yaxis()

ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
ax0.set_xticks([ 1e-2, 1e-1, 1e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax0.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


ax1.plot( lo_sm_R / 1e3, lo_sm_M, ls = '--', color = line_c[0], alpha = 0.85, label = fig_name[0] + '$ \\mid \; \\lambda$',)
ax1.fill_between( lo_sm_R / 1e3, y1 = lo_sm_M - lo_sm_err, y2 = lo_sm_M + lo_sm_err, color = line_c[0], alpha = 0.15,)

ax1.plot( hi_sm_R / 1e3, hi_sm_M, ls = '-', color = line_c[1], alpha = 0.85, label = fig_name[1] + '$ \\mid \; \\lambda$',)
ax1.fill_between( hi_sm_R / 1e3, y1 = hi_sm_M - hi_sm_err, y2 = hi_sm_M + hi_sm_err, color = line_c[1], alpha = 0.15,)

ax1.plot( cc_lo_R / 1e3, cc_lo_surf_M, ls = ':', color = 'b', alpha = 0.55, linewidth = 3, label = cc_fig_name[0],)
ax1.plot( cc_hi_R / 1e3, cc_hi_surf_M, ls = '-.', color = 'r', alpha = 0.55, linewidth = 3, label = cc_fig_name[1],)

ax1.plot( aveg_R / 1e3, aveg_surf_m, ls = (0,(5,5)), color = 'k', alpha = 0.85, label = 'All clusters',)
ax1.fill_between( aveg_R / 1e3, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'k', alpha = 0.15,)

ax1.fill_betweenx( y = np.logspace(3, 10, 250), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)

ax1.set_xlim( 3e-3, 1e0 )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim( 5e3, 2e9 )
ax1.legend( loc = 3, frameon = False, fontsize = 13,)
ax1.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 15,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


sub_ax1.plot( lo_eta_R / 1e3, lo_eta_M, ls = '--', color = line_c[0], alpha = 0.85,)
sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta_M - lo_eta_M_err, y2 = lo_eta_M + lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta_M, ls = '-', color = line_c[1], alpha = 0.85,)
sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta_M - hi_eta_M_err, y2 = hi_eta_M + hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( cc_lo_eta_R / 1e3, cc_lo_eta_M, ls = ':', color = line_c[0], alpha = 0.55, linewidth = 3,)
# sub_ax1.fill_between( cc_lo_eta_R / 1e3, y1 = cc_lo_eta_M - cc_lo_eta_M_err, 
# 		y2 = cc_lo_eta_M + cc_lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( cc_hi_eta_R / 1e3, cc_hi_eta_M, ls = '-.', color = line_c[1], alpha = 0.55, linewidth = 3,)
# sub_ax1.fill_between( cc_hi_eta_R / 1e3, y1 = cc_hi_eta_M - cc_hi_eta_M_err, 
# 		y2 = cc_hi_eta_M + cc_hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.axhline( y = 1, ls = (0,(5,5)), color = 'k', alpha = 0.85,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
sub_ax1.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax1.set_ylabel('$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{\\mathrm{All \; clusters} }$', fontsize = 15,)
sub_ax1.set_ylim( 0.55, 1.55 )
sub_ax1.set_yticks([0.5, 1, 1.5 ])
sub_ax1.set_yticklabels( labels = ['$\\mathrm{0.5}$','$\\mathrm{1.0}$', '$\\mathrm{1.5}$'] )

sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.set_xticklabels( labels = [],)

plt.savefig('/home/xkchen/%s_SB_SM_compare.png' % file_s, dpi = 300)
# plt.savefig('/home/xkchen/%s_SB_SM_compare.pdf' % file_s, dpi = 300)
plt.close()

