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
Da_ref = Test_model.angular_diameter_distance( z_ref ).value


band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

psf_FWHM = 1.32 # arcsec

phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3


### === ### 
cat_lis_0 = ['low-lgM20', 'hi-lgM20']
fig_name_0 = ['Low $ M_{\\ast, \, 20}$', 'High $ M_{\\ast, \, 20}$']

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']


### === ### subsamples without P_cen cut
low_r, low_sb, low_err = [], [], []
for kk in range( 3 ):
	with h5py.File( '/home/xkchen/figs/re_measure_SBs/BGs/' + 
		'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	low_r.append( tt_r )
	low_sb.append( tt_mag )
	low_err.append( tt_mag_err )

hi_r, hi_sb, hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( '/home/xkchen/figs/re_measure_SBs/BGs/' + 
		'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	hi_r.append( tt_r )
	hi_sb.append( tt_mag )
	hi_err.append( tt_mag_err )


#. lgM20 binned
low_M20_r, low_M20_sb, low_M20_err = [], [], []
for kk in range( 3 ):
	with h5py.File( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/BGs/' + 
		'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis_0[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	low_M20_r.append( tt_r )
	low_M20_sb.append( tt_mag )
	low_M20_err.append( tt_mag_err )

hi_M20_r, hi_M20_sb, hi_M20_err = [], [], []
for kk in range( 3 ):
	with h5py.File( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/BGs/' + 
		'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis_0[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	hi_M20_r.append( tt_r )
	hi_M20_sb.append( tt_mag )
	hi_M20_err.append( tt_mag_err )


## mass profile with deredden correction or not
# id_dered = False
# dered_str = ''

id_dered = True
dered_str = '_with-dered'


#. overall sample
dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi%s.csv' % dered_str)
aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


#. surface mass profiles
surf_M_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[0], dered_str),)
lo_sm_R = np.array( surf_M_dat['R'] )
lo_sm_M, lo_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )

surf_M_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[1], dered_str),)
hi_sm_R = np.array( surf_M_dat['R'] )
hi_sm_M, hi_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )


dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/surface_M/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis_0[0], dered_str),)
lo_M20_R, lo_M20_surf_M, lo_M20_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'])

dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/surface_M/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis_0[1], dered_str),)
hi_M20_R, hi_M20_surf_M, hi_M20_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'])


#. mass profile ratios (to the integral sample)
eta_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], dered_str),)
lo_eta_R, lo_eta_M, lo_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )

eta_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], dered_str),)
hi_eta_R, hi_eta_M, hi_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'] )


eta_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/surface_M/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis_0[0], dered_str),)
lo_M20_eta_R, lo_M20_eta_M, lo_M20_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'])

eta_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_SBs/surface_M/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis_0[1], dered_str),)
hi_M20_eta_R, hi_M20_eta_M, hi_M20_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), np.array( eta_dat['M/M_tot-err'])



### === ### subsamples with P_cen cut
surf_M_dat = pds.read_csv( '/home/xkchen/figs/Pcen_cut/surf_M/' + 
							'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[0], dered_str),)
Pcen_lo_sm_R = np.array( surf_M_dat['R'] )
Pcen_lo_sm_M, Pcen_lo_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )

surf_M_dat = pds.read_csv( '/home/xkchen/figs/Pcen_cut/surf_M/' + 
							'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[1], dered_str),)
Pcen_hi_sm_R = np.array( surf_M_dat['R'] )
Pcen_hi_sm_M, Pcen_hi_sm_err = np.array( surf_M_dat['medi_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )


#. mass profile ratios (to the integral sample)
eta_dat = pds.read_csv( '/home/xkchen/figs/Pcen_cut/surf_M/' + 
							'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], dered_str),)
Pcen_lo_eta_R, Pcen_lo_eta_M, Pcen_lo_eta_M_err = ( np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), 
														np.array( eta_dat['M/M_tot-err'] ) )

eta_dat = pds.read_csv( '/home/xkchen/figs/Pcen_cut/surf_M/' + 
							'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], dered_str),)
Pcen_hi_eta_R, Pcen_hi_eta_M, Pcen_hi_eta_M_err = ( np.array( eta_dat['R'] ), np.array( eta_dat['M/M_tot'] ), 
														np.array( eta_dat['M/M_tot-err'] ) )

#... surface brightness profile compare
Pcen_low_r, Pcen_low_sb, Pcen_low_err = [], [], []
for kk in range( 3 ):
	with h5py.File( '/home/xkchen/figs/Pcen_cut/BGs/' + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	Pcen_low_r.append( tt_r )
	Pcen_low_sb.append( tt_mag )
	Pcen_low_err.append( tt_mag_err )

Pcen_hi_r, Pcen_hi_sb, Pcen_hi_err = [], [], []
for kk in range( 3 ):
	with h5py.File( '/home/xkchen/figs/Pcen_cut/BGs/' + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	Pcen_hi_r.append( tt_r )
	Pcen_hi_sb.append( tt_mag )
	Pcen_hi_err.append( tt_mag_err )

#. P_cen properties
lo_pat = pds.read_csv('/home/xkchen/figs/Pcen_cut/low_BCG_star-Mass_gri-common_Pcen_cat.csv')
lo_Pcen = np.array( lo_pat['P_cen_0'] )

hi_pat = pds.read_csv('/home/xkchen/figs/Pcen_cut/high_BCG_star-Mass_gri-common_Pcen_cat.csv')
hi_Pcen = np.array( hi_pat['P_cen_0'] )



### === ### figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r']
line_s = [ '-', '--', ':']

Pcen_lim = 0.85
off_sb = 3.5


"""
fig = plt.figure( figsize = (6.4, 6.4) )
ax0 = fig.add_axes( [0.14, 0.10, 0.80, 0.83] )

for kk in range( 3 ):

	ax0.plot( low_r[kk] / 1e3, low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75, linewidth = 2, label = '%s band' % band[kk])
	ax0.fill_between( low_r[kk] / 1e3, y1 = low_sb[kk] - low_err[kk], y2 = low_sb[kk] + low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( hi_r[kk] / 1e3, hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, linewidth = 2,)
	ax0.fill_between( hi_r[kk] / 1e3, y1 = hi_sb[kk] - hi_err[kk], y2 = hi_sb[kk] + hi_err[kk], color = color_s[kk], alpha = 0.15,)


	ax0.plot( low_M20_r[kk] / 1e3, low_M20_sb[kk] - off_sb, ls = '--', color = color_s[kk], alpha = 0.75, linewidth = 1,)
	ax0.fill_between( low_M20_r[kk] / 1e3, y1 = low_M20_sb[kk] - off_sb - low_M20_err[kk], 
		y2 = low_M20_sb[kk] - off_sb + low_M20_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( hi_M20_r[kk] / 1e3, hi_M20_sb[kk] - off_sb, ls = '-', color = color_s[kk], alpha = 0.75, linewidth = 1,)
	ax0.fill_between( hi_M20_r[kk] / 1e3, y1 = hi_M20_sb[kk] - off_sb - hi_M20_err[kk], 
		y2 = hi_M20_sb[kk] - off_sb + hi_M20_err[kk], color = color_s[kk], alpha = 0.15,)


	ax0.plot( Pcen_low_r[kk] / 1e3, Pcen_low_sb[kk] + off_sb, ls = '--', color = color_s[kk], alpha = 0.75, linewidth = 3,)
	ax0.fill_between( Pcen_low_r[kk] / 1e3, y1 = Pcen_low_sb[kk] + off_sb - Pcen_low_err[kk], 
		y2 = Pcen_low_sb[kk] + off_sb + Pcen_low_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot( Pcen_hi_r[kk] / 1e3, Pcen_hi_sb[kk] + off_sb, ls = '-', color = color_s[kk], alpha = 0.75, linewidth = 3,)
	ax0.fill_between( Pcen_hi_r[kk] / 1e3, y1 = Pcen_hi_sb[kk] + off_sb - Pcen_hi_err[kk], 
		y2 = Pcen_hi_sb[kk] + off_sb + Pcen_hi_err[kk], color = color_s[kk], alpha = 0.15,)

legend_1 = ax0.legend( [ fig_name[0] + ' w/o $P_{cen}$ cut', fig_name[1] + ' w/o $P_{cen}$ cut', 
						 fig_name_0[0] + ' w/o $P_{cen}$ cut', fig_name_0[1] + ' w/o $P_{cen}$ cut',
						 fig_name[0] + ' with $P_{cen}$ cut', fig_name[1] + ' with $P_{cen}$ cut'],
						loc = 1, frameon = False, fontsize = 10,)
legend_0 = ax0.legend( loc = 3, frameon = False, fontsize = 10,)
ax0.add_artist( legend_1 )

ax0.fill_betweenx( y = np.linspace( 10, 40, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax0.text( 3.5e-3, 28, s = 'PSF', fontsize = 13,)
ax0.axvline( x = 0.3, ls = ':', color = 'k', alpha = 0.75, ymin = 0.0, ymax = 0.65)

ax0.set_xlim( 3e-3, 1e0 )
ax0.set_ylim( 16.5, 36.5 )
ax0.invert_yaxis()

ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
ax0.set_xticks([ 1e-2, 1e-1, 1e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax0.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/Mass-bin_SB_compare.png', dpi = 300)
plt.close()
"""


fig = plt.figure( figsize = (5.4, 5.4) )
ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )
sub_ax2 = fig.add_axes( [0.70, 0.68, 0.25, 0.25] )

ax1.plot( aveg_R / 1e3, aveg_surf_m, ls = '-.', color = 'k', alpha = 0.85, label = 'All clusters',)
# ax1.fill_between( aveg_R / 1e3, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'k', alpha = 0.15,)


ax1.plot( lo_sm_R / 1e3, lo_sm_M, ls = line_s[0], color = line_c[0], alpha = 0.85, label = fig_name[0])# + ' w/o $P_{cen}$ cut',)
ax1.fill_between( lo_sm_R / 1e3, y1 = lo_sm_M - lo_sm_err, y2 = lo_sm_M + lo_sm_err, color = line_c[0], alpha = 0.15,)

ax1.plot( lo_M20_R / 1e3, lo_M20_surf_M, ls = line_s[1], color = line_c[0], alpha = 0.75, label = fig_name_0[0])# + ' w/o $P_{cen}$ cut',)
# ax1.fill_between( lo_M20_R / 1e3, y1 = lo_M20_surf_M - lo_M20_surf_M_err, 
# 					y2 = lo_M20_surf_M + lo_M20_surf_M_err, color = line_c[0], alpha = 0.15,)

ax1.plot( Pcen_lo_sm_R / 1e3, Pcen_lo_sm_M, ls = line_s[2], color = line_c[0], alpha = 0.75, 
			label = fig_name[0] + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)
# ax1.fill_between( Pcen_lo_sm_R / 1e3, y1 = Pcen_lo_sm_M - Pcen_lo_sm_err, 
# 					y2 = Pcen_lo_sm_M + Pcen_lo_sm_err, color = line_c[0], alpha = 0.15,)


ax1.plot( hi_sm_R / 1e3, hi_sm_M, ls = line_s[0], color = line_c[1], alpha = 0.85, label = fig_name[1])# + ' w/o $P_{cen}$ cut',)
ax1.fill_between( hi_sm_R / 1e3, y1 = hi_sm_M - hi_sm_err, y2 = hi_sm_M + hi_sm_err, color = line_c[1], alpha = 0.15,)

ax1.plot( hi_M20_R / 1e3, hi_M20_surf_M, ls = line_s[1], color = line_c[1], alpha = 0.75, label = fig_name_0[1])# + ' w/o $P_{cen}$ cut',)
# ax1.fill_between( hi_M20_R / 1e3, y1 = hi_M20_surf_M - hi_M20_surf_M_err, 
# 					y2 = hi_M20_surf_M + hi_M20_surf_M_err, color = line_c[1], alpha = 0.15,)

ax1.plot( Pcen_hi_sm_R / 1e3, Pcen_hi_sm_M, ls = line_s[2], color = line_c[1], alpha = 0.75, 
			label = fig_name[1] + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)
# ax1.fill_between( Pcen_hi_sm_R / 1e3, y1 = Pcen_hi_sm_M - Pcen_hi_sm_err, 
# 					y2 = Pcen_hi_sm_M + Pcen_hi_sm_err, color = line_c[1], alpha = 0.15,)

ax1.set_xlim( 1e-2, 1e0 )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim( 5e3, 3e8 )
ax1.legend( loc = 3, frameon = False, fontsize = 11, )
ax1.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 15,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

bin_edge = np.linspace(0, 1, 25)

sub_ax2.hist( lo_Pcen, bins = bin_edge, density = True, color = 'b', histtype = 'step', alpha = 0.75,)
sub_ax2.hist( hi_Pcen, bins = bin_edge, density = True, color = 'r', histtype = 'step', alpha = 0.75,)
sub_ax2.axvline( x = Pcen_lim, ls = '--', color = 'k', alpha = 0.75,)
sub_ax2.set_xlabel('$P_{ \\mathrm{cen} }$', fontsize = 13,)
sub_ax2.set_ylabel('f', rotation = 'horizontal', fontsize = 13,)
sub_ax2.set_ylim( 4e-2, 5e1 )
sub_ax2.set_yscale('log')
sub_ax2.set_xlim( 0., 1. )

sub_ax2.text( 0.58, 0.10, s = '0.85', fontsize = 11,)
sub_ax2.set_yticks([0.1, 1, 10])
sub_ax2.set_yticklabels( labels = [] )
sub_ax2.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 10,)


sub_ax1.plot( lo_eta_R / 1e3, lo_eta_M, ls = line_s[0], color = line_c[0], alpha = 0.75,)
sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta_M - lo_eta_M_err, y2 = lo_eta_M + lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta_M, ls = line_s[0], color = line_c[1], alpha = 0.75,)
sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta_M - hi_eta_M_err, y2 = hi_eta_M + hi_eta_M_err, color = line_c[1], alpha = 0.15,)


sub_ax1.plot( lo_M20_eta_R / 1e3, lo_M20_eta_M, ls = line_s[1], color = line_c[0], alpha = 0.75,)
# sub_ax1.fill_between( lo_M20_eta_R / 1e3, y1 = lo_M20_eta_M - lo_M20_eta_M_err, 
# 				y2 = lo_M20_eta_M + lo_M20_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_M20_eta_R / 1e3, hi_M20_eta_M, ls = line_s[1], color = line_c[1], alpha = 0.75,)
# sub_ax1.fill_between( hi_M20_eta_R / 1e3, y1 = hi_M20_eta_M - hi_M20_eta_M_err, 
# 				y2 = hi_M20_eta_M + hi_M20_eta_M_err, color = line_c[1], alpha = 0.15,)


sub_ax1.plot( Pcen_lo_eta_R / 1e3, Pcen_lo_eta_M, ls = line_s[2], color = line_c[0], alpha = 0.75,)
# sub_ax1.fill_between( Pcen_lo_eta_R / 1e3, y1 = Pcen_lo_eta_M - Pcen_lo_eta_M_err, 
# 				y2 = Pcen_lo_eta_M + Pcen_lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( Pcen_hi_eta_R / 1e3, Pcen_hi_eta_M, ls = line_s[2], color = line_c[1], alpha = 0.75,)
# sub_ax1.fill_between( Pcen_hi_eta_R / 1e3, y1 = Pcen_hi_eta_M - Pcen_hi_eta_M_err, 
# 				y2 = Pcen_hi_eta_M + Pcen_hi_eta_M_err, color = line_c[1], alpha = 0.15,)


sub_ax1.fill_betweenx( y = np.logspace(-5, 5), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
sub_ax1.axhline( y = 1, ls = '-.', color = 'k', alpha = 0.85,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)
sub_ax1.set_ylim( 0.44, 1.62 )
sub_ax1.set_ylabel('$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{\\mathrm{All} }$', fontsize = 15,)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
sub_ax1.set_xticks( x_tick_arr )
sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.set_xticklabels( labels = [],)

# plt.savefig('/home/xkchen/M-bin_SM_compare%s.png' % dered_str, dpi = 300)
plt.savefig('/home/xkchen/M-bin_SM_compare.pdf', dpi = 300)
plt.close()

