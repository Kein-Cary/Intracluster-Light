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
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']

cat_lis_0 = ['low-lgM20', 'hi-lgM20']
fig_name_0 = ['Low $ M_{\\ast, \, 20}$', 'High $ M_{\\ast, \, 20}$']


### Mass profile compare
id_dered = True
dered_str = '_with-dered'

##. overall sample
dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi%s.csv' % dered_str)
aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


##. SM(R), ( low and high BCG Mstar bin )
surf_M_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[0], dered_str),)
lo_sm_R = np.array( surf_M_dat['R'] )
lo_sm_M, lo_sm_err = np.array( surf_M_dat['mean_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )

surf_M_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[1], dered_str),)
hi_sm_R = np.array( surf_M_dat['R'] )
hi_sm_M, hi_sm_err = np.array( surf_M_dat['mean_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )


eta_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], dered_str),)
lo_eta_R, lo_eta_M, lo_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['mean_ref_M/M_tot'] ), np.array( eta_dat['mean_ref_M/M_tot-err'] )

eta_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], dered_str),)
hi_eta_R, hi_eta_M, hi_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['mean_ref_M/M_tot'] ), np.array( eta_dat['mean_ref_M/M_tot-err'] )



##. SM( R ), divided by stellar mass within 20kpc (M20)
dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/aperM_bin/SM_profile/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis_0[0], dered_str),)
lo_M20_R, lo_M20_surf_M, lo_M20_surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'])

dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/aperM_bin/SM_profile/' + 
						'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis_0[1], dered_str),)
hi_M20_R, hi_M20_surf_M, hi_M20_surf_M_err = np.array( dat['R'] ), np.array( dat['mean_correct_surf_M'] ), np.array( dat['surf_M_err'])


eta_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/aperM_bin/SM_profile/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis_0[0], dered_str),)
lo_M20_eta_R, lo_M20_eta_M, lo_M20_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['mean_ref_M/M_tot'] ), np.array( eta_dat['mean_ref_M/M_tot-err'])

eta_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/aperM_bin/SM_profile/' + 
						'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis_0[1], dered_str),)
hi_M20_eta_R, hi_M20_eta_M, hi_M20_eta_M_err = np.array( eta_dat['R'] ), np.array( eta_dat['mean_ref_M/M_tot'] ), np.array( eta_dat['mean_ref_M/M_tot-err'])



##. SM( R ), P-cen limited subsamples
lo_pat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/low_BCG_star-Mass_gri-common_Pcen_cat.csv')
lo_Pcen = np.array( lo_pat['P_cen_0'] )

hi_pat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/high_BCG_star-Mass_gri-common_Pcen_cat.csv')
hi_Pcen = np.array( hi_pat['P_cen_0'] )


surf_M_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/' + 
							'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[0], dered_str),)
Pcen_lo_sm_R = np.array( surf_M_dat['R'] )
Pcen_lo_sm_M, Pcen_lo_sm_err = np.array( surf_M_dat['mean_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )

surf_M_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/' + 
							'%s_gri-band-based_corrected_aveg-jack_mass-Lumi%s.csv' % (cat_lis[1], dered_str),)
Pcen_hi_sm_R = np.array( surf_M_dat['R'] )
Pcen_hi_sm_M, Pcen_hi_sm_err = np.array( surf_M_dat['mean_correct_surf_M'] ), np.array( surf_M_dat['surf_M_err'] )


eta_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/' + 
							'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], dered_str),)
Pcen_lo_eta_R, Pcen_lo_eta_M, Pcen_lo_eta_M_err = ( np.array( eta_dat['R'] ), np.array( eta_dat['mean_ref_M/M_tot'] ), 
														np.array( eta_dat['mean_ref_M/M_tot-err'] ) )

eta_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/' + 
							'%s_gri-band_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], dered_str),)
Pcen_hi_eta_R, Pcen_hi_eta_M, Pcen_hi_eta_M_err = ( np.array( eta_dat['R'] ), np.array( eta_dat['mean_ref_M/M_tot'] ), 
														np.array( eta_dat['mean_ref_M/M_tot-err'] ) )


##. SM(r) ratio to entire sample SM(r) with Pcen cut
# lo_eat_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/' + 
# 							'%s_gri-band_Pcen_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[0], dered_str),)
# cp_lo_eta_R, cp_lo_eta_M, cp_lo_eta_M_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['mean_ref_M/M_tot']), np.array(lo_eat_dat['mean_ref_M/M_tot-err'])

# hi_eat_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_cut/SM_profile/' + 
# 							'%s_gri-band_Pcen_corrected-aveg-M-ratio_to_total-sample%s.csv' % (cat_lis[1], dered_str),)
# cp_hi_eta_R, cp_hi_eta_M, cp_hi_eta_M_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['mean_ref_M/M_tot']), np.array(hi_eat_dat['mean_ref_M/M_tot-err'])


lo_tmp_SM_F = interp.interp1d( Pcen_lo_sm_R, Pcen_lo_sm_M, kind = 'linear', fill_value = 'extrapolate',)
lo_tmp_err_F = interp.interp1d( Pcen_lo_sm_R, Pcen_lo_sm_err, kind = 'linear', fill_value = 'extrapolate',)

hi_tmp_SM_F = interp.interp1d( Pcen_hi_sm_R, Pcen_hi_sm_M, kind = 'linear', fill_value = 'extrapolate',)
hi_tmp_err_F = interp.interp1d( Pcen_hi_sm_R, Pcen_hi_sm_M, kind = 'linear', fill_value = 'extrapolate',)

tot_Pcen_R = Pcen_hi_sm_R + 0.
# tot_Pcen_SM = ( lo_tmp_SM_F( tot_Pcen_R ) * (1 / lo_tmp_err_F( tot_Pcen_R )**2 ) + Pcen_hi_sm_M * ( 1 / Pcen_hi_sm_err**2) ) / ( 1 / lo_tmp_err_F( tot_Pcen_R )**2 + 1 / Pcen_hi_sm_err**2 )
tot_Pcen_SM = ( lo_tmp_SM_F( tot_Pcen_R )  + Pcen_hi_sm_M ) / 2
tot_Pcen_SM_err = np.sqrt( lo_tmp_err_F( tot_Pcen_R )**2 + Pcen_hi_sm_err**2 )

cp_lo_eta_M = lo_tmp_SM_F( tot_Pcen_R ) / tot_Pcen_SM
cp_hi_eta_M = Pcen_hi_sm_M / tot_Pcen_SM




### === figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r']
line_s = [ '-', '--', ':']

Pcen_lim = 0.85


fig = plt.figure( figsize = (5.4, 5.4) )
ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )
sub_ax2 = fig.add_axes( [0.70, 0.68, 0.25, 0.25] )

# ax1.plot( aveg_R / 1e3, aveg_surf_m, ls = '-.', color = 'k', alpha = 0.85, label = 'All clusters',)
# ax1.fill_between( aveg_R / 1e3, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'k', alpha = 0.15,)


ax1.plot( lo_sm_R / 1e3, lo_sm_M, ls = line_s[0], color = line_c[0], alpha = 0.85, label = fig_name[0])# + ' w/o $P_{cen}$ cut',)
ax1.fill_between( lo_sm_R / 1e3, y1 = lo_sm_M - lo_sm_err, y2 = lo_sm_M + lo_sm_err, color = line_c[0], alpha = 0.15,)

ax1.plot( lo_M20_R / 1e3, lo_M20_surf_M, ls = line_s[1], color = line_c[0], alpha = 0.75, label = fig_name_0[0])# + ' w/o $P_{cen}$ cut',)
# ax1.fill_between( lo_M20_R / 1e3, y1 = lo_M20_surf_M - lo_M20_surf_M_err, 
# 					y2 = lo_M20_surf_M + lo_M20_surf_M_err, color = line_c[0], alpha = 0.15,)


ax1.plot( hi_sm_R / 1e3, hi_sm_M, ls = line_s[0], color = line_c[1], alpha = 0.85, label = fig_name[1])# + ' w/o $P_{cen}$ cut',)
ax1.fill_between( hi_sm_R / 1e3, y1 = hi_sm_M - hi_sm_err, y2 = hi_sm_M + hi_sm_err, color = line_c[1], alpha = 0.15,)

ax1.plot( hi_M20_R / 1e3, hi_M20_surf_M, ls = line_s[1], color = line_c[1], alpha = 0.75, label = fig_name_0[1])# + ' w/o $P_{cen}$ cut',)
# ax1.fill_between( hi_M20_R / 1e3, y1 = hi_M20_surf_M - hi_M20_surf_M_err, 
# 					y2 = hi_M20_surf_M + hi_M20_surf_M_err, color = line_c[1], alpha = 0.15,)


# ax1.plot( Pcen_lo_sm_R / 1e3, Pcen_lo_sm_M, ls = line_s[2], color = line_c[0], alpha = 0.75, 
# 			label = fig_name[0] + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)
# ax1.plot( Pcen_hi_sm_R / 1e3, Pcen_hi_sm_M, ls = line_s[2], color = line_c[1], alpha = 0.75, 
# 			label = fig_name[1] + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)

ax1.plot( Pcen_lo_sm_R / 1e3, Pcen_lo_sm_M, ls = line_s[2], color = line_c[0], alpha = 0.75, label = fig_name[0] + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)
ax1.plot( Pcen_hi_sm_R / 1e3, Pcen_hi_sm_M, ls = line_s[2], color = line_c[1], alpha = 0.75, label = fig_name[1] + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)

# ax1.plot( tot_Pcen_R / 1e3, tot_Pcen_SM, ls = ':', color = 'k', alpha = 0.65, lw = 2.75, label = 'All clusters' + '$\,( P_{cen} \\geq %.2f)$' % Pcen_lim,)


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

sub_ax2.set_xlabel('$P_{ \\mathrm{cen} }$', fontsize = 13,)
sub_ax2.set_ylabel('f', rotation = 'horizontal', fontsize = 13,)
sub_ax2.set_ylim( 4e-2, 5e1 )
sub_ax2.set_yscale('log')
sub_ax2.set_xlim( 0., 1. )

sub_ax2.axvline( x = Pcen_lim, ls = '--', color = 'k', alpha = 0.75,)

sub_ax2.set_yticks([0.1, 1, 10])
sub_ax2.set_yticklabels( labels = [] )
sub_ax2.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 10,)


sub_ax1.plot( lo_eta_R / 1e3, lo_eta_M, ls = line_s[0], color = line_c[0], alpha = 0.75,)
sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta_M - lo_eta_M_err, y2 = lo_eta_M + lo_eta_M_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta_M, ls = line_s[0], color = line_c[1], alpha = 0.75,)
sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta_M - hi_eta_M_err, y2 = hi_eta_M + hi_eta_M_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( lo_M20_eta_R / 1e3, lo_M20_eta_M, ls = line_s[1], color = line_c[0], alpha = 0.75,)
sub_ax1.plot( hi_M20_eta_R / 1e3, hi_M20_eta_M, ls = line_s[1], color = line_c[1], alpha = 0.75,)


##..
# sub_ax1.plot( Pcen_lo_eta_R / 1e3, Pcen_lo_eta_M, ls = line_s[2], color = line_c[0], alpha = 0.75,)
# sub_ax1.plot( Pcen_hi_eta_R / 1e3, Pcen_hi_eta_M, ls = line_s[2], color = line_c[1], alpha = 0.75,)

# sub_ax1.plot( cp_lo_eta_R / 1e3, cp_lo_eta_M, ls = line_s[2], color = line_c[0], alpha = 0.75,)
# sub_ax1.plot( cp_hi_eta_R / 1e3, cp_hi_eta_M, ls = line_s[2], color = line_c[1], alpha = 0.75,)

sub_ax1.plot( tot_Pcen_R / 1e3, cp_lo_eta_M, ls = line_s[2], color = line_c[0], alpha = 0.75,)
sub_ax1.plot( tot_Pcen_R / 1e3, cp_hi_eta_M, ls = line_s[2], color = line_c[1], alpha = 0.75,)


sub_ax1.fill_betweenx( y = np.logspace(-5, 5), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
sub_ax1.axhline( y = 1, ls = '-.', color = 'k', alpha = 0.85,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15,)

# sub_ax1.set_ylim( 0.44, 1.62 )
sub_ax1.set_ylim( 0.45, 1.55 )

sub_ax1.set_ylabel('$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{\\mathrm{All} }$', fontsize = 15,)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
sub_ax1.set_xticks( x_tick_arr )
sub_ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.set_xticklabels( labels = [],)

# plt.savefig('/home/xkchen/M-bin_SM_compare%s.png' % dered_str, dpi = 300)
plt.savefig('/home/xkchen/M_bin_SM_compare.pdf', dpi = 300)
plt.close()


##... profiles compare
dx0 = ( hi_eta_R >= 10 ) & ( hi_eta_R <= 200 )
D_eta_hi_M = hi_eta_M[ dx0 ]

dx1 = ( hi_M20_eta_R >= 10 ) & ( hi_M20_eta_R <= 200 )
D_eta_hi_M20 = hi_M20_eta_M[ dx1 ]

hi_dpx = ( D_eta_hi_M - D_eta_hi_M20 ) / D_eta_hi_M


dx0 = ( lo_eta_R >= 10 ) & ( lo_eta_R <= 200 )
D_eta_lo_M = lo_eta_M[ dx0 ]

dx1 = ( lo_M20_eta_R >= 10 ) & ( lo_M20_eta_R <= 200 )
D_eta_lo_M20 = lo_M20_eta_M[ dx1 ]

lo_dpx = ( D_eta_lo_M - D_eta_lo_M20 ) / D_eta_lo_M


