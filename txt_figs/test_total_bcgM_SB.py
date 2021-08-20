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

from color_2_mass import jk_sub_SB_func, jk_sub_Mass_func
from color_2_mass import aveg_mass_pro_func

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

BG_path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/'
path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/SBs/'

### === ### BG estimate and BG-subtraction (based on diag-err only)
"""
for kk in range( 3 ):

	with h5py.File( 
		path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[kk], 'r') as f:
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

	out_params_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]
	out_pros_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_diag-fit.csv' % band[kk]

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
	plt.savefig('/home/xkchen/tot-BCG-star-Mass_%s-band_SB_n=2.1-sersic.png' % band[kk], dpi = 300)
	plt.close()

N_bin = 30

for kk in range( 3 ):
	jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_%s-band_' % band[kk] + 'jack-sub-%d_SB-pro_z-ref.h5'

	sb_out_put = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk]
	BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]
	BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, band[ kk ], BG_file,)
"""

#... color of jack-sub sample and mean of jack average
def sub_color_f():

	N_bin = 30
	for kk in range( 3 ):

		jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_%s-band_' % band[kk] + 'jack-sub-%d_SB-pro_z-ref.h5'
		BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]
		out_sub_sb = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_' % band[kk] + 'jack-sub-%d_BG-sub_SB.csv'

		jk_sub_SB_func( N_bin, jk_sub_sb, BG_file, out_sub_sb )

		print('kk = ', kk)

	tmp_r, tmp_gr, tmp_gi, tmp_ri = [], [], [], []

	for ll in range( N_bin ):

		p_r_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_r-band_jack-sub-%d_BG-sub_SB.csv' % ll )
		tt_r_R, tt_r_sb, tt_r_err = np.array( p_r_dat['R'] ), np.array( p_r_dat['BG_sub_SB'] ), np.array( p_r_dat['sb_err'] )

		p_g_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_g-band_jack-sub-%d_BG-sub_SB.csv' % ll )
		tt_g_R, tt_g_sb, tt_g_err = np.array( p_g_dat['R'] ), np.array( p_g_dat['BG_sub_SB'] ), np.array( p_g_dat['sb_err'] )

		p_i_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_i-band_jack-sub-%d_BG-sub_SB.csv' % ll )
		tt_i_R, tt_i_sb, tt_i_err = np.array( p_i_dat['R'] ), np.array( p_i_dat['BG_sub_SB'] ), np.array( p_i_dat['sb_err'] )


		idR_lim = tt_r_R <= 1.2e3
		tt_r_R, tt_r_sb, tt_r_err = tt_r_R[ idR_lim], tt_r_sb[ idR_lim], tt_r_err[ idR_lim]

		idR_lim = tt_g_R <= 1.2e3
		tt_g_R, tt_g_sb, tt_g_err = tt_g_R[ idR_lim], tt_g_sb[ idR_lim], tt_g_err[ idR_lim] 

		idR_lim = tt_i_R <= 1.2e3
		tt_i_R, tt_i_sb, tt_i_err = tt_i_R[ idR_lim], tt_i_sb[ idR_lim], tt_i_err[ idR_lim] 

		gr_arr, gr_err = color_func( tt_g_sb, tt_g_err, tt_r_sb, tt_r_err )
		gi_arr, gi_err = color_func( tt_g_sb, tt_g_err, tt_i_sb, tt_i_err )
		ri_arr, ri_err = color_func( tt_r_sb, tt_r_err, tt_i_sb, tt_i_err )

		keys = [ 'R', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err' ]
		values = [ tt_g_R, gr_arr, gr_err, gi_arr, gi_err, ri_arr, ri_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( BG_path + 'tot-BCG-star-Mass_jack-sub-%d_color_profile.csv' % ll )

		tmp_r.append( tt_g_R )
		tmp_gr.append( gr_arr )
		tmp_gi.append( gi_arr )
		tmp_ri.append( ri_arr )

	aveg_R, aveg_gr, aveg_gr_err = arr_jack_func( tmp_gr, tmp_r, N_bin )[:3]
	aveg_R, aveg_gi, aveg_gi_err = arr_jack_func( tmp_gi, tmp_r, N_bin )[:3]
	aveg_R, aveg_ri, aveg_ri_err = arr_jack_func( tmp_ri, tmp_r, N_bin )[:3]

	keys = [ 'R_kpc', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err' ]
	values = [ aveg_R, aveg_gr, aveg_gr_err, aveg_gi, aveg_gi_err, aveg_ri, aveg_ri_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( BG_path + 'tot-BCG-star-Mass_color_profile.csv')

#...surface mass profile
def surf_M_func():

	N_samples = 30

	band_str = 'ri' # 'gi'
	c_inv = False

	sub_SB_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_jack-sub-%d_BG-sub_SB.csv'
	low_R_lim, up_R_lim = 1e0, 1.2e3
	out_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv'
	jk_sub_Mass_func( N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl_ref, z_ref, c_inv = c_inv)

	jk_sub_m_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv'
	jk_aveg_m_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str

	lgM_cov_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str
	M_cov_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % band_str
	aveg_mass_pro_func(N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file, M_cov_file = M_cov_file)	

	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
	aveg_R = np.array(dat['R'])

	aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
	aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])
	aveg_lumi, aveg_lumi_err = np.array(dat['lumi']), np.array(dat['lumi_err'])

	plt.figure()
	plt.title('%s-based surface mass density profile' % band_str )
	plt.plot( aveg_R, aveg_surf_m, ls = '-', color = 'r', alpha = 0.5,)
	plt.fill_between( aveg_R, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, 
		color = 'r', alpha = 0.12,)

	plt.xlim(1e0, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15)
	plt.yscale('log')
	plt.ylim(1e3, 2e9)
	plt.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/tot_%s-band_based_surface_mass_profile.png' % band_str, dpi = 300)
	plt.close()

sub_color_f()

# surf_M_func()

### === ### BG-sub SB and color
nbg_tot_r, nbg_tot_sb, nbg_tot_err = [], [], []
nbg_tot_mag, nbg_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_tot_r.append( tt_r )
	nbg_tot_sb.append( tt_sb )
	nbg_tot_err.append( tt_err )
	nbg_tot_mag.append( tt_mag )
	nbg_tot_mag_err.append( tt_mag_err )

nbg_tot_r = np.array( nbg_tot_r )
nbg_tot_r = nbg_tot_r / 1e3

##...Z05, SB profile
Z05_r, Z05_sb, Z05_mag = [], [], []

for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	Z05_r.append( R_obs )
	Z05_sb.append( flux_obs )
	Z05_mag.append( SB_obs )

Z05_r = np.array( Z05_r )
Z05_r = Z05_r / 1e3

##...Z05, color
pdat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_g-r.csv')
r_05_0, g2r_05 = np.array(pdat_0['(R)^(1/4)']), np.array(pdat_0['g-r'])
r_05_0 = r_05_0**4
r_05_0 = r_05_0 / 1e3

pdat_2 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_r-i.csv')
r_05_1, r2i_05 = np.array(pdat_2['(R)^(1/4)']), np.array(pdat_2['r-i'])
r_05_1 = r_05_1**4
r_05_1 = r_05_1 / 1e3

pdat_1 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zhang_SB/Z19_color.csv')
r_19, g2r_19 = np.array(pdat_1['R_kpc']), np.array(pdat_1['g-r'])
r_19 = r_19 / 1e3

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

## g-r color profile
c_dat = pds.read_csv( BG_path + 'tot-BCG-star-Mass_color_profile.csv' )
tot_c_R, tot_g2r, tot_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )

sm_tot_g2r = signal.savgol_filter( tot_g2r, 7, 3)
sm_tot_r = tot_c_R / 1e3
sm_tot_g2r_err = tot_g2r_err


## mass profile
dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_gi-band-based_corrected_aveg-jack_mass-Lumi.csv')
aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['correct_surf_M']), np.array(dat['surf_M_err'])

phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3

# fig = plt.figure( figsize = (10.6, 4.8) )
# ax0 = fig.add_axes([0.07, 0.12, 0.42, 0.85])
# ax1 = fig.add_axes([0.56, 0.12, 0.42, 0.85])

fig = plt.figure( figsize = (15.40, 4.8) )
ax0 = fig.add_axes([0.05, 0.12, 0.275, 0.85])
ax2 = fig.add_axes([0.38, 0.12, 0.275, 0.85])
ax1 = fig.add_axes([0.71, 0.12, 0.275, 0.85])

for kk in ( 2, 0, 1 ):

	ax0.plot( nbg_tot_r[kk], nbg_tot_mag[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax0.fill_between( nbg_tot_r[kk], y1 = nbg_tot_mag[kk] - nbg_tot_mag_err[kk], 
		y2 = nbg_tot_mag[kk] + nbg_tot_mag_err[kk], color = color_s[kk], alpha = 0.15,)
	ax0.plot(Z05_r[kk], Z05_mag[kk], ls = '--', color = color_s[kk], alpha = 0.75,)

legend_1 = ax0.legend( [ 'This work (redMaPPer)', '$\\mathrm{Zibetti}{+}2005$ (maxBCG)' ], loc = 3, frameon = False, fontsize = 15,)
legend_0 = ax0.legend( loc = 1, frameon = False, fontsize = 15,)
ax0.add_artist( legend_1 )

ax0.fill_betweenx( np.linspace(19, 36, 100), x1 = phyR_psf.max(), x2 = 0, color = 'k', alpha = 0.12,)
ax0.text( 3.5e-3, 26.5, s = 'PSF', fontsize = 15,)

ax0.set_ylim( 20, 34 )
ax0.invert_yaxis()

ax0.set_xlim( 3e-3, 1e0)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)

ax0.set_xticks([ 1e-2, 1e-1, 1e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax0.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)

ax1.plot( sm_tot_r, sm_tot_g2r, ls = '-', color = 'r', alpha = 0.75, label = 'This work (SDSS)')
ax1.fill_between( sm_tot_r, y1 = sm_tot_g2r - sm_tot_g2r_err, y2 = sm_tot_g2r + sm_tot_g2r_err, color = 'r', alpha = 0.12,)

ax1.plot(r_05_0, g2r_05, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{Zibetti}{+}2005$ (SDSS)')
ax1.plot(r_19, g2r_19, ls = '--', color = 'c', alpha = 0.75, label = '$\\mathrm{Zhang}{+}2019$ (DES)')
ax1.fill_betweenx( np.linspace(0, 2, 100), x1 = phyR_psf.max(), x2 = 0, color = 'k', alpha = 0.12,)


ax1.legend( loc = 3, frameon = False, fontsize = 15,)
ax1.set_ylim( 0.95, 1.55 )

ax1.set_xlim( 3e-3, 1e0)
ax1.set_xscale('log')
ax1.set_ylabel('$ g - r $', fontsize = 17,)
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)

ax1.set_xticks([ 1e-2, 1e-1, 1e0])
ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

ax2.plot( aveg_R / 1e3, aveg_surf_m, ls = '-', color = 'r', alpha = 0.5,)
ax2.fill_between( aveg_R / 1e3, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, 
	color = 'r', alpha = 0.12,)

ax2.set_xlim( 3e-3, 1e0)
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax2.set_yscale('log')
ax2.set_ylim(2e4, 2e9)
ax2.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2]$', fontsize = 15,)
ax2.legend( loc = 1, frameon = False, fontsize = 15,)

ax2.set_xticks([ 1e-2, 1e-1, 1e0])
ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax2.fill_betweenx( y = np.logspace(3, 9.8,200), x1 = phyR_psf.max(), x2 = 0, color = 'k', alpha = 0.12,)
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/total_sample_SB_compare.png', dpi = 300)
plt.close()

raise

last_Z05_r, last_Z05_sb = [], []

for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_sub_unmask.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	last_Z05_r.append( R_obs )
	last_Z05_sb.append( flux_obs )

###############################
##...pre, result
pre_path = '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/'

pre_tot_r, pre_tot_sb, pre_tot_err = [], [], []

for kk in range( 3 ):

	with h5py.File( pre_path + 'photo-z_total_star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	pre_tot_r.append( tt_r )
	pre_tot_sb.append( tt_sb )
	pre_tot_err.append( tt_err )


fig = plt.figure( figsize = (19.90, 5.0) )
ax0 = fig.add_axes([0.05, 0.13, 0.28, 0.80])
ax1 = fig.add_axes([0.38, 0.13, 0.28, 0.80])
ax2 = fig.add_axes([0.71, 0.13, 0.28, 0.80])

ylims = [ [7e-5, 8e0], [5e-5, 2e0], [1e-4, 1.5e1] ]

for kk in ( 1, 0, 2 ):

	if kk == 0:
		ax = ax1
	if kk == 1:
		ax = ax0
	if kk == 2:
		ax = ax2

	ax.plot(nbg_tot_r[kk], nbg_tot_sb[kk], ls = '-', color = color_s[kk], alpha = 0.5, label = 'this work')
	ax.fill_between(nbg_tot_r[kk], y1 = nbg_tot_sb[kk] - nbg_tot_err[kk], 
		y2 = nbg_tot_sb[kk] + nbg_tot_err[kk], color = color_s[kk], alpha = 0.12,)

	# ax.plot(pre_tot_r[kk], pre_tot_sb[kk], ls = ':', color = color_s[kk], alpha = 0.5, label = 'previous')
	# ax.fill_between(pre_tot_r[kk], y1 = pre_tot_sb[kk] - pre_tot_err[kk], 
	# 	y2 = pre_tot_sb[kk] + pre_tot_err[kk], color = color_s[kk], alpha = 0.12,)

	ax.plot(Z05_r[kk], Z05_sb[kk], ls = '--', color = 'k', alpha = 0.5, label = 'Z05, Raw BCG + ICL',)
	ax.plot( last_Z05_r[kk], last_Z05_sb[kk], ls = '-', color = 'k', alpha = 0.5, label = 'Z05, Pure BCG + ICL',)

	ax.annotate(text = '%s band' % band[kk], xy = (0.80, 0.90), xycoords = 'axes fraction', color = 'k', fontsize = 15,)
	ax.axvline( x = phyR_psf[kk], ls = ':', color = color_s[kk], alpha = 0.5, ymin = 0.75, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')

	ax.set_xlim(1e0, 1e3)
	ax.set_ylim( ylims[kk][0], ylims[kk][1] )

	ax.set_yscale('log')

	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]', fontsize = 15,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 15, )
	ax.legend( loc = 3, fontsize = 15, frameon = False,)

plt.savefig('/home/xkchen/total_BCG-M_BG-sub-SB.png', dpi = 300)
plt.close()

## g-i color
tot_g2i, tot_g2i_err = color_func( nbg_tot_sb[1], nbg_tot_err[1], nbg_tot_sb[2], nbg_tot_err[2] )

idnan = np.isnan( tot_g2i )
idx_lim = nbg_tot_r[0] < nbg_tot_r[0][idnan][0]
sm_tot_g2i = signal.savgol_filter( tot_g2i[idx_lim], 7, 3)

sm_tot_r = nbg_tot_r[0][idx_lim]
sm_tot_g2i_err = tot_g2i_err[idx_lim]

keys = [ 'R_kpc', 'g2i', 'g2i_err' ]
values = [ sm_tot_r, sm_tot_g2i, sm_tot_g2i_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( BG_path + 'total_g-i_color_profile.csv')


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( sm_tot_r, sm_tot_g2i, ls = '-', color = 'r', alpha = 0.5, linewidth = 1,)
ax.fill_between( sm_tot_r, y1 = sm_tot_g2i - sm_tot_g2i_err, y2 = sm_tot_g2i + sm_tot_g2i_err, color = 'r', alpha = 0.12,)

ax.axvline( x = phyR_psf[1], ls = ':', color = 'k', alpha = 0.5, ymin = 0.7, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')

ax.legend( loc = 3, frameon = False, fontsize = 15,)
ax.set_ylim( 1.5, 2.2 )

ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')
ax.set_ylabel('g - i', fontsize = 15,)
ax.set_xlabel('R [kpc]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/total_BCG-M_g2i_color_compare.png', dpi = 300)
plt.close()

