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
Omega_lambda = 1.- Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

band = ['r', 'g', 'i']
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === ###
def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

## fixed BCG-Mass bin

#...separated masking
pre_path = '/home/xkchen/mywork/ICL/code/BCG_M_based_cat/SB_pros/'
pre_BG_path = '/home/xkchen/mywork/ICL/code/BCG_M_based_cat/BG_pros/'

#...with combined masking
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'
# path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/SBs/'
# cat_lis = [ 'low-age', 'hi-age' ]
# fig_name = [ 'younger', 'older' ]
# file_s = 'age-bin'

BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'
path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/SBs/'
cat_lis = [ 'low-rich', 'hi-rich' ]
fig_name = [ 'low $\\lambda$', 'high $\\lambda$' ]
file_s = 'rich-bin'

### === mass profile
def surf_M_func():

	N_samples = 30

	for mm in range( 2 ):

		# band_str = 'gi'
		# for kk in (1,2): # g,i band based mass estimate

		# band_str = 'gr'
		# for kk in (1,0): # g,r band based mass estimate

		band_str = 'ri'
		for kk in (0,2): # r,i band based mass estimate

			jk_sub_sb = path + 'photo-z_match_gri-common_%s_%s-band_' % (cat_lis[ mm ], band[kk]) + 'jack-sub-%d_SB-pro_z-ref_pk-off.h5'
			BG_file = BG_path + 'photo-z_%s_%s-band_BG-profile_params_diag-fit.csv' % (cat_lis[ mm ], band[kk])
			out_sub_sb = BG_path + '%s_%s-band_' % (cat_lis[mm], band[kk]) + 'jack-sub-%d_BG-sub_SB.csv'

			jk_sub_SB_func( N_samples, jk_sub_sb, BG_file, out_sub_sb )

		sub_SB_file = BG_path + '%s_' % cat_lis[mm] + '%s-band_jack-sub-%d_BG-sub_SB.csv'
		# low_R_lim, up_R_lim = 1e1, 1e3
		low_R_lim, up_R_lim = 1e0, 1.2e3
		out_file = BG_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'

		jk_sub_Mass_func( N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl_ref, z_ref)

		jk_sub_m_file = BG_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'
		jk_aveg_m_file = BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str)
		lgM_cov_file = BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)

		aveg_mass_pro_func(N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file)

	### jack mean and figs
	dpt_R, dpt_L, dpt_M = [], [], []
	dpt_M_err, dpt_L_err = [], []
	dpt_cumu_M, dpt_cumu_M_err = [], []

	for mm in range( 2 ):

		dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
		aveg_R = np.array(dat['R'])

		aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])
		aveg_lumi, aveg_lumi_err = np.array(dat['lumi']), np.array(dat['lumi_err'])

		dpt_R.append( aveg_R )
		dpt_M.append( aveg_surf_m )
		dpt_M_err.append( aveg_surf_m_err )

		dpt_L.append( aveg_lumi )
		dpt_L_err.append( aveg_lumi_err )
		dpt_cumu_M.append( aveg_cumu_m )
		dpt_cumu_M_err.append( aveg_cumu_m_err )

	plt.figure()
	plt.title('%s-based surface mass density profile' % band_str )
	plt.plot( dpt_R[0], dpt_M[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name[0])
	plt.fill_between( dpt_R[0], y1 = dpt_M[0] - dpt_M_err[0], y2 = dpt_M[0] + dpt_M_err[0], color = line_c[0], alpha = 0.12,)
	plt.plot( dpt_R[1], dpt_M[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name[1])
	plt.fill_between( dpt_R[1], y1 = dpt_M[1] - dpt_M_err[1], y2 = dpt_M[1] + dpt_M_err[1], color = line_c[1], alpha = 0.12,)

	plt.xlim( 1e0, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15)
	plt.yscale('log')
	plt.ylim( 5e3, 2e9)
	plt.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/%s_%s-band_based_surface_mass_profile.png' % (file_s, band_str), dpi = 300)
	plt.close()


	plt.figure()
	plt.title('%s-based cumulative mass profile' % band_str )
	plt.plot( dpt_R[0], dpt_cumu_M[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name[0])
	plt.fill_between( dpt_R[0], y1 = dpt_cumu_M[0] - dpt_cumu_M_err[0], y2 = dpt_cumu_M[0] + dpt_cumu_M_err[0], color = line_c[0], alpha = 0.12,)
	plt.plot( dpt_R[1], dpt_cumu_M[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name[1])
	plt.fill_between( dpt_R[1], y1 = dpt_cumu_M[1] - dpt_cumu_M_err[1], y2 = dpt_cumu_M[1] + dpt_cumu_M_err[1], color = line_c[1], alpha = 0.12,)

	plt.xlim( 1e0, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15)
	plt.yscale('log')
	plt.ylim( 1e10, 7e11)
	plt.legend( loc = 4, frameon = False, fontsize = 15,)
	plt.ylabel('cumulative mass $[M_{\\odot}]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/%s_%s-band_based_cumulative_mass_profile.png' % (file_s, band_str), dpi = 300)
	plt.close()


	plt.figure()
	plt.title('%s band SB profile' % band_str[1] )
	plt.plot( dpt_R[0], dpt_L[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name[0])
	plt.fill_between( dpt_R[0], y1 = dpt_L[0] - dpt_L_err[0], y2 = dpt_L[0] + dpt_L_err[0], color = line_c[0], alpha = 0.12,)
	plt.plot( dpt_R[1], dpt_L[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name[1])
	plt.fill_between( dpt_R[1], y1 = dpt_L[1] - dpt_L_err[1], y2 = dpt_L[1] + dpt_L_err[1], color = line_c[1], alpha = 0.12,)

	plt.xlim( 1e0, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15)
	plt.yscale('log')
	plt.ylim( 1e3, 1e8)
	plt.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.ylabel('SB $[L_{\\odot} / kpc^2]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/%s_%s-band_based_%s-band_SB_profile.png' % (file_s, band_str, band_str[1]), dpi = 300)
	plt.close()

	return

def sub_color_f():

	N_samples = 30
	### average jack-sub sample color
	for mm in range( 2 ):

		tmp_r, tmp_gr, tmp_gi, tmp_ri = [], [], [], []
	
		for ll in range( N_samples ):

			p_r_dat = pds.read_csv( BG_path + '%s_r-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv' % ll )
			tt_r_R, tt_r_sb, tt_r_err = np.array( p_r_dat['R'] ), np.array( p_r_dat['BG_sub_SB'] ), np.array( p_r_dat['sb_err'] )

			p_g_dat = pds.read_csv( BG_path + '%s_g-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv' % ll )
			tt_g_R, tt_g_sb, tt_g_err = np.array( p_g_dat['R'] ), np.array( p_g_dat['BG_sub_SB'] ), np.array( p_g_dat['sb_err'] )

			p_i_dat = pds.read_csv( BG_path + '%s_i-band_' % cat_lis[mm] + 'jack-sub-%d_BG-sub_SB.csv' % ll )
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

			keys = [ 'R', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err']
			values = [ tt_g_R, gr_arr, gr_err, gi_arr, gi_err, ri_arr, ri_err ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( BG_path + '%s_jack-sub-%d_color_profile.csv' % (cat_lis[mm], ll),)

			tmp_r.append( tt_g_R )
			tmp_gr.append( gr_arr )
			tmp_gi.append( gi_arr )
			tmp_ri.append( ri_arr )

		aveg_R, aveg_gr, aveg_gr_err = arr_jack_func( tmp_gr, tmp_r, N_samples )[:3]
		aveg_R, aveg_gi, aveg_gi_err = arr_jack_func( tmp_gi, tmp_r, N_samples )[:3]
		aveg_R, aveg_ri, aveg_ri_err = arr_jack_func( tmp_ri, tmp_r, N_samples )[:3]

		keys = [ 'R_kpc', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err' ]
		values = [ aveg_R, aveg_gr, aveg_gr_err, aveg_gi, aveg_gi_err, aveg_ri, aveg_ri_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( BG_path + '%s_color_profile.csv' % cat_lis[mm] )

	return

# surf_M_func()

sub_color_f()

raise

###############################
### === before correction
pre_low_r, pre_low_sb, pre_low_err = [], [], []

for kk in range( 3 ):
	with h5py.File( pre_BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	pre_low_r.append( tt_r )
	pre_low_sb.append( tt_sb )
	pre_low_err.append( tt_err )

pre_hi_r, pre_hi_sb, pre_hi_err = [], [], []

for kk in range( 3 ):
	with h5py.File( pre_BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	pre_hi_r.append( tt_r )
	pre_hi_sb.append( tt_sb )
	pre_hi_err.append( tt_err )

pre_hi_g2r, pre_hi_g2r_err = color_func( pre_hi_sb[1], pre_hi_err[1], pre_hi_sb[0], pre_hi_err[0] )
pre_low_g2r, pre_low_g2r_err = color_func( pre_low_sb[1], pre_low_err[1], pre_low_sb[0], pre_low_err[0] )

idnan = np.isnan( pre_hi_g2r )
idx_lim = pre_hi_r[0] < pre_hi_r[0][idnan][0]
sm_pre_hi_g2r = signal.savgol_filter( pre_hi_g2r[idx_lim], 7, 3)

sm_pre_hi_r = pre_hi_r[0][idx_lim]
sm_pre_hi_g2r_err = pre_hi_g2r_err[idx_lim]

idnan = np.isnan( pre_low_g2r )
idx_lim = pre_low_r[0] < pre_low_r[0][idnan][0]
sm_pre_low_g2r = signal.savgol_filter( pre_low_g2r[idx_lim], 7, 3)

sm_pre_low_r = pre_low_r[0][idx_lim]
sm_pre_low_g2r_err = pre_low_g2r_err[idx_lim]

### === after correction
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

##.. g-r color
hi_g2r, hi_g2r_err = color_func( nbg_hi_sb[1], nbg_hi_err[1], nbg_hi_sb[0], nbg_hi_err[0] )
low_g2r, low_g2r_err = color_func( nbg_low_sb[1], nbg_low_err[1], nbg_low_sb[0], nbg_low_err[0] )

idnan = np.isnan( hi_g2r )
idx_lim = nbg_hi_r[0] < nbg_hi_r[0][idnan][0]
sm_hi_g2r = signal.savgol_filter( hi_g2r[idx_lim], 7, 3)

sm_hi_r = nbg_hi_r[0][idx_lim]
sm_hi_g2r_err = hi_g2r_err[idx_lim]

keys = [ 'R_kpc', 'g2r', 'g2r_err' ]
values = [ sm_hi_r, sm_hi_g2r, sm_hi_g2r_err ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( BG_path + '%s_g-r_color_profile.csv' % cat_lis[1] )


idnan = np.isnan( low_g2r )
idx_lim = nbg_low_r[0] < nbg_low_r[0][idnan][0]
sm_low_g2r = signal.savgol_filter( low_g2r[idx_lim], 7, 3)

sm_low_r = nbg_low_r[0][idx_lim]
sm_low_g2r_err = low_g2r_err[idx_lim]

keys = [ 'R_kpc', 'g2r', 'g2r_err' ]
values = [ sm_low_r, sm_low_g2r, sm_low_g2r_err ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( BG_path + '%s_g-r_color_profile.csv' % cat_lis[0] )

raise

#..figs
tmp_low_r, tmp_low_sb, tmp_low_com_sb, tmp_low_err = [], [], [], []

for kk in range( 3 ):

	fig = plt.figure( figsize = (6.8, 4.8) )
	ax = fig.add_axes([0.15, 0.25, 0.75, 0.70])
	ax1 = fig.add_axes([0.15, 0.10, 0.75, 0.15])

	ax.plot( pre_low_r[kk], pre_low_sb[kk], ls = '-', color = 'k', alpha = 0.5, label = 'separate masking, no offset correction')
	ax.fill_between( pre_low_r[kk], y1 = pre_low_sb[kk] - pre_low_err[kk], y2 = pre_low_sb[kk] + pre_low_err[kk], color = 'k', alpha = 0.12,)

	ax.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '-', color = color_s[kk], alpha = 0.5, label = 'combined masking, offset correction')
	ax.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.12,)

	idx_lim = pre_low_r[kk] <= 1.1e3
	intep_F = interp.interp1d( pre_low_r[kk][idx_lim], pre_low_sb[kk][idx_lim], kind = 'cubic', fill_value = 'extrapolate',)

	idu_lim = nbg_low_r[kk] <= 1.1e3
	lo_com_sb = intep_F( nbg_low_r[kk][idu_lim] )

	lo_cc_r = nbg_low_r[kk][idu_lim]
	lo_cc_sb = nbg_low_sb[kk][idu_lim]
	lo_cc_err = nbg_low_err[kk][idu_lim]

	tmp_low_r.append( lo_cc_r )
	tmp_low_sb.append( lo_cc_sb )
	tmp_low_com_sb.append( lo_com_sb )
	tmp_low_err.append( lo_cc_err )

	ax.set_xlim( 1e0, 2e3)

	if band[kk] == 'r':
		ax.set_ylim( 5e-5, 1e1 )
	if band[kk] == 'g':
		ax.set_ylim( 5e-5, 3e0 )
	if band[kk] == 'i':
		ax.set_ylim( 5e-5, 2e1 )

	ax.set_yscale('log')

	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax.annotate(text = '%s band' % band[kk] + '\n' + fig_name[0] + ' at fixed $M^{BCG}_{\\ast}$', 
		xy = (0.08, 0.35), xycoords = 'axes fraction', color = 'k', fontsize = 15,)

	ax.legend( loc = 1, frameon = False,)
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 12,)

	ax1.plot( pre_low_r[kk], pre_low_sb[kk] / pre_low_sb[kk], ls = '-', color = 'k', alpha = 0.5,)
	ax1.fill_between( pre_low_r[kk], y1 = (pre_low_sb[kk] - pre_low_err[kk]) / pre_low_sb[kk], 
						y2 = (pre_low_sb[kk] + pre_low_err[kk]) / pre_low_sb[kk], color = 'k', alpha = 0.12,)

	ax1.plot( lo_cc_r, lo_cc_sb / lo_com_sb, ls = '-', color = color_s[kk], alpha = 0.5,)
	ax1.fill_between( lo_cc_r, y1 = ( lo_cc_sb - lo_cc_err ) / lo_com_sb, 
		y2 = ( lo_cc_sb + lo_cc_err ) / lo_com_sb, color = color_s[kk], alpha = 0.12,)

	ax1.set_xlim( ax.get_xlim() )

	if band[kk] == 'g':
		ax1.set_ylim( 0.6, 1.10 )
	if band[kk] == 'i':
		ax1.set_ylim( 0.90, 1.10 )

	if band[kk] == 'r':
		ax1.set_ylim( 0.90, 1.10 )

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)
	ax1.set_ylabel('$ SB / SB_{w/o \; correction}$', fontsize = 12, labelpad = 10,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax.set_xticklabels( labels = [],)

	plt.savefig('/home/xkchen/%s_%s-band_sample_SB_compare.png' % (cat_lis[0], band[kk]), dpi = 300)
	plt.close()

for kk in range( 3 ):

	fig = plt.figure( figsize = (6.8, 4.8) )
	ax = fig.add_axes([0.15, 0.25, 0.75, 0.70])
	ax1 = fig.add_axes([0.15, 0.10, 0.75, 0.15])

	ax.plot( pre_hi_r[kk], pre_hi_sb[kk], ls = '-', color = 'k', alpha = 0.5, label = 'separate masking, no offset correction')
	ax.fill_between( pre_hi_r[kk], y1 = pre_hi_sb[kk] - pre_hi_err[kk], y2 = pre_hi_sb[kk] + pre_hi_err[kk], color = 'k', alpha = 0.12,)

	ax.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.5, label = 'combined masking, offset correction')
	ax.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.12,)

	idx_lim = pre_hi_r[kk] <= 1.1e3
	intep_F = interp.interp1d( pre_hi_r[kk][idx_lim], pre_hi_sb[kk][idx_lim], kind = 'cubic', fill_value = 'extrapolate',)

	idu_lim = nbg_hi_r[kk] <= 1.1e3
	com_sb = intep_F( nbg_hi_r[kk][idu_lim] )

	cc_r = nbg_hi_r[kk][idu_lim]
	cc_sb = nbg_hi_sb[kk][idu_lim]
	cc_err = nbg_hi_err[kk][idu_lim]

	ax.set_xlim( 1e0, 2e3)

	if band[kk] == 'r':
		ax.set_ylim( 5e-5, 1e1 )
	if band[kk] == 'g':
		ax.set_ylim( 5e-5, 3e0 )
	if band[kk] == 'i':
		ax.set_ylim( 5e-5, 2e1 )

	ax.set_yscale('log')

	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax.annotate(text = '%s band' % band[kk] + '\n' + fig_name[1] + ' at fixed $M^{BCG}_{\\ast}$', 
		xy = (0.08, 0.35), xycoords = 'axes fraction', color = 'k', fontsize = 15,)

	ax.legend( loc = 1, frameon = False,)
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]', fontsize = 12,)

	ax1.plot( pre_hi_r[kk], pre_hi_sb[kk] / pre_hi_sb[kk], ls = '-', color = 'k', alpha = 0.5,)
	ax1.fill_between( pre_hi_r[kk], y1 = (pre_hi_sb[kk] - pre_hi_err[kk]) / pre_hi_sb[kk], 
		y2 = (pre_hi_sb[kk] + pre_hi_err[kk]) / pre_hi_sb[kk], color = 'k', alpha = 0.12,)

	ax1.plot(cc_r, cc_sb / com_sb, ls = '-', color = color_s[kk], alpha = 0.5,)
	ax1.fill_between( cc_r, y1 = ( cc_sb - cc_err ) / com_sb, y2 = ( cc_sb + cc_err ) / com_sb, color = color_s[kk], alpha = 0.12,)

	ax1.plot( tmp_low_r[kk], tmp_low_sb[kk] / tmp_low_com_sb[kk], ls = '-', color = 'c', alpha = 0.5, label = fig_name[0],)
	# ax1.fill_between( tmp_low_r[kk], y1 = ( tmp_low_sb[kk] - tmp_low_err[kk] ) / tmp_low_com_sb[kk], 
	# 	y2 = ( tmp_low_sb[kk] + tmp_low_err[kk] ) / tmp_low_com_sb[kk], color = 'c', alpha = 0.12,)

	ax1.set_xlim( ax.get_xlim() )

	if band[kk] == 'g':
		ax1.set_ylim( 0.6, 1.10 )
	if band[kk] == 'i':
		ax1.set_ylim( 0.90, 1.10 )

	if band[kk] == 'r':
		ax1.set_ylim( 0.90, 1.10 )

	ax1.legend( loc = 3, frameon = False,)
	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)
	ax1.set_ylabel('$ SB / SB_{w/o \; correction}$', fontsize = 12, labelpad = 10,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax.set_xticklabels( labels = [],)

	plt.savefig('/home/xkchen/%s_%s-band_sample_SB_compare.png' % (cat_lis[1], band[kk]), dpi = 300)
	plt.close()

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( sm_pre_hi_r, sm_pre_hi_g2r, ls = '--', color = 'r', alpha = 0.75, linewidth = 1, label = 'separate masking, no offset correction')
ax.fill_between( sm_pre_hi_r, y1 = sm_pre_hi_g2r - sm_pre_hi_g2r_err, y2 = sm_pre_hi_g2r + sm_pre_hi_g2r_err, color = 'r', alpha = 0.075,)

ax.plot( sm_pre_low_r, sm_pre_low_g2r, ls = '--', color = 'b', alpha = 0.75, linewidth = 1,)
ax.fill_between( sm_pre_low_r, y1 = sm_pre_low_g2r - sm_pre_low_g2r_err, y2 = sm_pre_low_g2r + sm_pre_low_g2r_err, color = 'b', alpha = 0.075,)


ax.plot( sm_hi_r, sm_hi_g2r, ls = '-', color = 'r', alpha = 0.75, linewidth = 1, label = 'combined masking, offset correction')
ax.fill_between( sm_hi_r, y1 = sm_hi_g2r - sm_hi_g2r_err, y2 = sm_hi_g2r + sm_hi_g2r_err, color = 'r', alpha = 0.15,)

ax.plot( sm_low_r, sm_low_g2r, ls = '-', color = 'b', alpha = 0.75, linewidth = 1,)
ax.fill_between( sm_low_r, y1 = sm_low_g2r - sm_low_g2r_err, y2 = sm_low_g2r + sm_low_g2r_err, color = 'b', alpha = 0.15,)

ax.axvline( x = phyR_psf[1], ls = '--', color = 'k', alpha = 0.5, ymin = 0.7, ymax = 1.0, linewidth = 1.5, label = 'PSF scale')

legend_0 = plt.legend( [ fig_name[1] + ' at fixed $M^{BCG}_{\\ast}$', fig_name[0] + ' at fixed $M^{BCG}_{\\ast}$'], loc = 6, frameon = False,)
legend_1 = ax.legend( loc = 3, frameon = False,)
ax.add_artist( legend_0 )

ax.set_ylim( 0.5, 1.7 )

ax.set_xlim(1e0, 1e3)
ax.set_xscale('log')
ax.set_ylabel('g - r', fontsize = 12,)
ax.set_xlabel('R [kpc]', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/%s_g2r_compare.png' % file_s, dpi = 300)
plt.close()
