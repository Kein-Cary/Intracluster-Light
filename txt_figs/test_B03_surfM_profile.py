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

from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov
from scipy.interpolate import splev, splrep
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

from color_2_mass import jk_sub_SB_func
from img_BG_sub_SB_measure import BG_sub_sb_func, sub_color_func, B03_surfM_func


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


## === ## fixed rich bin
#... combine masking 
# cat_lis = [ 'younger', 'older' ]
# fig_name = [ 'younger', 'older' ]
# file_s = 'age-bin'
# pre_BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/BGs/'
# pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/SBs/'

cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast} ^{BCG}$', 'high $M_{\\ast} ^{BCG}$']
file_s = 'BCG_M-bin'
pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/SBs/'
pre_BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'


## fixed BCG-Mass bin
# cat_lis = [ 'low-age', 'hi-age' ]
# fig_name = [ 'younger', 'older' ]
# file_s = 'age-bin'
# pre_BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'
# pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/SBs/'

# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'low $\\lambda$', 'high $\\lambda$' ]
# file_s = 'rich-bin'
# pre_BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'
# pre_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/SBs/'


BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
path = '/home/xkchen/figs/re_measure_SBs/SBs/'

### === mass profile
N_bin = 30

#.(based on Bell 2003)
def surf_M_func():

	N_samples = 30

	for mm in range( 2 ):

		band_str = 'gi'
		c_inv = False
		# for kk in (1,2): # g,i band based mass estimate

		# band_str = 'gr'
		# c_inv = False
		# for kk in (1,0): # g,r band based mass estimate

		# band_str = 'ri'
		# c_inv = False
		# for kk in (0,2): # r,i band based mass estimate

		# band_str = 'ir'
		# c_inv = True
		# for kk in (2,0): # r,i band based mass estimate

		# band_str = 'ig'
		# c_inv = True
		# for kk in (2,0): # r,i band based mass estimate

		# band_str = 'rg'
		# c_inv = True
		# for kk in (2,0): # r,i band based mass estimate


		# band_str = 'gri'
		# c_inv = False
		# for kk in (1,0,2): # r,i band based mass estimate

		# band_str = 'gir'
		# c_inv = False
		# for kk in (1,2,0): # r,i band based mass estimate

		# band_str = 'rig'
		# c_inv = False

		# low_R_lim, up_R_lim = 1e1, 1e3
		low_R_lim, up_R_lim = 1e0, 1.2e3

		sub_sb_file = BG_path + '%s_' % cat_lis[mm] + '%s-band_jack-sub-%d_BG-sub_SB.csv'
		sub_sm_file = BG_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'

		aveg_jk_sm_file = BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str)
		lgM_cov_file = BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)

		B03_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_ref, z_ref,
    					aveg_jk_sm_file, lgM_cov_file, c_inv = c_inv )

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

	plt.xlim(1e0, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15)
	plt.yscale('log')
	# plt.ylim(1e3, 2e9)
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
	# plt.ylim( 1e10, 1e12)
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
	# plt.ylim( 5e2, 1e8)
	plt.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.ylabel('SB $[L_{\\odot} / kpc^2]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/%s_%s-band_based_%s-band_SB_profile.png' % (file_s, band_str, band_str[1]), dpi = 300)
	plt.close()

	return

# surf_M_func()

