import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import scipy.stats as sts

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.interpolate as interp
import scipy.stats as sts
from scipy.interpolate import splev, splrep
from fig_out_module import color_func
from scipy import integrate as integ

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from color_2_mass import get_c2mass_func
from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

## cosmology model
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

band = [ 'r', 'g', 'i' ]
l_wave = np.array( [6166, 4686, 7480] )
## solar Magnitude corresponding to SDSS filter
Mag_sun = [ 4.65, 5.11, 4.53 ]

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

# path = '/home/xkchen/mywork/ICL/code/rig_common_cat/'
# BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'

## bin number adjust
path = '/home/xkchen/mywork/ICL/code/rich_based_cat/tmp_mass_SBs/'
BG_path = '/home/xkchen/mywork/ICL/code/rich_based_cat/tmp_mass_BGs/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']


# path = '/home/xkchen/mywork/ICL/code/rig_common_cat/'
# BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin_BG/'

## bins number adjust
path = '/home/xkchen/mywork/ICL/code/rich_based_cat/tmp_age_SBs/'
BG_path = '/home/xkchen/mywork/ICL/code/rich_based_cat/tmp_age_BGs/'

cat_lis = ['younger', 'older']
fig_name = ['younger', 'older']

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']

N_samples = 30

def mass_pro_func():

	### measure mass for jack-sub sample
	for mm in range( 2 ):

		for nn in range( N_samples ):

			nt_r, nt_sb, nt_err = [], [], []

			for kk in range( 3 ):

				n_dat = pds.read_csv( BG_path + '%s_%s-band_jack-sub-%d_BG-sub_SB.csv' % (cat_lis[mm], band[kk], nn),)
				nn_r, nn_sb, nn_err = np.array( n_dat['R']), np.array( n_dat['BG_sub_SB']), np.array( n_dat['sb_err'])

				idvx = ( nn_r >= 10 ) & ( nn_r <= 1e3)

				nt_r.append( nn_r[idvx] )
				nt_sb.append( nn_sb[idvx] )
				nt_err.append( nn_err[idvx] )

			## gi band based case
			nt_g2i, nt_g2i_err = color_func( nt_sb[1], nt_err[1], nt_sb[2], nt_err[2] )

			nt_i_mag = 22.5 - 2.5 * np.log10( nt_sb[2] )
			nt_i_Mag = nt_i_mag - 5 * np.log10( Dl_ref * 10**6 / 10)

			out_m_file = BG_path + '%s_gi-band-based_jack-sub-%d_mass-Lumi.csv' % (cat_lis[mm], nn)
			band_str = 'gi'
			get_c2mass_func( nt_r[ 2 ], band_str, nt_i_Mag, nt_g2i, z_ref, out_file = out_m_file )

			## gr band based case
			nt_g2r, nt_g2r_err = color_func( nt_sb[1], nt_err[1], nt_sb[0], nt_err[0] )

			nt_r_mag = 22.5 - 2.5 * np.log10( nt_sb[0] )
			nt_r_Mag = nt_r_mag - 5 * np.log10( Dl_ref * 10**6 / 10)

			out_m_file_0 = BG_path + '%s_gr-band-based_jack-sub-%d_mass-Lumi.csv' % (cat_lis[mm], nn)
			band_str = 'gr'
			get_c2mass_func( nt_r[0], band_str, nt_r_Mag, nt_g2r, z_ref, out_file = out_m_file_0 )

			## ri band based case
			nt_r2i, nt_r2i_err = color_func( nt_sb[0], nt_err[0], nt_sb[2], nt_err[2] )
			out_m_file_1 = BG_path + '%s_ri-band-based_jack-sub-%d_mass-Lumi.csv' % (cat_lis[mm], nn)
			band_str = 'ri'
			get_c2mass_func( nt_r[ 2 ], band_str, nt_i_Mag, nt_r2i, z_ref, out_file = out_m_file_1 )

	### jack mean and figs
	for band_str in ('gi', 'gr', 'ri'):

		for mm in range( 2 ):

			tmp_r, tmp_mass = [], []
			tmp_c_mass, tmp_lumi = [], []

			for nn in range( N_samples ):

				if band_str == 'gi':
					o_dat = pds.read_csv( BG_path + '%s_gi-band-based_jack-sub-%d_mass-Lumi.csv' % (cat_lis[mm], nn),)

				if band_str == 'gr':
					o_dat = pds.read_csv( BG_path + '%s_gr-band-based_jack-sub-%d_mass-Lumi.csv' % (cat_lis[mm], nn),)

				if band_str == 'ri':
					o_dat = pds.read_csv( BG_path + '%s_ri-band-based_jack-sub-%d_mass-Lumi.csv' % (cat_lis[mm], nn),)

				tmp_r.append( o_dat['R'] )
				tmp_mass.append( o_dat['surf_mass'] )

				tmp_c_mass.append( o_dat['cumu_mass'] )
				tmp_lumi.append( o_dat['lumi'] )

			### jack-mean pf mass and lumi profile
			aveg_R, aveg_surf_m, aveg_surf_m_err = arr_jack_func( tmp_mass, tmp_r, N_samples)[:3]
			aveg_R, aveg_cumu_m, aveg_cumu_m_err = arr_jack_func( tmp_c_mass, tmp_r, N_samples)[:3]
			aveg_R, aveg_lumi, aveg_lumi_err = arr_jack_func( tmp_lumi, tmp_r, N_samples)[:3]

			keys = ['R', 'surf_mass', 'surf_mass_err', 'cumu_mass', 'cumu_mass_err', 'lumi', 'lumi_err']
			values = [ aveg_R, aveg_surf_m, aveg_surf_m_err, aveg_cumu_m, aveg_cumu_m_err, aveg_lumi, aveg_lumi_err]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)

			### cov_arr of mass profile

			#.. use lg_mass to calculate cov_arr to avoid to huge value occur 
			lg_M_arr = np.log10( tmp_mass )

			R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, list(lg_M_arr), id_jack = True)

			# with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'w') as f:
			with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'w') as f:
				f['R_kpc'] = np.array( R_mean )
				f['cov_MX'] = np.array( cov_MX )
				f['cor_MX'] = np.array( cor_MX )

			plt.figure()
			plt.imshow( cor_MX, origin = 'lower',)
			plt.show()
	raise
	"""
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

		plt.xlim(1e1, 1e3)
		plt.xscale('log')
		plt.xlabel('R[kpc]', fontsize = 15)
		plt.yscale('log')
		plt.ylim(1e3, 2e8)
		plt.legend( loc = 1, frameon = False, fontsize = 15,)
		plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
		plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
		plt.savefig('/home/xkchen/figs/%s-band_based_surface_mass_profile.png' % band_str, dpi = 300)
		plt.close()


		plt.figure()
		plt.title('%s-based cumulative mass profile' % band_str )
		plt.plot( dpt_R[0], dpt_cumu_M[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name[0])
		plt.fill_between( dpt_R[0], y1 = dpt_cumu_M[0] - dpt_cumu_M_err[0], y2 = dpt_cumu_M[0] + dpt_cumu_M_err[0], color = line_c[0], alpha = 0.12,)
		plt.plot( dpt_R[1], dpt_cumu_M[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name[1])
		plt.fill_between( dpt_R[1], y1 = dpt_cumu_M[1] - dpt_cumu_M_err[1], y2 = dpt_cumu_M[1] + dpt_cumu_M_err[1], color = line_c[1], alpha = 0.12,)

		plt.xlim(1e1, 1e3)
		plt.xscale('log')
		plt.xlabel('R[kpc]', fontsize = 15)
		plt.yscale('log')
		plt.ylim(4e10, 7e11)
		plt.legend( loc = 4, frameon = False, fontsize = 15,)
		plt.ylabel('cumulative mass $[M_{\\odot}]$', fontsize = 15,)
		plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
		plt.savefig('/home/xkchen/figs/%s-band_based_cumulative_mass_profile.png' % band_str, dpi = 300)
		plt.close()


		plt.figure()
		plt.title('%s band SB profile' % band_str[1] )
		plt.plot( dpt_R[0], dpt_L[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name[0])
		plt.fill_between( dpt_R[0], y1 = dpt_L[0] - dpt_L_err[0], y2 = dpt_L[0] + dpt_L_err[0], color = line_c[0], alpha = 0.12,)
		plt.plot( dpt_R[1], dpt_L[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name[1])
		plt.fill_between( dpt_R[1], y1 = dpt_L[1] - dpt_L_err[1], y2 = dpt_L[1] + dpt_L_err[1], color = line_c[1], alpha = 0.12,)

		plt.xlim(1e1, 1e3)
		plt.xscale('log')
		plt.xlabel('R[kpc]', fontsize = 15)
		plt.yscale('log')
		plt.ylim(5e2, 1e7)
		plt.legend( loc = 1, frameon = False, fontsize = 15,)
		plt.ylabel('SB $[L_{\\odot} / kpc^2]$', fontsize = 15,)
		plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
		plt.savefig('/home/xkchen/figs/%s-band_based_%s-band_SB_profile.png' % (band_str, band_str[1]), dpi = 300)
		plt.close()
	"""

# mass_pro_func()

### age-bin, mass-bin compare
line_c = ['b', 'r']
color_s = ['r', 'g', 'b']

# path_0 = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin_BG/'
path_0 = '/home/xkchen/mywork/ICL/code/rich_based_cat/tmp_age_BGs/'

cat_lis_0 = ['younger', 'older']
fig_name_0 = ['younger', 'older']


# path_1 = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'
path_1 = '/home/xkchen/mywork/ICL/code/rich_based_cat/tmp_mass_BGs/'

cat_lis_1 = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name_1 = ['low $M_{\\ast}^{BCG}$', 'high $M_{\\ast}^{BCG}$']


tmp_m_R, tmp_m_M = [], []
tmp_a_R, tmp_a_M = [], []

for band_str in ('gi', 'gr', 'ri'):

	dpt_R_0, dpt_M_0 = [], []
	dpt_M_err_0 = []
	dpt_cumu_M_0, dpt_cumu_M_err_0 = [], []

	for mm in range( 2 ):

		dat = pds.read_csv( path_0 + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis_0[mm], band_str),)
		aveg_R = np.array(dat['R'])

		aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])

		dpt_R_0.append( aveg_R )
		dpt_M_0.append( aveg_surf_m )
		dpt_M_err_0.append( aveg_surf_m_err )

		dpt_cumu_M_0.append( aveg_cumu_m )
		dpt_cumu_M_err_0.append( aveg_cumu_m_err )

		##..
		tmp_m_R.append( aveg_R )
		tmp_m_M.append( aveg_surf_m )


	dpt_R_1, dpt_M_1 = [], []
	dpt_M_err_1 = []
	dpt_cumu_M_1, dpt_cumu_M_err_1 = [], []

	for mm in range( 2 ):

		dat = pds.read_csv( path_1 + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis_1[mm], band_str),)
		aveg_R = np.array(dat['R'])

		aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])

		dpt_R_1.append( aveg_R )
		dpt_M_1.append( aveg_surf_m )
		dpt_M_err_1.append( aveg_surf_m_err )

		dpt_cumu_M_1.append( aveg_cumu_m )
		dpt_cumu_M_err_1.append( aveg_cumu_m_err )

		##..
		tmp_a_R.append( aveg_R )
		tmp_a_M.append( aveg_surf_m )

	plt.figure()

	plt.plot( dpt_R_0[0], dpt_M_0[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name_0[0],)
	plt.fill_between( dpt_R_0[0], y1 = dpt_M_0[0] - dpt_M_err_0[0], y2 = dpt_M_0[0] + dpt_M_err_0[0], color = line_c[0], alpha = 0.12,)

	plt.plot( dpt_R_0[1], dpt_M_0[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name_0[1],)
	plt.fill_between( dpt_R_0[1], y1 = dpt_M_0[1] - dpt_M_err_0[1], y2 = dpt_M_0[1] + dpt_M_err_0[1], color = line_c[1], alpha = 0.12,)

	plt.plot( dpt_R_1[0], dpt_M_1[0], ls = '--', color = line_c[0], alpha = 0.5, label = fig_name_1[0],)
	plt.fill_between( dpt_R_1[0], y1 = dpt_M_1[0] - dpt_M_err_1[0], y2 = dpt_M_1[0] + dpt_M_err_1[0], color = line_c[0], alpha = 0.12,)

	plt.plot( dpt_R_1[1], dpt_M_1[1], ls = '--', color = line_c[1], alpha = 0.5, label = fig_name_1[1],)
	plt.fill_between( dpt_R_1[1], y1 = dpt_M_1[1] - dpt_M_err_1[1], y2 = dpt_M_1[1] + dpt_M_err_1[1], color = line_c[1], alpha = 0.12,)

	plt.xlim(1e1, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15,)
	plt.yscale('log')
	plt.ylim(1e3, 3e8)
	plt.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/figs/mass-age-bin_%s-band_based_surface_mass_profile.png' % band_str, dpi = 300,)
	plt.close()

"""
	plt.figure()
	# plt.title( 'cumulative mass density profile' )
	plt.plot( dpt_R_0[0], dpt_cumu_M_0[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name_0[0],)
	plt.fill_between( dpt_R_0[0], y1 = dpt_cumu_M_0[0] - dpt_cumu_M_err_0[0], y2 = dpt_cumu_M_0[0] + dpt_cumu_M_err_0[0], color = line_c[0], alpha = 0.12,)

	plt.plot( dpt_R_0[1], dpt_cumu_M_0[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name_0[1],)
	plt.fill_between( dpt_R_0[1], y1 = dpt_cumu_M_0[1] - dpt_cumu_M_err_0[1], y2 = dpt_cumu_M_0[1] + dpt_cumu_M_err_0[1], color = line_c[1], alpha = 0.12,)

	plt.plot( dpt_R_1[0], dpt_cumu_M_1[0], ls = '--', color = line_c[0], alpha = 0.5, label = fig_name_1[0],)
	plt.fill_between( dpt_R_1[0], y1 = dpt_cumu_M_1[0] - dpt_cumu_M_err_1[0], y2 = dpt_cumu_M_1[0] + dpt_cumu_M_err_1[0], color = line_c[0], alpha = 0.12,)

	plt.plot( dpt_R_1[1], dpt_cumu_M_1[1], ls = '--', color = line_c[1], alpha = 0.5, label = fig_name_1[1],)
	plt.fill_between( dpt_R_1[1], y1 = dpt_cumu_M_1[1] - dpt_cumu_M_err_1[1], y2 = dpt_cumu_M_1[1] + dpt_cumu_M_err_1[1], color = line_c[1], alpha = 0.12,)

	plt.xlim(1e1, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15,)
	plt.yscale('log')
	plt.ylim(4e10, 7e11)
	plt.legend( loc = 4, frameon = False, fontsize = 15,)
	plt.ylabel('cumulative mass $[M_{\\odot}]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/figs/mass-age-bin_%s-band_based_cumulative_mass_profile.png' % band_str, dpi = 300)
	plt.close()
"""
