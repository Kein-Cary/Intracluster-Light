import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov

from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

from img_BG_sub_SB_measure import BG_sub_sb_func, sub_color_func, fit_surfM_func

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

### === ### subsamples
def sub_sample_SM():
	## fixed richness samples
	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
	fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
	file_s = 'BCG_Mstar_bin'
	cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'

	# cat_lis = ['younger', 'older']
	# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
	# file_s = 'BCG_age_bin'
	# cat_path = '/home/xkchen/mywork/ICL/data/cat_z_form/age_bin_cat/gri_common_cat/'


	## fixed BCG Mstar samples
	# cat_lis = [ 'low-rich', 'hi-rich' ]
	# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
	# file_s = 'rich_bin_fixed_BCG_M'
	# cat_path = '/home/xkchen/tmp_run/data_files/figs/'

	# cat_lis = [ 'low-age', 'hi-age' ]
	# fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
	# 			'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
	# file_s = 'age_bin_fixed_BCG_M'
	# cat_path = '/home/xkchen/tmp_run/data_files/figs/'


	#. flux scaling correction
	BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
	out_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'


	#. subsamples
	band_str = 'gri'
	fit_file = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv'

	# band_str = 'ri'
	# fit_file = '/home/xkchen/figs/L_Cri_M_test/least-square_M-to-i-band-Lumi&color.csv'

	N_samples = 30
	low_R_lim, up_R_lim = 1e0, 1.2e3


	for mm in range( 2 ):

		sub_sb_file = BG_path + '%s_' % cat_lis[mm] + '%s-band_jack-sub-%d_BG-sub_SB.csv'
		sub_sm_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'

		aveg_jk_sm_file = out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str)
		lgM_cov_file = out_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)
		M_cov_file = out_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)

		fit_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_ref, z_ref,
						aveg_jk_sm_file, lgM_cov_file, fit_file, M_cov_file = M_cov_file )


	#. surface mass profile with deredden color profile
	## lgM = a(g-r) + b(r-i) + c*lg_Li + d
	pfit_dat = pds.read_csv( fit_file )
	a_fit = np.array( pfit_dat['a'] )[0]
	b_fit = np.array( pfit_dat['b'] )[0]
	c_fit = np.array( pfit_dat['c'] )[0]
	d_fit = np.array( pfit_dat['d'] )[0]


	for mm in range( 2 ):

		E_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/' + '%s_photo-z-match_rgi-common_cat_dust-value.csv' % cat_lis[mm],)
		AL_r, AL_g, AL_i = np.array( E_dat['Al_r'] ), np.array( E_dat['Al_g'] ), np.array( E_dat['Al_i'] )

		mA_r = np.median( AL_r )
		mA_g = np.median( AL_g )
		mA_i = np.median( AL_i )

		tmp_r = []
		tmp_SM = []
		tmp_lgSM = []
		tmp_Li = []

		for nn in range( N_samples ):

			_sub_dat = pds.read_csv( out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv' % nn)
			_nn_R, _nn_surf_M, _nn_Li = np.array( _sub_dat['R'] ), np.array( _sub_dat['surf_mass'] ), np.array( _sub_dat['lumi'] )

			mf0 = a_fit * ( mA_r - mA_g )
			mf1 = b_fit * ( mA_i - mA_r )
			mf2 = c_fit * 0.4 * mA_i

			modi_surf_M = _nn_surf_M * 10**( mf0 + mf1 + mf2 )
			modi_Li = _nn_Li * 10** mf2

			keys = ['R', 'surf_mass', 'lumi']
			values = [ _nn_R, modi_surf_M, modi_Li ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi_with-dered.csv' % nn )

			tmp_r.append( _nn_R )	
			tmp_SM.append( modi_surf_M )
			tmp_lgSM.append( np.log10( modi_surf_M ) )
			tmp_Li.append( modi_Li )

		aveg_R_0, aveg_surf_m, aveg_surf_m_err = arr_jack_func( tmp_SM, tmp_r, N_samples )[:3]
		aveg_R_1, aveg_lumi, aveg_lumi_err = arr_jack_func( tmp_Li, tmp_r, N_samples )[:3]
		aveg_R_2, aveg_lgM, aveg_lgM_err = arr_jack_func( tmp_lgSM, tmp_r, N_samples )[:3]

		keys = ['R', 'surf_mass', 'surf_mass_err', 'lumi', 'lumi_err', 'lg_M', 'lg_M_err']
		values = [ aveg_R_0, aveg_surf_m, aveg_surf_m_err, aveg_lumi, aveg_lumi_err, aveg_lgM, aveg_lgM_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str) )

		#. covmatrix
		R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_lgSM, id_jack = True)
		with h5py.File( out_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'w') as f:
			f['R_kpc'] = np.array( R_mean )
			f['cov_MX'] = np.array( cov_MX )
			f['cor_MX'] = np.array( cor_MX )	

		R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_SM, id_jack = True)
		with h5py.File( out_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % (cat_lis[mm], band_str), 'w') as f:
			f['R_kpc'] = np.array( R_mean )
			f['cov_MX'] = np.array( cov_MX )
			f['cor_MX'] = np.array( cor_MX )


	##... mass profile compare
	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
	lo_m_R, lo_surf_M, lo_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )

	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
	hi_m_R, hi_surf_M, hi_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )


	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' % cat_lis[0] )
	cc_lo_m_R, cc_lo_surf_M, cc_lo_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )

	m_dat = pds.read_csv( out_path + '%s_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' % cat_lis[1] )
	cc_hi_m_R, cc_hi_surf_M, cc_hi_SM_err = np.array( m_dat['R'] ), np.array( m_dat['surf_mass'] ), np.array( m_dat['surf_mass_err'] )


	fig = plt.figure()
	fig = plt.figure( figsize = (5.8, 5.4) )
	ax = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
	sub_ax = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

	ax.plot( hi_m_R, hi_surf_M, ls = '-', color = 'r', alpha = 0.75, linewidth = 1, label = fig_name[1])
	ax.fill_between( hi_m_R, y1 = hi_surf_M - hi_SM_err, y2 = hi_surf_M + hi_SM_err, color = 'r', alpha = 0.15,)

	ax.plot( lo_m_R, lo_surf_M, ls = '-', color = 'b', alpha = 0.75, linewidth = 1, label = fig_name[0])
	ax.fill_between( lo_m_R, y1 = lo_surf_M - lo_SM_err, y2 = lo_surf_M + lo_SM_err, color = 'b', alpha = 0.15,)


	ax.plot( cc_hi_m_R, cc_hi_surf_M, ls = '--', color = 'r', alpha = 0.45, linewidth = 3, label = fig_name[1] + ', deredden')
	ax.fill_between( cc_hi_m_R, y1 = cc_hi_surf_M - cc_hi_SM_err, y2 = cc_hi_surf_M + cc_hi_SM_err, color = 'r', alpha = 0.15,)

	ax.plot( cc_lo_m_R, cc_lo_surf_M, ls = '--', color = 'b', alpha = 0.45, linewidth = 3, label = fig_name[0] + ', deredden')
	ax.fill_between( cc_lo_m_R, y1 = cc_lo_surf_M - cc_lo_SM_err, y2 = cc_lo_surf_M + cc_lo_SM_err, color = 'b', alpha = 0.15,)

	ax.legend( loc = 3, frameon = False,)
	ax.set_ylim( 6e3, 3e9 )
	ax.set_yscale( 'log' )
	ax.set_xlim( 1e0, 1e3 )
	ax.set_xscale('log')

	ax.set_ylabel('$M_{\\ast} [M_{\\odot}]$', fontsize = 12,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	sub_ax.plot( hi_m_R, cc_hi_surf_M / hi_surf_M, ls = '--', color = 'r', alpha = 0.75,)
	sub_ax.plot( lo_m_R, cc_lo_surf_M / lo_surf_M, ls = '--', color = 'b', alpha = 0.75,)

	sub_ax.set_ylim( 0.99, 1.02 )
	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xscale('log')

	sub_ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
	sub_ax.set_xlabel('R [kpc]', fontsize = 12,)
	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/%s_surf-M_compare.png' % file_s, dpi = 300)
	plt.close()

	return

sub_sample_SM()

raise

### === ### total sample
band_str = 'gri'
fit_file = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv'

# band_str = 'ri'
# fit_file = '/home/xkchen/figs/L_Cri_M_test/least-square_M-to-i-band-Lumi&color.csv'

#. surface mass profile with deredden color profile
## lgM = a(g-r) + b(r-i) + c*lg_Li + d
pfit_dat = pds.read_csv( fit_file )
a_fit = np.array( pfit_dat['a'] )[0]
b_fit = np.array( pfit_dat['b'] )[0]
c_fit = np.array( pfit_dat['c'] )[0]
d_fit = np.array( pfit_dat['d'] )[0]


N_samples = 30

BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
out_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'

low_R_lim, up_R_lim = 1e0, 1.2e3

sub_sb_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_jack-sub-%d_BG-sub_SB.csv'
sub_sm_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv'

aveg_jk_sm_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str
lgM_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str
M_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % band_str

fit_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_ref, z_ref,
				aveg_jk_sm_file, lgM_cov_file, fit_file, M_cov_file = M_cov_file )

#. mass estimation with deredden correction
gE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_g-band_dust_value.csv')
AL_g = np.array( gE_dat['A_l'] )

rE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_r-band_dust_value.csv')
AL_r = np.array( rE_dat['A_l'] )

iE_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_i-band_dust_value.csv')
AL_i = np.array( iE_dat['A_l'] )

# E_dat = pds.read_csv('/home/xkchen/figs/re_measure_SBs/tot-BCG-star-Mass_dust-value.csv')
# AL_r, AL_g, AL_i = np.array( E_dat['Al_r']), np.array( E_dat['Al_g']), np.array( E_dat['Al_i'])

mA_g = np.median( AL_g )
mA_r = np.median( AL_r )
mA_i = np.median( AL_i )

tmp_r, tmp_SM, tmp_lgSM, tmp_Li = [], [], [], []

for nn in range( N_samples ):

	_sub_dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv' % nn)
	_nn_R, _nn_surf_M, _nn_Li = np.array( _sub_dat['R'] ), np.array( _sub_dat['surf_mass'] ), np.array( _sub_dat['lumi'] )

	mf0 = a_fit * ( mA_r - mA_g )
	mf1 = b_fit * ( mA_i - mA_r )
	mf2 = c_fit * 0.4 * mA_i

	modi_surf_M = _nn_surf_M * 10**( mf0 + mf1 + mf2 )
	modi_Li = _nn_Li * 10** mf2

	keys = ['R', 'surf_mass', 'lumi']
	values = [ _nn_R, modi_surf_M, modi_Li ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi_with-dered.csv' % nn )

	tmp_r.append( _nn_R )
	tmp_SM.append( modi_surf_M )
	tmp_lgSM.append( np.log10( modi_surf_M ) )
	tmp_Li.append( modi_Li )

aveg_R_0, aveg_surf_m, aveg_surf_m_err = arr_jack_func( tmp_SM, tmp_r, N_samples )[:3]
aveg_R_1, aveg_lumi, aveg_lumi_err = arr_jack_func( tmp_Li, tmp_r, N_samples )[:3]
aveg_R_2, aveg_lgM, aveg_lgM_err = arr_jack_func( tmp_lgSM, tmp_r, N_samples )[:3]

keys = ['R', 'surf_mass', 'surf_mass_err', 'lumi', 'lumi_err', 'lg_M', 'lg_M_err']
values = [ aveg_R_0, aveg_surf_m, aveg_surf_m_err, aveg_lumi, aveg_lumi_err, aveg_lgM, aveg_lgM_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str )

#. covmatrix
R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_lgSM, id_jack = True)
with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr_with-dered.h5' % band_str, 'w') as f:
	f['R_kpc'] = np.array( R_mean )
	f['cov_MX'] = np.array( cov_MX )
	f['cor_MX'] = np.array( cor_MX )

R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_SM, id_jack = True)
with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr_with-dered.h5' % band_str, 'w') as f:
	f['R_kpc'] = np.array( R_mean )
	f['cov_MX'] = np.array( cov_MX )
	f['cor_MX'] = np.array( cor_MX )


dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
aveg_R = np.array(dat['R'])
aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
cc_aveg_R = np.array(dat['R'])
cc_aveg_surf_m, cc_aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


fig = plt.figure()
fig = plt.figure( figsize = (5.8, 5.4) )
ax = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

ax.plot( aveg_R, aveg_surf_m, ls = '-', color = 'r', alpha = 0.5, label = 'w/o deredden')
ax.fill_between( aveg_R, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, color = 'r', alpha = 0.12,)

ax.plot( cc_aveg_R, cc_aveg_surf_m, ls = '--', color = 'r', alpha = 0.5, label = 'deredden')
ax.fill_between( cc_aveg_R, y1 = cc_aveg_surf_m - cc_aveg_surf_m_err, 
				y2 = cc_aveg_surf_m + cc_aveg_surf_m_err, color = 'r', alpha = 0.12,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 6e3, 3e9 )
ax.set_yscale( 'log' )
ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')

ax.set_ylabel('$M_{\\ast} [M_{\\odot}]$', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

sub_ax.plot( aveg_R, aveg_surf_m / cc_aveg_surf_m, ls = '--', color = 'r', alpha = 0.75,)

sub_ax.set_ylim( 0.99, 1.02 )
sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xscale('log')

sub_ax.set_ylabel('$M_{\\ast}^{deredden} / M_{\\ast}$', fontsize = 12,)
sub_ax.set_xlabel('R [kpc]', fontsize = 12,)
sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/tot_%s-band_based_surface_mass_profile.png' % band_str, dpi = 300)
plt.close()

