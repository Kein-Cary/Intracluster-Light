import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.special as special
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import signal
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set
from color_2_mass import get_c2mass_func, gi_band_c2m_func
from fig_out_module import arr_jack_func

### === ### cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

rad2asec = U.rad.to(U.arcsec)
pixel = 0.396
band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )

		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

def SB_to_Lsun_func(sb_arr, obs_z, band_s):

	if band_s == 'r':
		Mag_dot = Mag_sun[0]

	if band_s == 'g':
		Mag_dot = Mag_sun[1]

	if band_s == 'i':
		Mag_dot = Mag_sun[2]

	Da_obs = Test_model.angular_diameter_distance( obs_z ).value
	Dl_obs = Test_model.luminosity_distance( obs_z ).value
	phy_S = 1 * Da_obs**2 * 1e6 / rad2asec**2

	sb_mag = 22.5 - 2.5 * np.log10( sb_arr )
	sb_Mag = sb_mag - 5 * np.log10( Dl_obs * 1e6 ) + 5

	L_sun = 10**( -0.4 * (sb_Mag - Mag_dot) )
	phy_SB = L_sun / phy_S # L_sun / kpc^2

	return phy_SB

def cp_SB_to_Lsun_func(sb_arr, obs_z, band_s):
	"""
	sb_arr : in unit of mag /arcsec^2,
	"""
	if band_s == 'r':
		Mag_dot = Mag_sun[0]

	if band_s == 'g':
		Mag_dot = Mag_sun[1]

	if band_s == 'i':
		Mag_dot = Mag_sun[2]

	#... surface brightness density, L_sun / pc^2
	#... 2.5 * np.log10( rad2arcsec^2 ) - 5 ~ 21.572
	# Lumi = 10**( -0.4 * (sb_arr - Mag_dot - 21.572 - 10*np.log10(obs_z + 1) ) )

	beta = 2.5 * np.log10( rad2asec**2 ) - 5
	Lumi = 10**( -0.4 * (sb_arr - Mag_dot - beta - 10*np.log10(obs_z + 1) ) )
	return Lumi

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)
Da_ref = Test_model.angular_diameter_distance( z_ref ).value


## fixed richness samples
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# file_s = 'BCG_Mstar_bin'
# cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'

# cat_lis = ['younger', 'older']
# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
# file_s = 'BCG_age_bin'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/BGs/'


## fixed BCG Mstar samples
# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'rich_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
file_s = 'age_bin_fixed_BCG_M'
cat_path = '/home/xkchen/tmp_run/data_files/figs/'
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'


out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'
band_str = 'gri'

#... total sample mass compare (for estimating apeture size to integrate BCG M)
# lo_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )
# lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
# lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['BCG_age'] )

# hi_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )
# hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
# hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['BCG_age'] )

# tot_lgM = np.r_[ lo_lgM, hi_lgM ]
# tot_lg_Mean = np.mean( 10**tot_lgM / h**2 )
# tot_lg_Medi = np.median( 10**tot_lgM / h**2 )

# dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
# aveg_R = np.array(dat['R'])
# aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

# N_grid = 250
# tot_cumu_M = cumu_mass_func( aveg_R, aveg_surf_m, N_grid = N_grid )
# intep_Mf = interp.interp1d( tot_cumu_M, aveg_R, kind = 'cubic',)
# R_bond_0 = intep_Mf( tot_lg_Mean )
# R_bond_1 = intep_Mf( tot_lg_Medi )

# keys = ['R_fixed_mean_M', 'R_fixed_medi_M']
# values = [ R_bond_0, R_bond_1 ]
# fill = dict( zip( keys, values) )
# out_data = pds.DataFrame( fill, index = ['k', 'v'])
# out_data.to_csv( out_path + 'BCGM-match_R.csv')


dt_R, dt_M, dt_Merr = [], [], []

for mm in range( 2 ):

	#... mass profile
	m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
	jk_R = np.array(m_dat['R'])
	surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
	surf_L = np.array( m_dat['lumi'] )

	N_grid = 250

	up_lim_R = 47.69

	cumu_M = cumu_mass_func( jk_R, surf_m, N_grid = N_grid )
	intep_Mf = interp.interp1d( jk_R, cumu_M, kind = 'cubic',)

	M_40 = intep_Mf( up_lim_R )

	#... catalog infor.
	p_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	p_obs_z, p_rich  = np.array( p_dat['z']), np.array( p_dat['rich'])
	p_lgM, p_age = np.array( p_dat['lg_Mstar']), np.array( p_dat['BCG_age'] )

	lg_Mean = np.log10( np.mean(10**p_lgM / h**2) )
	lg_Medi = np.log10( np.median(10**p_lgM / h**2) )

	devi_Mean = np.log10( M_40 ) - lg_Mean
	devi_Medi = np.log10( M_40 ) - lg_Medi

	print( 'lg_M devi*****' )
	print( 10**devi_Mean )
	print( 10**devi_Medi )

	dt_R.append( jk_R )
	dt_M.append( surf_m )
	dt_Merr.append( surf_m_err )

	#... cumulative flux
	with h5py.File( BG_path + 'photo-z_%s_i-band_BG-sub_SB.h5' % cat_lis[mm], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	# tt_mag_arr = 22.5 - 2.5 * np.log10( tt_sb ) ## SB in apparent magnitude
	# cc_surf_Lumi = cp_SB_to_Lsun_func( tt_mag_arr, z_ref, 'i')
	# cc_surf_Lumi = cc_surf_Lumi * 1e6


	#... cumulative Luminosity
	# cumu_L = cumu_mass_func( jk_R, surf_L, N_grid = N_grid )
	# intep_Lf = interp.interp1d( jk_R, cumu_L, kind = 'cubic',)
	# L_40 = intep_Lf( up_lim_R )


	# idx = tt_r <= 1.1e3
	# tt_deg_R = tt_r * rad2asec / (Da_ref * 1e3)

	# cumu_f = cumu_mass_func( tt_deg_R[idx], tt_sb[idx], N_grid = N_grid )
	# intep_ff = interp.interp1d( tt_deg_R[idx], cumu_f, kind = 'cubic',)

	# f_40 = intep_ff( up_lim_R )
	# mag_f40 = 22.5 - 2.5 * np.log10( f_40 )
	# Mag_f40 = mag_f40 - 5 * np.log10( Dl_ref * 1e6 ) + 5

	# cp_Lumi = SB_to_Lsun_func( tt_sb[idx], z_ref, band[2] )
	# cp_cumu_L = cumu_mass_func( tt_r[idx], cp_Lumi, N_grid = N_grid )
	# cp_intep_f = interp.interp1d( tt_r[idx], cp_cumu_L, kind = 'cubic',)
	# cp_L_40 = cp_intep_f( up_lim_R )


	# pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG_cmag.csv' % cat_lis[mm] )
	# i_cMag = np.array( pdat['i_cMag'] )
	# i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )

	# mean_Li = np.mean( i_Lumi )
	# medi_Li = np.median( i_Lumi )

	# print('*' * 10)
	# print( mean_Li / L_40 )
	# print( medi_Li / L_40 )

	# print('*' * 10)	
	# print( mean_Li / cp_L_40 )
	# print( medi_Li / cp_L_40 )

	# print('*' * 10)
	# print( np.median( i_cMag ) - Mag_f40 )
	# print( np.mean( i_cMag ) - Mag_f40 )

dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
tot_R = np.array(dat['R'])
tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

interp_M_f = interp.interp1d( tot_R, tot_surf_m, kind = 'linear',)

dt_eta_R, dt_eta, dt_eta_err = [], [], []

for mm in range( 2 ):

	N_samples = 30

	jk_sub_m_file = out_path + '%s_gri-band-based_' % cat_lis[mm] + 'jack-sub-%d_mass-Lumi.csv'

	tmp_r, tmp_ratio = [], []

	for nn in range( N_samples ):

		o_dat = pds.read_csv( jk_sub_m_file % nn,)

		tt_r = np.array( o_dat['R'] )
		tt_M = np.array( o_dat['surf_mass'] )

		idx_lim = ( tt_r >= np.nanmin( tot_R ) ) & ( tt_r <= np.nanmax( tot_R ) )

		lim_R = tt_r[ idx_lim ]
		lim_M = tt_M[ idx_lim ]

		com_M = interp_M_f( lim_R )

		sub_ratio = np.zeros( len(tt_r),)
		sub_ratio[ idx_lim ] = lim_M / com_M

		sub_ratio[ idx_lim == False ] = np.nan

		tmp_r.append( tt_r )
		tmp_ratio.append( sub_ratio )

	aveg_R, aveg_ratio, aveg_ratio_err = arr_jack_func( tmp_ratio, tmp_r, N_samples)[:3]

	dt_eta_R.append( aveg_R )
	dt_eta.append( aveg_ratio )
	dt_eta_err.append( aveg_ratio_err )


#...calibrated case
dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )


lo_eat_dat = pds.read_csv( out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[0] )
lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['M/M_tot']), np.array(lo_eat_dat['M/M_tot-err'])

hi_eat_dat = pds.read_csv( out_path + '%s_gri-band_aveg_M-ratio_to_total-sample.csv' % cat_lis[1] )
hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['M/M_tot']), np.array(hi_eat_dat['M/M_tot-err'])


line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

fig = plt.figure( figsize = (12.0, 4.8) )
ax0 = fig.add_axes([0.09, 0.41, 0.40, 0.56])
sub_ax0 = fig.add_axes([0.09, 0.13, 0.40, 0.28])

ax1 = fig.add_axes([0.58, 0.41, 0.40, 0.56])
sub_ax1 = fig.add_axes([0.58, 0.13, 0.40, 0.28])

for mm in range( 2 ):

	ax0.plot( dt_R[mm]/ 1e3, dt_M[mm], ls = line_s[mm], color = line_c[mm], alpha = 0.75, label = fig_name[mm])
	ax0.fill_between( dt_R[mm]/ 1e3, y1 = dt_M[mm] - dt_Merr[mm], y2 = dt_M[mm] + dt_Merr[mm], color = line_c[mm], alpha = 0.15,)

	sub_ax0.plot( dt_eta_R[mm]/ 1e3, dt_eta[mm], ls = line_s[mm], color = line_c[mm], alpha = 0.75)
	sub_ax0.fill_between( dt_eta_R[mm]/ 1e3, y1 = dt_eta[mm] - dt_eta_err[mm], y2 = dt_eta[mm] + dt_eta_err[mm], color = line_c[mm], alpha = 0.15)

ax0.plot( tot_R/ 1e3, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
ax0.fill_between( tot_R/ 1e3, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.15,)

ax0.set_xlim( 3e-3, 1e0 )
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_ylim( 5e3, 2e9 )
ax0.legend( loc = 3, frameon = False, fontsize = 16,)
ax0.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 16,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax0.annotate( text = 'Before calibration', xy = (0.55, 0.85), xycoords = 'axes fraction', fontsize = 16, color = 'k',)

sub_ax0.plot( tot_R / 1e3, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)
sub_ax0.set_xlim( ax0.get_xlim() )
sub_ax0.set_xscale( 'log' )
sub_ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 16,)

sub_ax0.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax0.set_ylim( 0.45, 1.20 )
sub_ax0.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax0.set_xticklabels( [] )

ax1.plot( lo_R/ 1e3, lo_surf_M, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0],)
ax1.fill_between( lo_R/ 1e3, y1 = lo_surf_M - lo_surf_M_err, 
	y2 = lo_surf_M + lo_surf_M_err, color = line_c[0], alpha = 0.15,)

ax1.plot( hi_R/ 1e3, hi_surf_M, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1],)
ax1.fill_between( hi_R/ 1e3, y1 = hi_surf_M - hi_surf_M_err, 
	y2 = hi_surf_M + hi_surf_M_err, color = line_c[1], alpha = 0.15,)

ax1.plot( tot_R/ 1e3, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
ax1.fill_between( tot_R/ 1e3, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.15,)

ax1.set_xlim( 3e-3, 1e0 )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim( 5e3, 2e9 )
ax1.legend( loc = 3, frameon = False, fontsize = 16,)
ax1.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 16,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax1.annotate( text = 'After calibration', xy = (0.55, 0.85), xycoords = 'axes fraction', fontsize = 16, color = 'k',)

sub_ax1.plot( lo_eta_R / 1e3, lo_eta, ls = '--', color = line_c[0], alpha = 0.75,)
sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta - lo_eta_err, y2 = lo_eta + lo_eta_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta, ls = '-', color = line_c[1], alpha = 0.75,)
sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta - hi_eta_err, y2 = hi_eta + hi_eta_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( tot_R / 1e3, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale( 'log' )
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 16,)

sub_ax1.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax1.set_ylim( 0.45, 1.20 )
sub_ax1.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax1.set_xticklabels( [] )

plt.savefig('/home/xkchen/%s_mass-reatio_compare.png' % file_s, dpi = 300)
plt.close()
