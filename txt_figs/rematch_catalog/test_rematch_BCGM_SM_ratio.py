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

from fig_out_module import arr_jack_func

### === ### cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

rad2asec = U.rad.to(U.arcsec)
pixel = 0.396
band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)
Da_ref = Test_model.angular_diameter_distance( z_ref ).value

### === ???
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


### === data load
all_out_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
band_str = 'gri'


### === ### estimate the aperture at which enclosed mass is equal to BCG mass
# lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
# 						'low_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
# lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
# lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['age'] )

# hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
# 						'high_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
# hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
# hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['age'] )

# tot_lgM = np.r_[ lo_lgM, hi_lgM ]
# tot_Mean = np.mean( 10**tot_lgM / h**2 )
# tot_Medi = np.median( 10**tot_lgM / h**2 )

# dat = pds.read_csv( all_out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
# aveg_R = np.array( dat['R'] )
# aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

# N_grid = 250
# tot_cumu_M = cumu_mass_func( aveg_R, aveg_surf_m, N_grid = N_grid )
# intep_Mf = interp.interp1d( tot_cumu_M, aveg_R, kind = 'linear',)

# R_bond_0 = intep_Mf( tot_Mean )
# R_bond_1 = intep_Mf( tot_Medi )

# print( R_bond_0, R_bond_1 )

# keys = ['R_fixed_mean_M', 'R_fixed_medi_M']
# values = [ R_bond_0, R_bond_1 ]
# fill = dict( zip( keys, values) )
# out_data = pds.DataFrame( fill, index = ['k', 'v'])
# out_data.to_csv( all_out_path + 'total-sample_%s-band_based_BCGM-match_R_with-dered.csv' % band_str )



### === subsamples mass profile
# cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['low $M_{\\ast} ^{BCG}$', 'high $M_{\\ast} ^{BCG}$']
# file_s = 'BCG_M-bin'
# cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/'

# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'low $\\lambda$', 'high $\\lambda$' ]
# file_s = 'rich-bin'
# cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/rich_bin_fixed_bcgM/'


##.. re-binned rich subsamples
cat_lis = [ 'low-rich', 'hi-rich' ]
fig_name = [ 'low $\\lambda$', 'high $\\lambda$' ]
file_s = 'rich-bin'
out_path = '/home/xkchen/figs/extend_bcgM_cat/re_bin_rich/SM_pros/'
cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/re_bin_rich_bin/'



### === SM(r) calibration for subsamples
fix_R_dat = pds.read_csv( all_out_path + 'total-sample_%s-band_based_BCGM-match_R_with-dered.csv' % band_str )
R_fixed_M = [ np.array( fix_R_dat[ 'R_fixed_mean_M' ] )[0], np.array( fix_R_dat[ 'R_fixed_medi_M' ] )[0] ]

cali_factor = np.array( [] )

for mm in range( 2 ):

	#... mass profile
	m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str),)
	jk_R = np.array(m_dat['R'])
	surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
	surf_L = np.array( m_dat['lumi'] )

	N_grid = 250

	up_lim_R_0 = R_fixed_M[0]  ## mean Mstar calibrated
	up_lim_R_1 = R_fixed_M[1]  ## median Mstar calibrated

	cumu_M = cumu_mass_func( jk_R, surf_m, N_grid = N_grid )

	intep_Mf = interp.interp1d( jk_R, cumu_M, kind = 'cubic',)

	M_c0 = intep_Mf( up_lim_R_0 )
	M_c1 = intep_Mf( up_lim_R_1 )

	#... catalog infor.
	p_dat = pds.read_csv( cat_path + '%s_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	p_obs_z, p_rich  = np.array( p_dat['z']), np.array( p_dat['rich'])

	#. Mass unit : M_sun / h^2
	p_lgM = np.array( p_dat['lg_Mstar'])


	lg_Mean = np.log10( np.mean(10**p_lgM / h**2) )
	lg_Medi = np.log10( np.median(10**p_lgM / h**2) )


	devi_Mean = lg_Mean - np.log10( M_c0 )
	devi_Medi = lg_Medi - np.log10( M_c1 )

	medi_off_surf_M = surf_m * 10**( devi_Medi )
	mean_off_surf_M = surf_m * 10**( devi_Mean )

	#. save the calibrated SM profiles
	keys = ['R', 'medi_correct_surf_M', 'mean_correct_surf_M', 'surf_M_err']
	values = [ jk_R, medi_off_surf_M, mean_off_surf_M, surf_m_err ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str),)

	cali_factor = np.r_[ cali_factor, [ devi_Medi, devi_Mean ] ]

	print( 'lg_M devi*****' )
	print( 10**devi_Mean )
	print( 10**devi_Medi )

#. save the calibrated factors
keys = ['low_medi_devi', 'low_mean_devi', 'high_medi_devi', 'high_mean_devi']
values = list( cali_factor )
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
out_data.to_csv( out_path + '%s_%s-band-based_M_calib-f_with-dered.csv' % (file_s, band_str),)


### === SM(r) ratio to the entire sample
dat = pds.read_csv( all_out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
tot_R = np.array(dat['R'])
tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

interp_M_f = interp.interp1d( tot_R, tot_surf_m, kind = 'linear',)


#. surface mass ratio with fixed-R mass correction
calib_cat = pds.read_csv( out_path + '%s_%s-band-based_M_calib-f_with-dered.csv' % (file_s, band_str),)

medi_M_offset = [ np.array(calib_cat['low_medi_devi'])[0], np.array(calib_cat['high_medi_devi'])[0] ]
mean_M_offset = [ np.array(calib_cat['low_mean_devi'])[0], np.array(calib_cat['high_mean_devi'])[0] ]


add_off = [0.01, -0.01] # based on the average BCG Mstar difference
# add_off = [0.0, -0.0]

for mm in range( 2 ):

	N_samples = 30
	jk_sub_m_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi_with-dered.csv'

	tmp_r, tmp_mean_ratio = [], []
	tmp_medi_ratio = []

	for nn in range( N_samples ):

		o_dat = pds.read_csv( jk_sub_m_file % nn,)

		tt_r = np.array( o_dat['R'] )
		tt_M = np.array( o_dat['surf_mass'] )

		tt_medi_off_M = tt_M * 10**(medi_M_offset[ mm ] + add_off[mm] )
		tt_mean_off_M = tt_M * 10**(mean_M_offset[ mm ] + add_off[mm] )

		idx_lim = ( tt_r >= np.nanmin( tot_R ) ) & ( tt_r <= np.nanmax( tot_R ) )

		lim_R = tt_r[ idx_lim ]
		lim_medi_off_M = tt_medi_off_M[ idx_lim ]
		lim_mean_off_M = tt_mean_off_M[ idx_lim ]

		com_M = interp_M_f( lim_R )

		sub_ratio = np.zeros( len(tt_r),)
		sub_ratio[ idx_lim ] = lim_medi_off_M / com_M
		sub_ratio[ idx_lim == False ] = np.nan

		sub_ratio_1 = np.zeros( len(tt_r),)
		sub_ratio_1[ idx_lim ] = lim_mean_off_M / com_M
		sub_ratio_1[ idx_lim == False ] = np.nan

		tmp_r.append( tt_r )
		tmp_medi_ratio.append( sub_ratio )
		tmp_mean_ratio.append( sub_ratio_1 )

	aveg_R, aveg_medi_ratio, aveg_medi_ratio_err = arr_jack_func( tmp_medi_ratio, tmp_r, N_samples)[:3]
	aveg_R, aveg_mean_ratio, aveg_mean_ratio_err = arr_jack_func( tmp_mean_ratio, tmp_r, N_samples)[:3]

	keys = ['R', 'medi_ref_M/M_tot', 'medi_ref_M/M_tot-err', 'mean_ref_M/M_tot', 'mean_ref_M/M_tot-err']
	values = [ aveg_R, aveg_medi_ratio, aveg_medi_ratio_err, aveg_mean_ratio, aveg_mean_ratio_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample_with-dered.csv' % (cat_lis[mm], band_str),)

#. surface mass ratio without fixed-R mass correction
dt_eta_R, dt_eta, dt_eta_err = [], [], []

for mm in range( 2 ):

	N_samples = 30

	jk_sub_m_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi_with-dered.csv'

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

	keys = ['R', 'M/M_tot', 'M/M_tot-err']
	values = [ aveg_R, aveg_ratio, aveg_ratio_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + '%s_%s-band_aveg_M-ratio_to_total-sample_with-dered.csv' % (cat_lis[mm], band_str),)



### === ### figs for comparison

dt_R, dt_M, dt_Merr = [], [], []

for mm in range( 2 ):

	#... mass profile
	m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[mm], band_str),)
	jk_R = np.array(m_dat['R'])
	surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
	surf_L = np.array( m_dat['lumi'] )

	dt_R.append( jk_R )
	dt_M.append( surf_m )
	dt_Merr.append( surf_m_err )


dt_eta_R, dt_eta, dt_eta_err = [], [], []

for mm in range( 2 ):

	dat = pds.read_csv( out_path + '%s_%s-band_aveg_M-ratio_to_total-sample_with-dered.csv' % (cat_lis[mm], band_str),)

	tt_R = np.array( dat['R'] )
	tt_eta = np.array( dat['M/M_tot'] )
	tt_eta_err = np.array( dat['M/M_tot-err'] )

	dt_eta_R.append( tt_R )
	dt_eta.append( tt_eta )
	dt_eta_err.append( tt_eta_err )


dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[0], band_str),)
lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )
lo_mean_surf_M = np.array( dat['mean_correct_surf_M'] )

dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[1], band_str),)
hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )
hi_mean_surf_M = np.array( dat['mean_correct_surf_M'] )


lo_eat_dat = pds.read_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample_with-dered.csv' % (cat_lis[0], band_str),)
lo_eta_R = np.array(lo_eat_dat['R'])
lo_medi_eta, lo_medi_eta_err = np.array(lo_eat_dat['medi_ref_M/M_tot']), np.array(lo_eat_dat['medi_ref_M/M_tot-err'])
lo_mean_eta, lo_mean_eta_err = np.array(lo_eat_dat['mean_ref_M/M_tot']), np.array(lo_eat_dat['mean_ref_M/M_tot-err'])


hi_eat_dat = pds.read_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample_with-dered.csv' % (cat_lis[1], band_str),)
hi_eta_R = np.array(hi_eat_dat['R'])
hi_medi_eta, hi_medi_eta_err = np.array( hi_eat_dat['medi_ref_M/M_tot']), np.array( hi_eat_dat['medi_ref_M/M_tot-err'])
hi_mean_eta, hi_mean_eta_err = np.array( hi_eat_dat['mean_ref_M/M_tot']), np.array( hi_eat_dat['mean_ref_M/M_tot-err'])


line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

fig = plt.figure( figsize = (18.4, 4.8) )

ax0 = fig.add_axes([0.05, 0.33, 0.27, 0.63])
sub_ax0 = fig.add_axes([0.05, 0.12, 0.27, 0.21])

ax1 = fig.add_axes([0.37, 0.33, 0.27, 0.63])
sub_ax1 = fig.add_axes([0.37, 0.12, 0.27, 0.21])

ax2 = fig.add_axes([0.69, 0.33, 0.27, 0.63])
sub_ax2 = fig.add_axes([0.69, 0.12, 0.27, 0.21])

#... ratio without aperture_Mass calibration
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
sub_ax0.set_ylim( 0.45, 1.55 )
sub_ax0.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax0.set_xticklabels( [] )

#... ratio without aperture_Mass calibration (median of BCG Mstar)
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
ax1.annotate( text = 'After calibration' + '\n' + '(median $M_{\\ast}^{BCG}$)', xy = (0.55, 0.75), 
				xycoords = 'axes fraction', fontsize = 16, color = 'k',)

sub_ax1.plot( lo_eta_R / 1e3, lo_medi_eta, ls = '--', color = line_c[0], alpha = 0.75,)
sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_medi_eta - lo_medi_eta_err, y2 = lo_medi_eta + lo_medi_eta_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_medi_eta, ls = '-', color = line_c[1], alpha = 0.75,)
sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_medi_eta - hi_medi_eta_err, y2 = hi_medi_eta + hi_medi_eta_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( tot_R / 1e3, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale( 'log' )
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 16,)

sub_ax1.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )
sub_ax1.set_ylim( 0.45, 1.55 )
sub_ax1.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax1.set_xticklabels( [] )

#... ratio without aperture_Mass calibration (mean of BCG Mstar)
ax2.plot( lo_R/ 1e3, lo_mean_surf_M, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0],)
ax2.fill_between( lo_R/ 1e3, y1 = lo_mean_surf_M - lo_surf_M_err, 
	y2 = lo_mean_surf_M + lo_surf_M_err, color = line_c[0], alpha = 0.15,)

ax2.plot( hi_R/ 1e3, hi_mean_surf_M, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1],)
ax2.fill_between( hi_R/ 1e3, y1 = hi_mean_surf_M - hi_surf_M_err, 
	y2 = hi_mean_surf_M + hi_surf_M_err, color = line_c[1], alpha = 0.15,)

ax2.plot( tot_R/ 1e3, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
ax2.fill_between( tot_R/ 1e3, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.15,)

ax2.set_xlim( 1e-2, 1e0 )
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim( 5e3, 2e9 )
ax2.legend( loc = 3, frameon = False, fontsize = 16,)
ax2.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 16,)
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax2.annotate( text = 'After calibration' + '\n' + '(mean $M_{\\ast}^{BCG}$)', xy = (0.55, 0.75), 
				xycoords = 'axes fraction', fontsize = 16, color = 'k',)

sub_ax2.plot( lo_eta_R / 1e3, lo_mean_eta, ls = '--', color = line_c[0], alpha = 0.75,)
sub_ax2.fill_between( lo_eta_R / 1e3, y1 = lo_mean_eta - lo_mean_eta_err, y2 = lo_mean_eta + lo_mean_eta_err, color = line_c[0], alpha = 0.15,)

sub_ax2.plot( hi_eta_R / 1e3, hi_mean_eta, ls = '-', color = line_c[1], alpha = 0.75,)
sub_ax2.fill_between( hi_eta_R / 1e3, y1 = hi_mean_eta - hi_mean_eta_err, y2 = hi_mean_eta + hi_mean_eta_err, color = line_c[1], alpha = 0.15,)

sub_ax2.plot( tot_R / 1e3, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)

sub_ax2.set_xlim( ax2.get_xlim() )
sub_ax2.set_xscale( 'log' )
sub_ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 16,)

sub_ax2.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )
sub_ax2.set_ylim( 0.45, 1.55 )
sub_ax2.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax2.set_xticklabels( [] )

plt.savefig('/home/xkchen/%s_mass-reatio_compare.png' % file_s, dpi = 300)
plt.close()

