import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.signal as signal
from scipy import interpolate as interp
from scipy import integrate as integ

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
from img_cat_param_match import match_func
from img_pre_selection import extra_match_func

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
band = ['r', 'g', 'i']

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

### === ### data load and figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

## fixed richness samples
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# file_s = 'BCG_Mstar_bin'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
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


#... sample properties
hi_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )
hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['BCG_age'] )

lo_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )
lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['BCG_age'] )

## mass profile calibration with BCG stellar mass comparison
lo_lg_medi_Mstar = np.log10( np.median( 10**lo_lgM / h**2 ) )
lo_lg_mean_Mstar = np.log10( np.mean( 10**lo_lgM / h**2 ) )

##
hi_lg_medi_Mstar = np.log10( np.median( 10**hi_lgM / h**2 ) )
hi_lg_mean_Mstar = np.log10( np.mean( 10**hi_lgM / h**2 ) )

# band_str = ['gi', 'gr', 'ri', 'ir', 'ig', 'rg', 'gri', 'gir', 'rig']

#... fitting test
band_str = ['gri']
out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'

for pp in range( 0,1 ):

	#...Mass profile
	# dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str[ pp]),)
	# lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	# dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str[ pp]),)
	# hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	#... fitting test
	dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str[ pp]),)
	lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str[ pp]),)
	hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )


	#... Mass profile integrate
	N_grid = 250

	# up_lim_R = 40
	up_lim_R = 47.69

	lo_cumu_M = cumu_mass_func( lo_R, lo_surf_M, N_grid = N_grid )
	lo_intep_Mf = interp.interp1d( lo_R, lo_cumu_M, kind = 'cubic', )

	lo_M_40 = lo_intep_Mf( up_lim_R )
	lo_lg_M40 = np.log10( lo_M_40 )

	lo_devi_mean = lo_lg_mean_Mstar - lo_lg_M40 # use for calibration
	lo_devi_medi = lo_lg_medi_Mstar - lo_lg_M40

	# lo_off_surf_M = lo_surf_M * 10**( lo_devi_mean )
	lo_off_surf_M = lo_surf_M * 10**( lo_devi_medi )

	keys = ['R', 'correct_surf_M', 'surf_M_err']
	values = [lo_R, lo_off_surf_M, lo_surf_M_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	# out_data.to_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str[ pp]),)
	out_data.to_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str[ pp]),)

	#... Mass profile integrate
	N_grid = 250
	hi_cumu_M = cumu_mass_func( hi_R, hi_surf_M, N_grid = N_grid )
	hi_intep_Mf = interp.interp1d( hi_R, hi_cumu_M, kind = 'cubic', )

	hi_M_40 = hi_intep_Mf( up_lim_R )
	hi_lg_M40 = np.log10( hi_M_40 )

	hi_devi_mean = hi_lg_mean_Mstar - hi_lg_M40
	hi_devi_medi = hi_lg_medi_Mstar - hi_lg_M40

	# hi_off_surf_M = hi_surf_M * 10**( hi_devi_mean )
	hi_off_surf_M = hi_surf_M * 10**( hi_devi_medi )

	keys = ['R', 'correct_surf_M', 'surf_M_err']
	values = [ hi_R, hi_off_surf_M, hi_surf_M_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	# out_data.to_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str[ pp]),)
	out_data.to_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str[ pp]),)


	#... save the calibration factor
	keys = ['low_medi_devi', 'low_mean_devi', 'high_medi_devi', 'high_mean_devi']
	values = [ lo_devi_medi, lo_devi_mean, hi_devi_medi, hi_devi_mean ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	# out_data.to_csv('/home/xkchen/tmp_run/data_files/figs/%s_gi-band-based_M_calib-f.csv' % file_s)
	out_data.to_csv(out_path + '%s_gri-band-based_M_calib-f.csv' % file_s)

raise

### ... for total cluster sample
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/'

for pp in range( 3 ):

	#...Mass profile
	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str[pp],)
	tt_R, tt_surf_M, tt_surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	#... Mass profile integrate
	N_grid = 250
	tt_cumu_M = cumu_mass_func( tt_R, tt_surf_M, N_grid = N_grid )
	tt_intep_Mf = interp.interp1d( tt_R, tt_cumu_M, kind = 'cubic', )

	tt_M_40 = tt_intep_Mf( 40 )
	tt_lg_M40 = np.log10( tt_M_40 )

	#... total sample
	tot_lgM = np.r_[ hi_lgM, lo_lgM ]

	medi_Mstar = np.log10( np.median( 10**tot_lgM / h**2 ) )
	mean_Mstar = np.log10( np.mean( 10**tot_lgM / h**2 ) )

	_devi_mean = mean_Mstar - tt_lg_M40
	_devi_medi = medi_Mstar - tt_lg_M40

	_off_surf_M = tt_surf_M * 10**( _devi_mean )
	keys = ['R', 'correct_surf_M', 'surf_M_err']
	values = [ tt_R, _off_surf_M, tt_surf_M_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % band_str[ pp],)
