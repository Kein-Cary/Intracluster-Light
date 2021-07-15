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
import scipy.stats as sts
from scipy.interpolate import splev, splrep
from scipy import integrate as integ

from fig_out_module import color_func
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func
from img_random_SB_fit import cc_rand_sb_func

##...constant
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2asec = U.rad.to(U.arcsec)
band = [ 'r', 'g', 'i' ]
l_wave = np.array( [6166, 4686, 7480] )
## solar Magnitude corresponding to SDSS filter
Mag_sun = [ 4.65, 5.11, 4.53 ]

### === color to mass (my fitting)
def gr_ri_band_c2m_func(g2r_arr, r2i_arr, i_lumi_arr, fit_params):

	a_i, b_i, c_i, d_i = fit_params[:]
	lg_Lumi = np.log10( i_lumi_arr )

	fit_lg_m = a_i * g2r_arr + b_i * r2i_arr + c_i * lg_Lumi + d_i
	M = 10**fit_lg_m
	return M

### === profile measurement
def SB_to_Lumi_func(sb_arr, obs_z, band_s):

	if band_s == 'r':
		Mag_dot = Mag_sun[0]

	if band_s == 'g':
		Mag_dot = Mag_sun[1]

	if band_s == 'i':
		Mag_dot = Mag_sun[2]

	Da_obs = Test_model.angular_diameter_distance( obs_z ).value
	phy_S = 1 * Da_obs**2 * 1e6 / rad2asec**2

	L_sun = 10**( -0.4 * (sb_arr - Mag_dot) )
	phy_SB = L_sun / phy_S # L_sun / kpc^2

	return phy_SB * 1e-6

def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )

	for ii in range( NR ):

		new_rp = np.logspace(0, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )
		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

def get_c2mass_func( r_arr, band_str, sb_arr, color_arr, z_obs, N_grid = 100, fit_file, out_file = None):
	"""
	band_str : use which bands as bsed luminosity to estimate, the second str is the band information.
	sb_arr : in terms of absolute magnitude
	color_arr : for two color arr case, the first one must be g-r
	"""
	band_id = band.index( band_str[-1] )

	t_Lumi = SB_to_Lumi_func( sb_arr, z_obs, band[ band_id ] ) ## in unit L_sun / pc^2
	t_Lumi = 1e6 * t_Lumi ## in unit L_sun / kpc^2

	fit_dat = pds.read_csv( fit_file )
	a_fit = np.array( fit_dat['a'] )[0]
	b_fit = np.array( fit_dat['b'] )[0]
	c_fit = np.array( fit_dat['c'] )[0]
	d_fit = np.array( fit_dat['d'] )[0]

	fit_params = [ a_fit, b_fit, c_fit, d_fit ]
	# print( fit_params )
	g2r_arr, r2i_arr = color_arr[0], color_arr[1]
	t_mass = gr_ri_band_c2m_func(g2r_arr, r2i_arr, t_Lumi, fit_params)

	## cumulative mass
	cumu_mass = cumu_mass_func( r_arr, t_mass, N_grid = N_grid )

	if out_file is not None:
		keys = ['R', 'surf_mass', 'cumu_mass', 'lumi']
		values = [r_arr, t_mass, cumu_mass, t_Lumi]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_file )

	return

### === SB profile and jackknife average
def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def jk_sub_SB_func(N_samples, jk_sub_sb, BG_file, out_sub_sb):

	### measure BG-sub SB for jack-sub sample
	for nn in range( N_samples ):

		with h5py.File( jk_sub_sb % nn, 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
			npix = np.array(f['npix'])

		id_Nul = npix < 1
		c_r_arr[ id_Nul ] = np.nan
		c_sb_arr[ id_Nul ] = np.nan
		c_sb_err[ id_Nul ] = np.nan

		## BG_file
		cat = pds.read_csv( BG_file )
		( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD) = ( np.array(cat['e_a'])[0], np.array(cat['e_b'])[0], 
				np.array(cat['e_x0'])[0], np.array(cat['e_A'])[0], np.array(cat['e_alpha'])[0], 
				np.array(cat['e_B'])[0], np.array(cat['offD'])[0] )

		I_e, R_e = np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
		sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

		full_r_fit = cc_rand_sb_func(c_r_arr, e_a, e_b, e_x0, e_A, e_alpha, e_B)
		full_BG = full_r_fit - offD + sb_2Mpc
		devi_sb = c_sb_arr - full_BG

		keys = ['R', 'BG_sub_SB', 'sb_err']
		values = [ c_r_arr, devi_sb, c_sb_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_sub_sb % nn )

	return

def jk_sub_Mass_func(N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl, z_obs, fit_file, sub_SB_file_item = None,):
	### measure surface mass of sub sample
	# Dl : the luminosity distance

	if sub_SB_file_item is None:
		R_item, SB_item, sb_err_item = ['R', 'BG_sub_SB', 'sb_err']

	if sub_SB_file_item is not None:
		R_item, SB_item, sb_err_item = sub_SB_file_item[:]

	for nn in range( N_samples ):

		#... r-band
		r_dat = pds.read_csv( sub_SB_file % ('r', nn),)
		r_R, r_sb, r_sb_err = np.array( r_dat[ R_item ]), np.array( r_dat[ SB_item ]), np.array( r_dat[ sb_err_item ])

		idx_lim = ( r_R >= low_R_lim ) & ( r_R <= up_R_lim )
		idnan = np.isnan( r_sb )
		id_lim = (idnan == False) & ( idx_lim )

		r_R, r_sb, r_sb_err = r_R[ id_lim ], r_sb[ id_lim ], r_sb_err[ id_lim ]

		#... g-band
		g_dat = pds.read_csv( sub_SB_file % ('g', nn),)
		g_R, g_sb, g_sb_err = np.array( g_dat[ R_item ]), np.array( g_dat[ SB_item ]), np.array( g_dat[ sb_err_item ])

		idx_lim = ( g_R >= low_R_lim ) & ( g_R <= up_R_lim )
		idnan = np.isnan( g_sb )
		id_lim = (idnan == False) & ( idx_lim )

		g_R, g_sb, g_sb_err = g_R[ id_lim ], g_sb[ id_lim ], g_sb_err[ id_lim ]

		#... i-band
		i_dat = pds.read_csv( sub_SB_file % ('i', nn),)
		i_R, i_sb, i_sb_err = np.array( i_dat[ R_item ]), np.array( i_dat[ SB_item ]), np.array( i_dat[ sb_err_item ])

		idx_lim = ( i_R >= low_R_lim ) & ( i_R <= up_R_lim )
		idnan = np.isnan( i_sb )
		id_lim = (idnan == False) & ( idx_lim )

		i_R, i_sb, i_sb_err = i_R[ id_lim ], i_sb[ id_lim ], i_sb_err[ id_lim ]

		gr_arr, gr_err = color_func( g_sb, g_sb_err, r_sb, r_sb_err )
		ri_arr, ri_err = color_func( r_sb, r_sb_err, i_sb, i_sb_err )

		c_arr = np.array( [ gr_arr, ri_arr ] )

		#... filter data used to calculate luminosity
		dat_nd = pds.read_csv( sub_SB_file % ( band_str[-1], nn),)
		r_nd, sb_nd, err_nd = np.array( dat_nd[ R_item ]), np.array( dat_nd[ SB_item ]), np.array( dat_nd[ sb_err_item ])

		idx_lim = ( r_nd >= low_R_lim ) & ( r_nd <= up_R_lim )
		idnan = np.isnan( sb_nd )
		id_lim = (idnan == False) & ( idx_lim )

		r_nd, sb_nd, err_nd = r_nd[ id_lim], sb_nd[ id_lim], err_nd[ id_lim]

		mag_arr = 22.5 - 2.5 * np.log10( sb_nd )
		Mag_arr = mag_arr - 5 * np.log10( Dl * 10**6 / 10)

		out_m_file = out_file % nn
		get_c2mass_func( r_nd, band_str, Mag_arr, c_arr, z_obs, fit_file = fit_file, out_file = out_m_file )

	return

def aveg_mass_pro_func(N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file, M_cov_file = None):

	### jack mean and figs
	tmp_r, tmp_mass = [], []
	tmp_c_mass, tmp_lumi = [], []

	for nn in range( N_samples ):

		o_dat = pds.read_csv( jk_sub_m_file % nn,)

		# tmp_r.append( o_dat['R'] )
		# tmp_mass.append( o_dat['surf_mass'] )

		# tmp_c_mass.append( o_dat['cumu_mass'] )
		# tmp_lumi.append( o_dat['lumi'] )

		sub_R, sub_mass = np.array( o_dat['R'] ), np.array( o_dat['surf_mass'] )
		sub_c_mass, sub_lumi = np.array( o_dat['cumu_mass'] ), np.array( o_dat['lumi'] )

		id_nn_0 = np.isnan( sub_mass )
		id_nn_1 = np.isnan( sub_c_mass )
		id_nn_2 = np.isnan( sub_lumi )

		id_nn = id_nn_0 | id_nn_1 | id_nn_2

		tmp_r.append( sub_R[ id_nn == False ] )
		tmp_mass.append( sub_mass[ id_nn == False ] )

		tmp_c_mass.append( sub_c_mass[ id_nn == False ] )
		tmp_lumi.append( sub_lumi[ id_nn == False ] )

	### jack-mean pf mass and lumi profile
	aveg_R, aveg_surf_m, aveg_surf_m_err = arr_jack_func( tmp_mass, tmp_r, N_samples)[:3]
	aveg_R, aveg_cumu_m, aveg_cumu_m_err = arr_jack_func( tmp_c_mass, tmp_r, N_samples)[:3]
	aveg_R, aveg_lumi, aveg_lumi_err = arr_jack_func( tmp_lumi, tmp_r, N_samples)[:3]
	aveg_R, aveg_lgM, aveg_lgM_err = arr_jack_func( np.log10(tmp_mass), tmp_r, N_samples)[:3]

	keys = ['R', 'surf_mass', 'surf_mass_err', 'cumu_mass', 'cumu_mass_err', 'lumi', 'lumi_err', 'lg_M', 'lg_M_err']
	values = [ aveg_R, aveg_surf_m, aveg_surf_m_err, aveg_cumu_m, aveg_cumu_m_err, aveg_lumi, aveg_lumi_err, aveg_lgM, aveg_lgM_err]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( jk_aveg_m_file )

	### cov_arr of mass profile
	#.. use lg_mass to calculate cov_arr to avoid to huge value occur 
	lg_M_arr = np.log10( tmp_mass )

	R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, list(lg_M_arr), id_jack = True)

	with h5py.File( lgM_cov_file, 'w') as f:
		f['R_kpc'] = np.array( R_mean )
		f['cov_MX'] = np.array( cov_MX )
		f['cor_MX'] = np.array( cor_MX )

	if M_cov_file is not None:

		R_mean, cov_MX, cor_MX = cov_MX_func( tmp_r, tmp_mass, id_jack = True)

		with h5py.File( M_cov_file, 'w') as f:
			f['R_kpc'] = np.array( R_mean )
			f['cov_MX'] = np.array( cov_MX )
			f['cor_MX'] = np.array( cor_MX )

	return

