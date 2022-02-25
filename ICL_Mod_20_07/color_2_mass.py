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
from surface_mass_density import cumu_mass_func


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

#... color profile
def color_func( flux_arr_0, flux_err_0, flux_arr_1, flux_err_1):

	mag_arr_0 = 22.5 - 2.5 * np.log10( flux_arr_0 )
	mag_arr_1 = 22.5 - 2.5 * np.log10( flux_arr_1 )

	color_pros = mag_arr_0 - mag_arr_1

	sigma_0 = 2.5 * flux_err_0 / (np.log(10) * flux_arr_0 )
	sigma_1 = 2.5 * flux_err_1 / (np.log(10) * flux_arr_1 )

	color_err = np.sqrt( sigma_0**2 + sigma_1**2 )

	return color_pros, color_err

### === ###... g-r color based
def gr_band_m2l_func(g2r_arr, r_lumi_arr):

	a_g2r = -0.306
	b_g2r = 1.097
	lg_m2l = a_g2r + b_g2r * g2r_arr
	return lg_m2l

def gr_band_c2m_func(g2r_arr, r_lumi_arr):
	a_g2r = -0.306
	b_g2r = 1.097

	lg_m2l = a_g2r + b_g2r * g2r_arr
	M = r_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

def rg_band_m2l_func(g2r_arr, g_lumi_arr):

	a_r2g = -0.499
	b_r2g = 1.519
	lg_m2l = a_r2g + b_r2g * g2r_arr
	return lg_m2l

def rg_band_c2m_func(g2r_arr, g_lumi_arr):

	a_r2g = -0.499
	b_r2g = 1.519

	lg_m2l = a_r2g + b_r2g * g2r_arr
	M = g_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

### === ###... g-i color based
def gi_band_m2l_func(g2i_arr, i_lumi_arr):

	a_g2i = -0.152
	b_g2i = 0.518
	lg_m2l = a_g2i + b_g2i * g2i_arr
	return lg_m2l

def gi_band_c2m_func(g2i_arr, i_lumi_arr):

	a_g2i = -0.152
	b_g2i = 0.518

	lg_m2l = a_g2i + b_g2i * g2i_arr
	M = i_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

def ig_band_m2l_func(g2i_arr, g_lumi_arr):

	a_i2g = -0.379
	b_i2g = 0.914
	lg_m2l = a_i2g + b_i2g * g2i_arr
	return lg_m2l

def ig_band_c2m_func(g2i_arr, g_lumi_arr):

	a_i2g = -0.379
	b_i2g = 0.914

	lg_m2l = a_i2g + b_i2g * g2i_arr
	M = g_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

### === ###... r-i color based
def ri_band_m2l_func(r2i_arr, i_lumi_arr):

	a_r2i = 0.006
	b_r2i = 1.114
	lg_m2l = a_r2i + b_r2i * r2i_arr
	return lg_m2l

def ri_band_c2m_func(r2i_arr, i_lumi_arr):

	a_r2i = 0.006
	b_r2i = 1.114

	lg_m2l = a_r2i + b_r2i * r2i_arr
	M = i_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

def ir_band_m2l_func(r2i_arr, r_lumi_arr):

	a_i2r = -0.022
	b_i2r = 1.431
	lg_m2l = a_i2r + b_i2r * r2i_arr
	return lg_m2l

def ir_band_c2m_func(r2i_arr, r_lumi_arr):

	a_i2r = -0.022
	b_i2r = 1.431

	lg_m2l = a_i2r + b_i2r * r2i_arr
	M = r_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

### === ###... three band combined
def gri_band_c2m_func(g2r_arr, i_lumi_arr):
	a_i = -0.222
	b_i = 0.864
	lg_m2l = a_i + b_i * g2r_arr

	M = i_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

def gir_band_c2m_func(g2i_arr, r_lumi_arr):
	a_r = -0.220
	b_r = 0.661
	lg_m2l = a_r + b_r * g2i_arr

	M = r_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

def rig_band_c2m_func(r2i_arr, g_lumi_arr):
	a_g = -0.106
	b_g = 1.982
	lg_m2l = a_g + b_g * r2i_arr

	M = g_lumi_arr * 10**( lg_m2l )
	## correction for h^2 term in Bell 2003
	M = M + 0.
	return M

### === ###... profile measurement
def SB_to_Lumi_func(sb_arr, obs_z, band_s):
	"""
	sb_arr : in unit of mag /arcsec^2, apparent magnitude
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

def get_c2mass_func( r_arr, band_str, sb_arr, color_arr, z_obs, N_grid = 100, out_file = None):
	"""
	band_str : use which bands as bsed luminosity to estimate, the second str is the band information.
	sb_arr : in terms of absolute magnitude 
	"""
	band_id = band.index( band_str[-1] )

	t_Lumi = SB_to_Lumi_func( sb_arr, z_obs, band[ band_id ] ) ## in unit L_sun / pc^2
	t_Lumi = 10**6 * t_Lumi ## in unit L_sun / kpc^2

	if band_str == 'gi':
		t_mass = gi_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2

	if band_str == 'ig':
		t_mass = ig_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2


	if band_str == 'gr':
		t_mass = gr_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2

	if band_str == 'rg':
		t_mass = rg_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2


	if band_str == 'ri':
		t_mass = ri_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2

	if band_str == 'ir':
		t_mass = ir_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2


	if band_str == 'gri':
		t_mass = gri_band_c2m_func( color_arr, t_Lumi )

	if band_str == 'gir':
		t_mass = gir_band_c2m_func( color_arr, t_Lumi )

	if band_str == 'rig':
		t_mass = rig_band_c2m_func( color_arr, t_Lumi )

	## cumulative mass
	cumu_mass = cumu_mass_func( r_arr, t_mass, N_grid = N_grid )

	if out_file is not None:
		keys = ['R', 'surf_mass', 'cumu_mass', 'lumi']
		values = [r_arr, t_mass, cumu_mass, t_Lumi]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_file )

	return

### === ### mean of jackknife sub-sample
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
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

def jk_sub_Mass_func(N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl, z_obs, 
	c_inv = False, sub_SB_file_item = None ):
	# sub_SB_file_item : the columns name of sub SB profile .csv files
	# Dl : the luminosity distance
	# c_inv : if change the position of the order of the two SB profile array

	### measure surface mass of sub sample
	for nn in range( N_samples ):
		#... sb array 1
		if sub_SB_file_item is None:
			dat_0 = pds.read_csv( sub_SB_file % ( band_str[0], nn),)
			r_0, sb_0, err_0 = np.array( dat_0['R']), np.array( dat_0['BG_sub_SB']), np.array( dat_0['sb_err'])

		if sub_SB_file_item is not None:
			R_item, SB_item, sb_err_item = sub_SB_file_item[:]
			dat_0 = pds.read_csv( sub_SB_file % ( band_str[0], nn),)
			r_0, sb_0, err_0 = np.array( dat_0[ R_item ]), np.array( dat_0[ SB_item ]), np.array( dat_0[ sb_err_item ])

		idx_lim = ( r_0 >= low_R_lim ) & ( r_0 <= up_R_lim )
		idnan = np.isnan( sb_0 )
		id_lim = (idnan == False) & ( idx_lim )

		r_0, sb_0, err_0 = r_0[ id_lim ], sb_0[ id_lim ], err_0[ id_lim ]

		#... SB array 2
		if sub_SB_file_item is None:
			dat_1 = pds.read_csv( sub_SB_file % ( band_str[1], nn),)
			r_1, sb_1, err_1 = np.array( dat_1['R']), np.array( dat_1['BG_sub_SB']), np.array( dat_1['sb_err'])

		if sub_SB_file_item is not None:
			R_item, SB_item, sb_err_item = sub_SB_file_item[:]
			dat_1 = pds.read_csv( sub_SB_file % ( band_str[1], nn),)
			r_1, sb_1, err_1 = np.array( dat_1[ R_item ]), np.array( dat_1[ SB_item ]), np.array( dat_1[ sb_err_item ])

		idx_lim = ( r_1 >= low_R_lim ) & ( r_1 <= up_R_lim )
		idnan = np.isnan( sb_1 )
		id_lim = (idnan == False) & ( idx_lim )

		r_1, sb_1, err_1 = r_1[ id_lim ], sb_1[ id_lim ], err_1[ id_lim ]


		if c_inv == True: 
			c_arr, c_err = color_func( sb_1, err_1, sb_0, err_0 )
		else:
			c_arr, c_err = color_func( sb_0, err_0, sb_1, err_1 )

		#... filter data used to calculate luminosity
		if sub_SB_file_item is None:
			dat_nd = pds.read_csv( sub_SB_file % ( band_str[-1], nn),)
			r_nd, sb_nd, err_nd = np.array( dat_nd['R']), np.array( dat_nd['BG_sub_SB']), np.array( dat_nd['sb_err'])

		if sub_SB_file_item is not None:
			R_item, SB_item, sb_err_item = sub_SB_file_item[:]
			dat_nd = pds.read_csv( sub_SB_file % ( band_str[-1], nn),)
			r_nd, sb_nd, err_nd = np.array( dat_nd[ R_item ]), np.array( dat_nd[ SB_item ]), np.array( dat_nd[ sb_err_item ])

		idx_lim = ( r_nd >= low_R_lim ) & ( r_nd <= up_R_lim )
		idnan = np.isnan( sb_nd )
		id_lim = (idnan == False) & ( idx_lim )

		r_nd, sb_nd, err_nd = r_nd[ id_lim], sb_nd[ id_lim], err_nd[ id_lim]

		#. use apparent magnitude for luminosity estimate
		mag_arr = 22.5 - 2.5 * np.log10( sb_nd )
		Mag_arr = mag_arr + 0.

		out_m_file = out_file % nn
		get_c2mass_func( r_1, band_str, Mag_arr, c_arr, z_obs, out_file = out_m_file )

	return

def aveg_mass_pro_func(N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file, M_cov_file = None):

	### jack mean and figs
	tmp_r, tmp_mass = [], []
	tmp_c_mass, tmp_lumi = [], []

	for nn in range( N_samples ):

		o_dat = pds.read_csv( jk_sub_m_file % nn,)
		sub_R, sub_mass = np.array( o_dat['R'] ), np.array( o_dat['surf_mass'] )
		sub_c_mass, sub_lumi = np.array( o_dat['cumu_mass'] ), np.array( o_dat['lumi'] )

		id_nn_0 = np.isnan( sub_mass )
		id_nn_1 = np.isnan( sub_c_mass )
		id_nn_2 = np.isnan( sub_lumi )

		id_nn = id_nn_0 | id_nn_1 | id_nn_2

		#. keep the array length and replace id_nn values
		sub_R[ id_nn ] = np.nan
		sub_mass[ id_nn ] = np.nan
		sub_c_mass[ id_nn ] = np.nan
		sub_lumi[ id_nn ] = np.nan

		tmp_r.append( sub_R )
		tmp_mass.append( sub_mass )
		tmp_c_mass.append( sub_c_mass )
		tmp_lumi.append( sub_lumi )

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
