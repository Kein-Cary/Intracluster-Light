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

## cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

band = [ 'r', 'g', 'i' ]
l_wave = np.array( [6166, 4686, 7480] )
## solar Magnitude corresponding to SDSS filter
Mag_sun = [ 4.65, 5.11, 4.53 ]

def gr_band_c2m_func(g2r_arr, r_lumi_arr):
	a_g2r = -0.306
	b_g2r = 1.097

	lg_m2l = a_g2r + b_g2r * g2r_arr
	M = r_lumi_arr * 10**( lg_m2l )

	## correction for h^2 term in Bell 2003
	M = M / h**2
	return M

def gi_band_c2m_func( g2i_arr, i_lumi_arr):

	a_g2i = -0.152
	b_g2i = 0.518

	lg_m2l = a_g2i + b_g2i * g2i_arr

	M = i_lumi_arr * 10**( lg_m2l )

	## correction for h^2 term in Bell 2003
	M = M / h**2
	return M

def SB_to_Lumi_func(sb_arr, obs_z, band_str):
	"""
	sb_arr : need in terms of absolute magnitude, in AB system
	"""
	if band_str == 'r':
		Mag_dot = Mag_sun[0]

	if band_str == 'g':
		Mag_dot = Mag_sun[1]

	if band_str == 'i':
		Mag_dot = Mag_sun[2]

	# luminosity, in unit of  L_sun / pc^2
	Lumi = 10**( -0.4 * (sb_arr - Mag_dot + 21.572 - 10 * np.log10( obs_z + 1 ) ) )

	return Lumi

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

def get_c2mass_func( r_arr, band_str, sb_arr, color_arr, z_obs, N_grid = 100, out_file = None):
	"""
	band_str : use which band as bsed luminosity to estimate
	sb_arr : in terms of absolute magnitude 
	"""
	band_id = band.index( band_str )

	t_Lumi = SB_to_Lumi_func( sb_arr, z_obs, band[ band_id ] ) ## in unit L_sun / pc^2
	t_Lumi = 10**6 * t_Lumi ## in unit L_sun / kpc^2

	if band_str == 'i':
		t_mass = gi_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2
	if band_str == 'r':
		t_mass = gr_band_c2m_func( color_arr, t_Lumi ) ## in unit M_sun / kpc^2

	## cumulative mass
	cumu_mass = cumu_mass_func( r_arr, t_mass, N_grid = N_grid )

	if out_file is not None:
		keys = ['R', 'surf_mass', 'cumu_mass', 'lumi']
		values = [r_arr, t_mass, cumu_mass, t_Lumi]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_file )

	return
