import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ


### === ### cosmology
def input_cosm_model( get_model = None ):

	global cosmo

	if get_model is not None:
		cosmo = get_model

	else:
		### cosmology
		cosmo = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)

	return cosmo

def cosmos_param():

	global H0, h, Omega_m, Omega_lambda, Omega_k

	## cosmology params
	H0 = cosmo.H0.value
	h = H0 / 100
	Omega_m = cosmo.Om0
	Omega_lambda = 1.-Omega_m
	Omega_k = 1.- (Omega_lambda + Omega_m)

	return


### === ### func.s
def Ms_to_Mh_func( z0, Mg_star ):
	"""
	stellar-to-halo mass relation of satellite in Moster, Naab, and White, 2013 (MNW13).
	all mass in unit of M_sun
	"""

	M10 = 11.590
	M11 = 1.195

	N10 = 0.0351
	N11 = -0.0247

	belt0 = 1.376
	belt1 = -0.826

	gama0 = 0.608
	gama1 = 0.329

	##.
	lg_Mz = M10 + M11 * ( z0 / (1 + z0) )
	Nz = N10 + N11 * ( z0 / (1 + z0) )
	belt_z = belt0 + belt1 * ( z0 / (1 + z0) )
	gama_z = gama0 + gama1 * ( z0 / (1 + z0) )

	Mh = ( Mg_star / ( 2 * Nz ) ) * ( (Mg_star / 10**lg_Mz)**(-1 * belt_z) + (Mg_star / 10**lg_Mz)**gama_z )

	return Mh


def Mh_rich_R_func( z_x, lamda ):
	"""
	use formula Eq.12 of Simet et al. 2017, but without Error estimation
	"""

	lg_M0 = 14.369    ###. M_sun / h
	lamda_pivo = 40
	alpha_0 = 1.30

	M_200m = 10**lg_M0 * ( lamda / lamda_pivo )**alpha_0
	M_200m = M_200m / h    ###. M_sun

	##.
	Qc = kpc2m / Msun2kg # correction fractor of rho_c
	Ez = np.sqrt(Omega_m * (1 + z_x)**3 + Omega_k * (1 + z_x)**2 + Omega_lambda)
	Hz = H0 * Ez

	rho_c = Qc * (3 * Hz**2) / (8 * np.pi * G) # in unit Msun/kpc^3
	omega_z = Test_model.Om( z_x ) # density parameter
	rho_m = V_num * rho_c * omega_z
	R_200m = ( 3 * M_200m / (4 * np.pi * rho_m) )**(1/3) # in unit kpc

	return M_200m, R_200m


def Mh_c_func( zx, Mhx ):
	"""
	Mass-concentration relation in Duffy et all. 2008, Eq.4, table 1	
	"""

	Am = 10.14
	Bm = -0.081
	Cm = -1.01

	M_pivo = 2e12  ## M_sun / h

	C_x = Am * ( Mhx / M_pivo )**Bm * ( 1 + zx )**Cm

	return C_x

