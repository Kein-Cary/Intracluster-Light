import h5py
import numpy as np
import pandas as pds
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy

from scipy import interpolate as interp
from scipy import optimize
from scipy.integrate import cumtrapz
from scipy import integrate as integ
from surface_mass_density import sigma_m, sigmam

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
# Test_model = apcy.Planck15.clone(H0 = 67.66, Om0 = 0.3111) # for test

H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

Omega_b = Test_model.Ob0
Omega_dm = Test_model.Odm0

DH = vc/H0
G = C.G.value
M_sun = C.M_sun.value # kg

# constant
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

rad2arcsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7 # (erg/s/cm^2)
Jy = 10**(-23) # (erg/s)/cm^2/Hz
F0 = 3.631 * 10**(-6) * Jy
L_speed = C.c.value # m/s

### === ### auto-correlation function
from halomod import DMHaloModel
import halomod
import hmf

### === ### Zu et al. 2020, section 3.
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""

	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def rhom_set( z ):

	Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
	Hz = H0 * Ez

	Qc = kpc2m / Msun2kg
	rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

	rho_c = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

	rho_m = rho_c * Omega_m

	return rho_m

def rho_nfw_func(r, z, c_mass, lgM, v_m = 200,):

	Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
	Hz = H0 * Ez

	Qc = kpc2m / Msun2kg
	rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

	rho_c = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

	delta_c = v_m * c_mass**3 / ( 3 * ( np.log(1 + c_mass) - c_mass / ( 1 + c_mass) ) )

	rho_mean = v_m * rho_c * Omega_m

	M = 10**lgM # in unit of M_sun / h

	R_mean = (3 * M / (4 * np.pi * rho_mean) )**(1/3)
	rs = R_mean / c_mass

	rho = delta_c * rho_c * Omega_m / ( (r / rs) * (1 + r / rs)**2 ) # in unit of M_sun * h^2 / kpc^3

	return rho

def xi_1h_func(r, z, c_mass, lgM, v_m):
	## r : in unit of kpc / h
	rho_m = rhom_set( z )
	xi_1h = rho_nfw_func( r, z, c_mass, lgM, v_m = v_m,) / rho_m - 1

	return xi_1h

def xi_2h_func( r, bias, z0):
	## r : in unit of kpc / h
	R = r / 1e3
	A_mode = DMHaloModel( rmin = R.min(), rmax = R.max(), rnum = len(R), z = z0, cosmo_params = {'Om0' : Omega_m, 'H0' : H0}, )
	xi_2h = A_mode.corr_2h_auto_matter

	return xi_2h * bias

def xi_hm_func(r, z0, c_mass, lgM, bias, v_m,):

	xi_1h = xi_1h_func( r, z0, c_mass, lgM, v_m = v_m,)
	xi_2h = xi_2h_func( r, bias, z0)

	xi_hm = np.max( [xi_1h, xi_2h], axis = 0)

	return xi_hm

def sigma_rp_func( rp, z, c_mass, lgM, bias, v_m,):

	rho_m = rhom_set( z )

	r_pi = np.linspace( -5e3, 5e3, 1e5) ## in unit of kpc/h
	d_r_pi = np.diff( r_pi )

	NR = len( rp )

	sigma_arr = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		r = np.sqrt( rp[ii]**2 + r_pi**2 )

		# xi_hm = rho_nfw_func( r, z, c_mass, lgM, v_m = v_m,) / rho_m - 1
		xi_hm = xi_hm_func(r, z, c_mass, lgM, bias, v_m,)

		integ_f = rho_m * xi_hm

		sigma_ii = integ.simps( integ_f, r_pi )

		sigma_arr[ ii ] = sigma_ii

	return sigma_arr

def off_sigma_func( rp, sigma_off, z, c_mass, lgM, bias, v_m,):

	theta = np.linspace( 0, 2 * np.pi, 100)
	d_theta = np.diff( theta )

	NR = len( rp )

	r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
	off_pdf = mis_p_func( r_off, sigma_off )

	NR_off = len( r_off )

	surf_dens_off = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		surf_dens_arr = np.zeros( NR_off, dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )

			surf_dens_of_theta = sigma_rp_func( r_cir, z, c_mass, lgM, bias, v_m)

			## integration on theta
			surf_dens_arr[jj] = integ.simps( surf_dens_of_theta, theta) / ( 2 * np.pi )

		## integration on r_off
		integ_f = surf_dens_arr * off_pdf

		surf_dens_ii = integ.simps( integ_f, r_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	return off_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, bias, v_m ):

	off_sigma = off_sigma_func( rp, sigma_off, z, c_mass, lgM, bias, v_m,)
	norm_sigma = sigma_rp_func( rp, z, c_mass, lgM, bias, v_m,)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

sigma_off = 230 # kpc/h
f_off = 0.37

# r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
# pdf = mis_p_func(r_off, sigma_off)
# cumu_pdf = integ.cumtrapz( pdf, r_off )

Mh0 = 14 # M_sun / h
R = np.logspace( -1, 1, 1000) * 10**3 # kpc / h

c_mass = 5
v_m = 200
z0 = 0
bias = 2.9

tt0 = time.time()

# no_off_surf_dens = sigmam( R, Mh0, z0, c_mass)
no_off_surf_dens = sigma_rp_func( R, z0, c_mass, Mh0, bias, v_m )
off_surf_dens = obs_sigma_func( R, f_off, sigma_off, z0, c_mass, Mh0, bias, v_m )

plt.figure( )
plt.plot( R, no_off_surf_dens, 'r-', alpha = 0.5, label = 'no offset')
plt.plot( R, off_surf_dens, 'b--', alpha = 0.5, label = 'with offset')
plt.yscale( 'log' )
plt.xscale( 'log' )
plt.xlabel( 'R[ kpc / h]' )
plt.ylabel( '$\\Sigma(r) [M_{\\odot} h / kpc^2]$' )
plt.savefig( '/home/xkchen/miscentering_nfw_test.png', dpi = 300 )
plt.close()

print( time.time() - tt0 )

