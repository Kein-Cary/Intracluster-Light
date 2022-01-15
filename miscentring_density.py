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
from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# cosmology model
# Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
Test_model = apcy.Planck15.clone(H0 = 67.66, Om0 = 0.3111)

H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

Omega_b = Test_model.Ob0
Omega_dm = Test_model.Odm0

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

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### setting for auto-correlation function
from halomod import DMHaloModel
import halomod
import hmf

import camb
from camb import model, initialpower
from cluster_toolkit import xi

def pk_at_r(R, z0, out_file,):

	r = R / 1e3 # Mpc / h
	K = 1 / r

	#Set cosmological parameters
	pars = camb.CAMBparams()
	pars.set_cosmology( H0 = H0, ombh2 = Omega_b * h**2, omch2 = Omega_dm * h**2)
	pars.set_dark_energy(w = -1.0)
	pars.InitPower.set_params( ns = 0.965)

	#This sets the k limits and specifies redshifts
	pars.set_matter_power( redshifts = [z0,], kmax = K.max(),)

	#Non-Linear spectra (Halofit)
	pars.NonLinear = model.NonLinear_both
	results = camb.get_results(pars)
	results.calc_power_spectra( pars )
	khnl_0, znl_0, pknl_0 = results.get_matter_power_spectrum( minkh = K.min(), maxkh = K.max(), npoints = len(R) )

	### set no-lim for larger range of k for xi_mm integrate
	khnl_1, znl_1, pknl_1 = results.get_nonlinear_matter_power_spectrum()

	out_arr = np.array( [ khnl_1, pknl_1[0] ] ).T
	np.savetxt( out_file, out_arr )

	return khnl_0, pknl_0[0]

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

	rho_m = rho_c * Omega_m * ( z + 1 )**3 / Ez**2 ## mean matter density of universe at z

	return rho_c, rho_m

def rho_nfw_delta_m(r, z, c_mass, lgM, v_m = 200,):

	Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
	Hz = H0 * Ez

	Qc = kpc2m / Msun2kg
	rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

	rhoc = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

	omega_mz = Omega_m * (z + 1)**3 / Ez**2
	rhom = rhoc * omega_mz

	delta_c = v_m * c_mass**3 / ( 3 * ( np.log(1 + c_mass) - c_mass / ( 1 + c_mass) ) ) 

	M = 10**lgM # in unit of M_sun / h

	r_200m = ( 3 * M / (4 * np.pi * rhom * v_m) )**(1/3)
	rs = r_200m / c_mass

	rho = delta_c * rhom / ( (r / rs) * (1 + r / rs)**2 ) # in unit of M_sun * h^2 / kpc^3

	return rho

def rho_nfw_delta_c(r, z, c_mass, lgM, v_m = 200):

	Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
	Hz = H0 * Ez

	Qc = kpc2m / Msun2kg
	rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

	rhoc = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

	delta_c = v_m * c_mass**3 / ( 3 * ( np.log(1 + c_mass) - c_mass / ( 1 + c_mass) ) ) 

	M = 10**lgM # in unit of M_sun / h

	r_200c = ( 3 * M / (4 * np.pi * rhoc * v_m) )**(1/3)
	rs = r_200c / c_mass

	rho = delta_c * rhoc / ( (r / rs) * (1 + r / rs)**2 ) # in unit of M_sun * h^2 / kpc^3

	return rho

def xi_1h_func(r, z, c_mass, lgM, v_m):
	## r : in unit of kpc / h
	rho_m = rhom_set( z )[1]
	xi_1h = rho_nfw_delta_m( r, z, c_mass, lgM, v_m = v_m,) / rho_m - 1

	return xi_1h

def xi_2h_func( r, bias, z0):

	## r : in unit of kpc / h
	R = r / 1e3
	cosmos = apcy.FlatLambdaCDM(H0 = H0, Om0 = Omega_m, Tcmb0 = Test_model.Tcmb0, Neff = 3.05, Ob0 = Omega_b,)
	A_mode = DMHaloModel( rmin = R.min(), rmax = R.max(), rnum = len(R), z = z0, cosmo_model = cosmos)
	# xi_2h = A_mode.corr_2h_auto_matter
	xi_2h = A_mode.corr_linear_mm

	return xi_2h * bias

def xi_hm_func(r, z0, c_mass, lgM, bias, v_m,):

	xi_1h = xi_1h_func( r, z0, c_mass, lgM, v_m = v_m,)
	xi_2h = xi_2h_func( r, bias, z0)

	xi_hm = np.max( [xi_1h, xi_2h], axis = 0)

	return xi_hm

def sigma_rp_func( rp, z, c_mass, lgM, bias, v_m):

	rho_m = rhom_set( z )

	NR = len( rp )

	def tmp_rho(x, rp, z, c_mass, lgM, bias, v_m):

		r = np.sqrt( x**2 + rp**2)

		return rho_nfw_func( r, z, c_mass, lgM, v_m = v_m,) / rho_m - 1

	sigma_arr = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		I_arr, I_err = integ.quad( tmp_rho, -np.inf, np.inf, args = (rp[ii], z, c_mass, lgM, bias, v_m), )

		sigma_ii = I_arr * rho_m

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

		surf_dens_arr = np.zeros( (NR_off, N_theta), dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )

			d_tt_0 = time.time()
			surf_dens_arr[jj,:] = sigma_rp_func( r_cir, z, c_mass, lgM, bias, v_m)
			print( time.time() - d_tt_0 )

		## integration on theta
		medi_surf_dens = ( surf_dens_arr[:,1:] + surf_dens_arr[:,:-1] ) / 2
		sum_theta_fdens = np.sum( medi_surf_dens * d_theta, axis = 1) / ( 2 * np.pi )

		## integration on r_off
		integ_f = sum_theta_fdens * off_pdf
		medi_integ_f = ( integ_f[1:] + integ_f[:-1] ) / 2

		surf_dens_ii = np.sum( medi_integ_f * dr_off )
		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	return off_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, bias, v_m ):

	off_sigma = off_sigma_func( rp, sigma_off, z, c_mass, lgM, bias, v_m,)
	norm_sigma = sigma_rp_func( rp, z, c_mass, lgM, bias, v_m,)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return norm_sigma, obs_sigma

def integ_pk_func(R, pk_file,):

	r = R / 1e3

	dd_arr = np.loadtxt( pk_file )

	dd_khnl = dd_arr[:, 0]
	dd_pknl = dd_arr[:, 1]

	intep_pk = interp.interp1d( dd_khnl, dd_pknl, kind = 'cubic', )

	k_min = dd_khnl.min()
	k_max = dd_khnl.max()

	new_k = np.logspace( np.log10( k_min ), np.log10( k_max ), 1e4)
	new_pk = intep_pk( np.log10( new_k ) )

	Nr = len(R)
	xi_arr = np.zeros( Nr, )

	for jj in range( Nr ):

		integ_f = new_k**2 * new_pk * ( np.sin(new_k * r[jj]) / (new_k * r[jj]) )

		xi_arr[jj] = integ.simps( integ_f, new_k ) / ( 2 * np.pi**2 )

	return xi_arr

sigma_off = 230 # kpc/h
f_off = 0.37

Mh0 = 14 # M_sun / h

c_mass = 5
v_m = 200
z0 = 0.65 #0.25
bias = 2.9


# R = np.logspace(0, 3.5, 1000) # kpc / h

### pk integrate
# ...??? oscillation
R = np.logspace( -3, np.log10(2e2), 1000) * 1e3 # kpc / h

out_pk_file = '/home/xkchen/figs/pk_arr.dat'
khnl, pknl = pk_at_r( R, z0, out_pk_file )


# pk_file = '/home/xkchen/figs/pk_arr.dat'
# xi_mm = integ_pk_func( R, pk_file,)


# plt.figure()
# plt.plot( R, xi_mm, 'r-', alpha = 0.5,)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('/home/xkchen/xi-mm_view.png', dpi = 300)
# plt.show()


### check xi_2h
xi_mm = xi.xi_mm_at_r( R / 1e3, khnl, pknl )
xi_nfw = xi.xi_nfw_at_r( R / 1e3, 1e14, c_mass, Omega_m)

xi_2h = xi_2h_func( R, 1, z0) ## set bias = 1

plt.figure()
plt.plot(R, xi_2h, 'r-', alpha = 0.5)
plt.plot(R, xi_mm, 'b--', alpha = 0.5)
plt.xscale('log')
plt.yscale('log')
plt.show()

xi_hm = xi_hm_func( R, z0, c_mass, Mh0, bias, v_m)
xi_2h = xi_2h_func( R, bias, z0)
xi_1h = xi_1h_func( R, z0, c_mass, Mh0, v_m)

plt.figure()
ax = plt.subplot( 111 )
ax.plot( R / 1e3, xi_hm, 'r-', alpha = 0.5, label = '$\\xi_{hm}$')
ax.plot( R / 1e3, xi_2h, 'g:', alpha = 0.5, label = '$\\xi_{2h}$')
ax.plot( R / 1e3, xi_1h, 'c-', alpha = 0.5, label = '$\\xi_{1h}$')

ax.legend( loc = 1 )

ax.set_xlim(1e-2, 1e1)
ax.set_ylim(1e0, 1e5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('r[Mpc/h]')

plt.savefig('/home/xkchen/xi_hm_test.png', dpi = 300)
plt.show()

