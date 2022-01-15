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
# Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311) # previous

## Zu 20
Test_model = apcy.Planck15.clone( H0 = 67.4, Om0 = 0.315,)

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

### === ### Einasto profile (colossus)
from colossus.cosmology import cosmology
from colossus.halo import profile_einasto
from colossus.halo import profile_nfw
from colossus.halo import profile_hernquist

cosmos_param = {'flat': True, 'H0': H0, 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8' : 0.811, 'ns': 0.965}
cosmology.addCosmology('myCosmo', cosmos_param )
cosmo = cosmology.setCosmology( 'myCosmo' )

### === ### moel calculation
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""

	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def cc_off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

	theta = np.linspace( 0, 2 * np.pi, 100)
	d_theta = np.diff( theta )
	N_theta = len( theta )

	try:
		NR = len( rp )
	except:
		rp = np.array( [rp] )
		NR = len( rp )

	r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
	off_pdf = mis_p_func( r_off, sigma_off )
	dr_off = np.diff( r_off )

	NR_off = len( r_off )

	surf_dens_off = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		surf_dens_arr = np.zeros( (NR_off, N_theta), dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )
			surf_dens_arr[jj,:] = sigmam( r_cir, lgM, z, c_mass,)

		## integration on theta
		medi_surf_dens = ( surf_dens_arr[:,1:] + surf_dens_arr[:,:-1] ) / 2
		sum_theta_fdens = np.sum( medi_surf_dens * d_theta, axis = 1) / ( 2 * np.pi )

		## integration on r_off
		integ_f = sum_theta_fdens * off_pdf

		medi_integ_f = ( integ_f[1:] + integ_f[:-1] ) / 2

		surf_dens_ii = np.sum( medi_integ_f * dr_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	if NR == 1:
		return off_sigma[0]
	return off_sigma

def aveg_sigma_func(rp, sigma_arr, N_grid = 1000):

	NR = len( rp )

	aveg_sigma = np.zeros( NR, dtype = np.float32 )

	tR = rp
	intep_sigma_F = interp.interp1d( tR , sigma_arr, kind = 'cubic', fill_value = 'extrapolate',)

	for ii in range( NR ):

		new_rp = np.logspace(-3, np.log10( tR[ii] ), N_grid)

		new_sigma = intep_sigma_F( new_rp )

		cumu_sigma = integ.simps( new_rp * new_sigma, new_rp)

		aveg_sigma[ii] = 2 * cumu_sigma / tR[ii]**2

	return aveg_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m, no_off_sigma):

	off_sigma = cc_off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * no_off_sigma

	return obs_sigma

### === ### test
def profile_check():

	Mh0 = 14.24 # M_sun / h
	c0 = [6.95, 5.87]
	sigma_off = [210, 230] ## kpc / h
	f_off = [0.20, 0.37]
	v_m = 200
	z0 = 0.242

	R = np.logspace(-2, np.log10(2e1), 100) * 10**3 # kpc / h

	N_grid = 100

	tmp_delta_sigma = []
	tmp_off_delta_sigma = []

	for ll in range( 2 ):

		norm_sigma = sigmam( R, Mh0, z0, c0[ll])
		mean_sigma = aveg_sigma_func( R, norm_sigma, N_grid = N_grid,)
		delta_sigma = mean_sigma - norm_sigma

		tmp_delta_sigma.append( delta_sigma )

		print( 'done!' )

		off_sigma = obs_sigma_func( R, f_off[ll], sigma_off[ll], z0, c0[ll], Mh0, v_m, norm_sigma)
		mean_off_sigma = aveg_sigma_func( R, off_sigma, N_grid = N_grid,)
		off_delta_sigma = mean_off_sigma - off_sigma

		tmp_off_delta_sigma.append( off_delta_sigma )

	plt.figure()

	plt.plot( R / 1e3, tmp_delta_sigma[0] * 1e-6, 'r--', label = 'no offset, high $M_{\\ast}$', alpha = 0.5,)
	plt.plot( R / 1e3, tmp_off_delta_sigma[0] * 1e-6, 'r-', label = 'with offset', alpha = 0.5,)

	plt.plot( R / 1e3, tmp_delta_sigma[1] * 1e-6, 'b--', label = 'no offset, low $M_{\\ast}$', alpha = 0.5,)
	plt.plot( R / 1e3, tmp_off_delta_sigma[1] * 1e-6, 'b-', label = 'with offset', alpha = 0.5,)

	# plt.plot( R / 1e3, h * tmp_delta_sigma[0] * 1e-6, 'r--', label = 'no offset, high $M_{\\ast}$', alpha = 0.5,)
	# plt.plot( R / 1e3, h * tmp_off_delta_sigma[0] * 1e-6, 'r-', label = 'with offset', alpha = 0.5,)

	# plt.plot( R / 1e3, h * tmp_delta_sigma[1] * 1e-6, 'b--', label = 'no offset, low $M_{\\ast}$', alpha = 0.5,)
	# plt.plot( R / 1e3, h * tmp_off_delta_sigma[1] * 1e-6, 'b-', label = 'with offset', alpha = 0.5,)

	plt.xscale('log')
	plt.yscale('log')
	plt.ylim( 9e-1, 3e2)
	plt.xlim( 1e-1, 2e1)
	plt.legend( loc = 1)
	plt.xlabel('R [Mpc/h]')
	plt.ylabel('$\\Sigma [h M_{\\odot} / pc^2]$')

	plt.savefig('/home/xkchen/delta_sigma_check.png', dpi = 300)
	plt.close()

	###
	tt_f_off = np.linspace(0, 1, 6)

	tt_off_sigma = []
	tt_off_delta_sigma = []

	off_sigma_kk = cc_off_sigma_func( R, sigma_off[0], z0, c0[0], Mh0, v_m)
	norm_sigma_kk = sigmam( R, Mh0, z0, c0[0])

	for kk in range( 6 ):

		obs_sigma = tt_f_off[kk] * off_sigma_kk + ( 1 - tt_f_off[kk] ) * norm_sigma_kk
		tt_off_sigma.append( obs_sigma )

		mm_sigma = aveg_sigma_func( R, obs_sigma, N_grid = N_grid,)
		delt_sigma = mm_sigma - obs_sigma
		tt_off_delta_sigma.append( delt_sigma )

	plt.figure()
	for kk in range( 6 ):
		plt.plot( R, tt_off_sigma[kk], ls = '-', color = mpl.cm.rainbow(kk/5), alpha = 0.5, label = 'f_off = %.1f' % tt_f_off[kk],)

	plt.xscale( 'log')
	plt.yscale( 'log')
	plt.xlabel( 'R[kpc/h]')
	plt.legend( loc = 3 )
	plt.ylabel('$\\Sigma [h M_{\\odot} / kpc^2]$')

	plt.savefig('/home/xkchen/sigma_check.png', dpi = 300)
	plt.close()


	plt.figure()
	for kk in range( 6 ):

		plt.plot( R / 1e3, tt_off_delta_sigma[kk] * 1e-6, ls = '-', color = mpl.cm.rainbow(kk/5), alpha = 0.5, label = 'f_off = %.1f' % tt_f_off[kk],)

	plt.xscale('log')
	plt.yscale('log')
	plt.ylim( 1e-1, 5e2)
	plt.xlim( 1e-2, 2e1)
	plt.legend( loc = 4)
	plt.xlabel('R [Mpc/h]')
	plt.ylabel('$\\Delta \\Sigma [h M_{\\odot} / pc^2]$')

	plt.savefig('/home/xkchen/off_delta_sigma_check.png', dpi = 300)
	plt.close()

	return

# profile_check()

Mh0 = 14.24 # M_sun / h
c0 = 6.95
sigma_off = 210 ## kpc / h
f_off = 0.20
v_m = 200
z0 = 0.242

# R = np.logspace(-2, np.log10(2e1), 1000) * 10**3 # kpc / h
R = np.logspace(0, np.log10(2e3), 1000) # kpc / h

N_grid = 100

norm_sigma = sigmam( R, Mh0, z0, c0,)
mean_off_sigma = aveg_sigma_func( R, norm_sigma, N_grid = N_grid,)
delta_sigma = mean_off_sigma - norm_sigma

p_einasto = profile_einasto.EinastoProfile( M = 10**(Mh0), c = c0, z = z0, mdef = '200m')


Ein_sigma = p_einasto.surfaceDensity( R )

delt_Ein_sigma = p_einasto.deltaSigma( R )

p_nfw = profile_nfw.NFWProfile( M = 10**(Mh0), c = c0, z = z0, mdef = '200m')
p_nfw_sigma = p_nfw.surfaceDensity( R )
delt_nfw_sigma = p_nfw.deltaSigma( R )

## compare differ halo profile
tt0 = time.time()

rho_Hern = profile_hernquist.HernquistProfile( M = 10**(Mh0), c = c0, z = z0, mdef = '200m')

tt0 = time.time()

sigm_Hern = rho_Hern.surfaceDensity( R )

print( time.time() - tt0 )

plt.figure()
plt.plot( R, p_nfw_sigma, 'r-', alpha = 0.5, label = 'NFW',)
plt.plot( R, sigm_Hern, 'g--', alpha = 0.5, label = 'Hernquist',)
plt.plot( R, Ein_sigma, 'b:', alpha = 0.5, label = 'Einasto',)
plt.legend( loc = 1,)
plt.xscale('log')
plt.yscale('log')
plt.savefig('/home/xkchen/sigma_compare.png', dpi = 300)
plt.show()

raise

# plt.figure()
# plt.plot( R, norm_sigma, 'r-', alpha = 0.5, label = 'NFW, mine')
# plt.plot( R, Ein_sigma, 'b--', alpha = 0.5, label = 'Einasto, colossus')
# plt.plot( R, p_nfw_sigma, 'g--', alpha = 0.5, label = 'NFW, colossus')

# plt.ylabel('$\\Sigma [h M_{\\odot} / kpc^2]$')
# plt.xlabel('R[kpc / h]')
# plt.legend( loc = 3 )
# plt.ylim( 1e4, 3e9)
# plt.xscale('log')
# plt.yscale('log')

# plt.savefig('/home/xkchen/sigma_check.png', dpi = 300)
# plt.close()


# plt.figure()

# plt.plot( R, delta_sigma * 1e-6, 'r-', alpha = 0.5, label = 'NFW, mine')
# plt.plot( R, delt_Ein_sigma * 1e-6, 'b--', alpha = 0.5, label = 'Einasto, colossus')
# plt.plot( R, delt_nfw_sigma * 1e-6, 'g--', alpha = 0.5, label = 'NFW, colossus')

# plt.ylabel('$\\Delta \\Sigma [h M_{\\odot} / pc^2]$')
# plt.xlabel('R[kpc / h]')
# plt.legend( loc = 3 )
# # plt.ylim( 1e-1, 6e2)
# plt.ylim( 1e1, 6e2)
# plt.xscale('log')
# plt.yscale('log')

# plt.savefig('/home/xkchen/delta_sigma_check.png', dpi = 300)
# plt.close()


## NFW
tt0 = time.time()

mis_sigma = cc_off_sigma_func( R, sigma_off, z0, c0, Mh0, v_m)

nfw_off_sigma = f_off * mis_sigma + ( 1 - f_off ) * norm_sigma

tt1 = time.time() - tt0
print( tt1 )

m_nfw_off_sigma = aveg_sigma_func( R, nfw_off_sigma, N_grid = N_grid,)

tt2 = time.time() - tt0 - tt1
print( tt2 )

delt_off_nfw_sigma = m_nfw_off_sigma - nfw_off_sigma

print( 'star here' )


plt.figure()
plt.plot( R, delt_off_nfw_sigma, 'r-', alpha = 0.5, label = 'NFW')
plt.xscale('log')
plt.yscale('log')
plt.show()

