import matplotlib as mpl
import matplotlib.pyplot as plt

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

from surface_mass_density import input_cosm_model, cosmos_param, rhom_set
from surface_mass_density import rho_nfw_delta_m

from colossus.cosmology import cosmology
from colossus.halo import profile_nfw

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_rho_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( rp[ii] ), N_grid)
		new_mass = intep_rho_F( new_rp )
		cumu_mass[ ii ] = integ.simps( 4 * np.pi * new_rp**2 * new_mass, new_rp)

	return cumu_mass

#. concentration-mass relation in Prada et al. 2012
def fa_func( x ):
	xf0 = x**(3/2)
	xf1 = (1 + x**3)**(3/2)
	return xf0 / xf1

def c_min_func( x_a ):

	c_0 = 3.681
	c_1 = 5.033
	alpa = 6.948
	x_0 = 0.424
	c_min_ = c_0 + (c_1 - c_0) * (np.arctan( alpa * (x_a - x_0) ) / np.pi + 0.5)

	return c_min_

def sigm_min_func( x_a ):

	sigm_0 = 1.047
	sigm_1 = 1.646
	belta = 7.386
	x_1 = 0.526
	sigm_min_ = sigm_0 + (sigm_0 + sigm_1) * (np.arctan( belta * (x_a - x_1) ) / np.pi + 0.5)

	return sigm_min_

def c_M_func( Mass, z0 ):

	M_piv = 1e12 # in unit of M_sun / h

	a_z0 = 1 / ( 1 + z0)
	x_a = (Omega_lambda / Omega_m)**(1/3) * a_z0

	integ_on_fa = integ.romberg( fa_func, 0, x_a )
	Df0 = np.sqrt(1 + x_a**3) / x_a**(3/2)

	D_a = 2.5 * (Omega_m / Omega_lambda)**(1/3) * Df0 * integ_on_fa

	c_min_x = c_min_func( x_a )
	c_min_0 = c_min_func( 1.393 )

	sigm_min_x = sigm_min_func( x_a )
	sigm_min_0 = sigm_min_func( 1.393 )

	B0_x = c_min_x / c_min_0
	B1_x = sigm_min_x / sigm_min_0

	y_m = (Mass / M_piv)**(-1)

	sigm_M_a = D_a * (16.9 * y_m**0.41 / (1 + 1.102 * y_m**0.20 + 6.22 * y_m**0.333) )
	sigm_prime = B1_x * sigm_M_a

	pa, pb, pc, pd = 2.881, 1.257, 1.022, 0.060
	f_prime = pa * ( (sigm_prime / pb)**pc + 1 ) * np.exp(pd / sigm_prime**2)

	c_m = B0_x * f_prime

	return c_m

# contini et al. 2014
def tidal_R_func( M_host, M_sate, d_cen ):

	eta_m = M_sate / M_host
	r_tid = ( eta_m / 3 )**(1/3) * d_cen

	return r_tid

def R_200m_func( z0, lg_Mh, v_m = 200 ):
	rho_c, rho_m = rhom_set( z0 ) # M_sun * h^2 / kpc^3
	_r_200m = ( 3 * 10**lg_Mh / (4 * v_m * rho_m * np.pi) )**(1/3) # kpc / h
	return _r_200m

def rho_e_func( R_e, M_in, r_m):
	# integral mass is M_in ( M_sun) wthin radius r_m ( kpc).
	mf0 = 4 * np.pi * ( 2 * R_e**3 - R_e * np.exp( -r_m / R_e ) * (r_m**2 + 2 * r_m * R_e + 2 * R_e**2) )
	_rho_e = M_in / mf0
	return _rho_e

def rho_bulge_func(r, a_B, lg_Mstar, eta_B2T):

	M_bulge = 10**lg_Mstar * B2T
	rho_B = ( M_bulge * a_B / ( 2 * np.pi ) ) / ( r * (r + a_B)**3)

	return rho_B

def rho_disc_func(r, r_e, lg_Mstar, eta_B2T):
	# r must be the largest size of the galaxy (or disc)

	M_bulge = 10**lg_Mstar * B2T
	M_disc = 10**lg_Mstar - M_bulge
	r_m = r[-1]

	_rho_e_disc = rho_e_func( r_e, M_disc, r_m )
	rho_d = _rho_e_disc * np.exp( - r / r_e )

	return rho_d

def rho_galaxy_func(r, a_B, r_e, lg_Mstar, eta_B2T):

	rho_B = rho_bulge_func(r, a_B, lg_Mstar, eta_B2T)
	rho_d = rho_disc_func(r, r_e, lg_Mstar, eta_B2T)
	rho_g = rho_B + rho_d

	return rho_g

#.. estimate halo mass of satellites (Moster, Naab, and White, 2013; Niemiec et al. 2017, observation check)
def Ms_to_Mh_func( z0, Mg_star ):
	# all mass in unit of M_sun
	M10 = 11.590
	M11 = 1.195

	N10 = 0.0351
	N11 = -0.0247

	belt0 = 1.376
	belt1 = -0.826

	gama0 = 0.608
	gama1 = 0.329

	lg_Mz = M10 + M11 * ( z0 / (1 + z0) )
	Nz = N10 + N11 * ( z0 / (1 + z0) )
	belt_z = belt0 + belt1 * ( z0 / (1 + z0) )
	gama_z = gama0 + gama1 * ( z0 / (1 + z0) )

	Mh = Mg_star * ( (Mg_star / 10**lg_Mz)**(-1 * belt_z) + 
					 (Mg_star / 10**lg_Mz)**gama_z ) / ( 2 * Nz )

	return Mh

### === ### infall galaxy mock
##. cluster profiles
z0 = 0.25
N_rx = 250

lg_Mh = 14 # M_sun / h, M_200m define
cm = 5
v_m = 200

R200m = R_200m_func( z0, lg_Mh )
R200m = R200m / h

rx = np.logspace(0, np.log10(R200m), N_rx)
nfw_rho = rho_nfw_delta_m( rx * h, z0, cm, lg_Mh )
nfw_rho = nfw_rho * h**2 # in unit of M_sun / kpc^3

N_grid = 250
integ_M = cumu_mass_func( rx, nfw_rho, N_grid = N_grid,) # 3D integrated mass

##. galaxy parameters
#...Contini et al. 2014, stripped or disrupted stars donated by galaxies in range 10^10~10^11 M_sun
lg_Msat = 10 # M_star, in unit of M_sun

r_d = 13
rg_max = 10 * r_d
B2T = 0.2 # Fig.14 of Weinzirl et al. 2009

M_bulg = 10**lg_Msat * B2T
a_B = 0.3

rg = np.logspace( -1, np.log10(rg_max), N_rx )
rho_g = rho_galaxy_func(rg, a_B, r_d, lg_Msat, B2T )
rho_B = rho_bulge_func(rg, a_B, lg_Msat, B2T )
rho_d = rho_disc_func(rg, r_d, lg_Msat, B2T )

plt.figure()
plt.plot( rg, rho_B, ls = ':', color = 'b', label = 'Bulge')
plt.plot( rg, rho_d, ls = '--', color = 'b', label = 'disk')
plt.plot( rg, rho_g, ls = '-', color = 'b', label = 'total')
plt.xscale('log')
plt.xlabel('R [kpc]')
plt.yscale('log')
plt.ylabel('$\\rho \;[M_{\\odot} / kpc^{3}]$')
plt.legend( loc = 1 )
plt.savefig('/home/xkchen/galaxy_Mstar_profile.png', dpi = 300)
plt.close()


#. dark matter halo of satellites
Mh_sat = Ms_to_Mh_func( z0, 10**lg_Msat )
lg_Mh_sat = np.log10( Mh_sat * h ) # M_sun / h, M_200m define

Rh_sat = R_200m_func( z0, lg_Mh_sat )
Rh_sat = Rh_sat / h

rg_h = np.logspace( -1, np.log10(Rh_sat), N_rx)
rho_g_h = rho_nfw_delta_m( rg_h * h, z0, cm, lg_Mh_sat )
rho_g_h = rho_g_h * h**2

integ_Mh_sat = cumu_mass_func( rg_h, rho_g_h, N_grid = N_grid,)

interp_Mh_sat = interp.interp1d( rg_h, integ_Mh_sat, kind = 'linear', fill_value = 'extrapolate',)
interp_Rh_sat = interp.interp1d( integ_Mh_sat[1:], rg_h[1:], kind = 'linear', fill_value = 'extrapolate',)
R_half_Mh = interp_Rh_sat( integ_Mh_sat[-1] / 2 )

M_gax = cumu_mass_func( rg, rho_g, N_grid = N_grid )
interp_M_g = interp.interp1d( rg, M_gax, kind = 'linear', fill_value = 'extrapolate',)
R_half_Md = 1.68 * r_d

##. infall process (case 1 : orphan galaxies, case 1 : galaxy with parsent halo)
res_rt = np.zeros( N_rx, )
res_M = np.zeros( N_rx, )
drop_M = np.zeros( N_rx, )

res_Mtot = np.zeros( N_rx, )
drop_Mtot = np.zeros( N_rx, ) 

res_M0 = integ_Mh_sat[-1]
res_r0 = rg_h[-1]

half_Rh_0 = R_half_Mh + 0.
half_Rd_0 = R_half_Md + 0.

#. case 2
for tt in range( 1, N_rx ):

	_pp_rt = tidal_R_func( integ_Mh_sat[ -tt ], res_M0, rx[ -tt ] )

	if _pp_rt >= res_r0:
		# no tripping
		res_rt[ -tt ] = res_r0
		res_M[ -tt ] = M_gax[-1]
		drop_M[ -tt ] = 0.
		res_Mtot[ -tt ] = res_M0
		drop_Mtot[ -tt ] = 0.

	else:

		if _pp_rt < a_B:
			# tidal radius is smaller than bulge size, galaxy is disrupted

			res_rt[ -tt ] = _pp_rt 
			drop_M[ -tt ] = M_gax[-1] + 0.
			res_M[ -tt ] = 0.
			res_Mtot[ -tt ] = 0.
			drop_Mtot[ -tt ] = integ_Mh_sat[-1]
			break

		identi = half_Rh_0 < half_Rd_0
		if identi:

			#. stripping dark matter
			tmp_rh = np.logspace( -1, np.log10( _pp_rt ), N_rx)
			limed_Mh = interp_Mh_sat( _pp_rt ) # in unit of M_sun

			drop_Mtot[ -tt ] = integ_Mh_sat[-1] - limed_Mh
			res_Mtot[ -tt ] = limed_Mh
			res_rt[ -tt ] = _pp_rt

			_ii_cm = cm + 0.
			_ii_rho_gh = rho_nfw_delta_m( tmp_rh * h, z0, _ii_cm, np.log10( limed_Mh * h ) )
			_ii_rho_gh = _ii_rho_gh * h**2

			_ii_integ_Mh_s = cumu_mass_func( tmp_rh, _ii_rho_gh, N_grid = N_grid,)

			interp_Mh_sat = interp.interp1d( tmp_rh, _ii_integ_Mh_s, kind = 'linear', fill_value = 'extrapolate',)
			_ii_interp_Rh_s = interp.interp1d( _ii_integ_Mh_s, tmp_rh, kind = 'linear', fill_value = 'extrapolate',)

			half_Rh_0 = _ii_interp_Rh_s( _ii_integ_Mh_s[-1] / 2 )
			res_M0 = limed_Mh + 0.
			res_r0 = _pp_rt + 0.

			#. stripping galaxy stars
			tmp_rg = np.logspace( -1, np.log10( _pp_rt ), N_rx)
			limed_Ms = interp_M_g( _pp_rt )

			drop_M[ -tt ] = M_gax[ -1 ] - limed_Ms
			res_M[ -tt ] = limed_Ms

			_ii_B2T = M_bulg / limed_Ms
			_ii_r_d = _pp_rt * 0.1

			_ii_rho_g = rho_galaxy_func( tmp_rg, a_B, _ii_r_d, np.log10( limed_Ms ), _ii_B2T )
			tmp_Mg = cumu_mass_func( tmp_rg, _ii_rho_g, N_grid = N_grid )
			interp_M_g = interp.interp1d( tmp_rg, tmp_Mg, kind = 'linear', fill_value = 'extrapolate',)

			half_Rd_0 = 1.68 * _ii_r_d

		else:
			#. stripping dark matter only
			tmp_rh = np.logspace( -1, np.log10( _pp_rt ), N_rx)
			limed_Mh = interp_Mh_sat( _pp_rt ) # in unit of M_sun

			drop_Mtot[ -tt ] = integ_Mh_sat[-1] - limed_Mh
			res_Mtot[ -tt ] = limed_Mh
			res_rt[ -tt ] = _pp_rt

			_ii_cm = cm + 0.
			_ii_rho_gh = rho_nfw_delta_m( tmp_rh * h, z0, _ii_cm, np.log10( limed_Mh * h ) )
			_ii_rho_gh = _ii_rho_gh * h**2

			_ii_integ_Mh_s = cumu_mass_func( tmp_rh, _ii_rho_gh, N_grid = N_grid,)

			interp_Mh_sat = interp.interp1d( tmp_rh, _ii_integ_Mh_s, kind = 'linear', fill_value = 'extrapolate',)
			_ii_interp_Rh_s = interp.interp1d( _ii_integ_Mh_s, tmp_rh, kind = 'linear', fill_value = 'extrapolate',)

			half_Rh_0 = _ii_interp_Rh_s( _ii_integ_Mh_s[-1] / 2 )
			res_M0 = limed_Mh + 0.
			res_r0 = _pp_rt + 0.

"""
res_rt = np.zeros( N_rx, )
res_M = np.zeros( N_rx, )
drop_M = np.zeros( N_rx, )

res_M0 = M_gax[-1]
res_r0 = rg_max

#. case 1 
for tt in range( 1, N_rx ):

	_pp_rt = tidal_R_func( integ_M[ -tt ], res_M0, rx[ -tt ] )

	if _pp_rt >= res_r0:
		res_rt[ -tt ] = res_r0
		res_M[ -tt ] = res_M0
		drop_M[-tt] = M_gax[-1] - res_M0

	else:

		if _pp_rt < a_B:
			# tidal radius is smaller than bulge size, galaxy is disrupted

			res_rt[ -tt ] = _pp_rt 
			drop_M[ -tt ] = M_gax[-1] + 0.
			res_M[ -tt ] = 0.
			break

		else:
			# tidal radius is larger than bulge size
			# assume disc stars droped out and bulge is stable

			tmp_rg = np.logspace( -1, np.log10( _pp_rt ), N_rx)
			limed_M = interp_M_g( _pp_rt )

			drop_M[ -tt ] = M_gax[ -1 ] - limed_M
			res_M[ -tt ] = limed_M
			res_rt[ -tt ] = _pp_rt

			_ii_B2T = M_bulg / limed_M
			_ii_r_d = _pp_rt * 0.1
			_ii_rho_g = rho_galaxy_func( tmp_rg, a_B, _ii_r_d, np.log10( limed_M ), _ii_B2T )

			tmp_Mg = cumu_mass_func( tmp_rg, _ii_rho_g, N_grid = N_grid )
			interp_M_g = interp.interp1d( tmp_rg, tmp_Mg, kind = 'linear', fill_value = 'extrapolate',)

			res_M0 = limed_M + 0.
			res_r0 = _pp_rt + 0.
"""

plt.figure()
plt.plot( rx, res_rt, ls = '-', color = 'r', label = 'w/o halo',)
plt.plot( rx, res_rt, ls = '-', color = 'b', label = 'with halo',)
plt.legend( loc = 2,)
plt.xscale('log')
plt.xlabel('R [kpc]')
plt.yscale('log')
plt.ylabel('$r_{\\tau} \; [kpc]$')
plt.savefig('/home/xkchen/tidal_radius.png', dpi = 200)
plt.close()

plt.figure()
plt.plot( rx, drop_M / M_gax[-1], ls = '-', color = 'r', label = 'w/o halo',)
plt.plot( rx, drop_Mtot / integ_Mh_sat[-1], ls = '--', color = 'b', label = 'with halo',)

plt.legend( loc = 3,)
plt.xscale('log')
plt.xlabel('R [kpc]')
plt.ylim(-0.01, 1.01)
plt.ylabel('$M_{\\ast}^{loss} \, / \, M_{\\ast}^{infall}$')
plt.savefig('/home/xkchen/mass_loss.png', dpi = 200)
plt.close()
