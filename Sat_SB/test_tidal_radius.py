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

### === concentration-mass relation in Prada et al. 2012
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

def rho_bulge_func(r, a_B, lg_Mstar, eta_B2T):
	# a_B : half mass radius of bulge
	M_bulge = 10**lg_Mstar * eta_B2T
	rho_B = ( M_bulge * a_B / ( 4 * np.pi ) ) / ( r**2 * (r + a_B)**2 )

	return rho_B

def rho_e_func( R_e, M_d ):
	# integral mass within R_e is hale of the disc mass M_d.
	_rho_e = M_d / ( 8 * np.pi * R_e**3)

	#... limit with cut radius
	# _f_rm = 4 * np.pi * ( 2 * R_e**3 - R_e * np.exp(-rm / R_e) * (rm**2 + 2 * rm * R_e + 2 * R_e**2) )
	# _rho_e = M_d / _f_rm

	return _rho_e

def rho_disc_func(r, r_e, lg_Mstar, eta_B2T):
	# r must be the largest size of the galaxy (or disc)

	M_disc = 10**lg_Mstar * (1 - eta_B2T)
	_rho_e_disc = rho_e_func( r_e, M_disc )
	rho_d = _rho_e_disc * np.exp( - r / r_e )

	return rho_d

def rho_galaxy_func(r, a_B, r_e, lg_Mstar, eta_B2T):

	rho_B = rho_bulge_func( r, a_B, lg_Mstar, eta_B2T )
	rho_d = rho_disc_func( r, r_e, lg_Mstar, eta_B2T )
	rho_g = rho_B + rho_d

	return rho_g

### === estimate halo mass of satellites (Moster, Naab, and White, 2013 (MNW13); Niemiec et al. 2017, observation check)
#... here we assume the M_star and M_halo is one-to-one relation, in realistic case, we need apply HOD model, or inverse 
#... the relation in MNW13 through Bayes relation
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

	Mh = Mg_star * ( (Mg_star / 10**lg_Mz)**(-1 * belt_z) + (Mg_star / 10**lg_Mz)**gama_z ) / ( 2 * Nz )

	return Mh

# this relation gives the lower limit of bulge effective radius (Shen et al. 2003)
#... in Shen et al, the relation is for a lowest limit of R_e of bulge
def bulg_size_func( Mass ):
	# Mass in unit of M_sun
	# radius in unit of kpc, return effective radius
	if Mass > 2 * 1e10:
		lg_Re = 0.56 * np.log10( Mass ) - 5.54

	if Mass <= 2 * 1e10:
		lg_Re = 0.14 * np.log10( Mass ) - 1.21

	return 10**lg_Re

### === the bulge/total ration, take from Hopkins et al. 2009
#... agree with observation of Balcells et al. (2007b)
def B_to_T_func( lg_Mass ):
	# lg_Mass = log( Mass ), Mass in unit of M_sun

	dat = pds.read_csv('/home/xkchen/tmp_run/data_files/R_t_test/B_to_T_mass_ratio.csv')
	_lg_Ms = np.array( dat['lg_Mstar'] )
	_B2T_arr = np.array( dat['B_to_T'] )

	interp_B2T_f = interp.interp1d( _lg_Ms, _B2T_arr, kind = 'linear', fill_value = 'extrapolate',)
	_mock_B2T = interp_B2T_f( lg_Mass )

	return _mock_B2T

### === disc size function, rely on mass and redshfit, taken from Shankar et al. 2014
#... based on Shen et al. 2003, but with some modifaction
def disc_size_func( Mass, z0 ):
	# mass in unit of M_sun, is the galaxy stellar mass
	# radius in unit of kpc
	R_0 = 0.1
	p_0 = 0.39
	k_0 = 0.14
	M_piv = 3.98 * 10**10 # M_sun

	rf0 = R_0 / (1 + z0)**0.4
	rf1 = (1 + Mass / M_piv)**(p_0 - k_0)
	#. half mass radius
	R_d = rf0 * rf1 * Mass**k_0

	return R_d

def color_pro_func( Mass, rx ):
	# Mass in unit M_sun
	# rx in unit kpc

	low_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/R_t_test/aveg_g-i_low_M.csv')
	low_g2i = np.array( low_dat['g-i'] )
	low_R = np.array( low_dat['R_kpc'] )

	low_intmp_g2i_F = interp.interp1d( low_R, low_g2i, kind = 'linear', fill_value = 'extrapolate',)


	hi_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/R_t_test/aveg_g-i_high_M.csv')
	hi_g2i = np.array( hi_dat['g-i'] )
	hi_R = np.array( hi_dat['R_kpc'] )

	hi_intmp_g2i_F = interp.interp1d( hi_R, hi_g2i, kind = 'linear', fill_value = 'extrapolate',)

	#. assume critical mass is 10.63
	if np.log10( Mass ) <= 10.63:
		mock_gi = low_intmp_g2i_F( rx )

	else:
		mock_gi = hi_intmp_g2i_F( rx )

	return mock_gi

### === ### infall mocking
def infall_process_func( zx, lg_M_clust, lg_Ms_sat, N_radii, id_halo = False):
	"""
	zx : redshift of mock
	lg_M_clust : cluster mass (M_gas + M_star + M_DM), unit of M_sun / h
	lg_Ms_sat : satellite stellar mass, unit of M_sun
	N_radii : the radii points number to divide the infall distance
	"""
	#.. cluster density profile and cumulative mass profile
	R200m = R_200m_func( zx, lg_M_clust )
	R200m = R200m / h

	cm = c_M_func( 10**lg_M_clust, zx )

	rx = np.logspace( 0, np.log10(R200m), N_radii )
	nfw_rho = rho_nfw_delta_m( rx * h, zx, cm, lg_M_clust )
	nfw_rho = nfw_rho * h**2 # in unit of M_sun / kpc^3

	integ_M = cumu_mass_func( rx, nfw_rho, N_grid =  N_radii,) # 3D integrated mass	


	#.. satellite stellar and total mass profile
	### Contini et al. 2014, stripped or disrupted stars donated by galaxies in range 10^10~10^11 M_sun

	r_d = disc_size_func( 10**lg_Ms_sat, zx )
	a_B = bulg_size_func( 10**lg_Ms_sat )

	rg_max = 10 * r_d

	B2T = B_to_T_func( lg_Ms_sat )

	M_bulg = 10**lg_Ms_sat * B2T

	rg = np.logspace( -1, np.log10(rg_max), N_radii )
	rho_g = rho_galaxy_func( rg, a_B, r_d, lg_Ms_sat, B2T )

	M_gax = cumu_mass_func( rg, rho_g, N_grid =  N_radii )
	interp_M_g = interp.interp1d( rg, M_gax, kind = 'linear', fill_value = 'extrapolate',)


	#... halo mass and profile of satellite
	Mh_sat = Ms_to_Mh_func( zx, 10**lg_Ms_sat )
	lg_Mh_sat = np.log10( Mh_sat * h ) # M_sun / h, M_200m define

	Rh_sat = R_200m_func( zx, lg_Mh_sat )
	Rh_sat = Rh_sat / h

	cm_sat = c_M_func( 10**lg_Mh_sat, zx )

	rg_h = np.logspace( -1, np.log10(Rh_sat), N_radii )
	rho_g_h = rho_nfw_delta_m( rg_h * h, zx, cm_sat, lg_Mh_sat )
	rho_g_h = rho_g_h * h**2

	integ_Mh_sat = cumu_mass_func( rg_h, rho_g_h, N_grid =  N_radii,)

	interp_Mh_sat = interp.interp1d( rg_h, integ_Mh_sat, kind = 'linear', fill_value = 'extrapolate',)
	interp_Rh_sat = interp.interp1d( integ_Mh_sat[1:], rg_h[1:], kind = 'linear', fill_value = 'extrapolate',)
	R_half_Mh = interp_Rh_sat( integ_Mh_sat[-1] / 2 )

	_rho_ds = rho_disc_func( rg, r_d, lg_Ms_sat, B2T )
	_M_ds = cumu_mass_func( rg, _rho_ds, N_grid =  N_radii )
	interp_Rh_dis = interp.interp1d( _M_ds, rg, kind = 'linear', fill_value = 'extrapolate',)
	R_half_Md = interp_Rh_dis( _M_ds[-1] / 2 )
	# R_half_Md = 1.68 * r_d


	#... input g-i color
	# ini_g2i = color_pro_func( 10**lg_Ms_sat, rg )

	##. infall process (case 1 : orphan galaxies, case 2 : galaxy with parsent halo)
	#. case 2
	if id_halo == True:

		res_rt = np.zeros( N_radii, )
		res_M = np.zeros( N_radii, )
		drop_M = np.zeros( N_radii, )
		drop_gi = np.zeros( N_radii, )

		res_Mtot = np.zeros( N_radii, )
		drop_Mtot = np.zeros( N_radii, ) 

		res_M0 = integ_Mh_sat[-1]
		res_r0 = rg_h[-1]

		half_Rh_0 = R_half_Mh + 0.
		half_Rd_0 = R_half_Md + 0.

		dex_start = 1
		for tt in range( dex_start, N_radii ):

			_pp_rt = tidal_R_func( integ_Mh_sat[ -tt ], res_M0, rx[ -tt ] )

			if _pp_rt >= res_r0:
				# no tripping
				res_rt[ -tt ] = res_r0
				res_M[ -tt ] = res_M[ -(tt-1) ] # M_gax[-1]
				drop_M[ -tt ] = drop_M[ -(tt-1) ]
				drop_gi[ -tt ] = drop_gi[ -(tt-1) ]

				res_Mtot[ -tt ] = res_Mtot[ -(tt-1) ] # res_M0
				drop_Mtot[ -tt ] = drop_Mtot[ -(tt-1) ]

			else:

				if _pp_rt < a_B:
					# tidal radius is smaller than bulge size, galaxy is disrupted

					res_rt[ -tt ] = _pp_rt 
					drop_M[ -tt ] = M_gax[-1] + 0.
					res_M[ -tt ] = 0.
					drop_gi[ -tt ] = color_pro_func( 10**lg_Ms_sat, a_B )

					res_Mtot[ -tt ] = 0.
					drop_Mtot[ -tt ] = integ_Mh_sat[-1]
					break

				#. once this happen, assume all dark matter have been dropped out
				identi = half_Rh_0 < half_Rd_0

				if identi == False:
					#. stripping dark matter only
					tmp_rh = np.logspace( -1, np.log10( _pp_rt ), N_radii)
					limed_Mh = interp_Mh_sat( _pp_rt ) # in unit of M_sun

					drop_Mtot[ -tt ] = integ_Mh_sat[-1] - limed_Mh
					res_Mtot[ -tt ] = limed_Mh
					res_rt[ -tt ] = _pp_rt

					# star component has no change
					res_M[ -tt ] = res_M[ -(tt-1) ]
					drop_M[ -tt ] = drop_M[ -(tt-1) ]
					drop_gi[ -tt ] = drop_gi[ -(tt-1) ]

					_ii_cm = c_M_func( limed_Mh, zx )

					_ii_rho_gh = rho_nfw_delta_m( tmp_rh * h, zx, _ii_cm, np.log10( limed_Mh * h ) )
					_ii_rho_gh = _ii_rho_gh * h**2

					_ii_integ_Mh_s = cumu_mass_func( tmp_rh, _ii_rho_gh, N_grid =  N_radii,)

					interp_Mh_sat = interp.interp1d( tmp_rh, _ii_integ_Mh_s, kind = 'linear', fill_value = 'extrapolate',)
					_ii_interp_Rh_s = interp.interp1d( _ii_integ_Mh_s, tmp_rh, kind = 'linear', fill_value = 'extrapolate',)

					half_Rh_0 = _ii_interp_Rh_s( _ii_integ_Mh_s[-1] / 2 )
					res_M0 = limed_Mh + 0.
					res_r0 = _pp_rt + 0.

					break_order = tt + 0
					continue

				#. change to star stripping state
				if ( identi == False ) and (tt == break_order):
					dex_start = break_order + 1
					res_M0 = M_gax[-1]
					res_r0 = rg_max
					continue

				#. stripping galaxy stars
				tmp_rg = np.logspace( -1, np.log10( _pp_rt ), N_radii )
				limed_Ms = interp_M_g( _pp_rt )

				res_M[ -tt ] = limed_Ms
				drop_M[ -tt ] = M_gax[ -1 ] - limed_Ms
				drop_gi[ -tt ] = color_pro_func( 10**lg_Ms_sat, _pp_rt )

				#.. condition for next stripping
				_ii_B2T = M_bulg / limed_Ms
				_ii_r_d = _pp_rt * 0.1

				_ii_rho_g = rho_galaxy_func( tmp_rg, a_B, _ii_r_d, np.log10( limed_Ms ), _ii_B2T )
				tmp_Mg = cumu_mass_func( tmp_rg, _ii_rho_g, N_grid =  N_radii )
				interp_M_g = interp.interp1d( tmp_rg, tmp_Mg, kind = 'linear', fill_value = 'extrapolate',)

				#. total stripped mass
				drop_Mtot[ -tt ] = (integ_Mh_sat[-1] - M_gax[-1]) + (M_gax[ -1 ] - limed_Ms)
				res_Mtot[ -tt ] = limed_Ms
				res_rt[ -tt ] = _pp_rt

				res_M0 = limed_Ms + 0.
				res_r0 = _pp_rt + 0.

		return rx, res_rt, res_M, drop_M, drop_gi

	#. case 1
	if id_halo == False:

		res_rt = np.zeros( N_radii, )
		res_M = np.zeros( N_radii, )
		drop_M = np.zeros( N_radii, )
		drop_gi = np.zeros( N_radii, )

		res_M0 = M_gax[-1]
		res_r0 = rg_max

		for tt in range( 1, N_radii ):

			_pp_rt = tidal_R_func( integ_M[ -tt ], res_M0, rx[ -tt ] )

			if _pp_rt >= res_r0:
				res_rt[ -tt ] = res_r0
				res_M[ -tt ] = res_M0
				drop_M[ -tt ] = M_gax[-1] - res_M0
				drop_gi[ -tt ] = color_pro_func( 10**lg_Ms_sat, res_r0 )

			else:

				if _pp_rt < a_B:
					# tidal radius is smaller than bulge size, galaxy is disrupted
					res_rt[ -tt ] = _pp_rt 
					drop_M[ -tt ] = M_gax[-1] + 0.
					res_M[ -tt ] = 0.
					drop_gi[ -tt ] = color_pro_func( 10**lg_Ms_sat, a_B )
					break

				else:
					# tidal radius is larger than bulge size
					# assume disc stars droped out and bulge is stable

					tmp_rg = np.logspace( -1, np.log10( _pp_rt ), N_radii)
					limed_M = interp_M_g( _pp_rt )

					drop_M[ -tt ] = M_gax[ -1 ] - limed_M
					res_M[ -tt ] = limed_M
					res_rt[ -tt ] = _pp_rt
					drop_gi[ -tt ] = color_pro_func( 10**lg_Ms_sat, _pp_rt )

					# _ii_B2T = M_bulg / limed_M
					# _ii_r_d = _pp_rt * 0.1
					# _ii_rho_g = rho_galaxy_func( tmp_rg, a_B, _ii_r_d, np.log10( limed_M ), _ii_B2T )

					# tmp_Mg = cumu_mass_func( tmp_rg, _ii_rho_g, N_grid =  N_radii )
					# interp_M_g = interp.interp1d( tmp_rg, tmp_Mg, kind = 'linear', fill_value = 'extrapolate',)

					res_M0 = limed_M + 0.
					res_r0 = _pp_rt + 0.

		return rx, res_rt, res_M, drop_M, drop_gi

### === CSMF Yang et al. 2012, and galaxy Mstar mock
def Schechter_func(x, M_pov, phi_pov, alpha_pov):
	pf0 = ( x / M_pov )**(alpha_pov + 1)
	pf1 = np.exp( -1 * x / M_pov)
	m_pdf = phi_pov * pf0 * pf1

	return m_pdf

def _modi_func(x, M_pov, phi_pov, alpha_pov):

	_pf0 = ( x / M_pov )**(alpha_pov + 1)
	_pf1 = np.exp( -1 * x / M_pov)
	_m_pdf = phi_pov * _pf0 * _pf1 / ( x * np.log(10) )

	return _m_pdf

def mock_galx_Mstar_func( N_galx, low_M_lim, up_M_lim ):
	"""
	N_galx : number of galaxy
	low_M_lim, up_M_lim : lg( M_* / (M_sun / h^2) ), mass range in which creat the galaxy sample
	"""

	csmf_param = np.loadtxt('/home/xkchen/tmp_run/data_files/figs/Yang_CSMF_fit-params.txt')
	M_mode, phi_mode, alpha_mode = csmf_param

	lg_Mx = np.linspace( low_M_lim, up_M_lim, N_galx )

	integ_N = np.zeros( N_galx,)

	for ii in range( N_galx ):

		pre_N0 = integ.romberg( _modi_func, 10**lg_Mx[0], 10**lg_Mx[ ii ], args = (M_mode, phi_mode, alpha_mode),)
		integ_N[ ii ] = pre_N0

	_tot_N = integ.romberg( _modi_func, 10**lg_Mx[0], 10**lg_Mx[ -1 ], args = (M_mode, phi_mode, alpha_mode),)\

	cdf = integ_N / _tot_N

	interp_galx_N_F = interp.interp1d( cdf, lg_Mx, kind = 'linear', fill_value = 'extrapolate',)

	tt0 = np.random.random( N_galx )
	tt_Ms_galx = interp_galx_N_F( tt0 )

	return tt_Ms_galx


### === ### infall galaxy mock
##. cluster profiles
z0 = 0.25
lg_Mh = 14.24 # M_sun / h, M_200m define
# cm = 5
# v_m = 200

def rho_mass_check():

	lg_Msat = 11
	N_grid = 250

	# B2T = 0.3
	B2T = B_to_T_func( lg_Msat )

	M_bulg = 10**lg_Msat * B2T
	M_ds = 10**lg_Msat * (1 - B2T)

	r_d = disc_size_func( 10**lg_Msat, z0 )
	a_B = bulg_size_func( M_bulg )

	rg_max = 11 * r_d
	rg = np.logspace( -1, np.log10(rg_max), N_grid )

	rho_g = rho_galaxy_func( rg, a_B, r_d, lg_Msat, B2T )
	rho_B = rho_bulge_func( rg, a_B, lg_Msat, B2T )
	rho_d = rho_disc_func( rg, r_d, lg_Msat, B2T )

	Md_r = cumu_mass_func( rg, rho_d, N_grid = N_grid )
	Mb_r = cumu_mass_func( rg, rho_B, N_grid = N_grid )
	Mg_r = cumu_mass_func( rg, rho_g, N_grid = N_grid )

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

	plt.figure()
	plt.plot( rg / r_d, Md_r / M_ds, 'r-', label = 'disc')
	plt.plot( rg / a_B, Mb_r / M_bulg, 'g-', label = 'bulge')
	plt.plot( rg / rg_max, Mg_r / 10**lg_Msat, 'b-', label = 'total')
	plt.legend( loc = 4 )
	plt.ylabel('$M(< r) \, / \, M_{tot}$')
	plt.xlabel('$r / r_{d}$, $r / r_B$, or $r / r_{ \\mathrm{max} }$')
	plt.savefig('/home/xkchen/cumu_Mass_check.png', dpi = 300)
	plt.close()

# rho_mass_check()


#. obs
obs_c_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/total_g-i_color_profile.csv' )
obs_R, obs_gi, obs_gi_err = np.array( obs_c_dat['R_kpc'] ), np.array( obs_c_dat['g2i'] ), np.array( obs_c_dat['g2i_err'] )

#... Contini et al. 2014, stripped or disrupted stars donated by galaxies in range 10^10~10^11 M_sun
#... mock galaxy stellar mass according CSMF Yang et al. 2012
low_M_lim, up_M_lim = 9, 12
N_galx = 11

# mock_M = mock_galx_Mstar_func( N_galx, low_M_lim, up_M_lim )
# lg_Msat = np.log10( 10**mock_M / h**2 )
# lg_Msat = np.sort( lg_Msat )

lg_Msat = np.linspace( 10, 11.5, N_galx )
N_grid = 250

no_halo_arr = []
w_halo_arr = []

for kk in range( N_galx ):

	id_halo_0 = False
	rx, res_rt, res_M, drop_M, drop_gi = infall_process_func( z0, lg_Mh, lg_Msat[kk], N_grid, id_halo = id_halo_0 )
	no_halo_arr.append( [rx, res_rt, res_M, drop_M, drop_gi] )

	id_halo_1 = True
	rx_1, res_rt_1, res_M_1, drop_M_1, drop_gi_1 = infall_process_func( z0, lg_Mh, lg_Msat[kk], N_grid, id_halo = id_halo_1 )
	w_halo_arr.append( [rx_1, res_rt_1, res_M_1, drop_M_1, drop_gi_1] )

plt.figure()
ax = plt.subplot(111)

for kk in range( N_galx ):

	ax.plot( no_halo_arr[kk][0], no_halo_arr[kk][1], ls = '-', color = mpl.cm.rainbow( kk / N_galx), 
		label = '$\\lg(M_{\\ast}/M_{\\odot}) = %.2f$' % lg_Msat[kk],)
	ax.plot( w_halo_arr[kk][0], w_halo_arr[kk][1], ls = '--', color = mpl.cm.rainbow( kk / N_galx),)

ax.legend( loc = 2, frameon = False,)
ax.set_xlim( 1e1, 2e3)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
ax.set_yscale('log')
ax.set_ylabel('$r_{\\tau} \; [kpc]$')
plt.savefig('/home/xkchen/tidal_radius.png', dpi = 200)
plt.close()


plt.figure()
ax = plt.subplot(111)

for kk in range( N_galx ):

	ax.plot( no_halo_arr[kk][0], no_halo_arr[kk][4], ls = '-', color = mpl.cm.rainbow( kk / N_galx), 
		label = '$\\lg(M_{\\ast}/M_{\\odot}) = %.2f$' % lg_Msat[kk],)
	# ax.plot( w_halo_arr[kk][0], w_halo_arr[kk][4], ls = '--', color = mpl.cm.rainbow( kk / N_galx),)

ax.plot( obs_R, obs_gi, 'k-', alpha = 0.75,)
ax.fill_between( obs_R, y1 = obs_gi - obs_gi_err, y2 = obs_gi + obs_gi_err, color = 'k', alpha = 0.15,)

# ax.legend( loc = 3, frameon = False,)
ax.set_xlim( 1e1, 2e3)
ax.set_xscale('log')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('g-i')
ax.set_ylim(0.0, 2.0)
plt.savefig('/home/xkchen/g-i_mock.png', dpi = 200)
plt.close()

