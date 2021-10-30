import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.special as special
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import signal
from scipy import ndimage
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac
from surface_mass_density import input_cosm_model, cosmos_param, rhom_set
from color_2_mass import get_c2mass_func, gi_band_c2m_func
from Mass_rich_radius import rich2R_Simet

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

### === ###
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

### === ### galaxy mass estimation

#... CSMF of Yang et al. 2012
def _interp_phi( lgM_x, split_x, low_func, up_func):

	idx0 = lgM_x < split_x
	_phi_f_0 = low_func( lgM_x[ idx0 ] )

	idx1 = lgM_x >= split_x
	_phi_f_1 = up_func( lgM_x[ idx1 ] )

	_phi_f = np.r_[ _phi_f_0, _phi_f_1 ]

	return _phi_f

def interp_csmf_func( N_norm, norm_lgM, id_norm = False, N_points = 250):

	csmf_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/cc_Yang_CSMF.txt', sep = ',')
	lgM_star = np.array( csmf_dat['logM_*'] )
	CSMF = np.array( csmf_dat['[14.2 14.5]'] ) # averaged halo mass 14.33, unit of M_sun / h
	CSMF_err = np.array( csmf_dat['[14.2 14.5]_err'] )
	d_log_M = 0.05

	id_lim_0 = lgM_star < 11.6
	id_lim_1 = CSMF > 0.
	id_lim = id_lim_0 & id_lim_1

	limd_lgM = lgM_star[ id_lim]
	limd_CSMF = CSMF[ id_lim]
	limd_CSMF_err = CSMF_err[ id_lim]

	filt_CSMF = ndimage.gaussian_filter1d( limd_CSMF[:-1], sigma = 1, mode = 'nearest', truncate = 1.,)
	interp_smf = interp.interp1d( limd_lgM[:-1], filt_CSMF, kind = 'linear', fill_value = 'extrapolate',)
	interp_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:], kind = 'linear', fill_value = 'extrapolate',)
	if id_norm == True:
		#. normalization
		x_lgM = np.linspace( norm_lgM, 11.5, N_points ) # lg( M_* / (M_sun / h^2) )
		x_CSMF = _interp_phi( x_lgM, limd_lgM[-6], interp_smf, interp_up_M )

		x_M = 10**x_lgM
		dx_lgM = np.mean( np.diff( x_lgM ) )
		dx_M = dx_lgM * np.log( 10 ) * x_M

		pre_N0 = integ.simps( x_CSMF / (np.log(10) * x_M), x_M, )
		c_shift = N_norm / pre_N0

		shift_smf = interp.interp1d( limd_lgM[:-1], filt_CSMF * c_shift, kind = 'linear', fill_value = 'extrapolate',)
		shift_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:] * c_shift, kind = 'linear', fill_value = 'extrapolate',)
		return shift_smf, shift_up_M

	return interp_smf, interp_up_M

"""
def aveg_SGs_Mstar_f( low_lim = None, up_lim = None, N_points = 250):

	csmf_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/cc_Yang_CSMF.txt', sep = ',')
	lgM_star = np.array( csmf_dat['logM_*'] )
	CSMF = np.array( csmf_dat['[14.2 14.5]'] ) # averaged halo mass 14.33
	CSMF_err = np.array( csmf_dat['[14.2 14.5]_err'] )
	d_log_M = 0.05

	id_lim_0 = lgM_star < 11.6
	id_lim_1 = CSMF > 0.
	id_lim = id_lim_0 & id_lim_1

	limd_lgM = lgM_star[ id_lim]
	limd_CSMF = CSMF[ id_lim]
	limd_CSMF_err = CSMF_err[ id_lim]

	filt_CSMF = ndimage.gaussian_filter1d( limd_CSMF[:-1], sigma = 1, mode = 'nearest', truncate = 1.,)
	interp_smf = interp.interp1d( limd_lgM[:-1], filt_CSMF, kind = 'linear', fill_value = 'extrapolate',)
	# interp_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:], kind = 'quadratic', fill_value = 'extrapolate',)
	interp_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:], kind = 'linear', fill_value = 'extrapolate',)

	if low_lim is None:
		low_lgM_x = 9.0
	else:
		low_lgM_x = low_lim + 0.

	if up_lim is None:
		up_lgM_x = limd_lgM[-2] # limd_lgM[-2]
	else:
		up_lgM_x = up_lim + 0.

	x_lgM = np.linspace( low_lgM_x, up_lgM_x, N_points ) # lg( M_* / (M_sun / h^2) )
	x_CSMF = _interp_phi( x_lgM, limd_lgM[-6], interp_smf, interp_up_M )

	x_M = 10**x_lgM
	dx_lgM = np.mean( np.diff( x_lgM ) )
	dx_M = dx_lgM * np.log( 10 ) * x_M

	integ_M_0 = integ.simps( x_CSMF * x_M / (np.log(10) * x_M), x_M,) # total mass of given mass interval
	integ_M_1 = integ.simps( x_CSMF / (np.log(10) * x_M), x_M, )
	aveg_M = integ_M_0 / integ_M_1

	return integ_M_0, integ_M_1, aveg_M

def shift_aveg_SGs_Mstar_f(N_norm, norm_lgM, low_lim = None, up_lim = None, N_points = 250,):

	csmf_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/cc_Yang_CSMF.txt', sep = ',')
	lgM_star = np.array( csmf_dat['logM_*'] )
	CSMF = np.array( csmf_dat['[14.2 14.5]'] ) # averaged halo mass 14.33
	CSMF_err = np.array( csmf_dat['[14.2 14.5]_err'] )
	d_log_M = 0.05

	id_lim_0 = lgM_star < 11.6
	id_lim_1 = CSMF > 0.
	id_lim = id_lim_0 & id_lim_1

	limd_lgM = lgM_star[ id_lim]
	limd_CSMF = CSMF[ id_lim]
	limd_CSMF_err = CSMF_err[ id_lim]

	filt_CSMF = ndimage.gaussian_filter1d( limd_CSMF[:-1], sigma = 1, mode = 'nearest', truncate = 1.,)
	interp_smf = interp.interp1d( limd_lgM[:-1], filt_CSMF, kind = 'linear', fill_value = 'extrapolate',)
	# interp_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:], kind = 'quadratic', fill_value = 'extrapolate',)
	interp_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:], kind = 'linear', fill_value = 'extrapolate',)

	if low_lim is None:
		low_lgM_x = 9.0
	else:
		low_lgM_x = low_lim + 0.

	if up_lim is None:
		up_lgM_x = limd_lgM[-2] # limd_lgM[-2]
	else:
		up_lgM_x = up_lim + 0.

	#. normalization
	x_lgM = np.linspace( norm_lgM, 11.5, N_points ) # lg( M_* / (M_sun / h^2) )
	x_CSMF = _interp_phi( x_lgM, limd_lgM[-6], interp_smf, interp_up_M )

	x_M = 10**x_lgM
	dx_lgM = np.mean( np.diff( x_lgM ) )
	dx_M = dx_lgM * np.log( 10 ) * x_M

	pre_N0 = integ.simps( x_CSMF / (np.log(10) * x_M), x_M, )
	c_shift = N_norm / pre_N0

	shift_smf = interp.interp1d( limd_lgM[:-1], filt_CSMF * c_shift, kind = 'linear', fill_value = 'extrapolate',)
	# shift_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:] * c_shift, kind = 'quadratic', fill_value = 'extrapolate',)
	shift_up_M = interp.interp1d( limd_lgM[-5:], limd_CSMF[-5:] * c_shift, kind = 'linear', fill_value = 'extrapolate',)

	#. shifted dN / d_lgM 
	x_lgM = np.linspace( low_lgM_x, up_lgM_x, N_points )
	x_CSMF = _interp_phi( x_lgM, limd_lgM[-6], interp_smf, interp_up_M )

	x_M = 10**x_lgM
	dx_lgM = np.mean( np.diff( x_lgM ) )
	dx_M = dx_lgM * np.log( 10 ) * x_M

	_x_CSMF = _interp_phi( x_lgM, limd_lgM[-6], shift_smf, shift_up_M )

	_all_M = integ.simps( _x_CSMF * x_M / (np.log(10) * x_M), x_M,) # total mass of given mass interval
	_all_N = integ.simps( _x_CSMF / (np.log(10) * x_M), x_M, )
	_aveg_M = _all_M / _all_N

	return _all_M, _all_N, _aveg_M

"""

def Schechter_func(x, M_pov, phi_pov, alpha_pov):
	pf0 = ( x / M_pov )**(alpha_pov + 1)
	pf1 = np.exp( -1 * x / M_pov)
	m_pdf = phi_pov * pf0 * pf1

	return m_pdf

def shift_csmf_func(x, M_pov, phi_pov, alpha_pov, f_scale):
	pf0 = ( x / M_pov )**(alpha_pov + 1)
	pf1 = np.exp( -1 * x / M_pov)
	m_pdf = phi_pov * pf0 * pf1 * f_scale

	return m_pdf

#... fitting formula
def aveg_SGs_Mstar_f( low_lim = None, up_lim = None, N_points = 250 ):
	# low_lim, up_lim : lg( M_* / (M_sun / h^2) )

	csmf_param = np.loadtxt('/home/xkchen/tmp_run/data_files/figs/Yang_CSMF_fit-params.txt')
	M_mode, phi_mode, alpha_mode = csmf_param

	if low_lim is None:
		low_lgM_x = 9.0
	else:
		low_lgM_x = low_lim + 0.

	if up_lim is None:
		up_lgM_x = 11.6
	else:
		up_lgM_x = up_lim + 0.

	def modi_func(x, M_pov, phi_pov, alpha_pov):

		_pf0 = ( x / M_pov )**(alpha_pov + 1)
		_pf1 = np.exp( -1 * x / M_pov)
		_m_pdf = phi_pov * _pf0 * _pf1 / ( x * np.log(10) )

		return _m_pdf

	def mass_weit_func(x, M_pov, phi_pov, alpha_pov):

		_pf0 = ( x / M_pov )**(alpha_pov + 1)
		_pf1 = np.exp( -1 * x / M_pov)
		_m_pdf = ( phi_pov * _pf0 * _pf1 / ( x * np.log(10) ) ) * x

		return _m_pdf

	# total mass of given mass interval
	integ_M_0 = integ.romberg( mass_weit_func, 10**low_lgM_x, 10**up_lgM_x, args = (M_mode, phi_mode, alpha_mode),)

	# total galaxy number of given mass interval
	integ_M_1 = integ.romberg( modi_func, 10**low_lgM_x, 10**up_lgM_x, args = (M_mode, phi_mode, alpha_mode),)

	# average galaxy mass
	aveg_M = integ_M_0 / integ_M_1

	return integ_M_0, integ_M_1, aveg_M

def shift_aveg_SGs_Mstar_f(N_norm, norm_lgM, low_lim = None, up_lim = None, N_points = 250 ):
	# low_lim, up_lim : lg( M_* / (M_sun / h^2) )

	csmf_param = np.loadtxt('/home/xkchen/tmp_run/data_files/figs/Yang_CSMF_fit-params.txt')
	M_mode, phi_mode, alpha_mode = csmf_param

	if low_lim is None:
		low_lgM_x = 9.0
	else:
		low_lgM_x = low_lim + 0.

	if up_lim is None:
		up_lgM_x = 11.6
	else:
		up_lgM_x = up_lim + 0.

	#. normalization
	def _modi_func(x, M_pov, phi_pov, alpha_pov):

		_pf0 = ( x / M_pov )**(alpha_pov + 1)
		_pf1 = np.exp( -1 * x / M_pov)
		_m_pdf = phi_pov * _pf0 * _pf1 / ( x * np.log(10) )

		return _m_pdf

	pre_N0 = integ.romberg( _modi_func, 10**norm_lgM, 10**11.6, args = (M_mode, phi_mode, alpha_mode),)
	c_shift = N_norm / pre_N0

	def cc_modi_func(x, M_pov, phi_pov, alpha_pov, f_scale):

		pre_smf = shift_csmf_func(x, M_pov, phi_pov, alpha_pov, f_scale)
		_m_pdf = pre_smf / ( x * np.log(10) )

		return _m_pdf

	def cc_mass_weit_func(x, M_pov, phi_pov, alpha_pov, f_scale):

		pre_smf = shift_csmf_func(x, M_pov, phi_pov, alpha_pov, f_scale)
		_m_pdf = ( pre_smf / ( x * np.log(10) ) ) * x

		return _m_pdf

	# total mass of given mass interval
	_all_M = integ.romberg( cc_mass_weit_func, 10**low_lgM_x, 10**up_lgM_x, args = (M_mode, phi_mode, alpha_mode, c_shift),)
	
	# total galaxy number of given mass interval
	_all_N = integ.romberg( cc_modi_func, 10**low_lgM_x, 10**up_lgM_x, args = (M_mode, phi_mode, alpha_mode, c_shift),)

	# average galaxy mass
	_aveg_M = _all_M / _all_N

	return _all_M, _all_N, _aveg_M

#... 2Mpc truncated effect on mass fraction
def M2mpc_sub_fraction():

	v_m = 200 # rho_mean = 200 * rho_c * omega_m
	c_mass = [5.87, 6.95]
	Mh0 = [14.24, 14.24]
	off_set = [230, 210] # in unit kpc / h
	f_off = [0.37, 0.20]

	mm = 1
	test_R = np.logspace(0, np.log10(2e3), 500)
	misNFW_sigma = obs_sigma_func( test_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h
	sigma_2Mpc = obs_sigma_func( 2e3 * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h

	sigma_dm = sigmam(test_R * h, Mh0[mm], z_ref, c_mass[mm])
	sigma_dm_2Mpc = sigmam(2e3 * h, Mh0[mm], z_ref, c_mass[mm])

	id_x = test_R < 2e3
	sub_M0 = misNFW_sigma - sigma_2Mpc
	sub_M1 = sigma_dm - sigma_dm_2Mpc

	N_grid = 200

	mis_M0 = cumu_mass_func( test_R[id_x], misNFW_sigma[id_x], N_grid = N_grid,)
	mis_M1 = cumu_mass_func( test_R[id_x], sub_M0[id_x], N_grid = N_grid,)
	int_M0 = cumu_mass_func( test_R[id_x], sigma_dm[id_x], N_grid = N_grid,)
	int_M1 = cumu_mass_func( test_R[id_x], sub_M1[id_x], N_grid = N_grid,)
	eta_mis = (mis_M1 - mis_M0) / mis_M1
	eta = (int_M1 - int_M0) / int_M1	

	return test_R, eta, eta_mis

################################ sample measurement
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

## ... conditional galaxy stellar mass function
csmf_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/cc_Yang_CSMF.txt', sep = ',')
bin_lgM = np.array( csmf_dat['logM_*'] ) # M_sun / h**2
CSMF = np.array( csmf_dat['[14.2 14.5]'] ) # averaged halo mass 14.33
CSMF_err = np.array( csmf_dat['[14.2 14.5]_err'] )

id_lim_0 = bin_lgM < 11.6
id_lim_1 = CSMF > 0.
id_lim = id_lim_0 & id_lim_1

bin_lgM, CSMF, CSMF_err = bin_lgM[id_lim], CSMF[id_lim], CSMF_err[id_lim]

# # .Schechter function fitting
# po = [10**10.673, 8e-2, -1.2 ]
# popt, pcov = optimize.curve_fit(Schechter_func, 10**bin_lgM, CSMF, p0 = np.array( po ), sigma = CSMF_err,)

# M_fit, phi_fit, alpha_fit = popt
# x_lgM = np.linspace(7, 12, 100)
# fit_line = Schechter_func( 10**x_lgM, M_fit, phi_fit, alpha_fit )

# param_out = np.array([M_fit, phi_fit, alpha_fit])
# np.savetxt('/home/xkchen/tmp_run/data_files/figs/Yang_CSMF_fit-params.txt', param_out )

# interp_smf, interp_up_M = interp_csmf_func( 64, 10.0 )
# pre_csmf = _interp_phi( x_lgM, bin_lgM[-5], interp_smf, interp_up_M )

# plt.figure()
# ax0 = plt.subplot(111)
# ax0.plot( bin_lgM, CSMF, 'r--', alpha = 0.50, label = '$\\mathrm{Yang}{+}2012$')
# ax0.fill_between( bin_lgM, y1 = CSMF - CSMF_err, y2 = CSMF + CSMF_err, color = 'r', alpha = 0.15,)

# ax0.plot( x_lgM, fit_line, 'b-',)
# ax0.plot( x_lgM, pre_csmf, 'g--',)
# ax0.axvline( x = 11.5, ls = ':', color = 'k')

# ax0.set_ylim(7,12)
# ax0.set_xlabel('$\\lg \, M_{\\ast}\;[M_{\\odot} / h^{2}]$')
# ax0.set_ylim(1e-2, 2e2)
# ax0.set_yscale('log')
# ax0.set_ylabel('$dN / d\\lgM_{\\ast} / halo$')
# plt.savefig('/home/xkchen/CSMF_fit.png', dpi = 300)
# plt.close()


## ... satellite number density
bin_R, siglow, errsiglow, sighig, errsighig, highoverlow, errhighoverlow = np.genfromtxt(
																			'/home/xkchen/tmp_run/data_files/figs/result_high_over_low.txt', unpack = True)
#. Physical coordinate
bin_R = bin_R * 1e3 * a_ref / h
siglow, errsiglow, sighig, errsighig = np.array( [siglow * h**2 / 1e6, errsiglow * h**2 / 1e6, sighig * h**2 / 1e6, errsighig * h**2 / 1e6] ) / a_ref**2

id_nan = np.isnan( bin_R )
bin_R = bin_R[ id_nan == False]
siglow, errsiglow, sighig, errsighig = siglow[ id_nan == False], errsiglow[ id_nan == False], sighig[ id_nan == False], errsighig[ id_nan == False]


## ... DM mass profile
lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = ( lo_xi * 1e3 * rho_m ) * h / a_ref**2
lo_rp = lo_rp * 1e3 * a_ref / h

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) * h / a_ref**2
hi_rp = hi_rp * 1e3 * a_ref / h

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )


#... cluster mass and radius 
dd_rich = np.array( [] ) 
dd_z_obs = np.array( [] )
dd_lg_Mstar = np.array( [] )
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

for mm in range( 2 ):

	#... lg_Mstar
	l_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm])
	l_rich = np.array( l_dat['rich'])
	l_lgM = np.array( l_dat['lg_Mstar'])

	#... mag
	pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG-color.csv' % cat_lis[mm] )
	p_z = np.array( pdat['z'] )

	dd_z_obs = np.r_[ dd_z_obs, p_z ]
	dd_rich = np.r_[ dd_rich, l_rich ]
	dd_lg_Mstar = np.r_[ dd_lg_Mstar, l_lgM - 2 * np.log10( h ) ]

M200m, R200m = rich2R_Simet( dd_z_obs, dd_rich,)


###... total mass profile and satellite number density
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

misNFW_sigma = xi_to_Mf( lo_rp )
sigma_2Mpc = xi_to_Mf( 2e3 )
lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )


sig_aveg = (siglow + sighig) / 2
err_aveg = np.sqrt( errsiglow**2 / 4 + errsighig**2 / 4)
sig_rho_f = interp.interp1d( bin_R, sig_aveg, kind = 'linear', fill_value = 'extrapolate',)

Ng_sigma = sig_rho_f( bin_R )
Ng_2Mpc = sig_rho_f( 2e3 )

N_grid = 500
integ_Ng = cumu_mass_func( bin_R, Ng_sigma, N_grid = N_grid )
fun_Ng = interp.interp1d( bin_R, integ_Ng, kind = 'linear', fill_value = 'extrapolate',)
N_sat = fun_Ng( np.median( R200m ) )


### limited magnitude and satellite mass estimation
#. lgM* = a*(g-r) + b*(r-i) + c*lg(Li) + d
p_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv' )
a_fit = np.array( p_dat['a'] )[0]
b_fit = np.array( p_dat['b'] )[0]
c_fit = np.array( p_dat['c'] )[0]
d_fit = np.array( p_dat['d'] )[0]

#. averaged g-r and r-i, Rykoff et al. 2014 (90percen member color)
mean_gr = 1.4
mean_ri = 0.5

#. faint galaxies mass estimation (low limitation of dwarf galaxy)
dwf_g_M = 1e7 * h**2 # M_sun / h^2

# #. limit from satellites number density profile
# lim_i_cmag = 21 # mag
# lim_i_Mag = lim_i_cmag - 5 * np.log10( Dl_ref * 1e6 ) + 5

lim_i_Mag = -19.43 + 5 * np.log10( h )
lim_Li = 10**( -0.4 * ( lim_i_Mag - Mag_sun[2] ) )


m_lg_M = a_fit * mean_gr + b_fit * mean_ri + c_fit * np.log10(lim_Li) + d_fit
lim_Mi = 10**m_lg_M * h**2

cc_sum_M, cc_Ng, cc_aveg_M = aveg_SGs_Mstar_f( np.log10( lim_Mi ) )
scal_F = N_sat / cc_Ng


#. limit from source detection
lim_L_sat = 22.08 # 22.8
dt_iMag = lim_L_sat - 5 * np.log10( Dl_ref * 1e6 ) + 5
dt_Li = 10**( -0.4 * ( dt_iMag - Mag_sun[2] ) )
dt_lgM = a_fit * mean_gr + b_fit * mean_ri + c_fit * np.log10(dt_Li) + d_fit
dt_Mi = 10**dt_lgM * h**2

sum_M, N_glx, aveg_M = shift_aveg_SGs_Mstar_f( N_sat, np.log10( lim_Mi ), low_lim = np.log10( dt_Mi ) )
lg_sum_M = np.log10( sum_M / h**2 )
lg_aveg_M = np.log10( aveg_M / h**2 )

sum_dwf_M, N_dwf, aveg_dwf_M = shift_aveg_SGs_Mstar_f( N_sat, np.log10( lim_Mi ),low_lim = np.log10( dwf_g_M ), up_lim = np.log10( dt_Mi ),)
lg_sum_dwf_M = np.log10( sum_dwf_M / h**2 )
lg_aveg_dwf_M = np.log10( aveg_dwf_M / h**2 )

#.. observed profile integrated mass
# M_ICL = 5.4*1e11 # unit : M_sun
# M_tran = 5.76*1e10
# M_BCG = 3.4*1e11
# #. total mass based on fitting profile integration
# M_halo = 2.432 * 1e14
# M_halo_no_sub = 3.13 * 1e14


#.. flux scaling correction mass
id_dered = True
dered_str = 'with-dered_'

# id_dered = False
# dered_str = ''


fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'
m_dat = pds.read_csv( fit_path + '%smass_contribution_estimator.csv' % dered_str )

lgM_ICL = np.array( m_dat['lg_M_ICL'] )[0]
M_ICL = 10**lgM_ICL

lgM_bcg = np.array( m_dat['lg_M_bcg'] )[0]
M_BCG = 10**lgM_bcg

lgM_trans = np.array( m_dat['lg_M_trans'] )[0]
M_tran = 10**lgM_trans

lgM_halo = np.array( m_dat['lg_M_all'] )[0]
M_halo = 10**lgM_halo

M_halo_BG_sub = 10**np.array( m_dat['lg_M_all_BG_sub'] )[0]


sum_Mstar = M_ICL + M_BCG + 10**lg_sum_M
pure_ICL = M_ICL - 10**lg_sum_dwf_M

eta_Mstar = np.array( [M_BCG, 10**lg_sum_M, pure_ICL - M_tran ] ) / sum_Mstar
eta_Mstar_1 = np.array( [10**lg_sum_dwf_M, M_tran] ) / sum_Mstar

eta_Mtot = np.array( [M_BCG, 10**lg_sum_M, pure_ICL - M_tran ] ) / M_halo
eta_Mtot_1 = np.array( [10**lg_sum_dwf_M, M_tran] ) / M_halo


###. figs
csmf_param = np.loadtxt('/home/xkchen/tmp_run/data_files/figs/Yang_CSMF_fit-params.txt')
M_mode, phi_mode, alpha_mode = csmf_param

lgM_bins = np.linspace( np.log10( dwf_g_M ), 11.65, 200) # the mass unit is M_sun / h^2

_csmf_fit = Schechter_func( 10**lgM_bins, M_mode, phi_mode, alpha_mode )
norm_csmf_ = shift_csmf_func( 10**lgM_bins, M_mode, phi_mode, alpha_mode, scal_F )

#. corresponding i_mag for mass bins
cc_lgM_bins = lgM_bins - 2 * np.log10( h ) # the mass unit is M_sun


"""
fig = plt.figure( figsize = (5.8, 5.4) )
ax0 = fig.add_axes( [0.155, 0.11, 0.80, 0.80] )

# ax0.plot( lgM_bins, norm_csmf_, ls = '-', color = 'darkred', alpha = 0.80, label = '$\\mathrm{Yang}{+}2012$')
# ax0.fill_betweenx( y = np.linspace(1e-4, 1e3, 500), x1 = np.log10( dt_Mi ), x2 = 0, color = 'teal', alpha = 0.15,)

ax0.plot( cc_lgM_bins, norm_csmf_, ls = '-', color = 'k', 
	label = 'Conditional SMF $({\\rm \\mathcal{lg} } [M_{h} \, / \, M_{\\odot}] = 14.41)$',)
ax0.fill_betweenx( y = np.linspace(1e-4, 1e3, 500), x1 = dt_lgM, x2 = 0, color = 'grey', alpha = 0.25,)

ax0.annotate( text = '$\\Sigma M_{\\ast}^{ \\mathrm{unmasked} }{=}10^{%.1f} M_{\\odot}$' % lg_sum_dwf_M, xy = (0.03, 0.25), 
	xycoords = 'axes fraction', fontsize = 14, color = 'k',)

ax0.legend( loc = 3, frameon = False, fontsize = 12,)
ax0.set_xlim( 7.0, 12 )
ax0.set_ylim( 1e-3, 3e2 )

ax0.set_ylabel('${\\rm d} N \, / \, {\\rm d} {\\rm \\mathcal{lg} } M_{\\ast} \; / \; \\mathrm{halo}$', fontsize = 15,)
ax0.set_yscale('log')

ax0.set_xlabel('${\\rm \\mathcal{lg} } \, [ M_{\\ast} \, / \, M_{\\odot} ]$', fontsize = 15,)
ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

sub_ax0 = ax0.twiny()
sub_ax0.set_xlim( ax0.get_xlim() )

x_ticks = ax0.get_xticks()

#. M_bin_ticks --> i_mag ticks
cc_lgM_2_Li = ( x_ticks - d_fit - a_fit * mean_gr - b_fit * mean_ri ) / c_fit
cc_iMag_x = (cc_lgM_2_Li / (-0.4) ) + Mag_sun[2]

label_lis = ['%.0f' % ll for ll in cc_iMag_x ]

sub_ax0.set_xticks( x_ticks )
sub_ax0.set_xticklabels( label_lis )
sub_ax0.set_xlabel('$\\mathrm{M}_{\\mathrm{i,\,cModel} }$', fontsize = 15,)
sub_ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

plt.savefig('/home/xkchen/sat_icl_mass_estimation.png', dpi = 300)
# plt.savefig('/home/xkchen/sat_icl_mass_estimation.pdf', dpi = 300)
plt.close()
"""


bar_x = np.array([ 1, 4, 2.5])
sum_eta_0 = np.sum( eta_Mstar ) + np.sum( eta_Mstar_1)
sum_eta_1 = np.sum( eta_Mtot ) + np.sum( eta_Mtot_1)
modi_lim = sum_eta_0 / ( 100 * sum_eta_1 )


fig = plt.figure( figsize = (6.1, 5.4) )
ax = fig.add_axes( [0.13, 0.08, 0.75, 0.85] )

ax.bar( bar_x, height = eta_Mstar, width = 1, color = 'none', edgecolor = 'k',)
ax.bar( bar_x[1], height = eta_Mstar_1[0], width = 1, bottom = eta_Mstar[1], color = 'dimgrey', edgecolor = 'dimgrey',)
ax.bar( bar_x[0], height = eta_Mstar_1[1], width = 1, bottom = eta_Mstar[0], color = 'dimgrey', edgecolor = 'dimgrey',)

ax.text( 3.5, 0.83, s = 'unmasked', fontsize = 15, color = 'dimgrey',)
ax.text( 0.5, 0.11, s = 'transition', fontsize = 15, color = 'dimgrey',)

#. '%.2f%%'
ax.text( 1.0, 0.18, s = '%.1f%%' % ( (eta_Mstar[0] + eta_Mstar_1[1]) * 100 ), fontsize = 15, horizontalalignment = 'center',)
ax.text( 2.5, 0.14, s = '%.1f%%' % (eta_Mstar[2] * 100), fontsize = 15, horizontalalignment = 'center',)
ax.text( 4.0, 0.9, s = '%.1f%%' % ( (eta_Mstar[1] + eta_Mstar_1[0]) * 100), fontsize = 15, horizontalalignment = 'center',)

# ax.text( 1.0, 0.18, s = '8.7%', fontsize = 15, horizontalalignment = 'center',)
# ax.text( 2.5, 0.14, s = '9.4%', fontsize = 15, horizontalalignment = 'center',)
# ax.text( 4.0, 0.9, s = '81.9%', fontsize = 15, horizontalalignment = 'center',)

ax.set_xticks( bar_x )
ax.set_xticklabels(labels = ['BCG', 'Satellites', 'ICL'])

top_y = 1.1
ax.set_ylim(0, top_y)

ax.axhline( y = 1, ls = '--', color = 'k',)
ax.set_ylabel('$M_{\\ast} \, / \, M_{\\ast}^{tot}$', fontsize = 15,)

ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

ax.text( 2, 1.02, s = '$M_{\\ast}^{tot} \, / \, M_{h} \, = \, %.2f $' % (100 * sum_eta_1) + '%', fontsize = 15,)

sub_ax = ax.twinx()

sub_ax.bar( bar_x[1], height = 100 * eta_Mtot_1[0], width = 1, bottom = 100 * eta_Mtot[1], color = 'dimgrey', edgecolor = 'dimgrey',)
sub_ax.bar( bar_x[0], height = 100 * eta_Mtot_1[1], width = 1, bottom = 100 * eta_Mtot[0], color = 'dimgrey', edgecolor = 'dimgrey',)
sub_ax.bar( bar_x, height = 100 * eta_Mtot, width = 1, color = 'none', edgecolor = 'k',)

sub_ax.set_ylabel('$100{\\times} \, M_{\\ast} \, / \, M_{h}$', fontsize = 15,)
sub_ax.set_ylim(0, top_y / modi_lim)
sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/%sICL_star_mass_fraction.png' % dered_str, dpi = 300)
plt.savefig('/home/xkchen/ICL_star_mass_fraction.pdf', dpi = 300)
plt.close()

