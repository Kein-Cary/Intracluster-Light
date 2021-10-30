import matplotlib as mpl
import matplotlib.pyplot as plt

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

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set
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

### === ### sersic profile
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

"""
def log_norm_func( r, Im, R_pk, L_trans, _eta_pk ):

	#... scaled version
	scl_r = r / R_pk       # r / R_crit
	scl_L = L_trans / R_pk # L_trans / R_crit

	# cen_p = 0.25 # 0.10, 0.25 # R_crit / R_pk
	cen_p = _eta_pk

	f1 = 1 / ( scl_r * scl_L * np.sqrt(2 * np.pi) )
	f2 = np.exp( -0.5 * (np.log( scl_r ) - cen_p )**2 / scl_L**2 )

	Ir = 10**Im * f1 * f2
	return Ir

"""
def log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * (np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_M = lg_SM0 - lg_A0 - lg_A1

	return 10**lg_M

### === ### miscentering nfw profile (Zu et al. 2020, section 3.)
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""

	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

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
		sum_theta_fdens = np.sum( medi_surf_dens * d_theta, axis = 1 ) / ( 2 * np.pi )

		## integration on r_off
		integ_f = sum_theta_fdens * off_pdf

		medi_integ_f = ( integ_f[1:] + integ_f[:-1] ) / 2

		surf_dens_ii = np.sum( medi_integ_f * dr_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	if NR == 1:
		return off_sigma[0]
	return off_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

	off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	norm_sigma = sigmam( rp, lgM, z, c_mass)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

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

### === ### loda obs data
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)


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


#...
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

misNFW_sigma = xi_to_Mf( lo_rp )
sigma_2Mpc = xi_to_Mf( 2e3 )
lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )


### === profile integration
id_dered = True
dered_str = 'with-dered_'

# id_dered = False
# dered_str = ''

#. path of obs. data
band_str = 'gri'
BG_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

if id_dered	== False:
	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
	obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

if id_dered == True:
	dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi_with-dered.csv' % band_str,)
	obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])	


id_rx = obs_R >= 1
obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]
lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )


# central part
p_dat = pds.read_csv( fit_path + '%stotal-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % (dered_str, band_str),)
c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

#... trans part fitting
mid_pat = pds.read_csv( fit_path + '%stotal_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (dered_str, band_str),)
lg_SM_fit, Rt_fit, sigm_tt_fit = np.array( mid_pat['lg_M0'] )[0], np.array( mid_pat['R_t'] )[0], np.array( mid_pat['sigma_t'] )[0]

# parameters of scaled relation
out_lim_R = 350 # 400

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % (dered_str, out_lim_R),)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]


#...trans part
new_R = np.logspace( 0, np.log10(2.5e3), 100)
cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
fit_cen_M = sersic_func( new_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc

devi_M = surf_M - ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi - ( sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc )

devi_lgM = np.log10( devi_M )
devi_err = lg_M_err
devi_R = obs_R


# mass ration within given radius
tot_interp_F = interp.interp1d( obs_R, surf_M, kind = 'linear', fill_value = 'extrapolate',)
cc_tot_M = tot_interp_F( new_R )

mid_M = log_norm_func( new_R, lg_SM_fit, Rt_fit, sigm_tt_fit ) - log_norm_func( 2e3, lg_SM_fit, Rt_fit, sigm_tt_fit )


#. set mid_M < 0 case as zero
id_zeros = mid_M < 0.
mid_M[ id_zeros ] = 0.

cc_out_M = xi_to_Mf( new_R ) * 10**lg_fb_gi
cc_DM_M = xi_to_Mf( new_R ) - sigma_2Mpc

all_M_no_sub = xi_to_Mf( new_R )


N_grid = 500
tot_integ_M = cumu_mass_func( new_R, cc_tot_M, N_grid = N_grid ) # BCG + ICL
cen_integ_M = cumu_mass_func( new_R, fit_cen_M, N_grid = N_grid )
out_integ_M = cumu_mass_func( new_R, cc_out_M, N_grid = N_grid )
mid_integ_M = cumu_mass_func( new_R, mid_M, N_grid = N_grid )

DM_integ_M = cumu_mass_func( new_R, cc_DM_M, N_grid = N_grid ) # stellar + gas + DM
all_integ_M = cumu_mass_func( new_R, all_M_no_sub, N_grid = N_grid ) # stellar + gas + DM, No 'background subtraction'

fun_tot_M = interp.interp1d( new_R, tot_integ_M, kind = 'linear', fill_value = 'extrapolate',)
fun_cen_M = interp.interp1d( new_R, cen_integ_M, kind = 'linear', fill_value = 'extrapolate',)
fun_mid_M = interp.interp1d( new_R, mid_integ_M, kind = 'linear', fill_value = 'extrapolate',)
fun_out_M = interp.interp1d( new_R, out_integ_M, kind = 'linear', fill_value = 'extrapolate',)

fun_DM_M = interp.interp1d( new_R, DM_integ_M, kind = 'linear', fill_value = 'extrapolate',)
fun_all_M = interp.interp1d( new_R, all_integ_M, kind = 'linear', fill_value = 'extrapolate',)

eta_bcg_ICL = tot_integ_M / DM_integ_M
eta_cen_M = cen_integ_M / DM_integ_M
eta_mid_M = mid_integ_M / DM_integ_M
eta_out_M = out_integ_M / DM_integ_M

#. mass estimates
medi_R200m = np.median( R200m )
mean_R200m = np.mean( R200m )

medi_M200m = np.median( M200m )
mean_M200m = np.mean( M200m )

#. aperture size estimation
fix_R_dat = pds.read_csv( BG_path + 'BCGM-match_R.csv' )
R_fixed_M = np.array( fix_R_dat[ 'R_fixed_medi_M' ] )[0]

mod_M_bcg = np.log10( fun_cen_M( R_fixed_M ) )
mod_all_M = np.log10( fun_all_M( medi_R200m ) )

mod_trans_M = np.log10( fun_mid_M( medi_R200m ) )
mod_out_ICL_M = np.log10( fun_out_M( medi_R200m ) )
mod_ICL_M = np.log10( fun_mid_M( medi_R200m ) + fun_out_M( medi_R200m ) )

mod_all_M_BG_sub = np.log10( fun_DM_M( medi_R200m ) )

obs_bcg_ICL_M = np.log10( fun_tot_M( medi_R200m ) )

#. 
keys = ['lg_M_bcg', 'lg_M_ICL', 'lg_M_trans', 'lg_M_out_ICL', 'lg_M_all', 'lg_M_all_BG_sub', 'obs_lg_M_bcg&ICL']
values = [ mod_M_bcg, mod_ICL_M, mod_trans_M, mod_out_ICL_M, mod_all_M, mod_all_M_BG_sub, obs_bcg_ICL_M ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
out_data.to_csv( fit_path + '%smass_contribution_estimator.csv' % dered_str )


plt.figure()
ax = plt.subplot(111)

ax.plot( new_R, eta_cen_M, 'b-.', label = 'BCG')
ax.plot( new_R, eta_mid_M, 'b:', label = 'ICL_trans')
ax.plot( new_R, eta_out_M, 'b--', label = 'ICL_out')
ax.plot( new_R, eta_bcg_ICL, 'b-', label = 'BCG + ICL')

ax.set_xscale('log')
ax.set_xlim(3, 1.1e3)
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e0)
ax.legend( loc = 1)
plt.savefig('/home/xkchen/%smass_ratio.png' % dered_str, dpi = 300)
plt.close()
