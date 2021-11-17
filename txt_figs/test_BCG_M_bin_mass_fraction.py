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

def log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * (np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_M = lg_SM0 - lg_A0 - lg_A1

	return 10**lg_M

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

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )


## ... obs BCG + ICL mass profile
#. mass estimation with deredden or not
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

file_s = 'BCG_Mstar_bin'
out_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'
band_str = 'gri'

#... cluster mass and radius 
dd_rich = []
dd_z_obs = []
dd_lg_Mstar = []

for mm in range( 2 ):

	#... lg_Mstar
	l_dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm])
	l_rich = np.array( l_dat['rich'])
	l_lgM = np.array( l_dat['lg_Mstar'])

	#... mag
	pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG-color.csv' % cat_lis[mm] )
	p_z = np.array( pdat['z'] )

	dd_z_obs.append( p_z )
	dd_rich.append( l_rich )
	dd_lg_Mstar.append( l_lgM - 2 * np.log10( h ) )

#. mass estimation with deredden or not
id_dered = True
dered_str = '_with-dered'

# id_dered = False
# dered_str = ''

if id_dered == False:
	#. surface mass profiles
	dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str),)
	lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str),)
	hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

if id_dered == True:
	#. surface mass profiles
	dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[0], band_str),)
	lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

	dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[1], band_str),)
	hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )


#. mass profile for central region
cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[0], band_str, dered_str) )
lo_Ie, lo_Re, lo_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]

cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[1], band_str, dered_str) )
hi_Ie, hi_Re, hi_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]


#... mass profile for the middle region
#. lognormal
mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[0], band_str, dered_str),)
lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]


mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[1], band_str, dered_str),)
hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]


## ... M200m and R200m estimate
lo_M200m, lo_R200m = rich2R_Simet( dd_z_obs[0], dd_rich[0],)

hi_M200m, hi_R200m = rich2R_Simet( dd_z_obs[1], dd_rich[1],)

lo_medi_R200m = np.median( lo_R200m )
lo_mean_R200m = np.mean( lo_R200m )

hi_medi_R200m = np.median( hi_R200m )
hi_mean_R200m = np.mean( hi_R200m )

lo_Mbcg = dd_lg_Mstar[0]
hi_Mbcg = dd_lg_Mstar[1]

#..
new_R = np.logspace( 0, np.log10(2.5e3), 100 )

lo_cen_M = sersic_func( new_R, 10**lo_Ie, lo_Re, lo_Ne) - sersic_func( 2e3, 10**lo_Ie, lo_Re, lo_Ne)
lo_mid_mass = log_norm_func( new_R, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit ) - log_norm_func( 2e3, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit )

hi_cen_M = sersic_func( new_R, 10**hi_Ie, hi_Re, hi_Ne) - sersic_func( 2e3, 10**hi_Ie, hi_Re, hi_Ne)
hi_mid_mass = log_norm_func( new_R, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit ) - log_norm_func( 2e3, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit )


N_grid = 500

lo_mid_integ_M = cumu_mass_func( new_R, lo_mid_mass, N_grid = N_grid )
lo_fun_mid_M = interp.interp1d( new_R, lo_mid_integ_M, kind = 'linear', fill_value = 'extrapolate',)

hi_mid_integ_M = cumu_mass_func( new_R, hi_mid_mass, N_grid = N_grid )
hi_fun_mid_M = interp.interp1d( new_R, hi_mid_integ_M, kind = 'linear', fill_value = 'extrapolate',)

lo_mod_trans_M = np.log10( lo_fun_mid_M( lo_medi_R200m ) )
hi_mod_trans_M = np.log10( hi_fun_mid_M( hi_medi_R200m ) )


