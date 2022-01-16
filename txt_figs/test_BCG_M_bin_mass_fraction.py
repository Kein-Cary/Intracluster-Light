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

def pdf_log_norm_func( r, Am, Rt, sigm_tt ):

	mf0 = r * sigm_tt * np.sqrt( 2 * np.pi )
	mf1 = -0.5 * ( np.log(r) - np.log(Rt) )**2 / sigm_tt**2
	Pdf = Am * np.exp( mf1 ) / mf0

	return Pdf

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


### === ### obs BCG + ICL mass profile

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

band_str = 'gri'

file_s = 'BCG_Mstar_bin'

out_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'


#... cluster BCG mass and richness 
dd_rich = []
dd_lg_Mstar = []

for mm in range( 2 ):

	#... lg_Mstar
	l_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'%s_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm],)
	l_rich = np.array( l_dat['rich'])
	l_lgM = np.array( l_dat['lg_Mstar'])

	dd_rich.append( l_rich )
	dd_lg_Mstar.append( l_lgM - 2 * np.log10( h ) )


def compare_to_pre_select_cat():
	"""
	check the average properties is the same or not
	"""
	# ... cluster catalog before image match and selection
	all_lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/low_BCG_star-Mass_fixed_rich_cluster.csv')
	all_lo_rich = np.array( all_lo_dat['Lambda'] )
	all_lo_Mstar = np.array( all_lo_dat['LgMstar'] ) - 2 * np.log10( h )

	all_hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/high_BCG_star-Mass_fixed_rich_cluster.csv')
	all_hi_rich = np.array( all_hi_dat['Lambda'] )
	all_hi_Mstar = np.array( all_hi_dat['LgMstar'] ) - 2 * np.log10( h )


	plt.figure()
	plt.title( 'Low $ M_{\\ast}^{\\mathrm{BCG}}$' )
	plt.hist( dd_rich[0], bins = 45, density = False, color = 'b', histtype = 'step', label = 'after selection')
	plt.axvline( x = np.median( dd_rich[0] ), ls = '--', alpha = 0.5, color = 'b', label = 'median')
	plt.axvline( x = np.mean( dd_rich[0] ), ls = '-', alpha = 0.5, color = 'b', label = 'mean')

	plt.hist( all_lo_rich, bins = 45, density = False, color = 'r', histtype = 'step', label = 'before selection')
	plt.axvline( x = np.median( all_lo_rich ), ls = '--', alpha = 0.5, color = 'r',)
	plt.axvline( x = np.mean( all_lo_rich ), ls = '-', alpha = 0.5, color = 'r',)

	plt.legend( loc = 1)
	plt.xscale('log')
	plt.xlabel('$\\lambda$')
	plt.yscale('log')
	plt.savefig('/home/xkchen/Low_BCG_M_rich_compare.png', dpi = 300)
	plt.close()

	plt.figure()
	plt.title( 'High $ M_{\\ast}^{\\mathrm{BCG}}$' )
	plt.hist( dd_rich[1], bins = 45, density = False, color = 'b', histtype = 'step', label = 'after selection')
	plt.axvline( x = np.median( dd_rich[1] ), ls = '--', alpha = 0.5, color = 'b', label = 'median')
	plt.axvline( x = np.mean( dd_rich[1] ), ls = '-', alpha = 0.5, color = 'b', label = 'mean')

	plt.hist( all_hi_rich, bins = 45, density = False, color = 'r', histtype = 'step', label = 'before selection')
	plt.axvline( x = np.median( all_hi_rich ), ls = '--', alpha = 0.5, color = 'r',)
	plt.axvline( x = np.mean( all_hi_rich ), ls = '-', alpha = 0.5, color = 'r',)

	plt.legend( loc = 1)
	plt.xscale('log')
	plt.xlabel('$\\lambda$')
	plt.yscale('log')
	plt.savefig('/home/xkchen/High_BCG_M_rich_compare.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.title( 'Low $ M_{\\ast}^{\\mathrm{BCG}}$' )
	plt.hist( dd_lg_Mstar[0], bins = 45, density = False, color = 'b', histtype = 'step', label = 'after selection')
	plt.axvline( x = np.median( dd_lg_Mstar[0] ), ls = '--', alpha = 0.5, color = 'b', label = 'median')
	plt.axvline( x = np.mean( dd_lg_Mstar[0] ), ls = '-', alpha = 0.5, color = 'b', label = 'mean')

	plt.hist( all_lo_Mstar, bins = 45, density = False, color = 'r', histtype = 'step', label = 'before selection')
	plt.axvline( x = np.median( all_lo_Mstar ), ls = '--', alpha = 0.5, color = 'r',)
	plt.axvline( x = np.mean( all_lo_Mstar ), ls = '-', alpha = 0.5, color = 'r',)

	plt.legend( loc = 2,)
	plt.xlabel('$\\lg M_{\\ast}^{BCG} [M_{\\odot}]$')
	plt.savefig('/home/xkchen/Low_BCG_M_Mstar_compare.png', dpi = 300)
	plt.close()

	plt.figure()
	plt.title( 'High $ M_{\\ast}^{\\mathrm{BCG}}$' )
	plt.hist( dd_lg_Mstar[1], bins = 45, density = False, color = 'b', histtype = 'step', label = 'after selection')
	plt.axvline( x = np.median( dd_lg_Mstar[1] ), ls = '--', alpha = 0.5, color = 'b', label = 'median')
	plt.axvline( x = np.mean( dd_lg_Mstar[1] ), ls = '-', alpha = 0.5, color = 'b', label = 'mean')

	plt.hist( all_hi_Mstar, bins = 45, density = False, color = 'r', histtype = 'step', label = 'before selection')
	plt.axvline( x = np.median( all_hi_Mstar ), ls = '--', alpha = 0.5, color = 'r',)
	plt.axvline( x = np.mean( all_hi_Mstar ), ls = '-', alpha = 0.5, color = 'r',)

	plt.legend( loc = 1,)
	plt.xlabel('$\\lg M_{\\ast}^{BCG} [M_{\\odot}]$')
	plt.savefig('/home/xkchen/High_BCG_M_Mstar_compare.png', dpi = 300)
	plt.close()

# compare_to_pre_select_cat()


### === M200m and R200m estimate
Mh_clus = 10**14.41 # M_sun
mrho_zref = rhom_set( z_ref )[1]
mrho_zref = mrho_zref * h**2  # M_sun / kpc^3

R200m = ( 3 * Mh_clus / (4 * np.pi * mrho_zref * 200) )**(1 / 3)


lo_Mbcg = dd_lg_Mstar[0]
hi_Mbcg = dd_lg_Mstar[1]

lo_lg_medi_Mbcg = np.log10( np.median( 10**lo_Mbcg ) )
hi_lg_medi_Mbcg = np.log10( np.median( 10**hi_Mbcg ) )

lo_lg_mean_Mbcg = np.log10( np.mean( 10**lo_Mbcg ) )
hi_lg_mean_Mbcg = np.log10( np.mean( 10**hi_Mbcg ) )


#. mass estimation with deredden or not
id_dered = True
dered_str = '_with-dered'

out_lim_R = 350

c_dat = pds.read_csv( fit_path + 'with-dered_total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R,)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]


#. mass profile for central region
cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[0], band_str, dered_str) )
lo_Ie, lo_Re, lo_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]

cen_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit%s.csv' % (cat_lis[1], band_str, dered_str) )
hi_Ie, hi_Re, hi_Ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]


### === fitting profiles
new_R = np.logspace( 0, np.log10(2.5e3), 100 )

#. mass profile on large scale
lo_out_SM = ( lo_interp_F( new_R ) - lo_xi2M_2Mpc ) * 10**lg_fb_gi
hi_out_SM = ( hi_interp_F( new_R ) - hi_xi2M_2Mpc ) * 10**lg_fb_gi

lo_cen_M = sersic_func( new_R, 10**lo_Ie, lo_Re, lo_Ne) - sersic_func( 2e3, 10**lo_Ie, lo_Re, lo_Ne)
hi_cen_M = sersic_func( new_R, 10**hi_Ie, hi_Re, hi_Ne) - sersic_func( 2e3, 10**hi_Ie, hi_Re, hi_Ne)


### === mass profile for the middle region

#. SM(r) fitting
# mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[0], band_str, dered_str),)
# lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
# lo_mid_mass = log_norm_func( lo_rp, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit ) - log_norm_func( 2e3, lo_lgSM_fit, lo_Rt_fit, lo_sigm_t_fit )

# id_nul = lo_mid_mass < 0.
# lo_mid_mass[ id_nul ] = 0.

# mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit%s.csv' % (cat_lis[1], band_str, dered_str),)
# hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['lg_M0'])[0], np.array( mid_dat['R_t'] )[0], np.array( mid_dat['sigma_t'] )[0]
# hi_mid_mass = log_norm_func( hi_rp, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit ) - log_norm_func( 2e3, hi_lgSM_fit, hi_Rt_fit, hi_sigm_t_fit )

# id_nul = hi_mid_mass < 0.
# hi_mid_mass[ id_nul ] = 0.


#. SM(r) ratio fitting
mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_mid-region_Lognorm_ratio-based_fit%s.csv' % (cat_lis[0], band_str, dered_str),)
lo_Am_fit, lo_Rt_fit, lo_sigm_t_fit = np.array( mid_dat['Am'])[0], np.array( mid_dat['Rt'] )[0], np.array( mid_dat['sigma_t'] )[0]

lo_mid_pdf = pdf_log_norm_func( new_R, lo_Am_fit, lo_Rt_fit, lo_sigm_t_fit)
lo_fit_sum = lo_out_SM + lo_cen_M
lo_mid_mass = lo_mid_pdf * lo_fit_sum / ( 1 - lo_mid_pdf )


mid_dat = pds.read_csv( fit_path + '%s_%s-band-based_mid-region_Lognorm_ratio-based_fit%s.csv' % (cat_lis[1], band_str, dered_str),)
hi_Am_fit, hi_Rt_fit, hi_sigm_t_fit = np.array( mid_dat['Am'])[0], np.array( mid_dat['Rt'] )[0], np.array( mid_dat['sigma_t'] )[0]

hi_mid_pdf = pdf_log_norm_func( new_R, hi_Am_fit, hi_Rt_fit, hi_sigm_t_fit)
hi_fit_sum = hi_out_SM + hi_cen_M
hi_mid_mass = hi_mid_pdf * hi_fit_sum / ( 1 - hi_mid_pdf )



N_grid = 250

lo_mid_integ_M = cumu_mass_func( new_R, lo_mid_mass, N_grid = N_grid )
lo_fun_mid_M = interp.interp1d( new_R, lo_mid_integ_M, kind = 'linear', fill_value = 'extrapolate',)

hi_mid_integ_M = cumu_mass_func( new_R, hi_mid_mass, N_grid = N_grid )
hi_fun_mid_M = interp.interp1d( new_R, hi_mid_integ_M, kind = 'linear', fill_value = 'extrapolate',)

lo_mod_trans_M = np.log10( lo_fun_mid_M( R200m ) )
hi_mod_trans_M = np.log10( hi_fun_mid_M( R200m ) )

print( lo_mod_trans_M, hi_mod_trans_M )

