import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

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
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set
from color_2_mass import get_c2mass_func, gi_band_c2m_func
from multiprocessing import Pool

import emcee

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

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### sersic profile
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def log_parabolic_func(r, Im, Rm, am, bas):
	Ir = 10**Im * ( r / Rm)**( -am - bas * np.log( r / Rm) )
	return Ir

def log_norm_func(r, Im, R_pk, L_trans):

	#... scaled version
	scl_r = r / R_pk       # r / R_crit
	scl_L = L_trans / R_pk # L_trans / R_crit

	cen_p = 0.25 # R_crit / R_pk

	f1 = 1 / ( scl_r * scl_L * np.sqrt(2 * np.pi) )
	f2 = np.exp( -0.5 * (np.log( scl_r ) - cen_p )**2 / scl_L**2 )

	Ir = 10**Im * f1 * f2
	return Ir

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

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

	off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	norm_sigma = sigmam( rp, lgM, z, c_mass)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

### === loda obs data
color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
mark_s = ['s', 'o']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

## ... satellite number density
bin_R, siglow, errsiglow, sighig, errsighig, highoverlow, errhighoverlow = np.genfromtxt(
																		 '/home/xkchen/tmp_run/data_files/figs/result_high_over_low.txt', unpack = True)
bin_R = bin_R * 1e3 * a_ref / h
siglow, errsiglow, sighig, errsighig = np.array( [siglow * h**2 / 1e6, errsiglow * h**2 / 1e6, sighig * h**2 / 1e6, errsighig * h**2 / 1e6] ) / a_ref**2

id_nan = np.isnan( bin_R )
bin_R = bin_R[ id_nan == False]
siglow, errsiglow, sighig, errsighig = siglow[ id_nan == False], errsiglow[ id_nan == False], sighig[ id_nan == False], errsighig[ id_nan == False]

lo_Ng_int_F = interp.interp1d( bin_R, siglow, kind = 'linear', fill_value = 'extrapolate')
hi_Ng_int_F = interp.interp1d( bin_R, sighig, kind = 'linear', fill_value = 'extrapolate')

## ... DM mass profile
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

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

band_str = 'gi'


## miscen params for high mass
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24]
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

### === subsample SM(r) decomposition
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'
# path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/SBs/'

# out_path = '/home/xkchen/tmp_run/data_files/figs/mass_pro_fit/'
# fit_path = '/home/xkchen/tmp_run/data_files/figs/mass_pro_fit/'

# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

"""
#... fitting params
for mm in range( 2 ):

	#... fitting for central part
	c_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_cen-deV_fit.csv' % (cat_lis[mm], band_str) )
	c_Ie, c_Re, c_ne = np.array( c_dat['Ie'] )[0], np.array( c_dat['Re'] )[0], np.array( c_dat['ne'] )[0]

	#... fitting for medille part
	tr_dat = pds.read_csv( fit_path + '%s_%s-band-based_xi2-sigma_mid-region_Lognorm-mcmc-fit.csv' % (cat_lis[mm], band_str),)
	Im_fit, Rm_fit, L_tr_fit = np.array( tr_dat['Ie'] )[0], np.array( tr_dat['R_pk'] )[0], np.array( tr_dat['L_trans'] )[0]

	out_lim_R = 250

	#... galaxy Number density scale
	file_name = out_path + '%s_%s-band-based_mass-profile_beyond-%dkpc_sigma-G_mcmc_fit.h5' % (cat_lis[0], band_str, out_lim_R)
	sampler = emcee.backends.HDFBackend( file_name )

	try:
		tau = sampler.get_autocorr_time()
		flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)

	mc_fits = []
	n_dim = flat_samples.shape[1]

	for oo in range( n_dim ):

		samp_arr = flat_samples[:, oo]
		mc_fit_oo = np.median( samp_arr )
		mc_fits.append( mc_fit_oo )
	lg_Ng = mc_fits[0]

	#... fitting for outer region
	pre_file = out_path + '%s_%s-band-based_xi2-sigma_beyond-%dkpc_fit_mcmc_fit.h5' % (cat_lis[0], band_str, out_lim_R)
	sampler = emcee.backends.HDFBackend( pre_file )

	try:
		tau = sampler.get_autocorr_time()
		flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain( discard = 3000, thin = 300, flat = True)

	mc_fits = []
	n_dim = flat_samples.shape[1]

	for oo in range( n_dim ):

		samp_arr = flat_samples[:, oo]
		mc_fit_oo = np.median( samp_arr )
		mc_fits.append( mc_fit_oo )

	lg_fb = mc_fits[0]

	keys = ['c_Ie', 'c_Re', 'c_ne', 'trans_Im', 'trans_Rp', 'trans_L', 'lg_DM_f', 'lg_Ng_f']
	values = [ c_Ie, c_Re, c_ne, Im_fit, Rm_fit, L_tr_fit, lg_fb, lg_Ng ]

	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( fit_path + '%s_%s-band-based_mass-profile_decompose-params.csv' % (cat_lis[mm], band_str),)
"""

"""
###... subsamples
fig = plt.figure( figsize = (10.20, 4.8) )
ax1 = fig.add_axes( [0.08, 0.12, 0.45, 0.85] )
ax2 = fig.add_axes( [0.53, 0.12, 0.45, 0.85] )

for mm in range( 2 ):

	dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
	obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

	id_rx = obs_R >= 0.5
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

	lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

	#... decomposition parameters
	p_dat = pds.read_csv( fit_path + '%s_%s-band-based_mass-profile_decompose-params.csv' % (cat_lis[mm], band_str),)
	c_Ie, c_Re, c_ne = np.array( p_dat['c_Ie'] )[0], np.array( p_dat['c_Re'] )[0], np.array( p_dat['c_ne'] )[0]
	tr_I, tr_R, tr_L = np.array( p_dat['trans_Im'] )[0], np.array( p_dat['trans_Rp'] )[0], np.array( p_dat['trans_L'] )[0]
	lg_fb, lg_Ng = np.array( p_dat['lg_DM_f'] )[0], np.array( p_dat['lg_Ng_f'] )[0]

	const = 10**(-lg_fb)

	new_R = np.logspace(0, 3.3, 250)
	#... projected DM profile
	# misNFW_sigma = obs_sigma_func( new_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m )
	# misNFW_sigma = misNFW_sigma * h # in unit of M_sun / kpc^2
	# sigma_2Mpc = obs_sigma_func( 2e3 * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h
	# lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	## .. use xi_hm for sigma estimation
	if mm == 0:
		misNFW_sigma = lo_interp_F( new_R )
		sigma_2Mpc = lo_xi2M_2Mpc + 0.
		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	if mm == 1:
		misNFW_sigma = hi_interp_F( new_R )
		sigma_2Mpc = hi_xi2M_2Mpc + 0.
		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	#... satellite number density
	if mm == 0:
		bin_R, sgN, sgN_err = bin_R, siglow, errsiglow
		sgN_2Mpc = lo_Ng_int_F( 2e3 )
	if mm == 1:
		bin_R, sgN, sgN_err = bin_R, sighig, errsighig
		sgN_2Mpc = hi_Ng_int_F( 2e3 )

	if mm == 0:
		ax = ax1
	if mm == 1:
		ax = ax2

	ax.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = mark_s[mm], color = line_c[mm], alpha = 0.80, label = '$\\Sigma_{\\ast}^{tot}$',)

	# ax.plot( obs_R / 1e3, surf_M, ls = '-', color = line_c[mm], alpha = 0.80, label = '$\\Sigma_{\\ast}^{tot}$',)
	# ax.fill_between( obs_R / 1e3, y1 = surf_M - surf_M_err, y2 = surf_M + surf_M_err, color = line_c[mm], alpha = 0.15,)

	ax.plot( bin_R / 1e3, (sgN - sgN_2Mpc) * 10**(lg_Ng), ls = '--', color = 'orange', alpha = 0.5, 
		label = '$ 10^{\\mathrm{%.0f}}\, M_{\\odot} \\times (\\mathrm{N}_{g} - \\mathrm{N}_{g}^{2Mpc}) $' % lg_Ng,)
	# ax.fill_between( bin_R / 1e3, y1 = (sgN - sgN_2Mpc - sgN_err) * 10**(lg_Ng), y2 = (sgN - sgN_2Mpc + sgN_err) * 10**(lg_Ng), 
	# 	color = 'orange', alpha = 0.15,)

	ax.plot( new_R / 1e3, 10**lg_M_sigma * 10**lg_fb, ls = '-', color = 'k', alpha = 0.5,
		label = '$ (\\Sigma_{dm} - \\Sigma_{dm}^{2Mpc}) / \\mathrm{%.0f} $' % const,)


	ax.plot( bin_R / 1e3, sgN * 10**(lg_Ng), ls = '--', color ='orange', alpha = 0.5, linewidth = 3, 
		label = '$ 10^{\\mathrm{%.0f}}\, M_{\\odot} \\times \\mathrm{N}_{g} $' % lg_Ng,)
	ax.fill_between( bin_R / 1e3, y1 = (sgN - sgN_err) * 10**(lg_Ng), y2 = (sgN + sgN_err) * 10**(lg_Ng), color = 'orange', alpha = 0.15,)

	if mm == 0:
		ax.plot( lo_rp / 1e3, lo_rho_m * 10**lg_fb, ls = '-', color = 'k', alpha = 0.5, linewidth = 3, label = '$ \\Sigma_{dm} / \\mathrm{%.0f} $' % const,)
	if mm == 1:
		ax.plot( hi_rp / 1e3, hi_rho_m * 10**lg_fb, ls = '-', color = 'k', alpha = 0.5, linewidth = 3, label = '$ \\Sigma_{dm} / \\mathrm{%.0f} $' % const,)

	handles,labels = ax.get_legend_handles_labels()
	handles = [ handles[4], handles[3], handles[2], handles[1], handles[0] ]
	labels = [ labels[4], labels[3], labels[2], labels[1], labels[0] ]
	ax.legend( handles, labels, loc = 3, frameon = False, fontsize = 14, )#markerfirst = False,)
	# ax.legend(loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

ax1.annotate( text = fig_name[0], xy = (0.65, 0.85), xycoords = 'axes fraction', color = line_c[0], fontsize = 15,)
ax2.annotate( text = fig_name[1], xy = (0.65, 0.85), xycoords = 'axes fraction', color = line_c[1], fontsize = 15,)

ax1.set_ylim( 4 * 10**3, 4 * 10**8 )

ax1.set_yscale('log')
ax1.set_ylabel( '$\\Sigma_{\\ast} \, [M_{\\odot} \, / \, kpc^2] $', fontsize = 15)

ax1.set_xlim( 9e-3, 10 )
ax1.set_xscale( 'log' )
ax1.set_xlabel( 'R [Mpc]', fontsize = 15,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

ax2.set_ylim( ax1.get_ylim() )
ax2.set_yscale( 'log' )
ax2.set_yticklabels( [] )

ax2.set_xlim( 9e-3, 10)
ax2.set_xscale( 'log' )
ax2.set_xlabel( 'R [Mpc]', fontsize = 15,)

ax2.set_xticks( [1e-2, 1e-1, 1e0, 1e1] )
ax2.set_xticklabels( labels = ['', '$\\mathrm{10}^{\\mathrm{-1}}$', '$\\mathrm{10}^{\\mathrm{0}}$', '$\\mathrm{10}^{\\mathrm{1}}$'],)

ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
# plt.savefig('/home/xkchen/mass-bin_%s-band_based_DM_sigma-G_ICL_compare.png' % band_str, dpi = 300)
plt.savefig('/home/xkchen/mass-bin_%s-band_based_DM_sigma-G_ICL_compare.pdf' % band_str, dpi = 300)
plt.close()
"""

### === total sample
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/'

BG_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'
fit_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'
band_str = 'gri'

# dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % band_str )
# obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
obs_R, surf_M, surf_M_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


#... central part
p_dat = pds.read_csv( fit_path + 'total-sample_%s-band-based_mass-profile_cen-deV_fit.csv' % band_str )
c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

new_R = np.logspace( 0, np.log10(2.5e3), 100)

cen_2Mpc = sersic_func( 2e3, 10**c_Ie, c_Re, c_ne)
fit_cen_M = sersic_func( new_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc

#...
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

misNFW_sigma = xi_to_Mf( lo_rp )
sigma_2Mpc = xi_to_Mf( 2e3 )
lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

#...
sig_aveg = (siglow + sighig) / 2
err_aveg = np.sqrt( errsiglow**2 / 4 + errsighig**2 / 4)

sig_rho_f = interp.interp1d( bin_R, sig_aveg, kind = 'linear', fill_value = 'extrapolate',)

Ng_sigma = sig_rho_f( bin_R )
Ng_2Mpc = sig_rho_f( 2e3 )
lg_Ng_sigma = np.log10( Ng_sigma - Ng_2Mpc )

out_lim_R = 400 # 250, 300, 350, 400

c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % out_lim_R,)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

c_dat = pds.read_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_SG_N-fit.csv' % out_lim_R,)
lg_Ng_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_Ng_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_Ng_ri = np.array( c_dat['lg_fb_ri'] )[0]

const = 10**(-1 * lg_fb_gi)

#... trans part
devi_M = surf_M - ( xi_to_Mf( obs_R) - sigma_2Mpc ) * 10**lg_fb_gi - ( sersic_func( obs_R, 10**c_Ie, c_Re, c_ne) - cen_2Mpc )
devi_err = surf_M_err

idx_lim = devi_M >= 10**4.6
devi_R = obs_R[ idx_lim ]
devi_M = devi_M[ idx_lim ]
devi_err = devi_err[ idx_lim ]

fit_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**lg_fb_gi
fit_sum = fit_out_M + fit_cen_M


fig_tx = plt.figure( figsize = (5.4, 5.4) )
ax1 = fig_tx.add_axes( [0.15, 0.11, 0.83, 0.83] )

ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
	label = '$\\Sigma_{\\ast}^{ \\mathrm{BCG}{+}\\mathrm{ICL} } $', capsize = 3, )

# ax1.plot( bin_R / 1e3, (Ng_sigma - Ng_2Mpc) * 10**lg_Ng_gi, ls = '--', color = 'k', alpha = 0.65, 
# 	label = '$ 10^{\\mathrm{%.0f}}\, M_{\\odot} \\times \\mathrm{N}_{g} $' % lg_Ng_gi,)
ax1.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
	label = '$ \\Sigma_{m} / {%.0f} $' % const,)

ax1.plot( new_R / 1e3, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{ \\mathrm{BCG} }$')
ax1.plot( new_R / 1e3, fit_sum, ls = '--', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
	label = '$\\Sigma_{\\ast}^{ \\mathrm{ \\mathrm{BCG} } } {+} \\Sigma_{m} / {%.0f} $' % const)

ax1.errorbar( devi_R / 1e3, devi_M, yerr = devi_err, ls = 'none', marker = 'o', ms = 8, mec = 'k', mfc = 'none', 
	ecolor = 'k', alpha = 0.75, capsize = 3,
	label = '$\\Sigma_{\\ast}^{tran} \, = \, \\Sigma_{\\ast}^{ \\mathrm{BCG}{+}\\mathrm{ICL} } {-} $' + 
			'$(\\Sigma_{\\ast}^{ \\mathrm{BCG} } {+} \\Sigma_{m} / \\mathrm{%.0f} )$' % const)

handles,labels = ax1.get_legend_handles_labels()

handles = [ handles[3], handles[4], handles[0], handles[2], handles[1] ]
labels = [ labels[3], labels[4], labels[0], labels[2], labels[1] ]
ax1.legend( handles, labels, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)

# ax1.legend( loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

ax1.set_ylim( 1e4, 3e8 )
ax1.set_yscale('log')
ax1.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

ax1.set_xlim( 9e-3, 2e0 )
ax1.set_xscale( 'log' )
ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)

ax1.set_xticks([ 1e-2, 1e-1, 1e0])
ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

ax1.set_xticks([ 2e0 ], minor = True,)
ax1.set_xticklabels( labels = ['$\\mathrm{2}$'], minor = True,)

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/DM_Ng_compare.pdf', dpi = 300)
plt.savefig('/home/xkchen/DM_Ng_compare.png', dpi = 300)
plt.close()


# fig_tx = plt.figure( figsize = (10.20, 4.8) )
# ax = fig_tx.add_axes( [0.08, 0.12, 0.41, 0.85] )
# ax1 = fig_tx.add_axes( [0.57, 0.12, 0.41, 0.85] )

# ax1.errorbar( obs_R / 1e3, surf_M, yerr = surf_M_err, ls = 'none', marker = 'o', ms = 8, color = 'k', alpha = 0.65, 
# 	label = '$\\Sigma_{\\ast}^{BCG{+}ICL} $', capsize = 3, )

# ax1.plot( bin_R / 1e3, (Ng_sigma - Ng_2Mpc) * 10**lg_Ng_gi, ls = '--', color = 'k', alpha = 0.65, 
# 	label = '$ 10^{\\mathrm{%.0f}}\, M_{\\odot} \\times \\mathrm{N}_{g} $' % lg_Ng_gi,)
# ax1.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65,
# 	label = '$ \\Sigma_{m} / {%.0f} $' % const,)

# ax1.plot( new_R / 1e3, fit_cen_M, ls = ':', color = 'k', label = '$\\Sigma_{\\ast}^{deV}$')
# ax1.plot( new_R / 1e3, fit_sum, ls = '-.', color = 'Gray', alpha = 0.95, linewidth = 3.0, 
# 	label = '$\\Sigma_{\\ast}^{deV} {+} \\Sigma_{m} / {%.0f} $' % const)

# ax1.errorbar( devi_R / 1e3, devi_M, yerr = devi_err, ls = 'none', marker = 'o', ms = 8, mec = 'k', mfc = 'none', 
# 	ecolor = 'k', alpha = 0.75, capsize = 3,
# 	label = '$\\Sigma_{\\ast}^{tran} \, = \, \\Sigma_{\\ast}^{BCG{+}ICL} {-} (\\Sigma_{\\ast}^{deV} {+} \\Sigma_{m} / \\mathrm{%.0f} )$' % const)

# handles,labels = ax1.get_legend_handles_labels()
# handles = [ handles[5], handles[3], handles[0], handles[1], handles[4], handles[2] ]
# labels = [ labels[5], labels[3], labels[0], labels[1], labels[4], labels[2] ]
# ax1.legend( handles, labels, loc = 1, frameon = False, fontsize = 13, markerfirst = False,)
# # ax1.legend( loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

# ax1.set_ylim( 1e4, 3e8 )
# ax1.set_yscale('log')
# ax1.set_ylabel('$\\Sigma_{\\ast} \, - \, \\Sigma_{\\ast, \, \\mathrm{2 \,M}pc } \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

# ax1.set_xlim( 9e-3, 2e0 )
# ax1.set_xscale( 'log' )
# ax1.set_xlabel( '$R \; [\\mathrm{M}pc] $', fontsize = 15,)

# ax1.set_xticks([ 1e-2, 1e-1, 1e0])
# ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

# ax1.set_xticks([ 2e0 ], minor = True,)
# ax1.set_xticklabels( labels = ['$\\mathrm{2}$'], minor = True,)

# ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


# aveg_lgM = 10.40 # M_sun, mean of satellite galaxy stellar mass (lim_mag = 22.08 mag in i-band)

# f_DM_Ng = 210
# scaled_Ng = f_DM_Ng * 10**aveg_lgM * Ng_sigma
# scaled_Ng_err = f_DM_Ng * 10**aveg_lgM * err_aveg

# ax.errorbar( bin_R / 1e3, scaled_Ng, yerr = scaled_Ng_err, ls = 'none', alpha = 0.80, marker = 's', mec = 'k', mfc = 'none', 
# 	ms = 8, capsize = 2.5, barsabove = True, ecolor = 'k',
# 	label = '$ {\\mathrm{%d} } {\\times} \\langle M_{\\ast}^{Sat} \\rangle \\times \\mathrm{N}_{g} $' % f_DM_Ng,)

# ax.plot( hi_rp / 1e3, misNFW_sigma, ls = '-', color = 'k', alpha = 0.45, linewidth = 4, 
# 	label = '$\\mathrm{Total \; mass} \, (lensing)$',)

# # ax.legend( loc = 3, frameon = False, fontsize = 14,)
# ax.legend( loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

# ax.set_ylim( 2e6, 1.5e9)
# ax.set_yscale('log')
# ax.set_ylabel( '$\\Sigma \; [M_{\\odot} \, / \, \\mathrm{k}pc^2] $', fontsize = 15)

# ax.set_xlim( 9e-3, 10 )
# ax.set_xscale( 'log' )
# ax.set_xlabel( '$R \; [\\mathrm{M}pc]$', fontsize = 15,)

# ax.set_xticks( [1e-2, 1e-1, 1e0, 1e1] )
# ax.set_xticklabels( labels = ['$\\mathrm{0.01}$', '$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{10}$'] )
# ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/DM_Ng_compare.pdf', dpi = 300)
# # plt.savefig('/home/xkchen/DM_Ng_compare_%d-kpc-outer.png' % out_lim_R, dpi = 300)
# plt.close()
