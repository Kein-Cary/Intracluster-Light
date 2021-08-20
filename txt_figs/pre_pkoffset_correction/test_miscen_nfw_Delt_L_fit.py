import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import emcee
import corner

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param

# from multiprocessing import Pool
from schwimmbad import MPIPool
import sys

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

# constant
M_sun = C.M_sun.value # in unit of kg
kpc2m = U.kpc.to(U.m)
Mpc2cm = U.Mpc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)

rad2arcsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7 # (erg/s/cm^2)
Jy = 10**(-23) # (erg/s)/cm^2/Hz
F0 = 3.631 * 10**(-6) * Jy
L_speed = C.c.value # m/s

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
## solar Magnitude corresponding to SDSS filter
Mag_sun = [ 4.65, 5.11, 4.53 ]

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### miscentering nfw profile (Zu et al. 2020, section 3.)
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""
	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

	theta = np.linspace( 0, 2 * np.pi, 100)
	d_theta = np.diff( theta )

	try:
		NR = len( rp )
	except:
		rp = np.array( [rp] )
		NR = len( rp )

	r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
	off_pdf = mis_p_func( r_off, sigma_off )

	NR_off = len( r_off )

	surf_dens_off = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		surf_dens_arr = np.zeros( NR_off, dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )

			surf_dens_of_theta = sigmam( r_cir, lgM, z, c_mass )

			## integration on theta
			surf_dens_arr[jj] = integ.simps( surf_dens_of_theta, theta) / ( 2 * np.pi )

		## integration on r_off
		integ_f = surf_dens_arr * off_pdf

		surf_dens_ii = integ.simps( integ_f, r_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	if NR == 1:
		return off_sigma[0]
	return off_sigma

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

def aveg_sigma_func(rp, sigma_arr, N_grid = 100):

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

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

	# off_sigma = off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	off_sigma = cc_off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)

	norm_sigma = sigmam( rp, lgM, z, c_mass)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

def delta_sigma_func(rp, f_off, sigma_off, z, c_mass, lgM, v_m, N_grid = 100):

	sigma_arr = obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m )
	aveg_sigma = aveg_sigma_func( rp, sigma_arr, N_grid = N_grid)
	delta_sigma = aveg_sigma - sigma_off

	return delta_sigma

### === ### SB profile to delta SB profile
def SB_to_Lumi_func(sb_arr, obs_z, band_str):
	"""
	sb_arr : need in terms of absolute magnitude, in AB system
	"""
	if band_str == 'r':
		Mag_dot = Mag_sun[0]

	if band_str == 'g':
		Mag_dot = Mag_sun[1]

	if band_str == 'i':
		Mag_dot = Mag_sun[2]

	# luminosity, in unit of  L_sun / pc^2
	lumi = 10**( -0.4 * (sb_arr - Mag_dot + 21.572 - 10 * np.log10( obs_z + 1 ) ) )

	Lumi = lumi * 1e6 # in unit of  L_sun / kpc^2

	return Lumi

def aveg_lumi_func(rp, lumi_arr, N_grid = 100):

	NR = len( rp )

	aveg_lumi = np.zeros( NR, dtype = np.float32 )

	intep_lumi_F = interp.interp1d( rp, lumi_arr, kind = 'linear', fill_value = 'extrapolate',)

	for ii in range( NR ):

		new_rp = np.logspace(-3, np.log10( rp[ii] ), N_grid)
		new_lumi = intep_lumi_F( new_rp )

		cumu_lumi = integ.simps( new_rp * new_lumi, new_rp)

		aveg_lumi[ii] = 2 * cumu_lumi / rp[ii]**2

	return aveg_lumi

def delta_SB_func( rp, lumi_arr, N_grid = 100):

	aveg_lumi = aveg_lumi_func(rp, lumi_arr, N_grid = N_grid)
	delta_lumi = aveg_lumi - lumi_arr

	return delta_lumi

### === ### SB profile
def sersic_func(r, Ie, re, ndex):

	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )

	return Ir

### === ### fitting function
def prior_p_func( p ):

	M0, c_mass, m2l, f_off, sigma_off, Ie, Re, n = p[:]

	identi_0 = ( 1e-2 < Ie < 1e1 ) & ( 5 < Re < 50) & ( 1 < n < 9 ) & (1 < c_mass < 50) & (2e2 < m2l < 5e3) 
	identi_1 = (13.5 <= M0 <= 15) & ( 0 < f_off < 1) & ( 10 < sigma_off < 400)

	if ( identi_0 & identi_1 ):
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )

	if not np.isfinite( pre_p ):
		return -np.inf
	else:
		M0, c_mass, m2l, f_off, sigma_off, Ie, Re, n = p[:]
		z0, cov_mx, v_m = params[:]

		## sersic
		I_r = sersic_func( x, Ie, Re, n )
		aveg_I = aveg_sigma_func( x, I_r )
		delta_I = aveg_I - I_r

		## miscen-nfw
		off_sigma = obs_sigma_func( x * h, f_off, sigma_off, z0, c_mass, M0, v_m) # unit M_sun * h / kpc^2
		mean_off_sigma = aveg_sigma_func( x * h, off_sigma )

		off_delta_sigma = mean_off_sigma - off_sigma
		off_D_sigma = off_delta_sigma * h * 1e-6 # unit M_sun / pc^2

		## model SB
		mode_mu = delta_I + off_D_sigma / m2l

		cov_inv = np.linalg.pinv( cov_mx )
		delta = mode_mu - y
		chi2 = -0.5 * delta.T.dot( cov_inv ).dot(delta)

		return pre_p + chi2

## use high mass bin for test
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$'] ## or line name

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
line_s = ['--', '-']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

### === ### MCMC fitting
Z0 = 0.25
a0 = 1 / (1 + Z0)
a_ref = 1 / (1 + z_ref)

v_m = 200 # rho_mean = 200 * rho_c * omega_m

path = '/home/xkchen/project/tmp_gri_joint_cat/'
out_path = '/home/xkchen/project/tmp_gri_joint_cat/'

pool = MPIPool()
if not pool.is_master():
	pool.wait()
	sys.exit(0)

for mm in range( 0,1 ):

	for kk in range( 0,1 ):

		d_dat = pds.read_csv( path + '%s_%s-band_aveg-jack_Lumi-pros.csv' % (cat_lis[mm], band[kk]) )
		dd_R, dd_L, dd_L_err = np.array( d_dat['R'] ), np.array( d_dat['Lumi'] ), np.array( d_dat['Lumi_err'] )
		dd_mL, dd_mL_err = np.array( d_dat['m_Lumi'] ), np.array( d_dat['m_Lumi_err'] )
		dd_DL, dd_DL_err = np.array( d_dat['d_Lumi'] ), np.array( d_dat['d_Lumi_err'] )

		## use delta_lumi for fitting, in unit L_sun / pc^2
		_dd_R = dd_R
		_DL = dd_DL * 1e-6
		_DL_err = dd_DL_err * 1e-6

		com_r = dd_R
		com_sb = dd_DL * 1e-6
		com_err = dd_DL_err * 1e-6

		## cov_arr
		with h5py.File( path + '%s_%s-band_Delta-Lumi-pros_cov-cor.h5' % (cat_lis[mm], band[kk]), 'r') as f:
			cov_MX = np.array(f['cov_Mx'])

		put_params = [ Z0, cov_MX, v_m ]

		n_walk = 70

		# identi_0 = ( 1e-2 < Ie < 1e1 ) & ( 5 < Re < 50) & ( 1 < n < 9 ) & (1 < c_mass < 50) & (2e2 < m2l < 5e3) 
		# identi_1 = (13.5 <= M0 <= 15) & ( 0 < f_off < 1) & ( 10 < sigma_off < 400)

		put_x0 = np.random.uniform( 13.5, 15, n_walk ) # Mass
		put_x1 = np.random.uniform( 1, 50, n_walk ) # c
		put_x2 = np.random.uniform( 2e2, 5e3, n_walk ) # m2l
		put_x3 = np.random.uniform( 0, 1, n_walk ) # f_off
		put_x4 = np.random.uniform( 10, 400, n_walk ) # sigma_off
		put_x5 = np.random.uniform( 1e-2, 1e1, n_walk ) # I_e
		put_x6 = np.random.uniform( 5, 5e1, n_walk ) # R_e
		put_x7 = np.random.uniform( 1, 9, n_walk ) # n

		param_labels = ['$Mh_{ M_{\\odot} / h }$', 'c', 'm2l', '$f_{off}$', '$\\sigma_{off}$', '$I_{e}$', '$R_{e}$', '$n$']

		pos = np.array([ put_x0, put_x1, put_x2, put_x3, put_x4, put_x5, put_x6, put_x7]).T

		n_dim = pos.shape[1]
		L_chains = 5e3

		file_name = out_path + '%s_%s-band_sersic+1h-miscen_Delta-L_mcmc_fit.h5' % (cat_lis[mm], band[kk])
		backend = emcee.backends.HDFBackend( file_name )
		backend.reset( n_walk, n_dim )

		# with Pool( 72 ) as pool:
		# 	sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = ( com_r, com_sb, put_params, com_err ), pool = pool, backend = backend)
		# 	sampler.run_mcmc(pos, L_chains, progress = True,)

		sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = ( com_r, com_sb, put_params, com_err ), pool = pool, backend = backend)
		sampler.run_mcmc(pos, L_chains, progress = True,)

		# sampler = emcee.backends.HDFBackend( file_name )
		try:
			tau = sampler.get_autocorr_time()
			print( tau )
			flat_samples = sampler.get_chain(discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
		except:
			flat_samples = sampler.get_chain(discard = 500, thin = 50, flat = True)

		fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
			levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
			plot_datapoints = True, plot_density = False, fill_contours = True,)

		mc_fits = []

		for oo in range( n_dim ):
			samp_arr = flat_samples[:, oo]

			mc_fit_oo = np.median( samp_arr )

			mc_fits.append( mc_fit_oo )

		axes = np.array( fig.axes ).reshape( (n_dim, n_dim) )

		for jj in range( n_dim ):
			ax = axes[jj, jj]
			ax.axvline( mc_fits[jj], color = 'r', ls = '-', alpha = 0.5,)

		for yi in range( n_dim ):
			for xi in range( yi ):
				ax = axes[yi, xi]

				ax.axvline( mc_fits[xi], color = 'r', ls = '-', alpha = 0.5,)
				ax.axhline( mc_fits[yi], color = 'r', ls = '-', alpha = 0.5,)

				ax.plot( mc_fits[xi], mc_fits[yi], 'ro', alpha = 0.5,)

		ax = axes[0, n_dim - 2 ]
		ax.set_title( fig_name[ mm ] + ', %s band' % band[kk] )
		plt.savefig('/home/xkchen/figs/%s_%s-band_sersic+1h-miscen_Delta-L_fit_params.png' % (cat_lis[mm], band[kk]), dpi = 300)
		plt.close()


		## save the fitting params
		keys = [ 'Mh_Msun2h', 'C_mass', 'M2L', 'f_off', 'sigma_off', 'Ie_0', 'Re_0', 'ne_0' ]

		values = []
		for oo in range( n_dim ):
			values.append( mc_fits[ oo ] )

		# values = [ mc_fits[0], mc_fits[1], mc_fits[2], mc_fits[3], mc_fits[4], mc_fits[5], mc_fits[6], mc_fits[7] ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill, index = ['k', 'v'])
		out_data.to_csv( out_path + '%s_%s-band_sersic+1h-miscen_Delta-L_fit_params.csv' % (cat_lis[mm], band[kk]),)

		## sersic
		I_r = sersic_func( com_r, mc_fits[5], mc_fits[6], mc_fits[7])
		aveg_I = aveg_sigma_func( com_r, I_r )
		delta_I = aveg_I - I_r

		## miscen-nfw
		off_sigma = obs_sigma_func( com_r * h, mc_fits[3], mc_fits[4], z_ref, mc_fits[1], mc_fits[0], v_m) # unit M_sun * h / kpc^2
		mean_off_sigma = aveg_sigma_func( com_r * h, off_sigma )
		off_delta_sigma = mean_off_sigma - off_sigma
		off_D_sigma = off_delta_sigma * h * 1e-6 # unit M_sun / pc^2

		## model SB
		mode_mu = delta_I + off_D_sigma / mc_fits[2]
		mu_2Mpc = I_2Mpc + m_2Mpc / mc_fits[2]

		cov_inv = np.linalg.pinv( cov_MX )
		delta = mode_mu - com_sb
		chi2 = delta.T.dot( cov_inv ).dot(delta)
		chi2nu = chi2 / ( len(com_sb) - n_dim )


		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[mm] + ',%s band' % band[kk])

		ax.plot( com_r, com_sb, ls = '-', color = 'k', alpha = 0.45, label = 'signal',)
		ax.fill_between( com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12,)

		ax.plot( com_r, mode_mu, ls = '-', color = 'b', alpha = 0.5, label = 'fit',)
		ax.plot( com_r, off_D_sigma / mc_fits[2], ls = '--', color = 'b', alpha = 0.5, label = '$ NFW_{mis} $',)
		ax.plot( com_r, delta_I, ls = '-.', color = 'b', alpha = 0.5, )

		ax.text(2e1, 5e-3, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k', alpha = 0.5,)
		ax.text(1e2, 1e0, s = '$c = %.2f, M/L = %.2f$' % ( mc_fits[1], mc_fits[2] ), color = 'k', alpha = 0.5,)
		ax.text(2e1, 1e-2, s = '$I_{e} = %.3f$' % mc_fits[5] + '\n' + '$R_{e} = %.3f$' % mc_fits[6] + '\n' + '$n=%.3f$' % mc_fits[7], color = 'b',)

		ax.set_xlim(1e1, 1e3)
		ax.set_ylim(1e-3, 1e1)
		ax.set_yscale('log')

		ax.legend( loc = 1)
		ax.set_xscale('log')
		ax.set_xlabel('R[kpc]')
		ax.set_ylabel('$ \\Delta\\mu $ $ [L_{\\odot} / pc^{2}] $')
		ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

		plt.savefig('/home/xkchen/figs/%s_%s-band_sersic+1h-miscen_Delta-L_mcmc_fit.png' % (cat_lis[mm], band[kk]), dpi = 300)
		plt.close()

pool.close()

