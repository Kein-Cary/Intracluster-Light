import matplotlib as mpl
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

from surface_mass_density import sigma_m, sigmam
from multiprocessing import Pool

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2cm = U.Mpc.to(U.cm)

rad2arcsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7 # (erg/s/cm^2)
Jy = 10**(-23) # (erg/s)/cm^2/Hz
F0 = 3.631 * 10**(-6) * Jy
L_speed = C.c.value # m/s

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def mode_flux_densi(rho_2d, m2l, z, wave_l_eff):

 	# h is factor for change unit M_sun * h / kpc^2 to M_sun / kpc^2
	m0 = h * Lsun / (np.pi * 4 * m2l)
	m1 = rho_2d / (z + 1)**4

	f_dens = m0 * m1 * Mpc2cm**(-2) * rad2arcsec**(-2)

	# erg/s cm^-2 Hz^-1 / arcsec^2 (use effective wave_length for estimation)
	fdens = f_dens * 10**6 * wave_l_eff / (L_speed * 10**10)

	f_out = fdens / F0 # scale with SDSS ZP
	return f_out

## NFW model case
def out_NFW_flux_densi( x, M0, z0, c_mass, m2l, wave_length):
	# radius is in unit kpc / h
	# M_sigma is in unit of M_sun * h / kpc^2

	#M_sigma = sigmam( x, M0, 0.0, c_mass) # projected NFW ( at z = 0 )

	M_sigma = sigmam( x, M0, z0, c_mass) # projected NFW ( at z = z0 )
	mock_f = mode_flux_densi( M_sigma, m2l, z0, wave_length ) # shifted to z = z0
	out_fdens = mock_f - 0
	return out_fdens

def err_combine_NFW_fit(p, x, y, params, yerr):

	Ie, Re, n, c_mass, m2l = p[:]
	M0, z0, wave_length, cov_mx = params[:]
	## a_0 is the scale factor
	a_0 = 1 / ( 1 + z0 )

	cen_fdens = sersic_func( x, Ie, Re, n)
	out_fdens = out_NFW_flux_densi( x * h / a_0, M0, z0, c_mass, m2l, wave_length)

	sb_2Mpc = sersic_func( 2e3, Ie, Re, n) + out_NFW_flux_densi( 2e3 * h / a_0, M0, z0, c_mass, m2l, wave_length)

	mode_sb = cen_fdens + out_fdens - sb_2Mpc

	cov_inv = np.linalg.pinv( cov_mx )
	delta = mode_sb - y

	chi2 = delta.T.dot( cov_inv ).dot(delta)
	# chi2 = np.sum( delta**2 / yerr**2 )

	return chi2

def prior_p_func( p ):

	Ie, Re, n, c_mass, m2l = p

	if ( 0. < Ie < 5e0 ) & ( 5 < Re < 50) & ( 1 < n < 10 ) & (1 < c_mass < 50) & (2e2 < m2l < 5e3):
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )

	if not np.isfinite( pre_p ):
		return -np.inf
	else:
		Ie, Re, n, c_mass, m2l = p[:]
		M0, z0, wave_length, cov_mx = params[:]
		a_0 = 1 / ( 1 + z0 )

		cen_fdens = sersic_func( x, Ie, Re, n)
		out_fdens = out_NFW_flux_densi( x * h / a_0, M0, z0, c_mass, m2l, wave_length)

		sb_2Mpc = sersic_func( 2e3, Ie, Re, n) + out_NFW_flux_densi( 2e3 * h / a_0, M0, z0, c_mass, m2l, wave_length)

		mode_sb = cen_fdens + out_fdens  - sb_2Mpc

		cov_inv = np.linalg.pinv( cov_mx )
		delta = mode_sb - y
		chi2 = -0.5 * delta.T.dot( cov_inv ).dot(delta)
		return pre_p + chi2

## fixed the outer region case
def cc_err_fit_func(p, x, y, params, yerr):
	Ie, Re, n = p[:]
	M0, z0, c_mass, m2l, wave_length, cov_mx = params[:]
	a_0 = 1 / ( 1 + z0 )

	cen_fdens = sersic_func( x, Ie, Re, n)
	out_fdens = out_NFW_flux_densi( x * h / a_0, M0, z0, c_mass, m2l, wave_length)

	sb_2Mpc = sersic_func( 2e3, Ie, Re, n) + out_NFW_flux_densi( 2e3 * h / a_0, M0, z0, c_mass, m2l, wave_length)

	mode_sb = cen_fdens + out_fdens  - sb_2Mpc

	cov_inv = np.linalg.pinv( cov_mx )
	delta = mode_sb - y

	chi2 = delta.T.dot( cov_inv ).dot(delta)
	# chi2 = np.sum( delta**2 / yerr**2 )

	return chi2

def likelihood_fix1_func(p, x, y, params, yerr):

	Ie, Re, n = p[:]
	M0, z0, c_mass, m2l, wave_length, cov_mx = params[:]
	a_0 = 1 / ( 1 + z0 )

	cen_fdens = sersic_func( x, Ie, Re, n)
	out_fdens = out_NFW_flux_densi( x * h / a_0, M0, z0, c_mass, m2l, wave_length)

	sb_2Mpc = sersic_func( 2e3, Ie, Re, n) + out_NFW_flux_densi( 2e3 * h / a_0, M0, z0, c_mass, m2l, wave_length)

	mode_sb = cen_fdens + out_fdens  - sb_2Mpc

	cov_inv = np.linalg.pinv( cov_mx )
	delta = mode_sb - y

	return -0.5 * delta.T.dot( cov_inv ).dot(delta)

def prior_p_fix1_func( p ):

	Ie, Re, n = p
	if ( 0. < Ie < 5e0 ) & ( 5 < Re < 50) & ( 1 < n < 10 ):
		return 0
	return -np.inf

def ln_p_fix1_func(p, x, y, params, yerr):

	pre_p = prior_p_fix1_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_fix1_func(p, x, y, params, yerr)

## load data
# path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/BG_estimate/'
# out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_fit_mcmc/'

path = '/home/xkchen/project/tmp_SB_fit/'
out_path = '/home/xkchen/project/tmp_SB_fit/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass', 'tot-BCG-star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$'] ## or line name
color_s = ['r', 'g', 'b']

# lower mass sample SB profiles
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []
for kk in range( 3 ):

	#with h5py.File( path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
	with h5py.File( path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_low_r.append( tt_r )
	nbg_low_sb.append( tt_sb )
	nbg_low_err.append( tt_err )

# higher mass sample SB profiles
nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []
for kk in range( 3 ):

	#with h5py.File( path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
	with h5py.File( path + 'photo-z_%s_%s-band_diag-fit-BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	nbg_hi_r.append( tt_r )
	nbg_hi_sb.append( tt_sb )
	nbg_hi_err.append( tt_err )

### model fitting
Mh0 = 14.21 # unit M_sun / h

Z0 = 0.25
z_ref = 0.25

a0 = 1 / (1 + Z0)
a_ref = 1 / (1 + z_ref)


### === low mass bin
mass_dex = 0

for kk in range( 3 ):

	idmx = nbg_low_r[kk] >= 10
	com_r, com_sb, com_err = nbg_low_r[kk][idmx], nbg_low_sb[kk][idmx], nbg_low_err[kk][idmx]

	## cov_mx
	#with h5py.File( path + '%s_%s-band_BG-sub_cov-cor_arr.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
	with h5py.File( path + '%s_%s-band_BG-sub_cov-1Mpc-h_cov-cor_arr.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
		cov_MX = np.array(f['cov_MX'])

	params = np.array([ Mh0, Z0, L_wave[kk], cov_MX ])

	#idx1 = com_r <= 1e3
	dtR = com_r * h / a_ref
	idx1 = (dtR >= 10) & (dtR <= 1e3)

	fx = com_r[idx1]
	fy = com_sb[idx1]
	ferr = com_err[idx1]

	mu_e1, re1, ndex1 = 5e-1, 50, 4.0
	po = [ mu_e1, re1, ndex1, 3.0, 5e2, ]

	bonds = [ [0, 1e1], [5, 50], [1, 10], [1, 50], [2e2, 5e3],]
	E_return = optimize.minimize( err_combine_NFW_fit, x0 = np.array( po ), args = (fx, fy, params, ferr), method = 'L-BFGS-B', bounds = bonds,)

	print( E_return )

	popt = E_return.x
	Ie_fit, Re_fit, ne_fit, C_fit, m2l_fit = popt

	print(popt)

	put_params = [ Mh0, Z0, L_wave[kk], cov_MX]

	n_walk = 120

	put_x0 = np.random.uniform( 0, 5e0, n_walk )
	put_x1 = np.random.uniform( 5, 50, n_walk )
	put_x2 = np.random.uniform( 1, 10, n_walk )
	put_x3 = np.random.uniform( 1, 50, n_walk )
	put_x4 = np.random.uniform( 2e2, 5e3, n_walk )

	param_labels = ['$I_{e}$', '$R_{e}$', '$n$', 'C', 'M/L $[M_{\\odot} / L_{\\odot}]$',]

	pos = np.array([ put_x0, put_x1, put_x2, put_x3, put_x4 ]).T
	n_dim = pos.shape[1]
	L_chains = 2e4

	#file_name = out_path + '%s_%s-band_sersic+NFW_fit_mcmc_arr.h5' % (cat_lis[mass_dex], band[kk])
	file_name = out_path + '%s_%s-band_sersic+NFW_fit_1Mpc-h_mcmc_arr.h5' % (cat_lis[mass_dex], band[kk])

	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 50 ) as pool:
		sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = (fx, fy, put_params, ferr), pool = pool, backend = backend)
		sampler.run_mcmc(pos, L_chains, progress = True,)

	#sampler = emcee.backends.HDFBackend( file_name )
	try:
		tau = sampler.get_autocorr_time()
		print( tau )
		flat_samples = sampler.get_chain(discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain(discard = 5000, thin = 100, flat = True)

	fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
		truths = [ Ie_fit, Re_fit, ne_fit, C_fit, m2l_fit ],
		levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
		plot_datapoints = True, plot_density = False, fill_contours = True,)

	fig.set_size_inches( 19.2, 14.4 )

	(Ie_arr, Re_arr, ne_arr, C_arr, m2l_arr) = (flat_samples[:,0], flat_samples[:,1], flat_samples[:,2],flat_samples[:,3], flat_samples[:,4])

	Ie_mc, Re_mc, ne_mc, C_mc, m2l_mc = np.median(Ie_arr), np.median(Re_arr), np.median(ne_arr), np.median(C_arr), np.median(m2l_arr)

	mc_fits = [Ie_mc, Re_mc, ne_mc, C_mc, m2l_mc ]

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

	g_xlims = [ [0, 0.6], [0, 20], [2, 10], [1, 50], [800, 3200] ]
	ri_xlims = [ [0.8, 2], [0, 16], [1, 8], [5, 25], [200, 1600] ]

	for xi in range( n_dim ):
		for yi in range( n_dim ):
			if kk == 1:
				ax = axes[yi, xi]
				ax.set_xlim( g_xlims[xi][0], g_xlims[xi][1] )
			else:
				ax = axes[yi, xi]
				ax.set_xlim( ri_xlims[xi][0], ri_xlims[xi][1] )

	for yi in range(1, n_dim):
		for xi in range( yi ):
			if kk == 1:
				ax = axes[yi, xi]
				ax.set_ylim( g_xlims[yi][0], g_xlims[yi][1] )
			else:
				ax = axes[yi, xi]
				ax.set_ylim( ri_xlims[yi][0], ri_xlims[yi][1] )

	ax = axes[1, 3]
	ax.set_title( fig_name[ mass_dex ] + ', %s band' % band[kk] )

	plt.subplots_adjust( hspace = 0.0, wspace = 0.0)
	plt.savefig('/home/xkchen/figs/%s_%s-ban_project-NFW_fit_params.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()


	## save the fitting params
	keys = ['Ie_0', 'Re_0', 'ne_0', 'C_mass', 'M2L']
	values = [ Ie_mc, Re_mc, ne_mc, C_mc, m2l_mc ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	#out_data.to_csv( out_path + 'low_BCG_star-Mass_%s-band_sersic+NFW_mcmc_fit.csv' % band[kk] )
	out_data.to_csv( out_path + 'low_BCG_star-Mass_%s-band_sersic+NFW_1Mpc-h_mcmc_fit.csv' % band[kk] )

	mock_cen_f = sersic_func( com_r, Ie_mc, Re_mc, ne_mc)
	mock_f_dens = out_NFW_flux_densi( com_r * h / a_ref, Mh0, Z0, C_mc, m2l_mc, L_wave[kk] )
	sb_2Mpc = sersic_func( 2e3, Ie_mc, Re_mc, ne_mc) + out_NFW_flux_densi( 2e3 * h / a_ref, Mh0, Z0, C_mc, m2l_mc, L_wave[kk] )
	sum_fit = mock_f_dens + mock_cen_f - sb_2Mpc

	M_sigma = sigmam( com_r * h / a_ref, Mh0, 0.0, C_mc )
	mock_f_nfw = mode_flux_densi( M_sigma, m2l_mc, Z0, L_wave[kk] )

	cov_inv = np.linalg.pinv( cov_MX )
	delta = sum_fit[idx1] - fy
	chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2nu = chi2 / ( len(fy) - n_dim )

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( fig_name[mass_dex] + ',%s band' % band[kk])

	ax.plot( com_r, com_sb, ls = '-', color = 'k', alpha = 0.45, label = 'signal',)
	ax.fill_between( com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12,)

	ax.plot( com_r, sum_fit, ls = '-', color = 'b', alpha = 0.5, label = 'fit',)
	ax.plot( com_r, mock_f_nfw, ls = '--', color = 'b', alpha = 0.5, label = 'projected NFW',)
	ax.plot( com_r, mock_cen_f, ls = '-.', color = 'b', alpha = 0.5, )

	#ax.plot( com_r, sersic_mode_f, ls = ':', color = 'r', alpha = 0.5, label = 'outer sersic',)
	ax.axhline( y = sb_2Mpc, ls = ':', color = 'b', alpha = 0.5, )

	ax.text(1e2, 1e-1, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k', alpha = 0.5,)
	ax.text(1e2, 3e-2, s = '$c = %.2f, M/L = %.2f$' % (C_mc, m2l_mc), color = 'k', alpha = 0.5,)
	ax.text(2e1, 1e-3, s = '$I_{e} = %.3f$' % Ie_mc + '\n' + '$R_{e} = %.3f$' % Re_mc + '\n' + '$n=%.3f$' % ne_mc, color = 'b',)

	ax.set_xlim(1e1, 2e3)
	ax.set_ylim(1e-5, 1e0)
	ax.set_yscale('log')

	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.savefig('/home/xkchen/figs/%s_%s-band_NFW_compare.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()

## === high mass bin
mass_dex = 1
for kk in range( 3 ):

	idmx = nbg_hi_r[kk] >= 10
	com_r, com_sb, com_err = nbg_hi_r[kk][idmx], nbg_hi_sb[kk][idmx], nbg_hi_err[kk][idmx]

	## cov_mx
	#with h5py.File( path + '%s_%s-band_BG-sub_cov-cor_arr.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
	with h5py.File( path + '%s_%s-band_BG-sub_cov-1Mpc-h_cov-cor_arr.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
		cov_MX = np.array(f['cov_MX'])

	## low mass params
	#p_dat = pds.read_csv( out_path + 'low_BCG_star-Mass_%s-band_sersic+NFW_mcmc_fit.csv' % band[ kk ] )
	p_dat = pds.read_csv( out_path + 'low_BCG_star-Mass_%s-band_sersic+NFW_1Mpc-h_mcmc_fit.csv' % band[kk] )

	p_c_mass, p_m2l = np.array( p_dat['C_mass'] )[0], np.array( p_dat['M2L'] )[0]

	#idx1 = com_r <= 1e3
	dtR = com_r * h / a_ref
	idx1 = (dtR >= 10) & (dtR <= 1e3)

	fx = com_r[idx1]
	fy = com_sb[idx1]
	ferr = com_err[idx1]

	params = np.array([ Mh0, Z0, p_c_mass, p_m2l, L_wave[kk], cov_MX ])

	mu_e1, re1, ndex1 = 5e-1, 50, 4.0
	po = [ mu_e1, re1, ndex1 ]

	bonds = [ [0, 1e1], [5, 50], [1, 10] ]
	E_return = optimize.minimize( cc_err_fit_func, x0 = np.array( po ), args = (fx, fy, params, ferr), method = 'L-BFGS-B', bounds = bonds,)

	print( E_return )
	popt = E_return.x
	Ie_fit, Re_fit, ne_fit = popt
	print(popt)

	put_params = [ Mh0, Z0, p_c_mass, p_m2l, L_wave[kk], cov_MX ]

	n_walk = 90
	put_x0 = np.random.uniform( 0, 5e0, n_walk )
	put_x1 = np.random.uniform( 5, 50, n_walk )
	put_x2 = np.random.uniform( 1, 10, n_walk )

	param_labels = ['$I_{e}$', '$R_{e}$', '$n$']

	pos = np.array([ put_x0, put_x1, put_x2 ]).T
	n_dim = pos.shape[1]
	L_chains = 2e4

	#file_name = out_path + '%s_%s-band_sersic+NFW_fit_mcmc_arr.h5' % (cat_lis[mass_dex], band[kk])
	file_name = out_path + '%s_%s-band_sersic+NFW_fit_1Mpc-h_mcmc_arr.h5' % (cat_lis[mass_dex], band[kk])

	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 50 ) as pool:
		sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_fix1_func, args = (fx, fy, put_params, ferr), pool = pool, backend = backend)
		sampler.run_mcmc(pos, L_chains, progress = True,)

	#sampler = emcee.backends.HDFBackend( file_name )
	try:
		tau = sampler.get_autocorr_time()
		print( tau )
		flat_samples = sampler.get_chain(discard = np.int( 2.5 * np.max(tau) ), thin = np.int( 0.5 * np.max(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain(discard = 5000, thin = 100, flat = True)

	fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
		truths = [ Ie_fit, Re_fit, ne_fit ],
		levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
		plot_datapoints = True, plot_density = False, fill_contours = True,)

	(Ie_arr, Re_arr, ne_arr ) = (flat_samples[:,0], flat_samples[:,1], flat_samples[:,2] )

	Ie_mc, Re_mc, ne_mc = np.median(Ie_arr), np.median(Re_arr), np.median(ne_arr)

	mc_fits = [ Ie_mc, Re_mc, ne_mc ]

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

	g_xlims = [ [0, 0.3], [10, 30], [8, 10] ]
	ri_xlims = [ [0, 1.0], [10, 24], [4, 9] ]

	for xi in range( n_dim ):
		for yi in range( n_dim ):
			if kk == 1:
				ax = axes[yi, xi]
				ax.set_xlim( g_xlims[xi][0], g_xlims[xi][1] )
			else:
				ax = axes[yi, xi]
				ax.set_xlim( ri_xlims[xi][0], ri_xlims[xi][1] )

	for yi in range(1, n_dim):
		for xi in range( yi ):
			if kk == 1:
				ax = axes[yi, xi]
				ax.set_ylim( g_xlims[yi][0], g_xlims[yi][1] )
			else:
				ax = axes[yi, xi]
				ax.set_ylim( ri_xlims[yi][0], ri_xlims[yi][1] )

	ax = axes[0, n_dim-1]
	ax.set_title( fig_name[ mass_dex ] + ', %s band' % band[kk] )
	plt.savefig('/home/xkchen/figs/%s_%s-ban_project-NFW_fit_params.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()

	## save the fitting params
	keys = ['Ie_0', 'Re_0', 'ne_0', 'C_mass', 'M2L']
	values = [ Ie_mc, Re_mc, ne_mc, p_c_mass, p_m2l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	#out_data.to_csv( out_path + 'high_BCG_star-Mass_%s-band_sersic+NFW_mcmc_fit.csv' % band[kk] )
	out_data.to_csv( out_path + 'high_BCG_star-Mass_%s-band_sersic+NFW_1Mpc-h_mcmc_fit.csv' % band[kk] )

	mock_cen_f = sersic_func( com_r, Ie_mc, Re_mc, ne_mc)
	mock_f_dens = out_NFW_flux_densi( com_r * h / a_ref, Mh0, Z0, p_c_mass, p_m2l, L_wave[kk] )
	sb_2Mpc = sersic_func( 2e3, Ie_mc, Re_mc, ne_mc) + out_NFW_flux_densi( 2e3 * h / a_ref, Mh0, Z0, p_c_mass, p_m2l, L_wave[kk] )
	sum_fit = mock_f_dens + mock_cen_f - sb_2Mpc

	M_sigma = sigmam( com_r * h / a_ref, Mh0, 0.0, p_c_mass )
	mock_f_nfw = mode_flux_densi( M_sigma, p_m2l, Z0, L_wave[kk] )

	cov_inv = np.linalg.pinv( cov_MX )
	delta = sum_fit[idx1] - fy
	chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2nu = chi2 / ( len(fy) - n_dim )

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( fig_name[mass_dex] + ',%s band' % band[kk])

	ax.plot( com_r, com_sb, ls = '-', color = 'k', alpha = 0.45, label = 'signal',)
	ax.fill_between( com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12,)

	ax.plot( com_r, sum_fit, ls = '-', color = 'b', alpha = 0.5, label = 'fit',)
	ax.plot( com_r, mock_f_nfw, ls = '--', color = 'b', alpha = 0.5, label = 'projected NFW',)
	ax.plot( com_r, mock_cen_f, ls = '-.', color = 'b', alpha = 0.5, )

	#ax.plot( com_r, sersic_mode_f, ls = ':', color = 'r', alpha = 0.5, label = 'outer sersic',)
	ax.axhline( y = sb_2Mpc, ls = ':', color = 'b', alpha = 0.5, )

	ax.text(1e2, 1e-1, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nu, color = 'k', alpha = 0.5,)
	ax.text(1e2, 3e-2, s = '$c = %.2f, M/L = %.2f$' % (p_c_mass, p_m2l), color = 'k', alpha = 0.5,)
	ax.text(2e1, 1e-3, s = '$I_{e} = %.3f$' % Ie_mc + '\n' + '$R_{e} = %.3f$' % Re_mc + '\n' + '$n=%.3f$' % ne_mc, color = 'b',)

	ax.set_xlim(1e1, 2e3)
	ax.set_ylim(1e-5, 1e0)
	ax.set_yscale('log')

	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.savefig('/home/xkchen/figs/%s_%s-band_NFW_compare.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()

### result compare
for kk in range( 3 ):

	## low mass bin
	#lo_dat = pds.read_csv( out_path + 'low_BCG_star-Mass_%s-band_sersic+NFW_mcmc_fit.csv' % band[kk] )
	lo_dat = pds.read_csv( out_path + 'low_BCG_star-Mass_%s-band_sersic+NFW_1Mpc-h_mcmc_fit.csv' % band[kk] )
	( lo_Ie_mc, lo_Re_mc, lo_ne_mc, lo_C_mc, lo_m2l_mc ) = ( np.array( lo_dat['Ie_0'])[0], np.array( lo_dat['Re_0'])[0], 
									np.array( lo_dat['ne_0'])[0], np.array( lo_dat['C_mass'])[0], np.array( lo_dat['M2L'])[0] )

	idmx = nbg_low_r[kk] >= 10
	lo_com_r, lo_com_sb, lo_com_err = nbg_low_r[kk][idmx], nbg_low_sb[kk][idmx], nbg_low_err[kk][idmx]

	dtR = lo_com_r * h / a_ref
	idx1 = (dtR >= 10) & (dtR <= 1e3)
	#idx1 = lo_com_r <= 1e3

	fy = lo_com_sb[idx1]
	ferr = lo_com_err[idx1]

	low_cen_f = sersic_func( lo_com_r, lo_Ie_mc, lo_Re_mc, lo_ne_mc)
	low_nfw_fdens = out_NFW_flux_densi( lo_com_r * h / a_ref, Mh0, Z0, lo_C_mc, lo_m2l_mc, L_wave[kk] )
	low_sb_2Mpc = sersic_func( 2e3, lo_Ie_mc, lo_Re_mc, lo_ne_mc) + out_NFW_flux_densi( 2e3 * h / a_ref, Mh0, Z0, lo_C_mc, lo_m2l_mc, L_wave[kk] )
	low_sum_fit = low_nfw_fdens + low_cen_f - low_sb_2Mpc

	low_M_sigma = sigmam( lo_com_r * h / a_ref, Mh0, 0.0, lo_C_mc )
	low_f_nfw = mode_flux_densi( low_M_sigma, lo_m2l_mc, Z0, L_wave[kk] )

	#with h5py.File( path + 'low_BCG_star-Mass_%s-band_BG-sub_cov-cor_arr.h5' % band[kk], 'r') as f:
	with h5py.File( path + 'low_BCG_star-Mass_%s-band_BG-sub_cov-1Mpc-h_cov-cor_arr.h5' % band[kk], 'r') as f:
		cov_MX = np.array(f['cov_MX'])

	cov_inv = np.linalg.pinv( cov_MX )
	delta = low_sum_fit[idx1] - fy
	lo_chi2 = delta.T.dot( cov_inv ).dot(delta)
	lo_chi2nu = lo_chi2 / ( len(fy) - 6 )


	## high mass bin
	#hi_dat = pds.read_csv( out_path + 'high_BCG_star-Mass_%s-band_sersic+NFW_mcmc_fit.csv' % band[kk] )
	hi_dat = pds.read_csv( out_path + 'high_BCG_star-Mass_%s-band_sersic+NFW_1Mpc-h_mcmc_fit.csv' % band[kk] )
	( hi_Ie_mc, hi_Re_mc, hi_ne_mc, hi_C_mc, hi_m2l_mc ) = ( np.array( hi_dat['Ie_0'])[0], np.array( hi_dat['Re_0'])[0], 
											np.array( hi_dat['ne_0'])[0], np.array( hi_dat['C_mass'])[0], np.array( hi_dat['M2L'])[0] )

	idmx = nbg_hi_r[kk] >= 10
	hi_com_r, hi_com_sb, hi_com_err = nbg_hi_r[kk][idmx], nbg_hi_sb[kk][idmx], nbg_hi_err[kk][idmx]

	dtR = hi_com_r * h / a_ref
	idx1 = (dtR >= 10) & (dtR <= 1e3)
	#idx1 = hi_com_r <= 1e3

	fy = hi_com_sb[idx1]
	ferr = hi_com_err[idx1]

	hi_cen_f = sersic_func( hi_com_r, hi_Ie_mc, hi_Re_mc, hi_ne_mc)
	hi_nfw_fdens = out_NFW_flux_densi( hi_com_r * h / a_ref, Mh0, Z0, hi_C_mc, hi_m2l_mc, L_wave[kk] )
	hi_sb_2Mpc = sersic_func( 2e3, hi_Ie_mc, hi_Re_mc, hi_ne_mc) + out_NFW_flux_densi( 2e3 * h / a_ref, Mh0, Z0, hi_C_mc, hi_m2l_mc, L_wave[kk] )
	hi_sum_fit = hi_nfw_fdens + hi_cen_f - hi_sb_2Mpc

	hi_M_sigma = sigmam( hi_com_r * h / a_ref, Mh0, 0.0, hi_C_mc )
	hi_f_nfw = mode_flux_densi( hi_M_sigma, hi_m2l_mc, Z0, L_wave[kk] )

	#with h5py.File( path + 'high_BCG_star-Mass_%s-band_BG-sub_cov-cor_arr.h5' % band[kk], 'r') as f:
	with h5py.File( path + 'high_BCG_star-Mass_%s-band_BG-sub_cov-1Mpc-h_cov-cor_arr.h5' % band[kk], 'r') as f:
		cov_MX = np.array(f['cov_MX'])

	cov_inv = np.linalg.pinv( cov_MX )
	delta = hi_sum_fit[idx1] - fy
	hi_chi2 = delta.T.dot( cov_inv ).dot(delta)
	hi_chi2nu = hi_chi2 / ( len(fy) - 3 )


	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( '%s band SB fitting' % band[kk])

	ax.plot( nbg_low_r[kk], nbg_low_sb[kk], ls = '-', color = 'b', alpha = 0.45, label = fig_name[0],)
	ax.fill_between( nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = 'b', alpha = 0.12,)

	ax.plot( lo_com_r, low_sum_fit, ls = '--', color = 'b', alpha = 0.5, label = 'fit',)
	ax.plot( lo_com_r, low_f_nfw, ls = '-.', color = 'k', alpha = 0.5, label = 'projected NFW',)
	ax.plot( lo_com_r, low_cen_f, ls = ':', color = 'b', alpha = 0.5, )

	ax.axhline( y = low_sb_2Mpc, ls = ':', color = 'b', xmin = 0.8, xmax = 1.0, alpha = 0.5, )

	ax.text(6e2, 1e-2, s = '$\\chi^{2} / \\nu = %.5f$' % lo_chi2nu, color = 'b', alpha = 0.5,)
	ax.text(1e2, 3e-2, s = '$c = %.2f, M/L = %.2f$' % (lo_C_mc, lo_m2l_mc), color = 'k', alpha = 0.5,)
	ax.text(2e1, 1e-4, s = '$I_{e} = %.3f$' % lo_Ie_mc + '\n' + '$R_{e} = %.3f$' % lo_Re_mc + '\n' + '$n=%.3f$' % lo_ne_mc, color = 'b',)


	ax.plot( nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = 'r', alpha = 0.45, label = fig_name[1],)
	ax.fill_between( nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = 'r', alpha = 0.12,)

	ax.plot( hi_com_r, hi_sum_fit, ls = '--', color = 'r', alpha = 0.5,)
	ax.plot( hi_com_r, hi_cen_f, ls = ':', color = 'r', alpha = 0.5, )

	ax.axhline( y = hi_sb_2Mpc, ls = ':', color = 'r', xmin = 0.8, xmax = 1.0, alpha = 0.5, )

	ax.text(1e2, 1e-1, s = '$\\chi^{2} / \\nu = %.5f$' % hi_chi2nu, color = 'r', alpha = 0.5,)
	ax.text(2e1, 4e-4, s = '$I_{e} = %.3f$' % hi_Ie_mc + '\n' + '$R_{e} = %.3f$' % hi_Re_mc + '\n' + '$n=%.3f$' % hi_ne_mc, color = 'r',)


	ax.set_xlim(1e1, 2e3)
	ax.set_ylim(1e-5, 1e0)
	ax.set_yscale('log')

	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.savefig('/home/xkchen/figs/%s-band_NFW+sersic_compare.png' % band[kk], dpi = 300 )
	plt.close()


