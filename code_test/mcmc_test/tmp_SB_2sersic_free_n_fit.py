import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

from light_measure import cov_MX_func
import corner
import emcee
from multiprocessing import Pool

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

## 2 sersic profile
def com_sersic(r, I1, r1, n1, I2, r2, n2):
	sb_1 = sersic_func(r, I1, r1, n1)
	sb_2 = sersic_func(r, I2, r2, n2)
	sb = sb_1 + sb_2
	return sb

def err_fix2_func(p, r, y, params, yerr):
	I1, r1, n1, I2, r2, n2 = p[:]

	cov_mx = params[:]
	cov_inv = np.linalg.pinv( cov_mx )
	model_sb = com_sersic(r, I1, r1, n1, I2, r2, n2)
	delta = model_sb - y

	return delta.T.dot( cov_inv ).dot(delta)
	#return np.sum( delta**2 / yerr**2 )

## mcmc function
def likelihood_func(p, x, y, params, yerr):

	mu0, re_0, ne_0, mu1, re_1, ne_1 = p[:]
	cov_mx = params[0]

	model_sb = com_sersic(x, mu0, re_0, ne_0, mu1, re_1, ne_1)

	cov_inv = np.linalg.pinv( cov_mx )
	delta = model_sb - y

	return -0.5 * delta.T.dot( cov_inv ).dot(delta)

def prior_p_func( p ):

	mu0, re_0, ne_0, mu1, re_1, ne_1 = p[:]
	if ( 0 < mu0 < 5e0 ) * ( 5 < re_0 < 5e1) * ( 1 < ne_0 < 10 ) * (0 < mu1 < 1e-2) * (1e2 < re_1 < 3e3) * (1 < ne_1 < 10):
		return 0.0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )

	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

## with one component fixed
def cc_err_fit_func(p, r, y, params, yerr):
	pt_i, pt_r, pt_n = p[:]
	fx_i, fx_r, fx_n, cov_mx = params[:]

	cov_inv = np.linalg.pinv( cov_mx )
	model_sb = com_sersic(r, pt_i, pt_r, pt_n, fx_i, fx_r, fx_n)
	delta = model_sb - y

	return delta.T.dot( cov_inv ).dot(delta)
	#return np.sum( delta**2 / yerr**2 )

def likelihood_fix1_func(p, x, y, params, yerr):

	mu0, re_0, ne_0 = p[:]
	mu1, re_1, ne_1, cov_mx = params[:]

	model_sb = com_sersic(x, mu0, re_0, ne_0, mu1, re_1, ne_1)

	cov_inv = np.linalg.pinv( cov_mx )
	delta = model_sb - y

	identi = (mu0 < 0) | (re_0 < 0) | ( ne_0 < 0.5)

	if identi == True:
		return -np.inf
	else:
		return -0.5 * delta.T.dot( cov_inv ).dot(delta)
		#return -0.5 * np.sum( delta**2 / yerr**2 )

def prior_p_fix1_func( p ):

	mu0, re_0, ne_0 = p[:]

	if ( 0 < mu0 < 5e0 ) * ( 5 < re_0 < 5e1) * ( 1 < ne_0 < 10 ):
		return 0
	return -np.inf

def ln_p_fix1_func(p, x, y, params, yerr):

	pre_p = prior_p_fix1_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_fix1_func(p, x, y, params, yerr)

### === ### data load
# #path = '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/'
# path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/BG_estimate/'
# out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_fit_mcmc/'

path = '/home/xkchen/project/tmp_SB_fit/'
out_path = '/home/xkchen/project/tmp_SB_fit/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass', 'tot-BCG-star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$', 'low $M_{\\ast}$ + high $M_{\\ast}$'] ## or line name
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

### === ### mcmc fitting
## low mass part
for tt in range( 3 ):

	com_r, com_sb, com_err = nbg_low_r[tt], nbg_low_sb[tt], nbg_low_err[tt]

	idvx = (com_r <= 1.0e3) & (com_r >= 10)
	fx, fy, ferr = com_r[idvx], com_sb[idvx], com_err[idvx]

	with h5py.File( path + 'low_BCG_star-Mass_%s-band_BG-sub_cov-cor_arr.h5' % band[tt], 'r') as f:
		cov_MX = np.array(f['cov_MX'])

	params_arr = [ cov_MX ]
	mu0, re_0, ne_0, mu1, re_1, ne_1 = 5e-1, 20, 4, 2e-3, 500, 2.1
	po = np.array( [ mu0, re_0, ne_0, mu1, re_1, ne_1 ] )

	bonds = ( (0, 5e0), (5, 50), (1, 10), (0, 1e-2), (1e2, 3e3), (1, 10) )
	E_return = optimize.minimize( err_fix2_func, x0 = np.array(po), args = (fx, fy, params_arr, ferr), method = 'L-BFGS-B', bounds = bonds,)

	popt = E_return.x

	print( E_return )
	print( popt )

	Ie1, Re1, Ne1, Ie2, Re2, Ne2 = popt

	## MCMC fitting
	put_params = [ cov_MX ]

	n_walk = 180

	put_x0 = np.random.uniform( 0, 5, n_walk )
	put_x1 = np.random.uniform( 5, 50, n_walk )
	put_x2 = np.random.uniform( 1, 10, n_walk )

	put_x3 = np.random.uniform( 0, 1e-2, n_walk )
	put_x4 = np.random.uniform( 1e2, 3e3, n_walk )
	put_x5 = np.random.uniform( 1, 10, n_walk )

	param_labels = ['$I_{e,1}$', '$R_{e,1}$', '$n_{1}$', '$I_{e,2}$', '$R_{e,2}$', '$n_{2}$']

	pos = np.array([ put_x0, put_x1, put_x2, put_x3, put_x4, put_x5]).T
	n_dim = pos.shape[1]
	L_chains = 5e4

	## array saving
	file_name = out_path + 'low_BCG_star-Mass_%s-band_2-sersic_free-n_fit_mcmc_arr.h5' % band[tt]
	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 10 ) as pool:
		sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = (fx, fy, put_params, ferr), pool = pool, backend = backend)
		sampler.run_mcmc(pos, L_chains, progress = True)

	# sampler = emcee.backends.HDFBackend( file_name )
	try:
		tau = sampler.get_autocorr_time()
		print( tau )
		flat_samples = sampler.get_chain(discard = np.int( 2.5 * np.mean(tau) ), thin = np.int( 0.5 * np.mean(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain(discard = 5000, thin = 200, flat = True)

	( Ie1_arr, Re1_arr, ne1_arr, Ie2_arr, Re2_arr, ne2_arr ) = ( flat_samples[:,0], flat_samples[:,1], flat_samples[:,2], 
																flat_samples[:,3], flat_samples[:,4], flat_samples[:,5] )
	Ie1_mc, Re1_mc, ne1_mc = np.median(Ie1_arr), np.median(Re1_arr), np.median( ne1_arr )
	Ie2_mc, Re2_mc, Ne2_mc = np.median(Ie2_arr), np.median( Re2_arr), np.median( ne2_arr )

	mc_fits = [ Ie1_mc, Re1_mc, ne1_mc, Ie2_mc, Re2_mc, Ne2_mc ]

	fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
		truths = [ Ie1, Re1, Ne1, Ie2, Re2, Ne2 ], levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), 
		show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f', plot_datapoints = False, plot_density = False, fill_contours = True,)

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
	ax = axes[0, n_dim-1]
	ax.set_title( fig_name[0] + ', %s band' % band[tt],)
	plt.savefig('/home/xkchen/low_BCG_star-Mass_%s-ban_2-sersic-fit_params.jpg' % band[tt], dpi = 300)
	plt.close()

	fit_line = com_sersic( com_r, Ie1_mc, Re1_mc, ne1_mc, Ie2_mc, Re2_mc, Ne2_mc)
	fit_1 = sersic_func( com_r, Ie1_mc, Re1_mc, ne1_mc )
	fit_2 = sersic_func( com_r, Ie2_mc, Re2_mc, Ne2_mc )

	## chi2 arr
	cov_inv = np.linalg.pinv( cov_MX )
	delta_v = fit_line[idvx] - fy

	chi2 = delta_v.T.dot( cov_inv ).dot( delta_v )
	chi2nv = chi2 / ( len(fy) - len(mc_fits) )

	## save the fitting params
	keys = ['Ie_0', 'Re_0', 'ne_0', 'Ie_1', 'Re_1', 'ne_1']
	values = [ Ie1_mc, Re1_mc, ne1_mc, Ie2_mc, Re2_mc, Ne2_mc ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_path + 'low_BCG_star-Mass_%s-band_free-n_mcmc_fit.csv' % band[tt] )


	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('low $M_{\\ast}$, %s band' % band[tt])

	ax.plot(com_r, com_sb, ls = '-', color = 'k', alpha = 0.5, label = 'signal')
	ax.fill_between(com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12,)

	ax.plot(com_r, fit_1, ls = '--', color = 'g', alpha = 0.5,)
	ax.plot(com_r, fit_2, ls = ':', color = 'r', alpha = 0.5,)
	ax.plot(com_r, fit_line, ls = '-', color = 'r', alpha = 0.5, label = 'fitting',)

	ax.text(1e2, 5e-1, s = '$I_{e} = %.5f$' % Ie1_mc + ',$R_{e} = %.3f$' % Re1_mc + '\n' + '$n_{0} = %.3f$' % ne1_mc, color = 'g', fontsize = 8,)
	ax.text(1e2, 1e-1, s = '$I_{e} = %.5f$' % Ie2_mc + ',$R_{e} = %.3f$' % Re2_mc + '\n' + ',$n_{0} = %.3f$' % Ne2_mc, color = 'r', fontsize = 8,)
	ax.text(1e2, 5e-2, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv, color = 'b',)

	ax.set_xlim(1e0, 2e3)
	ax.set_ylim(1e-4, 1e1)
	ax.set_yscale('log')
	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.subplots_adjust( left = 0.15,)
	plt.savefig('/home/xkchen/low-M-star_%s-band_M-star_sample_BG-sub-SB_fit.jpg' % band[ tt ], dpi = 300)
	plt.close()


## high mass part
for tt in range( 3 ):

	com_r, com_sb, com_err = nbg_hi_r[tt], nbg_hi_sb[tt], nbg_hi_err[tt]
	idvx = (com_r <= 1.0e3) & (com_r >= 10)
	fx, fy, ferr = com_r[idvx], com_sb[idvx], com_err[idvx]

	with h5py.File( path + 'high_BCG_star-Mass_%s-band_BG-sub_cov-cor_arr.h5' % band[tt], 'r') as f:
		cov_MX = np.array(f['cov_MX'])

	## read pre-fit params
	m_dat = pds.read_csv( out_path + 'low_BCG_star-Mass_%s-band_free-n_mcmc_fit.csv' % band[tt] )

	( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
											np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0] )

	params_arr = [m_I1, m_R1, m_n1, cov_MX ]

	mu0, re_0, ne_0 = 5e-1, 20, 4
	po = np.array([ mu0, re_0, ne_0 ])

	bonds = ( (0, 5e0), (5, 50), (1, 10) )
	E_return = optimize.minimize( cc_err_fit_func, x0 = np.array( po ), args = (fx, fy, params_arr, ferr), method = 'L-BFGS-B', bounds = bonds,)

	popt = E_return.x
	print( E_return )
	print( popt )
	Ie0, Re0, Ne0 = popt

	## mcmc fit
	put_params = [ m_I1, m_R1, m_n1, cov_MX ]

	n_walk = 60
	put_x0 = np.random.uniform( 0, 5, n_walk )
	put_x1 = np.random.uniform( 5, 50, n_walk )
	put_x2 = np.random.uniform( 1, 10, n_walk )

	param_labels = ['$I_{e,1}$', '$R_{e,1}$', '$n_{1}$' ]

	pos = np.array([ put_x0, put_x1, put_x2 ]).T
	n_dim = pos.shape[1]
	L_chains = 3e4

	## array saving
	file_name = out_path + 'high_BCG_star-Mass_%s-band_2-sersic_free-n_fit_mcmc_arr.h5' % band[tt]

	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 10 ) as pool:
		sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_fix1_func, args = (fx, fy, put_params, ferr), pool = pool, backend = backend)
		sampler.run_mcmc(pos, L_chains, progress = True)

	# sampler = emcee.backends.HDFBackend( file_name )
	try:
		tau = sampler.get_autocorr_time()
		print( tau )
		flat_samples = sampler.get_chain(discard = np.int( 2.5 * np.mean(tau) ), thin = np.int( 0.5 * np.mean(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain(discard = 5000, thin = 200, flat = True)

	( Ie0_arr, Re0_arr, ne0_arr ) = ( flat_samples[:,0], flat_samples[:,1], flat_samples[:,2] )
	Ie0_mc, Re0_mc, ne0_mc = np.median(Ie0_arr), np.median(Re0_arr), np.median(ne0_arr)
	mc_fits = [ Ie0_mc, Re0_mc, ne0_mc ]

	fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
		truths = [ Ie0, Re0, Ne0 ],
		levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f',
		plot_datapoints = True, plot_density = False, fill_contours = True,)

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
	ax = axes[0, n_dim-1]
	ax.set_title( fig_name[1] + ', %s band' % band[tt],)
	plt.savefig('/home/xkchen/high_BCG_star-Mass_%s-ban_2-sersic-fit_params.jpg' % band[tt], dpi = 300)
	plt.close()


	fit_line = com_sersic(com_r, Ie0_mc, Re0_mc, ne0_mc, m_I1, m_R1, m_n1)
	fit_1 = sersic_func(com_r, Ie0_mc, Re0_mc, ne0_mc)
	fit_2 = sersic_func(com_r, m_I1, m_R1, m_n1)
	## chi2 arr
	cov_inv = np.linalg.pinv( cov_MX )
	delta_v = fit_line[idvx] - fy

	chi2 = delta_v.T.dot( cov_inv ).dot( delta_v )
	chi2nv = chi2 / ( len(fy) - len(mc_fits) )

	## save the fitting params
	keys = ['Ie_0', 'Re_0', 'ne_0', 'Ie_1', 'Re_1', 'ne_1']
	values = [ Ie0_mc, Re0_mc, ne0_mc, m_I1, m_R1, m_n1]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_path + 'high_BCG_star-Mass_%s-band_free-n_mcmc_fit.csv' % band[ tt ])

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('high $M_{\\ast}$, %s band' % band[tt])

	ax.plot(com_r, com_sb, ls = '-', color = 'k', alpha = 0.5, label = 'signal')
	ax.fill_between(com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12,)

	ax.plot(com_r, fit_1, ls = '--', color = 'g', alpha = 0.5,)
	ax.plot(com_r, fit_2, ls = ':', color = 'r', alpha = 0.5,)
	ax.plot(com_r, fit_line, ls = '-', color = 'r', alpha = 0.5, label = 'fitting',)

	ax.text(1e2, 5e-1, s = '$I_{e} = %.5f$' % Ie0_mc + ',$R_{e} = %.3f$' % Re0_mc + '\n' + '$n_{0} = %.3f$' % ne0_mc, color = 'g', fontsize = 8,)
	ax.text(1e2, 1e-1, s = '$I_{e} = %.5f$' % m_I1 + ',$R_{e} = %.3f$' % m_R1 + '\n' + ',$n_{0} = %.3f$' % m_n1, color = 'r', fontsize = 8,)
	ax.text(1e2, 5e-2, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv, color = 'b',)

	ax.set_xlim(1e0, 2e3)
	ax.set_ylim(1e-5, 1e1)
	ax.set_yscale('log')
	ax.legend( loc = 1)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

	plt.subplots_adjust( left = 0.15,)
	plt.savefig('/home/xkchen/high-M-star_%s-band_M-star_sample_BG-sub-SB_fit.jpg' % band[ tt ], dpi = 300)
	plt.close()

raise

