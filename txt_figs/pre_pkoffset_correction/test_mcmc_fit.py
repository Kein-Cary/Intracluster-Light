import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

import corner
import emcee
from multiprocessing import Pool

# from mpi4py import MPI
# commd = MPI.COMM_WORLD
# rank = commd.Get_rank()
# cpus = commd.Get_size()

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

##### ===== ##### random image SB fitting part

## SB model for random image
def cc_inves_x2(x, x0, A, alpha, B):
	return A * (np.abs(x - x0))**(-1*alpha) + B

def cc_rand_sb_func(x, a, b, x0, A, alpha, B):
	pf0 = a * np.log10(x) + b
	pf1 = cc_inves_x2(np.log10(x), x0, A, alpha, B)
	pf = pf0 + pf1
	return pf

def err_func(p, x, y, yerr):
	a, b, x0, A, alpha, B = p[:]
	pf = cc_rand_sb_func(x, a, b, x0, A, alpha, B)
	return np.sum( (pf - y)**2 / yerr**2 )

## SB model for cluster component (in sersic formula, fixed sersic index)
def sersic_func(r, Ie, re):
	ndex = 2.1 # Zhang et a., 2019, for large scale, n~2.1
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def err_fit_func(p, x, y, params, yerr):

	a, b, x0, A, alpha, B, cov_inv = params 
	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)

	d_off, I_e, R_e = p[:]
	pf1 = sersic_func(x, 10**I_e, R_e)
	pf = pf0 + pf1 - 10**d_off

	delta = pf - y

	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

## MCMC fitting functions
def likelihood_func(p, x, y, params, yerr):

	a, b, x0, A, alpha, B, cov_inv = params
	d_off, I_e, R_e = p[:]

	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)
	pf1 = sersic_func(x, 10**I_e, 10**R_e)
	pf = pf0 + pf1 - 10**d_off

	delta = pf - y
	chi2 = delta.T.dot( cov_inv ).dot(delta)

	if np.isfinite( chi2 ):
		return -0.5 * chi2
	return -np.inf

def prior_p_func( p ):

	off_d, i_e, r_e = p[:]

	if ( 10**(-6) < 10**off_d < 10**(-1) ) & (10**(-6) < 10**i_e < 10**1) & ( 5e1 < 10**r_e < 3e3 ):
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):
	pre_p = prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

### ======== ###
color_s = ['r', 'g', 'b']

path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin/'
BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'

cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

# path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin/'
# BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin_BG/'

# fig_name = [ 'younger', 'older' ]
# cat_lis = [ 'younger', 'older' ]

cov_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/cov_arr/'
rand_path = '/home/xkchen/mywork/ICL/code/ref_BG_profile/'

out_path = '/home/xkchen/figs/'

R_low = 200 # betond 200 kpc
R_up = 1.4e3 # around point at which SB start to increase again

mass_dex = 0 # 0, 1

for kk in range( 3 ):

	with h5py.File( path + 'photo-z_match_gri-common_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	idmx = tt_r >= 10
	com_r = tt_r[idmx]
	com_sb = tt_sb[idmx]
	com_err = tt_err[idmx]

	with h5py.File( cov_path + 'photo-z_%s_%s-band_cov-cor_arr.h5' % ( cat_lis[mass_dex], band[kk] ), 'r') as f:
		cov_MX = np.array( f['cov_Mx'])
		cor_MX = np.array( f['cor_Mx'])
		R_mean = np.array( f['R_kpc'])

	cov_inv = np.linalg.pinv( cov_MX )

	## read params of random point SB profile
	p_dat = pds.read_csv( rand_path + '%s-band_random_SB_fit_params.csv' % band[kk] )
	( e_a, e_b, e_x0, e_A, e_alpha, e_B ) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0], 
											np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0],)

	idx1 = com_r >= R_low
	fx = com_r[idx1]
	fy = com_sb[idx1]
	ferr = com_err[idx1]

	## for cov_arr
	id_lim = np.where( idx1 == True )[0][0]
	lim_cov = cov_MX[ id_lim:, id_lim: ]
	lim_cov_inv = cov_inv[ id_lim:, id_lim: ]

	## select points for BG estimation fitting
	pre_dat = pds.read_csv( out_path + 'photo-z_%s_%s-band_BG-estimate-fit_chi2_compare.csv' % (cat_lis[mass_dex], band[kk]) )
	pre_chi2_cov = np.array( pre_dat['chi2_cov'] )
	pre_R = np.array( pre_dat['R'] )


	idvx = fx > 300
	lim_dex = np.where( idvx )[0][0]

	fit_r, fit_sb, fit_err = fx[idvx], fy[idvx], ferr[idvx]
	fit_cov = lim_cov[lim_dex:, lim_dex:]
	fit_cov_inv = lim_cov_inv[lim_dex:, lim_dex:]

	put_params = [e_a, e_b, e_x0, e_A, e_alpha, e_B, fit_cov_inv]

	n_walk = 60

	put_x0 = np.random.uniform( -6, -1, n_walk ) # off_D
	put_x1 = np.random.uniform( -6, 1, n_walk ) # Ie
	put_x2 = np.random.uniform( 1.7, 3.5, n_walk ) # Re

	L_chains = 2e4

	param_labels = ['$lgd_{off}$', '$lgI_{e}$', '$lgR_{e}$']

	pos = np.array([put_x0, put_x1, put_x2]).T
	n_dim = pos.shape[1]

	print(pos.shape)
	print(n_dim)

	file_name = out_path + '%s_%s-band_fix-n=2.1_BG-estimate_mcmc_fit_arr.h5' % (cat_lis[ mass_dex], band[kk])

	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	with Pool( 2 ) as pool:
		sampler = emcee.EnsembleSampler( n_walk, n_dim, ln_p_func, args = (fit_r, fit_sb, put_params, fit_err), pool = pool, backend = backend,)
		sampler.run_mcmc(pos, L_chains, progress = True, )

	# sampler = emcee.backends.HDFBackend( file_name )
	try:
		tau = sampler.get_autocorr_time()
		flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.mean(tau) ), thin = np.int( 0.5 * np.mean(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain( discard = 2000, thin = 200, flat = True)

	mc_fits = []

	for oo in range( n_dim ):
		samp_arr = flat_samples[:, oo]

		mc_fit_oo = np.median( samp_arr )

		mc_fits.append( mc_fit_oo )

	fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
		levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), 
		show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f', plot_datapoints = True, plot_density = False, fill_contours = True,)

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
	ax.set_title( fig_name[ mass_dex ] + ', %s band' % band[kk] )
	plt.savefig('/home/xkchen/%s_%s-band_n=2.1_BG-estimate_mcmc_fit_params.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()


	offD_mc, I_e_mc, R_e_mc = mc_fits
	sb_2Mpc = sersic_func( 2e3, 10**I_e_mc, 10**R_e_mc )

	## save params
	full_r = tt_r
	full_r_fit = cc_rand_sb_func( full_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)
	full_BG = full_r_fit - 10**offD_mc + sb_2Mpc

	keys = ['R_kpc', 'BG_sb']
	values = [full_r, full_BG ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'photo-z_%s_%s-band_BG-estimate_mcmc_fit_BG-profile.csv' % (cat_lis[mass_dex], band[kk]),)

	keys = ['e_a', 'e_b', 'e_x0', 'e_A', 'e_alpha', 'e_B', 'offD', 'I_e', 'R_e']
	values = [ e_a, e_b, e_x0, e_A, e_alpha, e_B, 10**offD_mc, 10**I_e_mc, 10**R_e_mc ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_path + 'photo-z_%s_%s-band_BG-estimate_mcmc_fit_params.csv' % (cat_lis[mass_dex], band[kk]),)


	idx2 = fit_r <= R_up
	lim_dex_2 = np.where( idx2 )[0][-1]

	params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B])
	sb_trunk = sersic_func( 2e3, 10**I_e_mc, 10**R_e_mc)

	fit_line = cc_rand_sb_func( fit_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( fit_r, 10**I_e_mc, 10**R_e_mc) - 10**offD_mc
	mode_sign = sersic_func( fit_r, 10**I_e_mc, 10**R_e_mc ) - sb_trunk

	BG_line = cc_rand_sb_func( fit_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - 10**offD_mc + sb_trunk

	delta = fit_line - fit_sb
	chi2_cov = delta.T.dot( fit_cov_inv ).dot( delta ) / ( len(fit_sb) - n_dim )
	chi2_diag = np.sum( delta**2 / fit_err**2 ) / ( len(fit_sb) - n_dim )

	sub_cov_inv = fit_cov_inv[ :lim_dex_2 + 1, :lim_dex_2 + 1]
	chi2_cov_lim = delta[idx2].T.dot( sub_cov_inv ).dot( delta[idx2] ) / ( len(fit_sb[idx2]) - n_dim )
	chi2_diag_lim = np.sum( delta[idx2]**2 / fit_err[idx2]**2 ) / ( len(fit_sb[idx2]) - n_dim )

	_sum_fit = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, 10**I_e_mc, 10**R_e_mc) - 10**offD_mc
	_mode_sign = sersic_func( com_r, 10**I_e_mc, 10**R_e_mc ) - sb_trunk

	_BG_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - 10**offD_mc + sb_trunk


	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( fig_name[ mass_dex ] + ', %s band' % band[kk] )

	ax.plot(com_r, com_sb, ls = '-', color = 'k', alpha = 0.5, label = 'signal')
	ax.fill_between(com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12)

	ax.plot( com_r, _sum_fit, color = 'r', ls = '-', alpha = 0.5, label = 'fit')
	ax.plot( com_r, _mode_sign, color = 'r', ls = '-.', alpha = 0.5,)
	ax.plot( com_r, _BG_line, color = 'r', ls = '--', alpha = 0.5,)

	ax.text(1e3, 5.5e-3, s = '$\\chi^2_{cov,limit} = %.5f$' % chi2_cov_lim, color = 'r',)
	ax.text(1e3, 5e-3, s = '$\\chi^2_{cov} = %.5f$' % chi2_cov, color = 'r', )

	ax.text(1e3, 4.5e-3, s = '$\\chi^2_{diag, limit} = %.5f$' % chi2_diag_lim, color = 'g',)
	ax.text(1e3, 4e-3, s = '$\\chi^2_{diag} = %.5f$' % chi2_diag, color = 'g', )

	ax.axvline(x = R_low, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)
	ax.axvline(x = R_up, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

	ax.set_xlim(1e2, 3e3)
	ax.set_xscale('log')
	ax.set_ylim(2e-3, 6e-3)

	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / arcsec^2]')
	ax.legend( loc = 3,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.subplots_adjust(left = 0.15, right = 0.9,)
	plt.savefig('/home/xkchen/%s_%s-band_n=2.1_BG-estimate_mcmc_fit.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()

	raise
