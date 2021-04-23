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

	a, b, x0, A, alpha, B, cov_mx = params 
	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)

	d_off, I_e, R_e = p[:]
	pf1 = sersic_func(x, I_e, R_e)
	pf = pf0 + pf1 - d_off

	cov_inv = np.linalg.pinv( cov_mx )
	delta = pf - y

	#return delta.T.dot( cov_inv ).dot(delta)
	return np.sum( delta**2 / yerr**2 )

## MCMC fitting functions
def likelihood_func(p, x, y, params, yerr):
	a, b, x0, A, alpha, B, cov_mx = params
	d_off, I_e, R_e = p[:]

	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)
	pf1 = sersic_func(x, I_e, R_e)
	pf = pf0 + pf1 - d_off
	delta = pf - y
	cov_inv = np.linalg.pinv( cov_mx )
	return -0.5 * delta.T.dot( cov_inv ).dot(delta)

def prior_p_func( p ):

	off_d, i_e, r_e = p

	if ( 0 < off_d < 1e-1 ) & (0 < i_e < 1e1) & ( 1e2 < r_e < 3e3 ):
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):
	pre_p = prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

def BG_pro_cov( jk_sub_sb, N_samples, out_file, R_lim0):

	from light_measure import cov_MX_func

	tmp_r = []
	tmp_sb = []

	for mm in range( N_samples ):

		with h5py.File( jk_sub_sb % mm, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

		idvx = npix < 1.
		sb_arr[idvx] = np.nan

		idux = r_arr >= R_lim0
		tt_r = r_arr[idux]
		tt_sb = sb_arr[idux]

		tmp_r.append( tt_r )
		tmp_sb.append( tt_sb )

	R_mean, cov_MX, cor_MX = cov_MX_func(tmp_r, tmp_sb, id_jack = True,)

	with h5py.File( out_file, 'w') as f:
		f['cov_Mx'] = np.array( cov_MX )
		f['cor_Mx'] = np.array( cor_MX )
		f['R_kpc'] = np.array( R_mean )

	return

### ======== ###
## 30 sub-sample case
# path = '/home/xkchen/mywork/ICL/code/photo_z_match_SB/'
# out_path = '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/'

path = '/home/xkchen/fig_tmp/stack_2_28/' 
out_path = '/home/xkchen/project/tmp_mcmc_mass_bin/'
# ...

## 80 sub-samples and 80 measurement points
# path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_pros/'
# out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/cov_arr/'

# path = '/home/xkchen/fig_tmp/test_jk_number/'
# out_path = '/home/xkchen/project/tmp_jk_number/'

low_r, low_sb, low_err = [], [], [] 
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_low_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	low_r.append( tt_r )
	low_sb.append( tt_sb )
	low_err.append( tt_err )

hi_r, hi_sb, hi_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_high_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	hi_r.append( tt_r )
	hi_sb.append( tt_sb )
	hi_err.append( tt_err )

## total sample
tot_r, tot_sb, tot_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tot_r.append( tt_r )
	tot_sb.append( tt_sb )
	tot_err.append( tt_err )

## ... labels
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

color_s = ['r', 'g', 'b']

"""
### === ### cov, cor arrays
R_tt = [300, 400, 500, 600, 700 ]
mass_dex = 0 # 0, 1, 2 --> low or high or total mass bin

N_bin = 80
path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_pros/'
out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/cov_arr/'

# N_bin = 30
# path = '/home/xkchen/mywork/ICL/code/photo_z_match_SB/'
# out_path = '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/'

for qq in range( len(R_tt) ):

	#R_low = R_tt[qq]
	R_low = 10

	for ll in range( 3 )

		if mass_dex == 0:
			jk_sub_sb = path + 'photo-z_match_low_BCG_star-Mass_%s-band_'  % band[ll] + 'jack-sub-%d_SB-pro_z-ref.h5'
			out_file = out_path + 'low_BCG-M-star_%s-band_%d-kpc-out_cov-cor_arr_before_BG-sub.h5' % (band[ll], R_low)

		elif mass_dex == 1:
			jk_sub_sb = path + 'photo-z_match_high_BCG_star-Mass_%s-band_' % band[ll] + 'jack-sub-%d_SB-pro_z-ref.h5'
			out_file = out_path + 'high_BCG-M-star_%s-band_%d-kpc-out_cov-cor_arr_before_BG-sub.h5' % (band[ll], R_low)
		else:
			jk_sub_sb = path + 'photo-z_match_tot-BCG-star-Mass_%s-band_' % band[ll] + 'jack-sub-%d_SB-pro_z-ref.h5'
			out_file = out_path + 'tot_BCG-M-star_%s-band_%d-kpc-out_cov-cor_arr_before_BG-sub.h5' % (band[ll], R_low)

		BG_pro_cov( jk_sub_sb, N_bin, out_file, R_low)

		with h5py.File( out_file, 'r') as f:
			cov_MX = np.array( f['cov_Mx'])
			cor_MX = np.array( f['cor_Mx'])
			R_mean = np.array( f['R_kpc'])

		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.45, 0.80])
		ax1 = fig.add_axes([0.50, 0.10, 0.45, 0.80])

		ax0.set_title( fig_name[ mass_dex ] + ', %s band, coV_arr / 1e-8' % band[ll] )
		tf = ax0.imshow(cov_MX / 1e-8, origin = 'lower', cmap = 'coolwarm', 
			norm = mpl.colors.SymLogNorm(linthresh = 1e0, linscale = 1e0, vmin = -3e0, vmax = 3e0, base = 5),)
		plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = '$ SB^2 $')

		ax0.set_xticklabels( labels = [] )

		ax0.set_ylim(0, len(R_mean) - 1 )
		yticks = ax0.get_yticks( )
		tik_lis = ['%.1f' % ll for ll in R_mean[ yticks[:-1].astype( np.int ) ] ]
		ax0.set_yticks( yticks[:-1] )
		ax0.set_yticklabels( labels = tik_lis, )
		ax0.set_ylim(-0.5, len(R_mean) - 0.5 )

		ax1.set_title( fig_name[ mass_dex ] + ', %s band, coR_arr' % band[ll] )
		tf = ax1.imshow(cor_MX, origin = 'lower', cmap = 'seismic', vmin = -1, vmax = 1,)

		ax1.set_xticklabels( labels = [] )

		ax1.set_ylim(0, len(R_mean) - 1 )
		yticks = ax1.get_yticks( )
		tik_lis = ['%.1f' % ll for ll in R_mean[ yticks[:-1].astype( np.int ) ] ]
		ax1.set_yticks( yticks[:-1] )
		ax1.set_yticklabels( labels = tik_lis, )
		ax1.set_ylim(-0.5, len(R_mean) - 0.5 )

		plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01,)
		plt.savefig('%s_%s-band_pre-BG-sub_%d-kpc-out_coV-coR_arr.jpg' % (cat_lis[mass_dex], band[ll], R_low), dpi = 300)
		plt.close()
"""

R_tt = [300, 400, 500, 600, 700 ]
R_r = 1.4e3

for mass_dex in range( 2 ):

	for qq in range( 5 ):

		R_low = R_tt[qq]

		for kk in range( 3 ):

			if mass_dex == 0:
				idmx = low_r[kk] >= 10
				com_r = low_r[kk][idmx]
				com_sb = low_sb[kk][idmx]
				com_err = low_err[kk][idmx]

				with h5py.File(out_path + 'low_BCG-M-star_%s-band_%d-kpc-out_cov-cor_arr_before_BG-sub.h5' % (band[kk], R_low), 'r') as f:
					cov_MX = np.array( f['cov_Mx'] )

			elif mass_dex == 1:
				idmx = hi_r[kk] >= 10
				com_r = hi_r[kk][idmx]
				com_sb = hi_sb[kk][idmx]
				com_err = hi_err[kk][idmx]

				with h5py.File(out_path + 'high_BCG-M-star_%s-band_%d-kpc-out_cov-cor_arr_before_BG-sub.h5' % (band[kk], R_low), 'r') as f:
					cov_MX = np.array( f['cov_Mx'] )

			else:
				idmx = tot_r[kk] >= 10
				com_r = tot_r[kk][idmx]
				com_sb = tot_sb[kk][idmx]
				com_err = tot_err[kk][idmx]

				with h5py.File(out_path + 'tot_BCG-M-star_%s-band_%d-kpc-out_cov-cor_arr_before_BG-sub.h5' % (band[kk], R_low), 'r') as f:
					cov_MX = np.array( f['cov_Mx'] )

			## read params of random point SB profile
			p_dat = pds.read_csv( out_path + '%s-band_random_SB_fit_params.csv' % band[kk],)
			( e_a, e_b, e_x0, e_A, e_alpha, e_B ) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0], 
													np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0],)

			idx1 = (com_r >= R_low)
			idx2 = (com_r >= R_low) & (com_r <= R_r)
			fx = com_r[idx1]
			fy = com_sb[idx1]
			ferr = com_err[idx1]

			params = np.array( [ e_a, e_b, e_x0, e_A, e_alpha, e_B, cov_MX ] )

			po = [ 2e-4, 4.8e-4, 6.8e2 ]
			bonds = [ [0, 1e-2], [0, 1e1], [2e2, 3e3] ]
			E_return = optimize.minimize(err_fit_func, x0 = np.array(po), args = (fx, fy, params, ferr), method = 'L-BFGS-B', bounds = bonds,)
			print(E_return)

			popt = E_return.x
			offD, I_e, R_e = popt
			print(popt)

			## fitting with emcee
			put_params = [e_a, e_b, e_x0, e_A, e_alpha, e_B, cov_MX]

			n_walk = 93
			put_off_x = np.random.uniform( 0, 1e-1, n_walk )
			put_I_e_x = np.random.uniform( 0, 1e1, n_walk )
			put_R_e_x = np.random.uniform( 1e2, 3e3, n_walk )
			L_chains = 1e5

			param_labels = ['$d_{off}$', '$I_{e}$', '$R_{e}$']

			pos = np.array([put_off_x, put_I_e_x, put_R_e_x]).T
			n_dim = pos.shape[1]

			print(pos.shape)
			print(n_dim)

			file_name = out_path + '%s_%s-band_fix-n=2.1_%d-kpc-out_fit_BG_mcmc_arr.h5' % (cat_lis[ mass_dex], band[kk], R_low)

			backend = emcee.backends.HDFBackend( file_name )
			backend.reset( n_walk, n_dim )

			with Pool( 20 ) as pool:
				sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = (fx, fy, put_params, ferr), pool = pool, backend = backend,)
				sampler.run_mcmc(pos, L_chains, progress = True, )

			# sampler = emcee.backends.HDFBackend( file_name )
			try:
				tau = sampler.get_autocorr_time()
				flat_samples = sampler.get_chain( discard = np.int( 2.5 * np.mean(tau) ), thin = np.int( 0.5 * np.mean(tau) ), flat = True)
			except:
				flat_samples = sampler.get_chain( discard = 2000, thin = 250, flat = True)

			## check the mcmc fitting result
			d_off_arr = flat_samples[:,0]
			I_e_arr = flat_samples[:,1]
			R_e_arr = flat_samples[:,2]

			offD_mc = np.median(d_off_arr)
			I_e_mc = np.median(I_e_arr)
			R_e_mc = np.median(R_e_arr)

			## normalize with SB value at 2Mpc of the model fitting
			sb_2Mpc = sersic_func( 2e3, I_e_mc, R_e_mc )

			# save the background profile and params
			if mass_dex == 0:
			    full_r = low_r[kk]
			elif mass_dex == 1:
			    full_r = hi_r[kk]
			else:
			    full_r = tot_r[kk]

			full_r_fit = cc_rand_sb_func( full_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)
			full_BG = full_r_fit - offD_mc + sb_2Mpc

			keys = ['R_kpc', 'BG_sb']
			values = [full_r, full_BG ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + 'photo-z_%s_%s-band_%d-kpc-out_fit_BG-profile.csv' % (cat_lis[mass_dex], band[kk], R_low),)

			keys = ['e_a', 'e_b', 'e_x0', 'e_A', 'e_alpha', 'e_B', 'offD', 'I_e', 'R_e']
			values = [ e_a, e_b, e_x0, e_A, e_alpha, e_B, offD_mc, I_e_mc, R_e_mc ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill, index = ['k', 'v'])
			out_data.to_csv( out_path + 'photo-z_%s_%s-band_%d-kpc-out_fit_BG-profile_params.csv' % (cat_lis[mass_dex], band[kk], R_low),)

			fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
				truths = [popt[0], popt[1], popt[2]], levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), 
				show_titles = True, smooth = 1, smooth1d = 1, title_fmt = '.5f', plot_datapoints = True, plot_density = False, fill_contours = True,)

			mc_fits = [ offD_mc, I_e_mc, R_e_mc ]

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
			plt.savefig('/home/xkchen/%s_%s-ban_n=2.1_%d-kpc-out_fit_params.jpg' % (cat_lis[mass_dex], band[kk], R_low), dpi = 300)
			plt.close()


			cov_inv = np.linalg.pinv( cov_MX )
			params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B])

			idx2 = (com_r >= R_low) & (com_r <= R_r) ## must use com_r, for match the dimension of cov_arr

			sb_trunk = sersic_func( 2e3, popt[1], popt[2])

			fit_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, popt[1], popt[2] ) - popt[0]
			mode_sign = sersic_func( com_r, popt[1], popt[2]) - sb_trunk
			BG_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - popt[0] + sb_trunk

			delta_0 = fit_line[idx1] - fy
			chi_cov_0 = np.sum( delta_0**2 / ferr**2 ) / ( len(fy) - n_dim )
			chi_diag_0 = np.sum( (fit_line[idx2] - com_sb[idx2])**2 / com_err[idx2]**2 ) / ( np.sum(idx2) - n_dim )


			sb_trunk_mc = sersic_func( 2e3, I_e_mc, R_e_mc)

			fit_line_mc = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, I_e_mc, R_e_mc) - offD_mc
			mode_sign_mc = sersic_func( com_r, I_e_mc, R_e_mc) - sb_trunk_mc
			BG_line_mc = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - offD_mc + sb_trunk_mc

			delta_1 = fit_line_mc[idx1] - fy
			chi_cov_1 = delta_1.T.dot( cov_inv ).dot( delta_1 ) / ( len(fx) - n_dim )

			id_lim = ( fx >= R_low ) & ( fx <= R_r )
			lim_dex = np.where( id_lim )[0]
			left_index = lim_dex[0]
			right_index = lim_dex[-1]

			cut_cov_inv = cov_inv[ left_index: right_index + 1, left_index: right_index + 1 ]
			cut_delta = delta_1[ id_lim ]
			chi_diag_1 = cut_delta.T.dot( cut_cov_inv ).dot( cut_delta ) / ( np.sum(idx2) - n_dim )

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title( fig_name[ mass_dex ] + ', %s band' % band[kk] )

			ax.plot(com_r, com_sb, ls = '-', color = 'k', alpha = 0.5, label = 'signal')
			ax.fill_between(com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12)

			ax.plot( com_r, fit_line, color = 'r', ls = '-', alpha = 0.5, label = 'minimize')
			ax.plot( com_r, mode_sign, color = 'r', ls = '-.', alpha = 0.5,)
			ax.plot( com_r, BG_line, color = 'r', ls = '--', alpha = 0.5,)

			ax.plot( com_r, fit_line_mc, color = 'g', ls = '-', alpha = 0.5, label = 'MCMC')
			ax.plot( com_r, mode_sign_mc, color = 'g', ls = '-.', alpha = 0.5,)
			ax.plot( com_r, BG_line_mc, color = 'g', ls = '--', alpha = 0.5,)

			ax.text(1e3, 5.5e-3, s = '$\\chi^2_{limit} = %.5f$' % chi_diag_0, color = 'r',)
			ax.text(1e3, 5e-3, s = '$\\chi^2 = %.5f$' % chi_cov_0, color = 'r', )
			ax.text(1e3, 4.5e-3, s = '$\\chi^2_{limit} = %.5f$' % chi_diag_1, color = 'g',)
			ax.text(1e3, 4e-3, s = '$\\chi^2 = %.5f$' % chi_cov_1, color = 'g', )

			ax.axvline(x = R_low, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)
			ax.axvline(x = R_r, ls = ':', color = 'r', alpha = 0.5, ymin = 0.0, ymax = 0.3,)

			ax.set_xlim(1e2, 3e3)
			ax.set_xscale('log')
			ax.set_ylim(2e-3, 6e-3)

			ax.set_xlabel('R [kpc]')
			ax.set_ylabel('SB [nanomaggies / arcsec^2]')
			ax.legend( loc = 3,)
			ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
			ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

			plt.subplots_adjust(left = 0.15, right = 0.9,)
			plt.savefig('/home/xkchen/%s_%s-band_SB_n=2.1_%d-kpc-out_fit.jpg' % (cat_lis[mass_dex], band[kk], R_low), dpi = 300)
			plt.close()

print('%d rank finished' % rank )

raise
