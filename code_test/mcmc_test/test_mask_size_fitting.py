import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
from scipy import optimize

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import cov_MX_func
from back_up_21_03 import random_SB_fit_func

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

import emcee
import corner

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

### === ###

## SB model for random image case
def cc_inves_x2(x, x0, A, alpha, B):
	return A * (np.abs(x - x0))**(-1*alpha) + B

def cc_rand_sb_func(x, a, b, x0, A, alpha, B):
	pf0 = a * np.log10(x) + b
	pf1 = cc_inves_x2(np.log10(x), x0, A, alpha, B)
	pf = pf0 + pf1
	return pf

## try fitting model SB(r) with fixed index
def sersic_func(r, Ie, re):
	ndex = 2.1 # 2.1, 1, 3, 4, 5, 
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def err_fit_func(p, x, y, params, yerr):
	# random img profiles
	a, b, x0, A, alpha, B, cov_mx = params 
	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)

	d_off, I_e, R_e = p[:]
	pf1 = sersic_func(x, I_e, R_e)
	pf = pf0 + pf1 - d_off

	cov_inv = np.linalg.pinv( cov_mx )
	delta = pf - y

	#return delta.T.dot( cov_inv ).dot(delta)
	return np.sum( delta**2 / yerr**2 )

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

	if (0 < off_d < 1e-1) & (0 < i_e < 1e1) & (1e2 < r_e < 3e3):
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):
	pre_p = prior_p_func( p )
	if not np.isfinite( pre_p ):
		return -np.inf
	return pre_p + likelihood_func(p, x, y, params, yerr)

### ==== ###
path = '/home/xkchen/mywork/ICL/code/mask_size_test/'
out_path = '/home/xkchen/mywork/ICL/code/BG_pros_3_03/'

#path = '/home/xkchen/fig_tmp/stack/'
#out_path = '/home/xkchen/project/tmp_mcmc_mask_size/'

size_arr = np.array([25, 50, 75, 100])
line_name = ['25 (FWHM/2)', '50 (FWHM/2)', '75 (FWHM/2)', '100 (FWHM/2)']
file_str = ['25-FWHM-ov2', '50-FWHM-ov2', '75-FWHM-ov2', '100-FWHM-ov2']
line_c = ['b', 'g', 'r', 'm']

N_bin = 30

#R_tt = np.array([3e2, 4e2, 5e2, 6e2, 650, 7e2])
R_tt = np.array([650, 7e2])

R_low = R_tt[ 1 ]

R_up = 1.4e3

"""
## cov_array
for pp in range( len(size_arr) ):

	tmp_r, tmp_sb = [], []

	for ll in range( N_bin ):
		if pp == 2:
			with h5py.File( path + 'Star-M_com_r-band_jack-sub-%d_SB-pro_z-ref_8-R-kron.h5' % ll, 'r') as f:
				r_arr = np.array(f['r'])[:-1]
				sb_arr = np.array(f['sb'])[:-1]
				sb_err = np.array(f['sb_err'])[:-1]
				npix = np.array(f['npix'])[:-1]
				nratio = np.array(f['nratio'])[:-1]
		else:
			with h5py.File( path + 'Star-M_com_r-band_jack-sub-%d_SB-pro_z-ref_%d-FWHM-ov2_bri-star.h5' % (ll, size_arr[pp]), 'r') as f:
				r_arr = np.array(f['r'])[:-1]
				sb_arr = np.array(f['sb'])[:-1]
				sb_err = np.array(f['sb_err'])[:-1]
				npix = np.array(f['npix'])[:-1]
				nratio = np.array(f['nratio'])[:-1]

		idvx = npix < 1.
		sb_arr[idvx] = np.nan

		idux = r_arr >= R_low # kpc
		#idux = (r_arr >= R_low) & (r_arr <= 3e3)
		tt_r = r_arr[idux]
		tt_sb = sb_arr[idux]

		tmp_r.append( tt_r )
		tmp_sb.append( tt_sb )

	R_mean, cov_MX, cor_MX = cov_MX_func(tmp_r, tmp_sb, id_jack = True,)

	with h5py.File( out_path + 'Star-M_com_r-band_cov-cor_arr_%d-kpc-out_pre-BG-sub_%d-FWHM-ov2.h5' % (R_low, size_arr[pp]), 'w') as f:
	#with h5py.File( out_path + 'Star-M_com_r-band_cov-cor_arr_%d-kpc-to-3Mpc_pre-BG-sub_%d-FWHM-ov2.h5' % (R_low, size_arr[pp]), 'w') as f:
		f['cov_Mx'] = np.array( cov_MX )
		f['cor_Mx'] = np.array( cor_MX )
		f['R_kpc'] = np.array( R_mean )

	#aa = np.cov( np.array(tmp_sb), bias = True, rowvar = False, ddof = 0,)
	#print( (aa * 29 + cov_MX) / cov_MX )

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.45, 0.80])
	ax1 = fig.add_axes([0.50, 0.10, 0.45, 0.80])

	ax0.set_title( '%d (FWHM/2)' % size_arr[pp] + ', r band, coV_arr / 1e-8')
	tf = ax0.imshow(cov_MX / 1e-8, origin = 'lower', cmap = 'coolwarm', 
		norm = mpl.colors.SymLogNorm(linthresh = 1e0, linscale = 1e0, vmin = -3e0, vmax = 3e0, base = 5),)

	plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = '$ SB^2 $')

	ax1.set_title( '%d (FWHM/2)' % size_arr[pp] + ', r band, coR_arr')
	tf = ax1.imshow(cor_MX, origin = 'lower', cmap = 'seismic', vmin = -1, vmax = 1,)
	plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01,)
	plt.savefig('r-band_%d-FWHM-ov2_%d-kpc-out_coV-coR_arr.jpg' % (size_arr[pp], R_low), dpi = 300)
	plt.close()
"""

#n_dex = 4 # 1, 3, 4, for 75 (FWHM/2) case only

#for pp in range( len(size_arr) ):
for pp in range( 2,3 ):

	if pp == 2:
		with h5py.File(path + 'Star-M_com_r-band_Mean_jack_SB-pro_z-ref_8-R-kron.h5', 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
	else:
		with h5py.File(path + 'Star-M_com_r-band_Mean_jack_SB-pro_z-ref_%d-FWHM-ov2_bri-star.h5' % size_arr[pp], 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

	id_Nul = c_r_arr > 0
	clus_r = c_r_arr[id_Nul]
	clus_sb = c_sb_arr[id_Nul]
	clus_err = c_sb_err[id_Nul]

	idmx = clus_r >= 10
	com_r = clus_r[idmx]
	com_sb = clus_sb[idmx]
	com_err = clus_err[idmx]

	idx1 = (com_r >= R_low)
	#idx1 = (com_r >= R_low) & (com_r <= 3e3) ## limit with 3Mpc
	fx = com_r[idx1]
	fy = com_sb[idx1]
	ferr = com_err[idx1]

	with h5py.File( out_path + 'Star-M_com_r-band_cov-cor_arr_%d-kpc-out_pre-BG-sub_%d-FWHM-ov2.h5' % (R_low, size_arr[pp]),'r') as f:
	#with h5py.File( out_path + 'Star-M_com_r-band_cov-cor_arr_%d-kpc-to-3Mpc_pre-BG-sub_%d-FWHM-ov2.h5' % (R_low, size_arr[pp]),'r') as f:
		cov_MX = np.array( f['cov_Mx'] )

	## pre-fitting
	BG_file = out_path + 'n=2.1_sersic_mean_BG_pros_params_%d-FWHM-ov2.csv' % size_arr[pp]
	cat = pds.read_csv(BG_file)
	e_a, e_b, e_x0 = np.array(cat['e_a'])[0], np.array(cat['e_b'])[0], np.array(cat['e_x0'])[0]
	e_A, e_alpha, e_B = np.array(cat['e_A'])[0], np.array(cat['e_alpha'])[0], np.array(cat['e_B'])[0]

	params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B, cov_MX])
	po = [ 2e-4, 4.8e-4, 6.8e2 ]
	bonds = ( (0, 1e-2), (0, 1e1), (3e2, 2.5e3) )
	E_return = optimize.minimize(err_fit_func, x0 = np.array(po), args = (fx, fy, params, ferr), method = 'L-BFGS-B', bounds = bonds,)
	# method = 'L-BFGS-B'
	print(E_return)
	popt = E_return.x
	offD, I_e, R_e = popt

	put_params = [e_a, e_b, e_x0, e_A, e_alpha, e_B, cov_MX ]

	n_walk = 93
	L_chains = 1e5
	put_off_x = np.random.uniform( 0, 1e-1, n_walk )
	put_I_e_x = np.random.uniform( 0, 1e1, n_walk )
	put_R_e_x = np.random.uniform( 1e2, 3e3, n_walk )

	param_labels = ['$d_{off}$', '$I_{e}$', '$R_{e}$']

	pos = np.array([ put_off_x, put_I_e_x, put_R_e_x ]).T
	n_dim = pos.shape[1]

	## run and save
	file_name = out_path + 'r-band_%d-FWHM-ov2_%d-kpc-out_fix-n=2.1_fit_BG_mcmc_arr.h5' % (size_arr[pp], R_low)
	#file_name = out_path + 'r-band_%d-FWHM-ov2_%d-kpc-to-3Mpc_fix-n=2.1_fit_BG_mcmc_arr.h5' % (size_arr[pp], R_low)
	# ... n != 2.1
	#file_name = out_path + 'r-band_%d-FWHM-ov2_%d-kpc-out_fix-n=%.1f_fit_BG_mcmc_arr.h5' % (size_arr[pp], R_low, n_dex)

	backend = emcee.backends.HDFBackend( file_name )
	backend.reset( n_walk, n_dim )

	sampler = emcee.EnsembleSampler(n_walk, n_dim, ln_p_func, args = (fx, fy, put_params, ferr), backend = backend,)
	sampler.run_mcmc(pos, L_chains, progress = True, )

	try:
		tau = sampler.get_autocorr_time()
		print( '%d rank' % rank )
		print( tau )
		flat_samples = sampler.get_chain(discard = np.int( 2.5 * np.mean(tau) ), thin = np.int( 0.5 * np.mean(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain(discard = 1000, thin = 500, flat = True)

	d_off_arr = flat_samples[:,0]
	I_e_arr = flat_samples[:,1]
	R_e_arr = flat_samples[:,2]

	offD_mc = np.median( d_off_arr )
	I_e_mc = np.median( I_e_arr )
	R_e_mc = np.median( R_e_arr )

	## save the fitting result
	sb_2Mpc = sersic_func( 2e3, I_e_mc, R_e_mc)

	full_r = clus_r
	full_r_fit = cc_rand_sb_func( full_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)
	full_BG = full_r_fit - offD_mc + sb_2Mpc

	keys = ['R_kpc', 'BG_sb']
	values = [full_r, full_BG ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + '%d_FWHM-ov2_%d-kpc-out_n=2.1_sersic_fit_BG-profile.csv' % (size_arr[pp], R_low) )
	#out_data.to_csv( out_path + '%d_FWHM-ov2_%d-kpc-to-3Mpc_n=2.1_sersic_fit_BG-profile.csv' % (size_arr[pp], R_low) )
	# ... n != 2.1
	#out_data.to_csv( out_path + '%d_FWHM-ov2_%d-kpc-out_n=%.1f_sersic_fit_BG-profile.csv' % (size_arr[pp], R_low, n_dex) )


	keys = [ 'e_a', 'e_b', 'e_x0', 'e_A', 'e_alpha', 'e_B', 'offD', 'I_e', 'R_e' ]
	values = [ e_a, e_b, e_x0, e_A, e_alpha, e_B, offD_mc, I_e_mc, R_e_mc ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_path + '%d_FWHM-ov2_%d-kpc-out_n=2.1_sersic_fit_BG-params.csv' % (size_arr[pp], R_low) )
	#out_data.to_csv( out_path + '%d_FWHM-ov2_%d-kpc-to-3Mpc_n=2.1_sersic_fit_BG-params.csv' % (size_arr[pp], R_low) )
	# ... n != 2.1
	#out_data.to_csv( out_path + '%d_FWHM-ov2_%d-kpc-out_n=%.1f_sersic_fit_BG-params.csv' % (size_arr[pp], R_low, n_dex) )

	fig = corner.corner(flat_samples, bins = [100] * n_dim, labels = param_labels, quantiles = [0.16, 0.84], 
		truths = [ popt[0], popt[1], popt[2] ], levels = (1 - np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-4.5) ), show_titles = True, 
		smooth = 1, smooth1d = 1, title_fmt = '.5f', plot_datapoints = True, plot_density = False, fill_contours = True,)

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
	ax.set_title( '%d (FWHM/2)' % size_arr[pp] )

	plt.savefig('/home/xkchen/%d_FWHM-ov2_%d-kpc-out_n=2.1_sersic_fit_test.jpg' % (size_arr[pp], R_low), dpi = 300)
	#plt.savefig('/home/xkchen/%d_FWHM-ov2_%d-kpc-out_n=%.1f_sersic_fit_test.jpg' % (size_arr[pp], R_low, n_dex), dpi = 300)
	plt.close()


	### figure
	idx2 = ( com_r >= R_low ) & ( com_r <= R_up )
	'''
	pdat = pds.read_csv( out_path + '%d_FWHM-ov2_%d-kpc-out_n=2.1_sersic_fit_BG-params.csv' % (size_arr[pp], R_low),)
	#pdat = pds.read_csv( out_path + '%d_FWHM-ov2_%d-kpc-to-3Mpc_n=2.1_sersic_fit_BG-params.csv' % (size_arr[pp], R_low),)
	(e_a, e_b, e_x0, e_A, e_alpha, e_B, offD_mc, I_e_mc, R_e_mc) = ( np.array(pdat['e_a'])[0], np.array(pdat['e_b'])[0], 
						np.array(pdat['e_x0'])[0], np.array(pdat['e_A'])[0], np.array(pdat['e_alpha'])[0], 
						np.array(pdat['e_B'])[0], np.array(pdat['offD'])[0], np.array(pdat['I_e'])[0], np.array(pdat['R_e'])[0] )
	'''
	cov_inv = np.linalg.pinv( cov_MX )

	params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B])
	sb_trunk = sersic_func(2e3, popt[1], popt[2])

	fit_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, popt[1], popt[2] ) - popt[0]
	mode_sign = sersic_func( com_r, popt[1], popt[2]) - sb_trunk
	BG_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - popt[0] + sb_trunk

	delta_0 = fit_line[idx1] - fy
	chi_cov_0 = np.sum( delta_0**2 / ferr**2 ) / ( len(fy) - n_dim )
	chi_diag_0 = np.sum( (fit_line[idx2] - com_sb[idx2])**2 / com_err[idx2]**2 ) / ( np.sum(idx2) - n_dim )


	sb_trunk_mc = sersic_func(2e3, I_e_mc, R_e_mc)

	fit_line_mc = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, I_e_mc, R_e_mc) - offD_mc
	mode_sign_mc = sersic_func( com_r, I_e_mc, R_e_mc) - sb_trunk_mc
	BG_line_mc = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - offD_mc + sb_trunk_mc

	delta_1 = fit_line_mc[idx1] - fy
	chi_cov_1 = delta_1.T.dot( cov_inv ).dot( delta_1 ) / ( len(fx) - n_dim )

	id_lim = ( fx >= R_low ) & ( fx <= R_up )
	lim_dex = np.where( id_lim )[0]
	left_index = lim_dex[0]
	right_index = lim_dex[-1]

	cut_cov_inv = cov_inv[ left_index: right_index + 1, left_index: right_index + 1 ]
	cut_delta = delta_1[ id_lim ]
	chi_diag_1 = cut_delta.T.dot( cut_cov_inv ).dot( cut_delta ) / ( np.sum(idx2) - n_dim )


	plt.figure()
	ax = plt.subplot(111)
	ax.plot(clus_r, clus_sb, ls = '-', color = 'k', alpha = 0.5, label = line_name[pp],)
	ax.fill_between(clus_r, y1 = clus_sb - clus_err, y2 = clus_sb + clus_err, color = 'k', alpha = 0.15,)

	ax.plot( com_r, fit_line, color = 'r', ls = '-', alpha = 0.5, label = 'minimize')
	ax.plot( com_r, mode_sign, color = 'r', ls = '-.', alpha = 0.5,)
	ax.plot( com_r, BG_line, color = 'r', ls = '--', alpha = 0.5,)

	ax.plot( com_r, fit_line_mc, color = 'g', ls = '-', alpha = 0.5, label = 'MCMC')
	ax.plot( com_r, mode_sign_mc, color = 'g', ls = '-.', alpha = 0.5,)
	ax.plot( com_r, BG_line_mc, color = 'g', ls = '--', alpha = 0.5,)

	ax.axvline( x = R_low, ls = ':', ymin = 0.0, ymax = 0.2, color = 'r', alpha = 0.5,)
	ax.axvline( x = R_up, ls = ':', ymin = 0.0, ymax = 0.2, color = 'r', alpha = 0.5,)

	ax.text(1e3, 5.5e-3, s = '$\\chi^2_{limit} = %.5f$' % chi_diag_0, color = 'r',)
	ax.text(1e3, 5e-3, s = '$\\chi^2 = %.5f$' % chi_cov_0, color = 'r', )
	ax.text(1e3, 4.5e-3, s = '$\\chi^2_{limit} = %.5f$' % chi_diag_1, color = 'g',)
	ax.text(1e3, 4e-3, s = '$\\chi^2 = %.5f$' % chi_cov_1, color = 'g', )

	ax.set_ylim(2e-3, 7e-3)
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xlim(1e2, 4e3)
	ax.set_xscale('log')
	ax.set_xlabel('kpc')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
	ax.legend( loc = 1)

	plt.subplots_adjust( left = 0.15 )

	plt.savefig('/home/xkchen/%d_FWHM-ov2_%d-kpc-out_SB_fitting_compare.jpg' % (size_arr[pp], R_low), dpi = 300)
	#plt.savefig('/home/xkchen/%d_FWHM-ov2_%d-kpc-out_SB_n=%.1f_fitting_compare.jpg' % (size_arr[pp], R_low, n_dex), dpi = 300)
	plt.close()

raise

### === ### BG-sub SB profile measurement
from img_BG_sub_SB_measure import BG_sub_sb_func
from back_up_21_03 import random_SB_fit_func

path = '/home/xkchen/mywork/ICL/code/mask_size_test/'
out_path = '/home/xkchen/mywork/ICL/code/BG_pros_3_03/'

size_arr = np.array([25, 50, 75, 100])
line_name = ['25 (FWHM/2)', '50 (FWHM/2)', '75 (FWHM/2)', '100 (FWHM/2)']
file_str = ['25-FWHM-ov2', '50-FWHM-ov2', '75-FWHM-ov2', '100-FWHM-ov2']
line_c = ['b', 'g', 'r', 'm']
N_bin = 30

clus_r, clus_sb, clus_err = [], [], []

for pp in range( len(size_arr) ):

	if pp == 2:
		with h5py.File(path + 'Star-M_com_r-band_Mean_jack_SB-pro_z-ref_8-R-kron.h5', 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
	else:
		with h5py.File(path + 'Star-M_com_r-band_Mean_jack_SB-pro_z-ref_%d-FWHM-ov2_bri-star.h5' % size_arr[pp], 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

	id_Nul = c_r_arr > 0
	clus_r.append( c_r_arr[id_Nul] )
	clus_sb.append( c_sb_arr[id_Nul] )
	clus_err.append( c_sb_err[id_Nul] )

BG_r, BG_pros = [], []
for kk in range( len(size_arr) ):
	pdat = pds.read_csv( out_path + '%d_FWHM-ov2_700-kpc-out_n=2.1_sersic_fit_BG-profile.csv' % (size_arr[kk]),)
	bg_r = np.array( pdat['R_kpc'] )
	bg_sb = np.array( pdat['BG_sb'] )

	BG_r.append( bg_r )
	BG_pros.append( bg_sb )

plt.figure()
for tt in range( len(size_arr) ):

	plt.plot(clus_r[tt], clus_sb[tt], ls = '-', color =  line_c[tt], alpha = 0.5, label = line_name[tt],)
	plt.fill_between(clus_r[tt], y1 = clus_sb[tt] - clus_err[tt], y2 = clus_sb[tt] + clus_err[tt], color =  line_c[tt], alpha = 0.12)
	plt.plot(BG_r[tt], BG_pros[tt], ls = '--', color =  line_c[tt], alpha = 0.5,)

plt.ylim(2.5e-3, 5e-3)
plt.ylabel('SB [nanomaggies / $arcsec^2$]')
plt.xlim(1e2, 4e3)
plt.xlabel('R [kpc]')
plt.xscale('log')
plt.grid(which = 'both', axis = 'both', alpha = 0.25,)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
plt.legend( loc = 2)
plt.savefig('n=2.1_BG_profile_trunc-signal_2Mpc.jpg', dpi = 300)
plt.close()

R_low = 7e2
R_up = 1.4e3

## signal and background compare
"""
trunk_R = np.array( [ 1.3e3, 1.4e3, 1.5e3, 1e3 / h ] )

for qq in range( 4 ):

	tmp_bg_r, tmp_bg_sb = [], []

	for pp in range( len(size_arr) ):

		pdat = pds.read_csv( out_path + '%d_FWHM-ov2_%d-kpc-out_n=2.1_sersic_fit_BG-params.csv' % (size_arr[pp], R_low),)
		(e_a, e_b, e_x0, e_A, e_alpha, e_B, offD_mc, I_e_mc, R_e_mc) = ( np.array(pdat['e_a'])[0], np.array(pdat['e_b'])[0], 
			np.array(pdat['e_x0'])[0], np.array(pdat['e_A'])[0], np.array(pdat['e_alpha'])[0], 
			np.array(pdat['e_B'])[0], np.array(pdat['offD'])[0], np.array(pdat['I_e'])[0], np.array(pdat['R_e'])[0])

		idmx = clus_r[pp] >= 10  
		com_r = clus_r[pp][idmx]
		com_sb = clus_sb[pp][idmx]
		com_err = clus_err[pp][idmx]

		sb_trunk_mc = sersic_func( trunk_R[qq], I_e_mc, R_e_mc) # sersic_func( 2e3, I_e_mc, R_e_mc) # 
		fit_line_mc = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, I_e_mc, R_e_mc) - offD_mc
		mode_sign_mc = sersic_func( com_r, I_e_mc, R_e_mc) - sb_trunk_mc
		BG_line_mc = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - offD_mc + sb_trunk_mc

		tmp_bg_r.append( com_r )
		tmp_bg_sb.append( BG_line_mc )

		_shift_D = np.min( com_sb[ com_r >= 500 ] )

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( line_name[pp] + ',fitting beyond %dkpc' % R_low)

		ax.plot(com_r, com_sb - _shift_D, ls = '-', color = 'r', alpha = 0.5, label = 'signal (measured)')
		ax.fill_between(com_r, y1 = com_sb - _shift_D - com_err, y2 = com_sb - _shift_D + com_err, color = 'r', alpha = 0.12)

		ax.plot( com_r, fit_line_mc, color = 'k', ls = '-', alpha = 0.5, label = 'fit')
		ax.plot( com_r, mode_sign_mc, color = 'k', ls = '-.', alpha = 0.5, label = 'signal (model)')
		ax.plot( com_r, BG_line_mc - _shift_D, color = 'k', ls = '--', alpha = 0.5, label = 'background')

		ax.axvline( x = R_low, ymin = 0, ymax = 0.25, ls = '-.', color = 'g', alpha = 0.5,)
		ax.axvline( x = R_up, ymin = 0, ymax = 0.2, ls = '-.', color = 'g', alpha = 0.5,)

		ax.set_ylim(-1e-4, 8e-4)
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax.set_xlim(5e2, 4e3)
		ax.set_xscale('log')
		ax.set_xlabel('R [kpc]')
		ax.legend( loc = 'upper center',)
		ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

		plt.subplots_adjust( left = 0.15, )
		plt.savefig('signal_fit_compare_%s_norm-R_%.2fMpc.jpg' % (file_str[pp], trunk_R[qq] / 1e3), dpi = 300)
		plt.close()

	plt.figure()
	for tt in range( len(size_arr) ):

		plt.plot(clus_r[tt], clus_sb[tt], ls = '-', color =  line_c[tt], alpha = 0.5, label = line_name[tt],)
		plt.fill_between(clus_r[tt], y1 = clus_sb[tt] - clus_err[tt], y2 = clus_sb[tt] + clus_err[tt], color =  line_c[tt], alpha = 0.12)
		plt.plot(tmp_bg_r[tt], tmp_bg_sb[tt], ls = '--', color =  line_c[tt], alpha = 0.5,)

	plt.ylim(2.5e-3, 5e-3)
	plt.ylabel('SB [nanomaggies / $arcsec^2$]')
	plt.xlim(1e2, 4e3)
	plt.xlabel('R [kpc]')
	plt.xscale('log')
	plt.grid(which = 'both', axis = 'both', alpha = 0.25,)
	plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
	plt.legend( loc = 2)
	plt.savefig('n=2.1_BG_profile_trunc-signal_norm-R_%.2fMpc.jpg' % (trunk_R[qq] / 1e3), dpi = 300)
	plt.close()
"""

## BG-sub SB profile
star_dex = 1

trunk_R = np.array( [ 1.3e3, 1.4e3, 1.5e3, 1e3 / h, 2e3] )
"""
for qq in range( len(trunk_R) ):

	for tt in range( len(size_arr) ):
		if tt == 2:
			jk_sub_sb = path + 'Star-M_com_r-band_jack-sub-%d_SB-pro_z-ref_8-R-kron.h5'
		else:
			if star_dex == 0:
				jk_sub_sb = path + 'Star-M_com_r-band_jack-sub-%d_SB-pro_z-ref' + '_%d-FWHM-ov2.h5' % size_arr[tt]
			else:
				jk_sub_sb = path + 'Star-M_com_r-band_jack-sub-%d_SB-pro_z-ref' + '_%d-FWHM-ov2_bri-star.h5' % size_arr[tt]
		if star_dex == 0:
			sb_out_put = out_path + 'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2.h5' % size_arr[tt]
		else:
			#sb_out_put = out_path + 'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2_bri-star.h5' % size_arr[tt]
			sb_out_put = out_path + 'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2_bri-star_norm-R_%.2fMpc.h5' % (size_arr[tt], trunk_R[qq] / 1e3)

		BG_params = out_path + '%d_FWHM-ov2_%d-kpc-out_n=2.1_sersic_fit_BG-params.csv' % (size_arr[tt], R_low)

		BG_sub_sb_func(N_bin, jk_sub_sb, sb_out_put, 'r', BG_params, trunk_R = trunk_R[ qq ],)

	bg_sub_r, bg_sub_sb, bg_sub_err = [], [], []

	for tt in range( len(size_arr) ):
		if star_dex == 0:
			with h5py.File( out_path + 'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2.h5' % size_arr[tt], 'r') as f:
				tt_jk_r = np.array( f['r'] )
				tt_jk_sb = np.array( f['sb'] )
				tt_jk_err = np.array( f['sb_err'] )
		else:
			#with h5py.File( out_path + 'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2_bri-star.h5' % size_arr[tt], 'r') as f:
			with h5py.File( out_path + 
				'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2_bri-star_norm-R_%.2fMpc.h5' % (size_arr[tt], trunk_R[qq] / 1e3), 'r') as f:
				tt_jk_r = np.array( f['r'] )
				tt_jk_sb = np.array( f['sb'] )
				tt_jk_err = np.array( f['sb_err'] )

		bg_sub_r.append( tt_jk_r )
		bg_sub_sb.append( tt_jk_sb )
		bg_sub_err.append( tt_jk_err )

	plt.figure()
	gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	for tt in range( len(size_arr) ):

		ax.plot(bg_sub_r[tt], bg_sub_sb[tt], ls = '-', color = line_c[tt], alpha = 0.5, label = line_name[tt],)
		ax.fill_between(bg_sub_r[tt], y1 = bg_sub_sb[tt] - bg_sub_err[tt], y2 = bg_sub_sb[tt] + bg_sub_err[tt],
			color = line_c[tt], alpha = 0.12,)

		bx.plot(bg_sub_r[tt], bg_sub_sb[tt] / bg_sub_sb[2], ls = '-', color = line_c[tt], alpha = 0.5)
		if tt == 2:
			bx.fill_between(bg_sub_r[tt], y1 = (bg_sub_sb[tt] - bg_sub_err[tt]) / bg_sub_sb[2],
				y2 = (bg_sub_sb[tt] + bg_sub_err[tt]) / bg_sub_sb[2], color = line_c[tt], alpha = 0.12)

	ax.legend( loc = 1,)
	ax.set_xlim(1e2, 2e3)
	ax.set_ylim(1e-5, 2e-2)
	ax.set_yscale('log')
	ax.set_xscale('log')
	#ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	bx.set_xlim( ax.get_xlim() )
	bx.set_xscale('log')
	bx.set_xlabel('R [kpc]')
	bx.grid(which = 'both', axis = 'both', alpha = 0.25,)
	bx.set_ylim( 0.5, 1.5)
	bx.axhline( y = 0.95, ls = ':', color = 'k', alpha = 0.25,)
	bx.axhline( y = 1.05, ls = ':', color = 'k', alpha = 0.25,)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax.set_xticklabels( labels = [] )

	plt.subplots_adjust( hspace = 0.05 )
	plt.savefig('n=2.1_mask-size_BG-sub_SB-compare_norm-R_%.2fMpc.jpg' % (trunk_R[qq] / 1e3), dpi = 300)
	plt.close()
"""

for pp in range( len(size_arr) ):

	bg_sub_r, bg_sub_sb, bg_sub_err = [], [], []

	for qq in range( len(trunk_R) ):

		with h5py.File( out_path + 
			'Star-M_com_r-band_Mean_jack_BG-sub_SB_z-ref_%d-FWHM-ov2_bri-star_norm-R_%.2fMpc.h5' % (size_arr[pp], trunk_R[qq] / 1e3), 'r') as f:
			tt_jk_r = np.array( f['r'] )
			tt_jk_sb = np.array( f['sb'] )
			tt_jk_err = np.array( f['sb_err'] )

		bg_sub_r.append( tt_jk_r )
		bg_sub_sb.append( tt_jk_sb )
		bg_sub_err.append( tt_jk_err )

	plt.figure()
	gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	ax.set_title( line_name[pp] )

	for tt in range( len(trunk_R) ):

		ax.plot(bg_sub_r[tt], bg_sub_sb[tt], ls = '-', color = mpl.cm.rainbow(tt/4), alpha = 0.5, 
			label = 'normalized R = %.2fMpc' % (trunk_R[tt] / 1e3),)

		if tt == len(trunk_R) - 1:
			ax.fill_between(bg_sub_r[tt], y1 = bg_sub_sb[tt] - bg_sub_err[tt], y2 = bg_sub_sb[tt] + bg_sub_err[tt],
				color = mpl.cm.rainbow(tt/4), alpha = 0.10,)

		bx.plot(bg_sub_r[tt], bg_sub_sb[tt] / bg_sub_sb[-1], ls = '-', color = mpl.cm.rainbow(tt/4), alpha = 0.5)
		if tt == len(trunk_R) - 1:
			bx.fill_between(bg_sub_r[tt], y1 = (bg_sub_sb[tt] - bg_sub_err[tt]) / bg_sub_sb[-1],
				y2 = (bg_sub_sb[tt] + bg_sub_err[tt]) / bg_sub_sb[-1], color = mpl.cm.rainbow(tt/4), alpha = 0.10)

	ax.legend( loc = 1,)
	ax.set_xlim(1e2, 2e3)
	ax.set_ylim(1e-5, 2e-2)
	ax.set_yscale('log')
	ax.set_xscale('log')

	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	bx.set_xlim( ax.get_xlim() )
	bx.set_xscale('log')
	bx.set_xlabel('R [kpc]')
	bx.grid(which = 'both', axis = 'both', alpha = 0.25,)
	bx.set_ylim( 0.65, 1.05)
	bx.axhline( y = 0.95, ls = ':', color = 'k', alpha = 0.25,)
	bx.axhline( y = 1.05, ls = ':', color = 'k', alpha = 0.25,)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax.set_xticklabels( labels = [] )

	plt.subplots_adjust( hspace = 0.05 )
	plt.savefig('%d_FWHM-ov2_n=2.1_mask-size_BG-sub_SB-compare_norm-R_%.2fMpc.jpg' % (size_arr[pp], trunk_R[qq] / 1e3), dpi = 300)
	plt.close()

