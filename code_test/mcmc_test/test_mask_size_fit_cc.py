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
	ndex = 2.1
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
path = '/home/xkchen/fig_tmp/stack/'
out_path = '/home/xkchen/project/tmp_mcmc_mask_size/'

size_arr = np.array([25, 50, 75, 100])
line_name = ['25 (FWHM/2)', '50 (FWHM/2)', '75 (FWHM/2)', '100 (FWHM/2)']
file_str = ['25-FWHM-ov2', '50-FWHM-ov2', '75-FWHM-ov2', '100-FWHM-ov2']
line_c = ['b', 'g', 'r', 'm']

N_bin = 30

R_tt = np.array([3e2, 4e2, 5e2, 6e2, 650, 7e2]) ## for 75 (FWHM/2) only
#R_tt = np.array([650, 7e2]) ## for all size
R_low = R_tt[ rank ]
R_up = 1.4e3

for pp in range( 2,3 ): #len(size_arr) ):

	if pp == 2:
		with h5py.File(path + 'Star-M_com_r-band_Mean_jack_SB-pro_z-ref_8-R-kron.h5', 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
	else:
		with h5py.File(path + 
			'Star-M_com_r-band_Mean_jack_SB-pro_z-ref_%d-FWHM-ov2_bri-star.h5' % size_arr[pp], 'r') as f:
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

	## MCMC result
	param_labels = ['$d_{off}$', '$I_{e}$', '$R_{e}$']
	n_dim = len( param_labels )

	file_name = out_path + 'r-band_%d-FWHM-ov2_%d-kpc-out_fix-n=2.1_fit_BG_mcmc_arr.h5' % (size_arr[pp], R_low)
	#file_name = out_path + 'r-band_%d-FWHM-ov2_%d-kpc-to-3Mpc_fix-n=2.1_fit_BG_mcmc_arr.h5' % (size_arr[pp], R_low)

	sampler = emcee.backends.HDFBackend( file_name )

	try:
		tau = sampler.get_autocorr_time()
		print( '%d rank' % rank )
		print( tau )
		flat_samples = sampler.get_chain(discard = np.int( 2 * np.mean(tau) ), thin = np.int( 0.5 * np.mean(tau) ), flat = True)
	except:
		flat_samples = sampler.get_chain(discard = 1000, thin = 500, flat = True)

	d_off_arr = flat_samples[:,0]
	I_e_arr = flat_samples[:,1]
	R_e_arr = flat_samples[:,2]

	offD_mc = np.median(d_off_arr)
	I_e_mc = np.median(I_e_arr)
	R_e_mc = np.median(R_e_arr)

	### figure
	idx2 = ( com_r >= R_low ) & ( com_r <= R_up )
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

	## 1-sigma region
	tmp_line_arr = np.zeros( (len(flat_samples[:,0]), len(com_r) ), )
	tmp_bg_lines = np.zeros( (len(flat_samples[:,0]), len(com_r) ), )

	for ii in range( len(flat_samples[:,0]) ):

		tt_offd, tt_ie, tt_re = flat_samples[:,0][ii], flat_samples[:,1][ii], flat_samples[:,2][ii]

		tt_fit_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, tt_ie, tt_re) - tt_offd
		tt_trunk_sb = sersic_func( 2e3, tt_ie, tt_re )
		tt_mode_sign = sersic_func( com_r, tt_ie, tt_re) - tt_trunk_sb
		tt_mode_bg = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - tt_offd + tt_trunk_sb

		tmp_line_arr[ii, :] = tt_fit_line
		tmp_bg_lines[ii, :] = tt_mode_bg

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

	ax.fill_between( com_r, y1 = np.percentile(tmp_bg_lines, 16, axis = 0,), 
							y2 = np.percentile(tmp_bg_lines, 84, axis = 0,), color = 'c', alpha = 0.12,)

	ax.plot( com_r, np.percentile(tmp_line_arr, 16, axis = 0,), ls = ':', color = 'c', lw = 1, alpha = 0.5,)
	ax.plot( com_r, np.percentile(tmp_line_arr, 84, axis = 0,), ls = ':', color = 'c', lw = 1, alpha = 0.5,)

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
	plt.close()

