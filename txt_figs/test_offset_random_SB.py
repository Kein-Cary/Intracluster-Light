import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.signal as signal

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov
from fig_out_module import cc_grid_img

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
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

color_s = ['r', 'g', 'b']

# rand_path = '/home/xkchen/tmp_run/data_files/jupyter/random_ref_SB/'
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

rand_r, rand_sb, rand_err = [], [], []

#... 1D profile

for ii in range( 3 ):

	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_SB-pro_z-ref-aveg.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	rand_r.append( tt_r )
	rand_sb.append( tt_sb )
	rand_err.append( tt_err )

	## fitting random image 
	po = [-1.55 * 10**(-5), 0.0032, 3.7, 6 * 10**(-5), 4.5, -1e-5]

	fit_file = rand_path + '%s-band_random_SB_fit_params.csv' % band[ii]

	random_SB_fit_func( tt_r, tt_sb, tt_err, po, fit_file, end_point = 2, R_psf = 10,)
	# random_SB_fit_func( tt_r, tt_sb, tt_err, po, fit_file, end_point = 1, R_psf = 10,)

	## figs
	idx_lim = tt_r >= 10

	fit_r, fit_sb, fit_err = tt_r[idx_lim], tt_sb[idx_lim], tt_err[idx_lim]

	p_dat = pds.read_csv( fit_file )
	( e_a, e_b, e_x0, e_A, e_alpha, e_B) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0],
											np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0])

	tmp_line = cc_rand_sb_func( fit_r, e_a, e_b, e_x0, e_A, e_alpha, e_B)

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( '%s band' % band[ii] )

	ax.plot( tt_r, tt_sb, color = 'k', ls = '-', alpha = 0.45, label = 'signal')
	ax.fill_between( tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'k', alpha = 0.15)

	ax.plot( fit_r, tmp_line, color = 'b', ls = '-', alpha = 0.5, label = 'fit')
	ax.text(x = 12, y = 0.00425,s = 'fit line : ' + '\n' + 
	'$ SB(r) = %.5f * log(r) + %.5f + %.5f * |log(r) - %.5f|^{(-%.5f)}$' % (e_a, e_b, e_A, e_x0, e_alpha), color = 'b', fontsize = 7,)

	ax.set_ylim( 2.5e-3, 5.5e-3 )
	ax.set_xlim( 1e1, 5e3 )
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 2, fontsize = 8.0,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax.set_xlabel('R [kpc]')

	plt.subplots_adjust( left = 0.15 )
	plt.savefig('/home/xkchen/%s-band_random-SB_fit.png' % band[ii], dpi = 300)
	plt.close()

raise

#... 2D img
for ii in range( 3 ):

	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[ii], 'r') as f:
		tt_img = np.array( f['a'] )

	xn, yn = tt_img.shape[1] / 2, tt_img.shape[0] / 2	

	id_nan = np.isnan( tt_img )
	idy, idx = np.where( id_nan == False )

	y_low, y_up = np.min( idy ), np.max( idy )
	x_low, x_up = np.min( idx ), np.max( idx )

	cut_img = tt_img[ y_low : y_up + 1, x_low : x_up + 1]

	N_step = 200
	patch_mean = cc_grid_img( cut_img, N_step, N_step)[0]


	fig = plt.figure( figsize = (12.8, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.50, 0.10, 0.40, 0.80])

	tf = ax0.imshow( cut_img / pixel**2, origin = 'lower', cmap = 'bwr', vmin = -1e-2, vmax = 1e-2, )

	tf = ax1.imshow( patch_mean / pixel**2, origin = 'lower', cmap = 'bwr', vmin = -1e-2, vmax = 1e-2,)

	plt.colorbar( tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB')

	plt.savefig('/home/xkchen/%s-band_random-2D_flux.png' % band[ii], dpi = 300)
	plt.close()

