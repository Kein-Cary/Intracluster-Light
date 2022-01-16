import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from scipy import ndimage
from astropy import cosmology as apcy

#.
from light_measure import jack_SB_func
from fig_out_module import arr_jack_func
from img_jack_stack import SB_pros_func


import time

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

band = ['r', 'g', 'i']

pixel = 0.396
z_ref = 0.25

Dl_ref = Test_model.luminosity_distance( z_ref ).value
Da_ref = Test_model.angular_diameter_distance( z_ref ).value

psf_FWHM = 1.32 # arcsec
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )




### === data load (gravity)
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

out_path = '/home/xkchen/figs/' 

n_rbins = 30
N_sample = 30
id_Z0 = False  # not in angle coordinate

band_str = band[ rank ]

cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']


for ll in range( 2 ):

	flux_img = load + 'Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref_pk-off.h5'
	pix_cont_img = load + 'Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_pk-off.h5'
	sub_sb_file = out_path + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref_pk-off.h5'

	SB_pros_func(flux_img, pix_cont_img, sub_sb_file, N_sample, n_rbins, id_Z0, z_ref)

	##... average of jackknife
	jack_SB_file = out_path + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref_pk-off.h5'

	tmp_sb = []
	tmp_r = []

	for nn in range( N_sample ):

		with h5py.File( sub_sb_file % nn, 'r') as f:

			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

		idvx = npix < 1.
		sb_arr[idvx] = np.nan
		r_arr[idvx] = np.nan

		tmp_sb.append(sb_arr)
		tmp_r.append(r_arr)

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func( tmp_sb, tmp_r, band_str, N_sample )[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jack_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

