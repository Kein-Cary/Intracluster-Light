import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy.ndimage import gaussian_filter

# pipe-code
from img_pre_selection import cat_match_func, get_mu_sigma
from fig_out_module import cc_grid_img, grid_img
from fig_out_module import zref_BCG_pos_func
from img_mask import source_detect_func, mask_func
from img_mask_adjust import adjust_mask_func

from img_jack_stack import jack_main_func
from img_jack_stack import SB_pros_func
from light_measure import jack_SB_func
from img_jack_stack import lim_SB_pros_func, zref_lim_SB_adjust_func
from img_jack_stack import weit_aveg_img

from img_condition_select import img_condition_select_func
from cc_block_select import diffuse_identi_func
from img_pre_selection import map_mu_sigma_func
from img_resample import resamp_func
from img_jack_stack import aveg_stack_img

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

# constant
rad2asec = U.rad.to(U.arcsec)
pixel = 0.396
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

z_ref = 0.25

# stacking in angle coordinate
kk = 0

id_cen = 0
n_rbins = 100
N_bin = 30

S2N = 5
edg_bins = 4
'''
dat = pds.read_csv('/home/xkchen/project/random_r-band_tot_remain_mock-BCG-pos_%d-rank.csv' % rank,)
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

# XXX
sub_img = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_img' + '_%d-rank.h5' % rank
sub_pix_cont = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont' + '_%d-rank.h5' % rank
sub_sb = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro' + '_%d-rank.h5' % rank
# XXX

J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' + '_%d-rank.h5' % rank
J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' + '_%d-rank.h5' % rank
J_sub_sb = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro' + '_%d-rank.h5' % rank

jack_SB_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro' + '_%d-rank.h5' % rank
jack_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img' + '_%d-rank.h5' % rank
jack_cont_arr = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont' + '_%d-rank.h5' % rank

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = False, N_edg = None, id_Z0 = True, z_ref = None, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

commd.Barrier()

if rank == 0:

	N_sample = 10 # random 10 times

	for pp in range( N_bin ):

		data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' % pp + '_%d-rank.h5'
		out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' % pp + '-aveg.h5'
		aveg_stack_img(N_sample, data_file, out_file,)

		data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' % pp + '_%d-rank.h5'
		out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' % pp + '-aveg.h5'
		aveg_stack_img(N_sample, data_file, out_file,)

	data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img' + '_%d-rank.h5'
	out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img' + '-aveg.h5'
	aveg_stack_img(N_sample, data_file, out_file,)

	data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont' + '_%d-rank.h5'
	out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont' + '-aveg.h5'
	aveg_stack_img(N_sample, data_file, out_file,)
'''

# average of the random cases
if rank == 0:

	J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img-aveg.h5'
	J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont-aveg.h5'
	J_sub_sb = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro-aveg.h5'
	jack_SB_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro-aveg.h5'

	id_Z0 = True
	SB_pros_func(J_sub_img, J_sub_pix_cont, J_sub_sb, N_bin, n_rbins, id_Z0, z_ref)

	## final jackknife SB profile
	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):
		with h5py.File(J_sub_sb % nn, 'r') as f:
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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ kk ], N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jack_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

print('angle-coordinate_no-cut finished')
commd.Barrier()

"""
dat = pds.read_csv('/home/xkchen/project/random_r-band_tot_remain_mock-BCG-pos_%d-rank.csv' % rank,)
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

# cut-out edge pixels
Nedg = 500
# XXX
sub_img = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_img' + '_%d-rank_cut.h5' % rank
sub_pix_cont = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont' + '_%d-rank_cut.h5' % rank
sub_sb = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro' + '_%d-rank_cut.h5' % rank
# XXX

J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' + '_%d-rank_cut-%d.h5' % (rank, Nedg)
J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' + '_%d-rank_cut-%d.h5' % (rank, Nedg)
J_sub_sb = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro' + '_%d-rank_cut-%d.h5' % (rank, Nedg)

jack_SB_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro' + '_%d-rank_cut-%d.h5' % (rank, Nedg)
jack_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img' + '_%d-rank_cut-%d.h5' % (rank, Nedg)
jack_cont_arr = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont' + '_%d-rank_cut-%d.h5' % (rank, Nedg)

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = True, N_edg = Nedg, id_Z0 = True, z_ref = None, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

commd.Barrier()

if rank == 0:

	N_sample = 10 # random 10 times

	for pp in range( N_bin ):

		data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' % pp + '_%d-rank' + '_cut-%d.h5' % Nedg
		out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' % pp + '_cut-%d-aveg.h5' % Nedg
		aveg_stack_img(N_sample, data_file, out_file,)

		data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' % pp + '_%d-rank' + '_cut-%d.h5' % Nedg
		out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' % pp + '_cut-%d-aveg.h5' % Nedg
		aveg_stack_img(N_sample, data_file, out_file,)

	data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_%d-rank' + '_cut-%d.h5' % Nedg
	out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_cut-%d-aveg.h5' % Nedg
	aveg_stack_img(N_sample, data_file, out_file,)

	data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_%d-rank' + '_cut-%d.h5' % Nedg
	out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_cut-%d-aveg.h5' % Nedg
	aveg_stack_img(N_sample, data_file, out_file,)

# average of the random cases
if rank == 0:

	J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img' + '_cut-%d-aveg.h5' % Nedg
	J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont' + '_cut-%d-aveg.h5' % Nedg
	J_sub_sb = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro' + '_cut-%d-aveg.h5' % Nedg
	jack_SB_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro_cut-%d-aveg.h5' % Nedg

	id_Z0 = True

	SB_pros_func(J_sub_img, J_sub_pix_cont, J_sub_sb, N_bin, n_rbins, id_Z0, z_ref)

	## final jackknife SB profile
	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):
		with h5py.File(J_sub_sb % nn, 'r') as f:
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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ kk ], N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jack_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

print('angle-coordinate_cut-500 finished')
"""

'''
# stacking in physical coordinate
dat = pds.read_csv('/home/xkchen/project/random_r-band_tot_remain_mock-BCG-pos_zref_%d-rank.csv' % rank)
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

d_file = home + 'tmp_stack/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
# XXX
sub_img = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_img' + '_%d-rank.h5' % rank
sub_pix_cont = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont' + '_%d-rank.h5' % rank
sub_sb = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro' + '_%d-rank.h5' % rank
# XXX

J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref' + '_%d-rank.h5' % rank
J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref' + '_%d-rank.h5' % rank
J_sub_sb = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro_z-ref' + '_%d-rank.h5' % rank

jack_SB_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro_z-ref' + '_%d-rank.h5' % rank
jack_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_z-ref' + '_%d-rank.h5' % rank
jack_cont_arr = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref' + '_%d-rank.h5' % rank

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img, sub_pix_cont, sub_sb,
	J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr, id_cut = False, N_edg = None, id_Z0 = False,
	z_ref = z_ref, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

commd.Barrier()

if rank == 0:

	N_sample = 10 # random 10 times

	for pp in range( N_bin ):

		data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref' % pp + '_%d-rank.h5'
		out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref' % pp + '-aveg.h5'
		aveg_stack_img(N_sample, data_file, out_file,)

		data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref' % pp + '_%d-rank.h5'
		out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref' % pp + '-aveg.h5'
		aveg_stack_img(N_sample, data_file, out_file,)

	data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_z-ref' + '_%d-rank.h5'
	out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_z-ref' + '-aveg.h5'
	aveg_stack_img(N_sample, data_file, out_file,)

	data_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref' + '_%d-rank.h5'
	out_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref' + '-aveg.h5'
	aveg_stack_img(N_sample, data_file, out_file,)

print('finished rand-stack !')
'''

## average of random cases
if rank == 0:

	J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref-aveg.h5'
	J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref-aveg.h5'
	J_sub_sb = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro_z-ref-aveg.h5'
	jack_SB_file = load + 'stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro_z-ref-aveg.h5'

	id_Z0 = False
	SB_pros_func(J_sub_img, J_sub_pix_cont, J_sub_sb, N_bin, n_rbins, id_Z0, z_ref)

	## final jackknife SB profile
	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):
		with h5py.File(J_sub_sb % nn, 'r') as f:
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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ kk ], N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jack_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

print('finished z_ref rand-stack !')
commd.Barrier()

