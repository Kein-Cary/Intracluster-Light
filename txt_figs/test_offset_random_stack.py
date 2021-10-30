import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from groups import groups_find_func
from img_mask_adjust import adjust_mask_func
from img_mask import mask_func
from img_condition_select import img_condition_select_func
from img_pre_selection import map_mu_sigma_func

from light_measure import flux_recal
from resample_modelu import sum_samp
from resample_modelu import down_samp
from img_resample import resamp_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

band = ['r', 'g', 'i']
z_ref = 0.25
pixel = 0.396

from img_jack_stack import jack_main_func
from img_jack_stack import SB_pros_func
from light_measure import jack_SB_func
from img_jack_stack import weit_aveg_img
from img_jack_stack import aveg_stack_img

def aveg_SB_pros(img_file, weit_file, sub_sb_file, aveg_SB_file, N_sample, n_bins, id_Z0, z_ref, band_str,):

	SB_pros_func(img_file, weit_file, sub_sb_file, N_sample, n_bins, id_Z0, z_ref)

	## final jackknife SB profile
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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_sample)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( aveg_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

	return

### 2_28
id_cen = 0
N_bin = 30
n_rbins = 90 # 100

N_times = 10
out_path = '/home/xkchen/project/rand_center/'

for kk in range( 3 ):

	dat = pds.read_csv( out_path + 'mock_random_bcg_2_28/random_%s-band_tot_remain_mock-BCG-pos_z-ref_%d-rank.csv' % (band[kk], rank),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	d_file = home + 'bcg_mask_img/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	# XXX
	sub_img = load + 'pkoffset_stack/random_%s-band' % band[kk] + '_sub-%d_img' + '_%d-rank.h5' % rank
	sub_pix_cont = load + 'pkoffset_stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont' + '_%d-rank.h5' % rank
	sub_sb = load + 'pkoffset_stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro' + '_%d-rank.h5' % rank
	# XXX

	J_sub_img = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref_' + '%d-rank.h5' % rank
	J_sub_pix_cont = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref_' + '%d-rank.h5' % rank
	J_sub_sb = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro_z-ref_' + '%d-rank.h5' % rank

	jack_SB_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro_z-ref_' + '%d-rank.h5' % rank
	jack_img = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_z-ref_' + '%d-rank.h5' % rank
	jack_cont_arr = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref_' + '%d-rank.h5' % rank

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img, sub_pix_cont, sub_sb,
		J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr, id_cut = True, N_edg = 1, id_Z0 = False,
		z_ref = z_ref, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

	commd.Barrier()


	if rank == 0:

		N_sample = 10

		for pp in range( N_bin ):

			data_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref_' % pp + '%d-rank.h5'
			out_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref' % pp + '-aveg.h5'
			aveg_stack_img(N_sample, data_file, out_file,)

			data_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref_' % pp + '%d-rank.h5'
			out_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref' % pp + '-aveg.h5'
			aveg_stack_img(N_sample, data_file, out_file,)

		data_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_z-ref_' + '%d-rank.h5'
		out_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_img_z-ref' + '-aveg.h5'
		aveg_stack_img(N_sample, data_file, out_file,)

		data_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref' + '_%d-rank.h5'
		out_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref' + '-aveg.h5'
		aveg_stack_img(N_sample, data_file, out_file,)

	print('finished rand-stack !')

	## average of random cases
	if rank == 0:

		J_sub_img = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_img_z-ref' + '-aveg.h5'
		J_sub_pix_cont = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref' + '-aveg.h5'
		J_sub_sb = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_jack-sub-%d_SB-pro_z-ref' + '-aveg.h5'
		jack_SB_file = load + 'pkoffset_stack/random_%s-band_rand-stack' % band[kk] + '_Mean_jack_SB-pro_z-ref' + '-aveg.h5'

		id_Z0 = False
		aveg_SB_pros(J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, N_bin, n_rbins, id_Z0, z_ref, band[kk],)

	print('finished z_ref rand-stack !')
	commd.Barrier()

print('%d rank, finished !' % rank)

