import h5py
import pandas as pds
import numpy as np

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits

import scipy.stats as sts
import scipy.special as special
from scipy import optimize
from scipy.stats import binned_statistic as binned
## mode
from light_measure import light_measure_Z0_weit
from light_measure import light_measure_weit
from light_measure import jack_SB_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

# constant
rad2asec = U.rad.to(U.arcsec)
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])
pixel = 0.396

def tmp_SB_func(R_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, band_id, jack_sb, pixel,):

	r_bins = R_bins

	for nn in range( N_bin ):

		with h5py.File( J_sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])

		with h5py.File( J_sub_pix_cont % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		xn, yn = np.int( tmp_img.shape[1] / 2), np.int( tmp_img.shape[0] / 2)

		Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit( tmp_img, tmp_cont, pixel, xn, yn, r_bins)
		sb_arr, sb_err_arr = Intns, Intns_err
		r_arr = Angl_r

		with h5py.File( J_sub_sb % nn, 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err_arr)
			f['nratio'] = np.array(nratio)
			f['npix'] = np.array(npix)

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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ band_id ], N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jack_sb, 'w') as f:
		f['r'] = np.array( tt_jk_R )
		f['sb'] = np.array( tt_jk_SB )
		f['sb_err'] = np.array( tt_jk_err )
		f['lim_r'] = np.array( sb_lim_r )

	return

def zref_tmp_SB_func(R_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, band_id, jack_sb, pixel, z_ref,):

	r_bins = R_bins

	for nn in range( N_bin ):

		with h5py.File( J_sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])

		with h5py.File( J_sub_pix_cont % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		xn, yn = np.int( tmp_img.shape[1] / 2), np.int( tmp_img.shape[0] / 2)

		Intns, phy_r, Intns_err, npix, nratio = light_measure_weit( tmp_img, tmp_cont, pixel, xn, yn, z_ref, r_bins)
		sb_arr, sb_err_arr = Intns, Intns_err
		r_arr = phy_r

		with h5py.File( J_sub_sb % nn, 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err_arr)
			f['nratio'] = np.array(nratio)
			f['npix'] = np.array(npix)

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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ band_id ], N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jack_sb, 'w') as f:
		f['r'] = np.array( tt_jk_R )
		f['sb'] = np.array( tt_jk_SB )
		f['sb_err'] = np.array( tt_jk_err )
		f['lim_r'] = np.array( sb_lim_r )

	return

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
z_ref = 0.25

##### r-band img stacking, (combine low and high BCG star Mass sample)
kk = 0

id_cen = 0
n_rbins = 100
N_bin = 30

'''
##**************## stacking in angle coordinate
r_bins = np.logspace(0, np.log10(2520), n_rbins)

if rank == 0:
	J_sub_img = load + 'stack/cluster_%s-band_BCG-stack' % band[ kk ] + '_jack-sub-%d_img.h5'
	J_sub_pix_cont = load + 'stack/cluster_%s-band_BCG-stack' % band[ kk ] + '_jack-sub-%d_pix-cont.h5'
	J_sub_sb = '/home/xkchen/project/stack/cluster_%s-band_BCG-stack' % band[ kk ] + '_jack-sub-%d_SB-pro.h5'
	jack_SB_file = '/home/xkchen/project/stack/cluster_%s-band_BCG-stack' % band[ kk ] + '_Mean_jack_SB-pro.h5'

	tmp_SB_func( r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel,)

	print('cluster img finished !')

	#### random img (BCG-stacking)
	J_sub_img = load + 'stack/random_%s-band_BCG-stack' % band[ kk ] + '_jack-sub-%d_img.h5'
	J_sub_pix_cont = load + 'stack/random_%s-band_BCG-stack' % band[ kk ] + '_jack-sub-%d_pix-cont.h5'
	J_sub_sb = '/home/xkchen/project/stack/random_%s-band_BCG-stack' % band[ kk ] + '_jack-sub-%d_SB-pro.h5'
	jack_SB_file = '/home/xkchen/project/stack/random_%s-band_BCG-stack' % band[ kk ] + '_Mean_jack_SB-pro.h5'

	tmp_SB_func( r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel,)

	## random img (aveged random-stacking, 10 times) 
	J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[ kk ] + '_jack-sub-%d_img-aveg.h5'
	J_sub_count = load + 'stack/random_%s-band_rand-stack' % band[ kk ] + '_jack-sub-%d_pix-cont-aveg.h5'
	J_sub_sb = '/home/xkchen/project/stack/random_%s-band_rand-stack' % band[ kk ] + '_jack-sub-%d_SB-pro-aveg.h5'
	jack_SB_file = '/home/xkchen/project/stack/random_%s-band_rand-stack' % band[ kk ] + '_Mean_jack_SB-pro-aveg.h5'

	tmp_SB_func( r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel,)

print('random img finished !')
commd.Barrier()

#### random img (random-stacking case)
## random-center stacking
J_sub_img = load + 'stack/random_%s-band_rand-stack' % band[ kk ] + '_jack-sub-%d_img' + '_%d-rank.h5' % rank
J_sub_pix_cont = load + 'stack/random_%s-band_rand-stack' % band[ kk ] + '_jack-sub-%d_pix-cont' + '_%d-rank.h5' % rank
J_sub_sb = '/home/xkchen/project/stack/random_%s-band_rand-stack' % band[ kk ] + '_jack-sub-%d_SB-pro' + '_%d-rank.h5' % rank
jack_SB_file = '/home/xkchen/project/stack/random_%s-band_rand-stack' % band[ kk ] + '_Mean_jack_SB-pro' + '_%d-rank.h5' % rank

tmp_SB_func( r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel,)

print('fixed R measure finished !')
'''

##**************## stacking in physical coordinate (z_ref)
R0 = 2840
#r_bins = np.logspace(0, np.log10(R0), 110)
r_bins = np.logspace(0, np.log10(R0), 50)

Da0 = Test_model.angular_diameter_distance( z_ref ).value
phy_R = Da0 * 1e3 * r_bins * pixel / rad2asec

if rank == 0:

	## random img
	J_sub_img = load + 'stack/random_r-band_BCG-stack' + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = load + 'stack/random_r-band_BCG-stack' + '_jack-sub-%d_pix-cont_z-ref.h5'

	#J_sub_sb = '/home/xkchen/project/stack/random_r-band_BCG-stack' + '_jack-sub-%d_SB-pro_z-ref.h5'
	#jack_SB_file = '/home/xkchen/project/stack/random_r-band_BCG-stack' + '_Mean_jack_SB-pro_z-ref.h5'
	J_sub_sb = '/home/xkchen/project/tmp/random_r-band_BCG-stack' + '_jack-sub-%d_SB-pro_z-ref.h5'
	jack_SB_file = '/home/xkchen/project/tmp/random_r-band_BCG-stack' + '_Mean_jack_SB-pro_z-ref.h5'

	zref_tmp_SB_func(r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel, z_ref,)


	J_sub_img = load + 'stack/random_r-band_rand-stack' + '_jack-sub-%d_img_z-ref-aveg.h5'
	J_sub_pix_cont = load + 'stack/random_r-band_rand-stack' + '_jack-sub-%d_pix-cont_z-ref-aveg.h5'

	#J_sub_sb = '/home/xkchen/project/stack/random_r-band_rand-stack' + '_jack-sub-%d_SB-pro_z-ref-aveg.h5'
	#jack_SB_file = '/home/xkchen/project/stack/random_r-band_rand-stack' + '_Mean_jack_SB-pro_z-ref-aveg.h5'
	J_sub_sb = '/home/xkchen/project/tmp/random_r-band_rand-stack' + '_jack-sub-%d_SB-pro_z-ref-aveg.h5'
	jack_SB_file = '/home/xkchen/project/tmp/random_r-band_rand-stack' + '_Mean_jack_SB-pro_z-ref-aveg.h5'

	zref_tmp_SB_func(r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel, z_ref,)

	## cluster img
	J_sub_img = load + 'stack/com-BCG-star-Mass_r-band' + '_jack-sub-%d_img_z-ref_with-selection.h5'
	J_sub_pix_cont = load + 'stack/com-BCG-star-Mass_r-band' + '_jack-sub-%d_pix-cont_z-ref_with-selection.h5'

	#J_sub_sb = '/home/xkchen/project/stack/com-BCG-star-Mass_r-band' + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'
	#jack_SB_file = '/home/xkchen/project/stack/com-BCG-star-Mass_r-band' + '_Mean_jack_SB-pro_z-ref_with-selection.h5'
	J_sub_sb = '/home/xkchen/project/tmp/com-BCG-star-Mass_r-band' + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'
	jack_SB_file = '/home/xkchen/project/tmp/com-BCG-star-Mass_r-band' + '_Mean_jack_SB-pro_z-ref_with-selection.h5'

	zref_tmp_SB_func(r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel, z_ref,)

	print('finish cluster img !')

print('finish rank 0 part !')
commd.Barrier()

## random rand-stack case
J_sub_img = load + 'stack/random_r-band_rand-stack' + '_jack-sub-%d_img_z-ref' + '_%d-rank.h5' % rank
J_sub_pix_cont = load + 'stack/random_r-band_rand-stack' + '_jack-sub-%d_pix-cont_z-ref' + '_%d-rank.h5' % rank

#J_sub_sb = '/home/xkchen/project/stack/random_r-band_rand-stack' + '_jack-sub-%d_SB-pro_z-ref' + '_%d-rank.h5' % rank
#jack_SB_file = '/home/xkchen/project/stack/random_r-band_rand-stack' + '_Mean_jack_SB-pro_z-ref' + '_%d-rank.h5' % rank
J_sub_sb = '/home/xkchen/project/tmp/random_r-band_rand-stack' + '_jack-sub-%d_SB-pro_z-ref' + '_%d-rank.h5' % rank
jack_SB_file = '/home/xkchen/project/tmp/random_r-band_rand-stack' + '_Mean_jack_SB-pro_z-ref' + '_%d-rank.h5' % rank

zref_tmp_SB_func(r_bins, N_bin, J_sub_img, J_sub_pix_cont, J_sub_sb, kk, jack_SB_file, pixel, z_ref,)

print('rand-stack finished !')

raise


