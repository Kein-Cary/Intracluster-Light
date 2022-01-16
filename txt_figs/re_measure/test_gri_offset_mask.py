import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.interpolate as interp
from astropy import cosmology as apcy

from img_resample import resamp_func
from fig_out_module import zref_BCG_pos_func
from img_mask import source_detect_func, mask_func

# from img_mask_adjust import adjust_mask_func
from img_mask_tmp import adjust_mask_func

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### === ### gri combined masking
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
out_path = home + 'photo_files/pos_offset_correct_imgs/'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'younger', 'older' ]
"""
for mm in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		dat = pds.read_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat.csv' % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		zN = len( z )
		print( zN )

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]

		## masking
		d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
		cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'
		offset_file = out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

		gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
		bcg_photo_file = home + 'photo_files/BCG_photometry/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'

		out_file = out_path + 'mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

		bcg_mask = 0

		if band_str == 'r':
			extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
						  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			extra_img = [ home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
						  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

		if band_str == 'g':
			extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
						  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
						  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

		if band_str == 'i':
			extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat',
						  home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
						  home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

		adjust_mask_func( d_file, cat_file, set_z, set_ra, set_dec, band_str, gal_file, out_file, bcg_mask,
			offset_file = offset_file, bcg_photo_file = bcg_photo_file, extra_cat = extra_cat, extra_img = extra_img,)

		print( '%d, %s band, done !' % (mm, band_str),)

# commd.Barrier()
"""

### === ### resampling
for mm in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		dat = pds.read_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat.csv' % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		zN = len( z )
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		d_file = out_path + 'mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
		out_file = out_path + 'resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

		resamp_func( d_file, z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], 
			clus_x[N_sub0 : N_sub1], clus_y[N_sub0 : N_sub1], band_str, out_file, z_ref, stack_info = None, pixel = 0.396, id_dimm = True,)

		print('%d, %s band finished !' % (mm, band_str) )

commd.Barrier()


### === ### masking and resampling for random imgs
for kk in range( 3 ):

	band_str = band[ kk ]

	dat = pds.read_csv( load + 'random_cat/2_28/random_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band_str )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	print( zN )

	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
	set_imgx, set_imgy = clus_x[N_sub0 : N_sub1], clus_y[N_sub0 : N_sub1]

	# ## masking
	# d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	# cat_file = home + 'new_sql_star_cat/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
	# offset_file = out_path + 'offset/random_%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

	# gal_file = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

	# out_file = home + 'bcg_mask_img/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	# bcg_mask = 1

	# if band_str == 'r':
	# 	extra_cat = [ home + 'source_detect_cat/random_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
	# 				  home + 'source_detect_cat/random_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

	# 	extra_img = [ home + 'redMap_random/rand_img-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
	# 				  home + 'redMap_random/rand_img-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

	# if band_str == 'g':
	# 	extra_cat = [ home + 'source_detect_cat/random_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
	# 				  home + 'source_detect_cat/random_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

	# 	extra_img = [ home + 'redMap_random/rand_img-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
	# 				  home + 'redMap_random/rand_img-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

	# if band_str == 'i':
	# 	extra_cat = [ home + 'source_detect_cat/random_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
	# 				  home + 'source_detect_cat/random_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

	# 	extra_img = [ home + 'redMap_random/rand_img-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
	# 				  home + 'redMap_random/rand_img-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

	# adjust_mask_func( d_file, cat_file, set_z, set_ra, set_dec, band_str, gal_file, out_file, bcg_mask,
	# 	offset_file = offset_file, extra_cat = extra_cat, extra_img = extra_img,)

	#.. pixel resample
	mask_file = home + 'bcg_mask_img/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	resamp_file = home + 'bcg_mask_img/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	resamp_func( mask_file, set_z, set_ra, set_dec, set_imgx, set_imgy, band_str, resamp_file, z_ref,
		stack_info = None, pixel = 0.396, id_dimm = True,)

	print( '%s, %d done !' % (band_str, rank) )

