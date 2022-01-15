import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from astropy import cosmology as apcy
from img_mask import source_detect_func, mask_func
from img_mask_adjust import adjust_mask_func
from img_resample import resamp_func

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


### random imgs
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
'''
# source detected
for kk in range( 3 ):

	dat = pds.read_csv( home + 'selection/rand_%s_band_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	out_file0 = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

	source_detect_func(d_file, z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], band[kk], out_file0,)

commd.Barrier()
print('finished random imgs')
'''

'''
# source masking
for kk in range( 3 ):

	dat = pds.read_csv( home + 'selection/rand_%s_band_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	cat_file = home + 'corrected_star_cat/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

	out_file0 = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	out_file1 = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	bcg_mask = 1

	mask_func(d_file, cat_file, z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], 
		band[kk], out_file0, out_file1, bcg_mask, stack_info = None, pixel = 0.396, source_det = False,)
'''


# pixel resampling
for kk in range( 3 ):
	dat = pds.read_csv('/home/xkchen/random_tot-%s-band_norm-img_cat.csv' % band[ kk ],)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	out_file = home + 'tmp_stack/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	resamp_func(d_file, z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], clus_x[N_sub0:N_sub1], clus_y[N_sub0:N_sub1],
		band[ kk ], out_file, z_ref, stack_info = None, pixel = 0.396, id_dimm = True,)

	print('%d rank finished !' % rank)

print('finished random imgs')

raise

### cluster imgs
'''
# source detected
for kk in range( 3 ):
	dat = pds.read_csv( home + 'selection/%s_band_sky_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	out_file0 = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

	source_detect_func(d_file, z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], band[kk], out_file0,)	

print('finished cluster imgs')
'''

'''
# source masking
for kk in range( 3 ):

	dat = pds.read_csv( home + 'selection/%s_band_sky_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	cat_file = home + 'corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

	out_file0 = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	out_file1 = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	bcg_mask = 1 # BCG will be masked

	mask_func(d_file, cat_file, z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], 
		band[kk], out_file0, out_file1, bcg_mask, stack_info = None, pixel = 0.396, source_det = False,)	

print('finished cluster imgs')
'''

'''
# source masking but not applied on BCGs
for kk in range( 3 ):

	dat = pds.read_csv( home + 'selection/%s_band_sky_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	cat_file = home + 'corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

	gal_file = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	out_file = home + 'mask_imgs/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	bcg_photo_file = home + 'BCG_photometric/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'

	bcg_mask = 0

	adjust_mask_func(d_file, cat_file, z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], band[ kk ], 
		gal_file, out_file, bcg_mask, bcg_photo_file = bcg_photo_file,)
	#	extra_cat = None, alter_fac = None, alt_bright_R = None, alt_G_size = None, stack_info = None, pixel = 0.396)

print('masking finished')
'''


# pixel resampling
for kk in range( 3 ):

	dat = pds.read_csv( load + 'img_cat_2_28/cluster_tot-%s-band_norm-img_cat.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'mask_imgs/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	out_file = home + 'pix_resamp_imgs/z_ref_0.25/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	resamp_func(d_file, z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], clus_x[N_sub0:N_sub1], clus_y[N_sub0:N_sub1],
		band[ kk ], out_file, z_ref, stack_info = None, pixel = 0.396, id_dimm = True,)

	print('%d rank finished !' % rank)

print('finished cluster imgs')

