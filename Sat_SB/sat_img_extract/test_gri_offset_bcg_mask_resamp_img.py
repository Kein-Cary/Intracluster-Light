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

from io import StringIO
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

##.
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


### === ### data load
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

img_path = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/nobcg_mask_img/'
out_path = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/nobcg_resamp_img/'


##.
for jj in range( 4 ):

	if jj == 0:
		cat_lis = [ 'low-rich', 'hi-rich' ]

	if jj == 1:
		cat_lis = [ 'low-age', 'hi-age' ]

	if jj == 2:
		cat_lis = [ 'younger', 'older' ]

	if jj == 3:
		cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

	for kk in range( 3 ):

		band_str = band[ kk ]

		for mm in range( 2 ):

			dat = pds.read_csv(load + 'pkoffset_cat/%s_%s-band_no-corrected_yet_pk-offset_cat.csv' % (cat_lis[mm], band_str),)
			ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
			clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)    ###. position have applied offset correction

			zN = len( z )
			# print( zN )

			m, n = divmod(zN, cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n

			set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
			set_imgx, set_imgy = clus_x[N_sub0 : N_sub1], clus_y[N_sub0 : N_sub1]

			#.
			d_file = img_path + 'photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
			out_file = out_path + 'photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

			resamp_func( d_file, set_z, set_ra, set_dec, set_imgx, set_imgy, band_str, out_file, z_ref, stack_info = None, pixel = pixel, id_dimm = True,)

print('%d rank !' % rank,)
# commd.Barrier()


### === ### gri combined masking~( extended sample )
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

for kk in range( 3 ):

	band_str = band[ kk ]

	for mm in range( 2 ):

		##.
		dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_pre-diffi_BCG_cat.csv' % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		sub_coord = SkyCoord( ra * U.deg, dec * U.deg )

		#. match the position of BCGs
		ref_dat = pds.read_csv( load + 
								'Extend_Mbcg_cat/%s_%s-band_photo-z-match_pk-offset_cat.csv' % (cat_lis[mm], band_str),)
		ref_ra, ref_dec = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] )
		ref_bcgx, ref_bcgy = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )

		ref_coord = SkyCoord( ref_ra * U.deg, ref_dec * U.deg )

		idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
		id_lim = sep.value < 2.7e-4

		clus_x, clus_y = ref_bcgx[ idx[ id_lim ] ], ref_bcgy[ idx[ id_lim ] ]   ###. position have applied offset correction

		zN = len( ra )

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
		set_imgx, set_imgy = clus_x[N_sub0 : N_sub1], clus_y[N_sub0 : N_sub1]

		#.
		d_file = img_path + 'photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
		out_file = out_path + 'photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

		resamp_func( d_file, set_z, set_ra, set_dec, set_imgx, set_imgy, band_str, out_file, z_ref, stack_info = None, pixel = pixel, id_dimm = True,)

print('%d rank !' % rank,)

raise
