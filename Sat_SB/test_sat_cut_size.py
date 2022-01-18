import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle

import numpy as np
import pandas as pds
import h5py

import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from img_sat_extract import sate_Extract_func
from img_sat_extract import sate_surround_mask_func
from img_sat_resamp import resamp_func


import time

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### === image cut
band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'


### === satellites image cut (with larger cutsize, previous is R_cut = 320 pixels)
dat = pds.read_csv(home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

clus_ID = np.array( dat['clus_ID'] )

set_IDs = np.array( list( set( clus_ID ) ) )
set_IDs = set_IDs.astype( int )


Ns = len( set_IDs )

m, n = divmod( Ns, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

sub_clusID = set_IDs[N_sub0 : N_sub1]


N_clus = len( sub_clusID )

##... image cutout
"""
# R_cut = np.int( 320 * 1.5 )  ## wider
R_cut = np.int( 320 / 2.0 )  ## smaller


for tt in range( 3 ):

	band_str = band[ tt ]

	for kk in range( N_clus ):

		kk_ID = sub_clusID[ kk ]

		id_vx = clus_ID == kk_ID

		ra_g, dec_g, z_g = bcg_ra[ id_vx ][0], bcg_dec[ id_vx ][0], bcg_z[ id_vx ][0]
		lim_ra, lim_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]

		print('%d-rank, %d member' % (rank, len(lim_ra) ) )


		d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
		gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
		offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

		##... image mask
		cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'

		# out_mask_file = home + 'member_files/cutsize_test/mask_img/Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_wider.fits'
		out_mask_file = home + 'member_files/cutsize_test/mask_img/Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_small.fits'

		stack_cat = '/home/xkchen/project/tmp_obj_cat/clus_%s-band_ra%.3f_dec%.3f_z%.3f_Sat-cat.csv'

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

		tt2 = time.time()

		sate_surround_mask_func( d_file, cat_file, ra_g, dec_g, z_g, lim_ra, lim_dec, band_str, gal_file, out_mask_file, R_cut, 
								offset_file = offset_file, extra_cat = extra_cat, extra_img = extra_img, stack_info = stack_cat )

		print( time.time() - tt2 )

"""

##... pixel resampling
for ll in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		if ll == 0:
			d_file = home + 'member_files/cutsize_test/mask_img/Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_small.fits'

			out_file = ( home + 'member_files/cutsize_test/resamp_img/' + 
								'Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_small-resamp.fits',)[0]

			dat = pds.read_csv( load + 'Extend_Mbcg_sat_cutsize/pos_cat/' + 
								'frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_member_pos.csv' % band_str )

		if ll == 1:
			d_file = home + 'member_files/cutsize_test/mask_img/Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_wider.fits'

			out_file = ( home + 'member_files/cutsize_test/resamp_img/' + 
								'Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_wider-resamp.fits',)[0]

			dat = pds.read_csv( load + 'Extend_Mbcg_sat_cutsize/pos_cat/' + 
								'frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_member_pos.csv' % band_str )


		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )


		id_dimm = True

		_Ns_ = len( sat_ra )

		m, n = divmod( _Ns_, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]

		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

		img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]

		resamp_func( d_file, sub_z, sub_ra, sub_dec, ra_set, dec_set, img_x, img_y, band_str, out_file, z_ref, id_dimm = id_dimm )

		print( '%s band finished!' % band_str )

