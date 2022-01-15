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

"""
dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
clus_ID = np.array( dat['clust_ID'])

Ns = len( ra )

m, n = divmod( Ns, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

sub_ra, sub_dec, sub_z = ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], z[N_sub0 : N_sub1]
sub_clusID = clus_ID[N_sub0 : N_sub1]

N_clus = len( sub_ra )


dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')
s_ra, s_dec, s_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z_spec'] )
s_host_ID = np.array( dat['clus_ID'] )
s_host_ID = s_host_ID.astype( int )


for tt in range( 3 ):

	band_str = band[ tt ]

	for kk in range( N_clus ):

		ra_g, dec_g, z_g = sub_ra[ kk ], sub_dec[ kk ], sub_z[ kk]

		kk_ID = sub_clusID[ kk ]

		id_vx = s_host_ID == kk_ID
		lim_ra, lim_dec, lim_z = s_ra[ id_vx ], s_dec[ id_vx ], s_z[ id_vx ]

		# R_cut = 2.5  ## scaled case
		R_cut = 320

		d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
		gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
		offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

		##... image mask
		cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'
		out_mask_file = home + 'member_files/mask_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

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

		sate_surround_mask_func(d_file, cat_file, ra_g, dec_g, z_g, lim_ra, lim_dec, lim_z, band_str, gal_file, out_mask_file, R_cut, 
								offset_file = offset_file, extra_cat = extra_cat, extra_img = extra_img, stack_info = stack_cat )

		print( time.time() - tt2 )

raise
"""

"""
### === satellite position records
dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
clus_ID = np.array( dat['clust_ID'])

N_clus = len( ra )


keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'ori_imgx', 'ori_imgy', 'cut_cx', 'cut_cy' ]

N_ks = len( keys )

for kk in range( 3 ):

	band_str = band[ kk ]
	print( band_str )

	ra_g, dec_g, z_g = ra[0], dec[0], z[0]
	dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/' + 'clus_%s-band_ra%.3f_dec%.3f_z%.3f_Sat-cat.csv' % (band_str, ra_g, dec_g, z_g), )

	tmp_array = []

	for ll in range( N_ks ):
		tmp_array.append( np.array( dat[ '%s' % keys[ll] ] ) )


	for pp in range( 1, N_clus ):

		ra_g, dec_g, z_g = ra[ pp ], dec[ pp ], z[ pp ]

		dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/' + 
					'clus_%s-band_ra%.3f_dec%.3f_z%.3f_Sat-cat.csv' % (band_str, ra_g, dec_g, z_g), )

		for ll in range( N_ks ):

			tmp_array[ ll ] = np.r_[ tmp_array[ ll ], np.array( dat[ '%s' % keys[ll] ] ) ]

	fill = dict( zip( keys, tmp_array ) )
	data = pds.DataFrame( fill )
	data.to_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str )

	print( tmp_array[0].shape )

"""


##... image resampling
for tt in range( 3 ):

	band_str = band[ tt ]

	dat = pds.read_csv( load + 'Extend_Mbcg_sat_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )


	d_file = home + 'member_files/mask_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'
	# out_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'
	out_file = '/home/xkchen/project/tmp/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

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

