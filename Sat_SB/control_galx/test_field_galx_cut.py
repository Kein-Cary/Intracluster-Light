from math import isfinite
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import mechanize
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

#.
from img_sat_extract import sate_surround_mask_func
from img_sat_extract import contrl_galx_surround_mask_func
from img_sat_resamp import contrl_galx_resamp_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25


### === ### data load
home = '/home/xkchen/data/SDSS/'

R_cut = 320

"""
# for kk in range( 3 ):
for kk in range( 1 ):  ##. r-band for test

	band_str = band[ kk ]

	#.
	# keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_objID', 'sat_x', 'sat_y' ]
	dat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
						'random_field-galx_map_%s-band_cat.csv' % band_str )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec, sat_z = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] ), np.array( dat['sat_z'] )

	Ns = len( bcg_ra )

	m, n = divmod( Ns, cpus)

	for mm in range( rank, rank + 1):

		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0: N_sub1], bcg_dec[N_sub0: N_sub1], bcg_z[N_sub0: N_sub1]
		lim_ra, lim_dec, lim_z = sat_ra[N_sub0: N_sub1], sat_dec[N_sub0: N_sub1], sat_z[N_sub0: N_sub1]

		N_pp = len( sub_ra )
		print( '%d-rank, N_pp = ' % rank, N_pp )

		orin_x, orin_y = np.array([]), np.array([])
		cut_sx, cut_sy = np.array([]), np.array([])

		for pp in range( N_pp ):

			p_ra, p_dec, p_z = sub_ra[ pp ], sub_dec[ pp ], sub_z[ pp ]
			ps_ra, ps_dec, ps_z = lim_ra[ pp ], lim_dec[ pp ], lim_z[ pp ]

			##.
			d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

			offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/random_%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

			gal_file = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

			##... image mask
			cat_file = home + 'new_sql_star_cat/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

			out_mask_file = home + 'member_files/redMap_contral_galx/mask_img/ctrl-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

			stack_cat = '/home/xkchen/project/tmp_obj_cat/control_%s-band_ra%.3f_dec%.3f_z%.3f_Sat-cat.csv'

			if band_str == 'r':
				extra_cat = [ home + 'source_detect_cat/random_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'source_detect_cat/random_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'redMap_random/rand_img-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'redMap_random/rand_img-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			if band_str == 'g':
				extra_cat = [ home + 'source_detect_cat/random_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'source_detect_cat/random_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'redMap_random/rand_img-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'redMap_random/rand_img-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			if band_str == 'i':
				extra_cat = [ home + 'source_detect_cat/random_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'source_detect_cat/random_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'redMap_random/rand_img-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'redMap_random/rand_img-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			tmp_ini_x, tmp_ini_y, tmp_cut_x, tmp_cut_y = contrl_galx_surround_mask_func( 
																			d_file, cat_file, p_ra, p_dec, p_z, ps_ra, ps_dec, ps_z, band_str, gal_file, out_mask_file, R_cut,
																			offset_file = offset_file, extra_cat = extra_cat, extra_img = extra_img,)# stack_info = stack_cat )

			#.
			orin_x = np.r_[ orin_x, tmp_ini_x ]
			orin_y = np.r_[ orin_y, tmp_ini_y ]

			cut_sx = np.r_[ cut_sx, tmp_cut_x ]
			cut_sy = np.r_[ cut_sy, tmp_cut_y ]

		#. save
		keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 
					'ori_imgx', 'ori_imgy', 'cut_cx', 'cut_cy' ]

		values = [ sub_ra, sub_dec, sub_z, lim_ra, lim_dec, lim_z, 
					orin_x, orin_y, cut_sx, cut_sy ]

		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv('/home/xkchen/project/tmp_obj_cat/control_%s-band_Sat-cat_%d-rank.csv' % (band_str, rank),)

print( '%s band finished!' % band_str )

raise
"""


"""
##... img_cut information for stacking
N_cpu = 288

band_str = band[ 0 ]

for kk in range( 1 ):

	#.
	dat = pds.read_csv( '/home/xkchen/project/tmp_obj_cat/control_%s-band_Sat-cat_0-rank.csv' % band_str,)

	keys = dat.columns[1:]

	N_ks = len( keys )

	tmp_arr = []

	for dd in range( N_ks ):

		tmp_arr.append( np.array( dat[ keys[ dd ] ] ) )

	#.
	for pp in range( 1, N_cpu ):

		dat = pds.read_csv( '/home/xkchen/project/tmp_obj_cat/control_%s-band_Sat-cat_%d-rank.csv' % (band_str, pp),)

		for dd in range( N_ks ):

			tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], np.array( dat[ keys[ dd ] ] ) ]

	#.
	fill = dict( zip( keys, tmp_arr ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
					'redMap_map_control_galx_%s-band_cut_pos.csv' % band_str,)

raise
"""


##... image resampling

for tt in range( 1 ):

	##.
	band_str = band[ tt ]

	dat = pds.read_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
						'redMap_map_control_galx_%s-band_cut_pos.csv' % band_str, )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

	sat_z = np.array( dat['sat_z'] )

	coord_sat = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	##.
	ref_cat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
						'random_field-galx_map_%s-band_cat_params.csv' % band_str,)

	cp_s_ra, cp_s_dec, cp_s_z = np.array( ref_cat['ra'] ), np.array( ref_cat['dec'] ), np.array( ref_cat['z'] )
	cp_clus_z = np.array( ref_cat['map_clus_z'] )

	cp_coord_sat = SkyCoord( ra = cp_s_ra * U.deg, dec = cp_s_dec * U.deg )

	idx, d2d, d3d = coord_sat.match_to_catalog_sky( cp_coord_sat )
	id_lim = d2d.value < 2.7e-4

	##. use for resampling
	ref_clus_z = cp_clus_z[ idx[id_lim] ]


	##. control galaxy images
	d_file = home + 'member_files/redMap_contral_galx/mask_img/ctrl-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'
	out_file = home + 'member_files/redMap_contral_galx/resamp_img/ctrl-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

	id_dimm = True

	_Ns_ = len( sat_ra )

	m, n = divmod( _Ns_, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	sub_ra, sub_dec = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1]

	ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

	img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]

	sub_z = bcg_z[N_sub0 : N_sub1]

	# z_set = sat_z[N_sub0 : N_sub1]
	z_set = ref_clus_z[N_sub0 : N_sub1]

	contrl_galx_resamp_func( d_file, sub_z, sub_ra, sub_dec, ra_set, dec_set, z_set, img_x, img_y, band_str, out_file, z_ref, id_dimm = id_dimm )

	print( '%s band finished!' % band_str )

raise
