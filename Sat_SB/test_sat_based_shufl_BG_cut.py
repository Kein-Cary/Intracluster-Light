"""
shuffle each satellite individually, and each satellite must have a
shuffle-mapping location
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from img_sat_resamp import resamp_func
from img_sat_resamp import BG_resamp_func
from img_sat_BG_extract_tmp import origin_img_cut_func

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']
rad2arcsec = U.rad.to( U.arcsec )



### === image cut and resampling
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

#. masking
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/shufl_cat/'

R_cut = 320   ##. pixels

N_shufl = 20      ###. shuffle times
list_order = 13   ###. pre-cord (14, )
print( 'list_order = ', list_order )



img_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
out_file = ( home + 'member_files/rich_binned_shufl_img/mask_img/' + 
				'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]

for tt in range( 1 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##. shuffle table
		rand_cat = pds.read_csv( out_path + 
					'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[tt], bin_rich[tt+1], band_str, list_order),)

		bcg_ra, bcg_dec, bcg_z = np.array( rand_cat['bcg_ra'] ), np.array( rand_cat['bcg_dec'] ), np.array( rand_cat['bcg_z'] )
		sat_ra, sat_dec = np.array( rand_cat['sat_ra'] ), np.array( rand_cat['sat_dec'] )

		shufl_sx, shufl_sy = np.array( rand_cat['cp_sx'] ), np.array( rand_cat['cp_sy'] )

		set_IDs = np.array( rand_cat['orin_cID'] )
		rand_IDs = np.array( rand_cat['shufl_cID'] )

		set_IDs = set_IDs.astype( int )		
		rand_mp_IDs = rand_IDs.astype( int )


		#. shuffle cutout images
		N_cc = len( set_IDs )

		m, n = divmod( N_cc, cpus )
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_clusID = set_IDs[N_sub0 : N_sub1]
		sub_rand_mp_ID = rand_mp_IDs[N_sub0 : N_sub1]

		sub_bcg_ra, sub_bcg_dec, sub_bcg_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
		sub_sat_ra, sub_sat_dec = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

		sub_cp_sx, sub_cp_sy = shufl_sx[N_sub0 : N_sub1], shufl_sy[N_sub0 : N_sub1]

		#. clust cat_file
		clust_cat_file = cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1])

		origin_img_cut_func( clust_cat_file, img_file, band_str, sub_clusID, sub_rand_mp_ID, sub_bcg_ra, sub_bcg_dec, sub_bcg_z, 
							sub_sat_ra, sub_sat_dec, sub_cp_sx, sub_cp_sy, R_cut, pixel, out_file )

	print('%s, %d-rank, cut Done!' %(sub_name[tt], rank), )

commd.Barrier()



#. resampling... 
for tt in range( 1 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##. satellite table
		dat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos.csv' % 
				(bin_rich[tt], bin_rich[tt+1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

		pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )


		##. shuffle table~( background information)
		pat = pds.read_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[tt], bin_rich[tt+1], band_str, list_order),)

		p_bcg_ra, p_bcg_dec, p_bcg_z = np.array( pat['bcg_ra'] ), np.array( pat['bcg_dec'] ), np.array( pat['bcg_z'] )
		p_sat_ra, p_sat_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )

		shufl_ra, shufl_dec, shufl_z = np.array( pat['cp_bcg_ra'] ), np.array( pat['cp_bcg_dec'] ), np.array( pat['cp_bcg_z'] )

		p_coord = SkyCoord( ra = p_sat_ra * U.deg, dec = p_sat_dec * U.deg )


		idx, sep, d3d = pre_coord.match_to_catalog_sky( p_coord )
		id_lim = sep.value < 2.7e-4

		z_bg = shufl_z[ idx[ id_lim ] ]


		##.
		_Ns_ = len( sat_ra )

		m, n = divmod( _Ns_, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
		sub_z_bg = z_bg[N_sub0 : N_sub1]

		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]
		img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]


		id_dimm = True
		d_file = home + 'member_files/rich_binned_shufl_img/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
		out_file = home + 'member_files/rich_binned_shufl_img/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'

		BG_resamp_func( d_file, sub_z, sub_ra, sub_dec, ra_set, dec_set, img_x, img_y, band_str, out_file, 
					sub_z_bg, z_ref, id_dimm = id_dimm )

	print( '%d rank, done!' % rank )

raise
