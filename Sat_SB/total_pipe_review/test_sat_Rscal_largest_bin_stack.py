import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func
#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === ###
cat_path = '/home/xkchen/figs/out_bin_cat/'
out_path = '/home/xkchen/figs/out_bin_stack/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35


### === ### subsample sample stacking
"""
##. sat_img without BCG
img_path = '/home/xkchen/data/SDSS/member_files/sat_woBCG/resamp_imgs/'
d_file = img_path + 'Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

N_bin = 50  ## number of jackknife subsample

bin_rich = [ 20, 30, 50, 210 ]

##. test for inner bins SB compare
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#.
for kk in range( 1 ):

	band_str = band[ kk ]

	##. entire catalog
	dat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem-%s-band_pos-zref.csv' % band_str,)

	bcg_ra = np.array( dat['bcg_ra'] )
	bcg_dec = np.array( dat['bcg_dec'] )
	bcg_z = np.array( dat['bcg_z'] )

	sat_ra = np.array( dat['sat_ra'] )
	sat_dec = np.array( dat['sat_dec'] )

	img_x = np.array( dat['sat_x'] )
	img_y = np.array( dat['sat_y'] )

	d_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	#. Rs binned
	for id_part in range( 2 ):

		pat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_Rs-cut-%d_mem_cat.csv' % id_part,)

		p_ra, p_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )

		p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

		#.
		idx, sep, d3d = p_coord.match_to_catalog_sky( d_coord )
		id_lim = sep.value < 2.7e-4

		dd_bcg_ra, dd_bcg_dec, dd_bcg_z = bcg_ra[idx[ id_lim ] ], bcg_dec[idx[ id_lim ] ], bcg_z[idx[ id_lim ] ]
		dd_sat_ra, dd_sat_dec = sat_ra[idx[ id_lim ] ], sat_dec[idx[ id_lim ] ]
		dd_imgx, dd_imgy = img_x[idx[ id_lim ] ], img_y[idx[ id_lim ] ]

		##.
		print('N_sample = ', len( bcg_ra ) )

		# XXX
		sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_sub-%d_img.h5',)[0]
		sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_sub-%d_pix-cont.h5',)[0]
		sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_sub-%d_SB-pro.h5',)[0]
		# XXX

		J_sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_jack-sub-%d_img_z-ref.h5',)[0]
		J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]
		J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

		jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_Mean_jack_SB-pro_z-ref.h5',)[0]
		jack_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_Mean_jack_img_z-ref.h5',)[0]
		jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half' % (band_str, id_part) + '_Mean_jack_pix-cont_z-ref.h5',)[0]

		sat_img_fast_stack_func( dd_bcg_ra, dd_bcg_dec, dd_bcg_z, dd_sat_ra, dd_sat_dec, dd_imgx, dd_imgy, 
				d_file, band_str, id_cen, N_bin, n_rbins, sub_img, sub_pix_cont, sub_sb, 
				J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
				rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )
raise
"""


### === ### background stacking

##. sat_img without BCG
d_file = '/home/xkchen/figs/imgs/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'

#. medi-rich subsample
list_order = 13

N_bin = 50   ## number of jackknife subsample

bin_rich = [ 20, 30, 50, 210 ]

##. test for inner bins SB compare
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#.
for kk in range( 1 ):

	band_str = band[ kk ]

	##. entire catalog
	dat = pds.read_csv( cat_path + 
		'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem-%s-band_pos-zref.csv' % band_str,)

	bcg_ra = np.array( dat['bcg_ra'] )
	bcg_dec = np.array( dat['bcg_dec'] )
	bcg_z = np.array( dat['bcg_z'] )

	sat_ra = np.array( dat['sat_ra'] )
	sat_dec = np.array( dat['sat_dec'] )

	img_x = np.array( dat['sat_x'] )
	img_y = np.array( dat['sat_y'] )

	d_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	##. Ng counts
	pat = pds.read_csv( cat_path + 
		'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem_%s-band_sat_fixRs-shufl-13_shufl-Ng.csv' % band_str,)

	c_ra, c_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
	
	orin_Ng = np.array( pat['orin_Ng'] )
	pos_Ng = np.array( pat['shufl_Ng'] )

	c_coord = SkyCoord( ra = c_ra * U.deg, dec = c_dec * U.deg )

	#.
	for id_part in range( 2 ):

		#.
		pat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_Rs-cut-%d_mem_cat.csv' % id_part,)

		p_ra, p_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
		p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

		#.
		idx, sep, d3d = p_coord.match_to_catalog_sky( d_coord )
		id_lim = sep.value < 2.7e-4

		dd_bcg_ra, dd_bcg_dec, dd_bcg_z = bcg_ra[idx[ id_lim ] ], bcg_dec[idx[ id_lim ] ], bcg_z[idx[ id_lim ] ]
		dd_sat_ra, dd_sat_dec = sat_ra[idx[ id_lim ] ], sat_dec[idx[ id_lim ] ]
		dd_imgx, dd_imgy = img_x[idx[ id_lim ] ], img_y[idx[ id_lim ] ]

		#.
		idx, sep, d3d = p_coord.match_to_catalog_sky( c_coord )
		id_lim = sep.value < 2.7e-4

		weit_Ng = pos_Ng[ idx[ id_lim ] ] / orin_Ng[ idx[ id_lim ] ]


		# XXX
		sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_sub-%d_img.h5',)[0]

		sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_sub-%d_pix-cont.h5',)[0]

		sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_sub-%d_SB-pro.h5',)[0]

		# XXX
		J_sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_jack-sub-%d_img_z-ref.h5',)[0]

		J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]

		J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

		jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_Mean_jack_SB-pro_z-ref.h5',)[0]

		jack_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_Mean_jack_img_z-ref.h5',)[0]

		jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
					'_%s-band_%d-half_shufl-%d_BG' % (band_str, id_part, list_order) + '_Mean_jack_pix-cont_z-ref.h5',)[0]

		sat_BG_fast_stack_func( dd_bcg_ra, dd_bcg_dec, dd_bcg_z, dd_sat_ra, dd_sat_dec, dd_imgx, dd_imgy, 
				d_file, band_str, id_cen, N_bin, n_rbins, sub_img, sub_pix_cont, sub_sb, 
				J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
				rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, Ng_weit = weit_Ng )

print('%d-rank, Done' % rank )

raise

