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

	##. rand_dex
	rand_dex = np.loadtxt( cat_path + 'random_set_dex.txt')
	rand_dex = rand_dex.astype( int )

	N_rnd = rand_dex.shape[0]

	for dd in range( N_rnd ):

		sub_dex = rand_dex[ dd ]

		sub_bcg_ra, sub_bcg_dec, sub_bcg_z = bcg_ra[ sub_dex ], bcg_dec[ sub_dex ], bcg_z[ sub_dex ]
		sub_sat_ra, sub_sat_dec = sat_ra[ sub_dex ], sat_dec[ sub_dex ]
		sub_img_x, sub_img_y = img_x[ sub_dex ], img_y[ sub_dex ]

		#.
		N_gx = len( sat_ra )

		cut_dex = np.int( N_gx / 2 )

		#. Rs binned
		for id_part in range( 2 ):

			if id_part == 0:
				dd_bcg_ra, dd_bcg_dec, dd_bcg_z = sub_bcg_ra[ :cut_dex ], sub_bcg_dec[ :cut_dex ], sub_bcg_z[ :cut_dex ]
				dd_sat_ra, dd_sat_dec = sub_sat_ra[ :cut_dex ], sub_sat_dec[ :cut_dex ]
				dd_imgx, dd_imgy = sub_img_x[ :cut_dex ], sub_img_y[ :cut_dex ]

			if id_part == 1:
				dd_bcg_ra, dd_bcg_dec, dd_bcg_z = sub_bcg_ra[cut_dex: ], sub_bcg_dec[cut_dex: ], sub_bcg_z[cut_dex: ]
				dd_sat_ra, dd_sat_dec = sub_sat_ra[cut_dex: ], sub_sat_dec[cut_dex: ]
				dd_imgx, dd_imgy = sub_img_x[cut_dex: ], sub_img_y[cut_dex: ]

			##.
			print('N_sample = ', len( bcg_ra ) )

			# XXX
			sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_sub-%d_img.h5',)[0]
			sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_sub-%d_pix-cont.h5',)[0]
			sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_sub-%d_SB-pro.h5',)[0]
			# XXX

			J_sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_jack-sub-%d_img_z-ref.h5',)[0]
			J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]
			J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_Mean_jack_SB-pro_z-ref.h5',)[0]
			jack_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_Mean_jack_img_z-ref.h5',)[0]
			jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d' % (band_str, dd, id_part) + '_Mean_jack_pix-cont_z-ref.h5',)[0]

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


	##. rand_dex
	rand_dex = np.loadtxt( cat_path + 'random_set_dex.txt')
	rand_dex = rand_dex.astype( int )

	N_rnd = rand_dex.shape[0]

	for dd in range( N_rnd ):

		sub_dex = rand_dex[ dd ]

		sub_bcg_ra, sub_bcg_dec, sub_bcg_z = bcg_ra[ sub_dex ], bcg_dec[ sub_dex ], bcg_z[ sub_dex ]
		sub_sat_ra, sub_sat_dec = sat_ra[ sub_dex ], sat_dec[ sub_dex ]
		sub_img_x, sub_img_y = img_x[ sub_dex ], img_y[ sub_dex ]

		sub_orin_N = orin_Ng[ sub_dex ]
		sub_pos_Ng = pos_Ng[ sub_dex ]

		#.
		N_gx = len( sat_ra )

		cut_dex = np.int( N_gx / 2 )

		#. Rs binned
		for id_part in range( 2 ):

			if id_part == 0:
				dd_bcg_ra, dd_bcg_dec, dd_bcg_z = sub_bcg_ra[ :cut_dex ], sub_bcg_dec[ :cut_dex ], sub_bcg_z[ :cut_dex ]
				dd_sat_ra, dd_sat_dec = sub_sat_ra[ :cut_dex ], sub_sat_dec[ :cut_dex ]
				dd_imgx, dd_imgy = sub_img_x[ :cut_dex ], sub_img_y[ :cut_dex ]

				weit_Ng = sub_pos_Ng[ :cut_dex ] / sub_orin_N[ :cut_dex ]

			if id_part == 1:
				dd_bcg_ra, dd_bcg_dec, dd_bcg_z = sub_bcg_ra[cut_dex: ], sub_bcg_dec[cut_dex: ], sub_bcg_z[cut_dex: ]
				dd_sat_ra, dd_sat_dec = sub_sat_ra[cut_dex: ], sub_sat_dec[cut_dex: ]
				dd_imgx, dd_imgy = sub_img_x[cut_dex: ], sub_img_y[cut_dex: ]

				weit_Ng = sub_pos_Ng[cut_dex: ] / sub_orin_N[cut_dex: ]

			# XXX
			sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_sub-%d_img.h5',)[0]

			sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_sub-%d_pix-cont.h5',)[0]

			sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_sub-%d_SB-pro.h5',)[0]

			# XXX
			J_sub_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_jack-sub-%d_img_z-ref.h5',)[0]

			J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]

			J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_Mean_jack_SB-pro_z-ref.h5',)[0]

			jack_img = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_Mean_jack_img_z-ref.h5',)[0]

			jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_all_0.56-1.00R200m' + 
						'_%s-band_rand-%d_half-%d_shufl-%d_BG' % (band_str, dd, id_part, list_order) + '_Mean_jack_pix-cont_z-ref.h5',)[0]

			sat_BG_fast_stack_func( dd_bcg_ra, dd_bcg_dec, dd_bcg_z, dd_sat_ra, dd_sat_dec, dd_imgx, dd_imgy, 
					d_file, band_str, id_cen, N_bin, n_rbins, sub_img, sub_pix_cont, sub_sb, 
					J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
					rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, Ng_weit = weit_Ng )

print('%d-rank, Done' % rank )

raise

