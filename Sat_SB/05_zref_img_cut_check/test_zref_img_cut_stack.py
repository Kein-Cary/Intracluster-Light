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


### === ### Background image stacking test
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

cat_path = load + 'Extend_Mbcg_rich_rebin_sat_cat/'
out_path = '/home/xkchen/img_zref_cut_test/stack/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35


##. background stacking
img_path = '/home/xkchen/img_zref_cut_test/imgs/'
d_file = img_path + 'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'


#. medi-rich subsample
list_order = 13

N_bin = 100   ## number of jackknife subsample

bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

R_bins = np.array( [ 0, 300, 400, 550, 5000] )


for ll in range( 1,2 ):

	for kk in range( 1 ):

		band_str = band[ kk ]

		for tt in range( len(R_bins) - 1 ):

			dat = pds.read_csv( cat_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem-%s-band_pos-zref.csv'
					% ( bin_rich[ ll ], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

			pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )
			print('N_sample = ', len( bcg_ra ) )


			# ##. read the cut information at z_ref and adjust centeral points
			# cat = pds.read_csv( '/home/xkchen/img_zref_cut_test/cat/' + 
			# 		'clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str, list_order),)

			# cc_ra, cc_dec = np.array( cat['sat_ra'] ), np.array( cat['sat_dec'] )
			# cc_Rcut = np.array( cat['cut_size'] )

			# cc_coord = SkyCoord( ra = cc_ra * U.deg, dec = cc_dec * U.deg )

			# idx, sep, d3d = pre_coord.match_to_catalog_sky( cc_coord )
			# id_lim = sep.value < 2.7e-4

			# mp_eta = 320 / cc_Rcut[ idx[ id_lim ] ]

			# ##. stacking center points
			# mp_cx, mp_cy = cc_Rcut[ idx[ id_lim ] ], cc_Rcut[ idx[ id_lim ] ]
			# img_x, img_y = mp_cx - 1 / mp_eta, mp_cy - 1 / mp_eta


			##. N_g for weight
			pat = pds.read_csv( cat_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_%s-band_sat-shufl-%d_shufl-Ng.csv' % 
					(bin_rich[ ll ], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str, list_order),)


			p_ra, p_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
			p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			idx, sep, d3d = pre_coord.match_to_catalog_sky( p_coord )
			id_lim = sep.value < 2.7e-4

			orin_Ng = np.array( pat['orin_Ng'] )
			pos_Ng = np.array( pat['shufl_Ng'] )

			weit_Ng = pos_Ng[ idx[ id_lim ] ] / orin_Ng[ idx[ id_lim ] ]


			# XXX
			sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_img.h5',)[0]

			sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_pix-cont.h5',)[0]

			sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_SB-pro.h5',)[0]

			# XXX
			J_sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_img_z-ref.h5',)[0]

			J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]

			J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

			jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5',)[0]

			jack_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_img_z-ref.h5',)[0]

			jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
						'_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_pix-cont_z-ref.h5',)[0]

			sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
					sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
					rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, Ng_weit = weit_Ng )

print('%d-rank, Done' % rank )

raise

