import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

from scipy import interpolate as interp
from scipy import integrate as integ

#.
from img_sat_BG_extract_tmp import origin_img_cut_func
from img_sat_BG_extract_tmp import zref_img_cut_func

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()



###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
z_ref = 0.25
pixel = 0.396
a_ref = 1 / (z_ref + 1)



### ========= ### satellite location at z_ref (images after pixel resampling)
"""
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/zref_imgcut_check/cat/'
shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/'

###... catalog for BG_img cut

##. cluster (30 < rich < 50)
R_bins = np.array( [ 0, 300, 400, 550, 5000] )
bin_rich = [ 20, 30, 50, 210 ]

R_cut = 320

N_shufl = 20

##. use r-band, 30-50 richness, R_sat~(30, 400, 550) as test
for dd in range( 1 ):

	band_str = band[ dd ]

	for kk in range( 3 ):

		for tt in range( N_shufl ):

			cat = pds.read_csv( shufl_path + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, tt),)

			clus_ID = np.array( cat['orin_cID'] )
			clus_ID = clus_ID.astype( int )

			bcg_ra, bcg_dec, bcg_z = np.array( cat['bcg_ra'] ), np.array( cat['bcg_dec'] ), np.array( cat['bcg_z'] )

			sat_ra, sat_dec = np.array( cat['sat_ra'] ), np.array( cat['sat_dec'] )

			Rsat_phy = np.array( cat['orin_Rsat_phy'] )

			##. satellite shuffle information
			cp_sx, cp_sy, cp_PA = np.array( cat['cp_sx'] ), np.array( cat['cp_sy'] ), np.array( cat['cp_PA'] )
			cp_Rpix, cp_Rsat_phy = np.array( cat['cp_Rpix'] ), np.array( cat['cp_Rsat_phy'] )

			is_symP = np.array( cat['is_symP'] )

			cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array( cat['cp_bcg_ra'] ), np.array( cat['cp_bcg_dec'] ), np.array( cat['cp_bcg_z'] )

			cp_clus_ID = np.array( cat['shufl_cID'] )


			##. refer to information at z_ref
			R_cut = 320

			Da_z = Test_model.angular_diameter_distance( cp_bcg_z ).value
			Da_ref = Test_model.angular_diameter_distance( z_ref ).value

			L_ref = Da_ref * pixel / rad2arcsec
			L_z = Da_z * pixel / rad2arcsec
			eta = L_ref / L_z

			ref_sx = cp_sx / eta
			ref_sy = cp_sy / eta

			ref_R_cut = R_cut / eta
			ref_R_pix = cp_Rpix / eta

			##. 
			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_cID', 'orin_Rsat_phy', 
					 'shufl_cID', 'cp_sx', 'cp_sy', 'cp_PA', 'cp_Rpix', 'cp_Rsat_phy', 'is_symP',
					 'cp_bcg_ra', 'cp_bcg_dec', 'cp_bcg_z', 'cut_size']

			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, clus_ID, Rsat_phy, 
					cp_clus_ID, ref_sx, ref_sy, cp_PA, ref_R_pix, cp_Rsat_phy, is_symP,
					cp_bcg_ra, cp_bcg_dec, cp_bcg_z, ref_R_cut ]

			fill = dict( zip( keys, values ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, tt),)

raise
"""


### ========= ### zref_img (image after pixel resampling) cut
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/project/tmp_obj_cat/'

##. cluster image without BCG
# img_file = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/nobcg_resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

##. cluster image with BCG
img_file = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

##.
out_file = ('/home/xkchen/data/SDSS/member_files/shufl_img_wBCG/resamp_img/' + 
				'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]


##. cluster (30 < rich < 50)
R_bins = np.array( [ 0, 300, 400, 550, 5000] )
bin_rich = [ 20, 30, 50, 210 ]

R_cut = 320

N_shufl = 20

##. use r-band, 30-50 richness, R_sat~(30, 400, 550) as test
for dd in range( 1 ):

	band_str = band[ dd ]

	for kk in range( 3 ):

		for tt in range( 13, 14 ):

			##. shuffle table
			rand_cat = pds.read_csv( out_path + 
						'clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv' % 
						(bin_rich[kk], bin_rich[kk + 1], band_str, tt),)

			bcg_ra, bcg_dec, bcg_z = np.array( rand_cat['bcg_ra'] ), np.array( rand_cat['bcg_dec'] ), np.array( rand_cat['bcg_z'] )
			sat_ra, sat_dec = np.array( rand_cat['sat_ra'] ), np.array( rand_cat['sat_dec'] )

			shufl_sx, shufl_sy = np.array( rand_cat['cp_sx'] ), np.array( rand_cat['cp_sy'] )

			set_IDs = np.array( rand_cat['orin_cID'] )
			rand_IDs = np.array( rand_cat['shufl_cID'] )

			set_IDs = set_IDs.astype( int )
			rand_mp_IDs = rand_IDs.astype( int )

			R_cut_pix = np.array( rand_cat['cut_size'] ) 


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
			sub_R_cut = R_cut_pix[N_sub0 : N_sub1]


			#. clust cat_file
			clust_cat_file = cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[kk], bin_rich[kk + 1])

			zref_img_cut_func( clust_cat_file, img_file, band_str, sub_clusID, sub_rand_mp_ID, sub_bcg_ra, sub_bcg_dec, sub_bcg_z, 
								sub_sat_ra, sub_sat_dec, sub_cp_sx, sub_cp_sy, sub_R_cut, pixel, out_file )

print('%d-rank, cut Done!' % rank )

