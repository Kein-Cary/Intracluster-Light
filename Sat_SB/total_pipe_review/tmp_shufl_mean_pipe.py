"""
Use to combine satellite background image extraction stacking, including
------------------------------------------------------------------------

1). image extract for each shuffle ~ (PS: shuffle list is build on entire sample)
2). image stacking according to given stacking catalog
3). remove median process data, save the SB profile and jack-knife mean image for each shuffle

------------------------------------------------------------------------
pre-process for above
1). build shuffle list based on over all sample
2). build the stacking catalog of given sample bins

PS: for Background : shuffle align with BCG

"""
import matplotlib as mpl
mpl.use('Agg')
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
import time
import subprocess as subpro

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

##.
from img_sat_BG_extract_tmp import zref_img_cut_func

##.
from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func


### === ### constant
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



### === ### shell variables input~(kdx is loop for 20-shuffle random location)
import sys
kdx = sys.argv

list_order = np.int( kdx[1] )

if rank == 0:
	print( 'shufl-ID = ', list_order )


### === ### image extract
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_rich_rebin_sat_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/shufl_woBCG_wPA/zref_cut_cat/'

img_file = ( '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/nobcg_resamp_img/' + 
			'photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits',)[0]

out_file = ( '/home/xkchen/figs/tt_imgs/' + 
				'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]

#.
bin_rich = [ 20, 30, 50, 210 ]

R_cut = 320

N_shufl = 20

##. r-band only
for dd in range( 1 ):

	band_str = band[ dd ]

	for kk in range( 3 ):

		for tt in range( list_order, list_order + 1 ):

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

			zref_img_cut_func( clust_cat_file, img_file, band_str, sub_clusID, sub_rand_mp_ID, 
							sub_bcg_ra, sub_bcg_dec, sub_bcg_z, sub_sat_ra, sub_sat_dec, 
							sub_cp_sx, sub_cp_sy, sub_R_cut, pixel, out_file )

print('%d-rank, cut Done!' % rank )
commd.Barrier()


### === ### image stacking
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_rich_rebin_sat_cat/'
out_path = '/home/xkchen/figs/tt_stack/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35


##. sat_img without BCG
img_path = '/home/xkchen/figs/tt_imgs/' 
d_file = img_path + 'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'

N_bin = 50   ## number of jackknife subsample

bin_rich = [ 20, 30, 50, 210 ]

##. fixed R for all richness subsample
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#.
for kk in range( 1 ):

	band_str = band[ kk ]

	for tt in range( len(R_bins) - 1 ):

		#.
		bcg_ra, bcg_dec, bcg_z = np.array([]), np.array([]), np.array([])
		sat_ra, sat_dec = np.array([]), np.array([])

		img_x, img_y = np.array([]), np.array([])
		weit_Ng = np.array([])

		#.
		for ll in range( 3 ):

			dat = pds.read_csv( cat_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem-%s-band_pos-zref.csv' 
						% (bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

			bcg_ra = np.r_[ bcg_ra, np.array( dat['bcg_ra'] ) ] 
			bcg_dec = np.r_[ bcg_dec, np.array( dat['bcg_dec'] ) ]
			bcg_z = np.r_[ bcg_z, np.array( dat['bcg_z'] ) ]
			
			sat_ra =  np.r_[ sat_ra, np.array( dat['sat_ra'] ) ]
			sat_dec = np.r_[ sat_dec, np.array( dat['sat_dec'] ) ]

			img_x = np.r_[ img_x, np.array( dat['sat_x'] ) ]
			img_y = np.r_[ img_y, np.array( dat['sat_y'] ) ]

			pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )


			##. N_g for weight
			pat = pds.read_csv( cat_path + 
						'frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_%s-band_wBCG-PA_sat-shufl-%d_shufl-Ng.csv' 
						% ( bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str, list_order),)

			p_ra, p_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
			p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			idx, sep, d3d = pre_coord.match_to_catalog_sky( p_coord )
			id_lim = sep.value < 2.7e-4

			orin_Ng = np.array( pat['orin_Ng'] )
			pos_Ng = np.array( pat['shufl_Ng'] )

			_dd_Ng = pos_Ng[ idx[ id_lim ] ] / orin_Ng[ idx[ id_lim ] ]
			weit_Ng = np.r_[ weit_Ng, _dd_Ng ]

		# XXX
		sub_img = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_img.h5',)[0]

		sub_pix_cont = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_pix-cont.h5',)[0]

		sub_sb = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_SB-pro.h5',)[0]

		# XXX
		J_sub_img = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_img_z-ref.h5',)[0]

		J_sub_pix_cont = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]

		J_sub_sb = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

		jack_SB_file = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5',)[0]

		jack_img = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_img_z-ref.h5',)[0]

		jack_cont_arr = ( out_path + 'Sat-all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
					'_%s-band_wBCG-PA_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_pix-cont_z-ref.h5',)[0]

		sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
				sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
				rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, Ng_weit = weit_Ng )

print('%d-rank, Done' % rank )
commd.Barrier()

if rank == 0:

	#.
	cmd = 'rm -r /home/xkchen/figs/tt_stack/*_sub-*'
	pa = subpro.Popen(cmd, shell = True)
	pa.wait()

	#.
	cmd = 'rm -r /home/xkchen/figs/tt_stack/*jack-sub-*img*'
	pa = subpro.Popen(cmd, shell = True)
	pa.wait()

	#.
	cmd = 'rm -r /home/xkchen/figs/tt_stack/*jack-sub-*pix*'
	pa = subpro.Popen(cmd, shell = True)
	pa.wait()

commd.Barrier()
