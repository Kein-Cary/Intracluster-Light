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


### === ### galaxy image stacking
"""
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_stack/'


id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35


img_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/resamp_img/'
d_file = img_path + 'ctrl-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'


N_bin = 100   ## number of jackknife subsample

bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']


# ##. fixed R for all richness subsample
# R_str = 'phy'
# # R_bins = np.array( [ 0, 300, 400, 550, 5000] )
# R_bins = np.array( [ 0, 150, 300, 400, 550, 5000] )


##. average shuffle test
R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m, for rich + sR bin

#.
for ll in range( 3 ):

	for tt in range( len(R_bins) - 1 ):

		for kk in range( 1 ):

			band_str = band[ kk ]

			##.
			if R_str == 'phy':

				dat = pds.read_csv( cat_path + 
					'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat_zref-pos.csv' % 
					(bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

				bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
				sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
				img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

				print('N_sample = ', len( bcg_ra ) )

				# XXX
				sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_sub-%d_img.h5',)[0]
				sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_sub-%d_pix-cont.h5',)[0]
				sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_sub-%d_SB-pro.h5',)[0]
				# XXX

				J_sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5',)[0]
				J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]
				J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

				jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5',)[0]
				jack_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_Mean_jack_img_z-ref.h5',)[0]
				jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5',)[0]

				sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
						sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
						rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )


			##.
			if R_str == 'scale':

				dat = pds.read_csv( cat_path + 
						'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat_zref-pos.csv' % 
						(bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

				bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
				sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
				img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

				print('N_sample = ', len( bcg_ra ) )

				# XXX
				sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_sub-%d_img.h5',)[0]
				sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_sub-%d_pix-cont.h5',)[0]
				sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_sub-%d_SB-pro.h5',)[0]
				# XXX

				J_sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5',)[0]
				J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]
				J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

				jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5',)[0]
				jack_img = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_Mean_jack_img_z-ref.h5',)[0]
				jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5',)[0]

				sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
						sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
						rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise
"""


### === ### Background stacking
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_stack/'


id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

img_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/BG_resamp_img/'
d_file = img_path + 'ctrl-tract_BG_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'


N_bin = 100   ## number of jackknife subsample

bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']


# ##. fixed R for all richness subsample
# R_str = 'phy'
# # R_bins = np.array( [ 0, 300, 400, 550, 5000] )
# R_bins = np.array( [ 0, 150, 300, 400, 550, 5000] )


##. average shuffle test
R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m, for rich + sR bin


#.
for ll in range( 3 ):

	for tt in range( len(R_bins) - 1 ):

		for kk in range( 1 ):

			band_str = band[ kk ]

			##.
			if R_str == 'phy':

				dat = pds.read_csv( cat_path + 
					'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat_zref-pos.csv' % 
					(bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

				bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
				sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
				img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

				print('N_sample = ', len( bcg_ra ) )

				# XXX
				sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_sub-%d_img.h5',)[0]
				sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_sub-%d_pix-cont.h5',)[0]
				sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_sub-%d_SB-pro.h5',)[0]
				# XXX

				J_sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_jack-sub-%d_img_z-ref.h5',)[0]
				J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]
				J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

				jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_Mean_jack_SB-pro_z-ref.h5',)[0]
				jack_img = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_Mean_jack_img_z-ref.h5',)[0]
				jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_Mean_jack_pix-cont_z-ref.h5',)[0]

				sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
						sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
						rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )


			##.
			if R_str == 'scale':

				dat = pds.read_csv( cat_path + 
						'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat_zref-pos.csv' % 
						(bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

				bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
				sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
				img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

				print('N_sample = ', len( bcg_ra ) )

				# XXX
				sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_sub-%d_img.h5',)[0]
				sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_sub-%d_pix-cont.h5',)[0]
				sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_sub-%d_SB-pro.h5',)[0]
				# XXX

				J_sub_img = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_jack-sub-%d_img_z-ref.h5',)[0]
				J_sub_pix_cont = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5',)[0]
				J_sub_sb = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5',)[0]

				jack_SB_file = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_Mean_jack_SB-pro_z-ref.h5',)[0]
				jack_img = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_Mean_jack_img_z-ref.h5',)[0]
				jack_cont_arr = ( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_BG_' % band_str + '_Mean_jack_pix-cont_z-ref.h5',)[0]

				sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
						sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
						rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise
