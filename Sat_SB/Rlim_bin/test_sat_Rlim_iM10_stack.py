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
from scipy.stats import binned_statistic as binned

from img_sat_jack_stack import jack_main_func
from img_sat_stack import cut_stack_func, stack_func
from img_sat_fast_stack import sat_img_fast_stack_func


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


### ===
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

cat_path = load + 'Extend_Mbcg_sat_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_sat_stack/'

id_cen = 0
N_edg = 1
n_rbins = 35

d_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'


### === divided by centric radius, but make sure the i-Mag within 10kpc is almost the same
N_bin = 100   ## number of jackknife subsample

cat_lis = ['inner', 'middle', 'outer']

for ll in range( 3 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		dat = pds.read_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_%s-band_pos-zref.csv' % (cat_lis[ll], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		print('N_sample = ', len( bcg_ra ) )

		# XXX
		sub_img = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img.h5'
		sub_pix_cont = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
		sub_sb = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
		# XXX

		J_sub_img = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
		J_sub_sb = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

		jack_SB_file = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
		jack_img = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
		jack_cont_arr = out_path + 'Extend_BCGM_gri-common_iMag10-fix_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

		sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
							sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
							rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise


### === divided by centric radius + fixed the i-Mag within 10kpc + red-Sequence

N_bin = 50  ## number of jackknife subsample

cat_lis = ['inner', 'middle', 'outer']

id_redQ = ['below', 'above']

for mm in range( 2 ):

	for ll in range( 3 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			dat = pds.read_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_%s-red_member_%s-band_pos-zref.csv' % (cat_lis[ll], id_redQ[ mm ], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

			print('N_sample = ', len( bcg_ra ) )

			# XXX
			sub_img = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_sub-%d_img.h5'
			sub_pix_cont = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
			sub_sb = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
			# XXX

			J_sub_img = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
			J_sub_pix_cont = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
			J_sub_sb = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

			jack_SB_file = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
			jack_img = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
			jack_cont_arr = out_path + 'Extend_BCGM_gri-common_iMag10-fix_%s_%s' % (cat_lis[ll], id_redQ[mm]) + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

			sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
								sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
								rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise