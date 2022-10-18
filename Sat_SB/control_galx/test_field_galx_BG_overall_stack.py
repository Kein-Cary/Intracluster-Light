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


### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ###

### === ### stacking control galaxy mapping to redMapper 
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

for kk in range( 1 ):

	band_str = band[ kk ]

	##.
	dat = pds.read_csv( cat_path + 'contrl-galx_Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_cat_zref-pos.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	print('N_sample = ', len( bcg_ra ) )	

	# XXX
	sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
			sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
			rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise
"""


### === background stacking
"""
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_stack/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

img_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/BG_resamp_img/'
d_file = img_path + 'ctrl-tract_BG_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

N_bin = 100   ## number of jackknife subsample

for kk in range( 1 ):

	band_str = band[ kk ]

	##.
	dat = pds.read_csv( cat_path + 'contrl-galx_Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_cat_zref-pos.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	print('N_sample = ', len( bcg_ra ) )

	# XXX
	sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_BG' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
			sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
			rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise
"""


### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ###
"""
### === ### mapped to redMapper member but radius binned only
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_stack/'


id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35


img_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/resamp_img/'
d_file = img_path + 'ctrl-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

N_bin = 100   ## number of jackknife subsample

R_str = 'scale'
R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

for kk in range( 1 ):

	band_str = band[ kk ]

	for nn in range( len( R_bins ) - 1 ):

		##.
		dat = pds.read_csv( cat_path + 'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_%.2f-%.2fR200m_%s-band_cat_zref-pos.csv' % 
							(R_bins[nn], R_bins[nn + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		print('N_sample = ', len( bcg_ra ) )

		# XXX
		sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_sub-%d_img.h5'
		sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_sub-%d_pix-cont.h5'
		sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_sub-%d_SB-pro.h5'
		# XXX

		J_sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_jack-sub-%d_pix-cont_z-ref.h5'
		J_sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_jack-sub-%d_SB-pro_z-ref.h5'

		jack_SB_file = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_Mean_jack_SB-pro_z-ref.h5'
		jack_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_Mean_jack_img_z-ref.h5'
		jack_cont_arr = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % (R_bins[nn], R_bins[nn + 1], band_str) + '_Mean_jack_pix-cont_z-ref.h5'

		sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
				sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
				rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise
"""


### === background stacking
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_stack/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

img_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/BG_resamp_img/'
d_file = img_path + 'ctrl-tract_BG_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

N_bin = 100   ## number of jackknife subsample

R_str = 'scale'
R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

for kk in range( 1 ):

	band_str = band[ kk ]

	for nn in range( len( R_bins ) - 1 ):

		##.
		dat = pds.read_csv( cat_path + 'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_%.2f-%.2fR200m_%s-band_cat_zref-pos.csv' % 
							(R_bins[nn], R_bins[nn + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		print('N_sample = ', len( bcg_ra ) )

		# XXX
		sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_sub-%d_img.h5'
		sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_sub-%d_pix-cont.h5'
		sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_sub-%d_SB-pro.h5'
		# XXX

		J_sub_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_jack-sub-%d_pix-cont_z-ref.h5'
		J_sub_sb = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_jack-sub-%d_SB-pro_z-ref.h5'

		jack_SB_file = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_Mean_jack_SB-pro_z-ref.h5'
		jack_img = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_Mean_jack_img_z-ref.h5'
		jack_cont_arr = out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG' % (R_bins[nn], R_bins[nn + 1], band_str) + '_Mean_jack_pix-cont_z-ref.h5'

		sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
				sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
				rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

raise

