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
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

##.
from light_measure import light_measure_weit
from img_sat_resamp import resamp_func
from img_sat_BG_extract import origin_img_cut_func

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func


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


### === satellite images stacking
"""
home = '/home/xkchen/data/SDSS/'

cat_path = '/home/xkchen/T200_test/cat/'
out_path = '/home/xkchen/T200_test/SBs/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

N_bin = 30


##. sat_cut with BCG
img_path = '/home/xkchen/data/SDSS/member_files/rich_binned_sat_test/resamp_img/'
d_file = img_path + 'Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

for kk in range( 3 ):

	band_str = band[ kk ]

	##.
	dat = pds.read_csv( cat_path + 'clust_rich_30-50_%s-band_sat-shufl_T200-test_sat-pos-zref.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	print('N_sample = ', len( bcg_ra ) )


	##. satellite SB
	# XXX
	sub_img = out_path + 'T200-test_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = out_path + 'T200-test_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = out_path + 'T200-test_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = out_path + 'T200-test_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = out_path + 'T200-test_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = out_path + 'T200-test_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = out_path + 'T200-test_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = out_path + 'T200-test_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = out_path + 'T200-test_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	sat_img_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
						sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
						rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

print('%d-rank, Done' % rank )

"""



### === Background images stacking
home = '/home/xkchen/data/SDSS/'

cat_path = '/home/xkchen/T200_test/cat/'
out_path = '/home/xkchen/T200_test/BGs/'

id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

N_bin = 30


##. background image cut with BCG
d_file = ( home + 'member_files/rich_binned_shufl_img/resamp_img/' + 
			'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]

mask_file = ( home + 'member_files/rich_binned_sat_test/resamp_img/' + 
			'Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits',)[0]

for kk in range( 3 ):

	band_str = band[ kk ]

	##. satellite catalog
	dat = pds.read_csv( cat_path + 'clust_rich_30-50_%s-band_sat-shufl_T200-test_sat-pos-zref.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )


	##. N_g for weight
	pat = pds.read_csv( cat_path + 'clust_rich_30-50_%s-band_sat-shufl_T200-test_shufl-Ng.csv' % band_str,)

	orin_Ng = np.array( pat['orin_Ng'] )
	pos_Ng = np.array( pat['shufl_Ng'] )

	weit_Ng = pos_Ng / orin_Ng


	# XXX
	sub_img = out_path + 'T200_test_%s-band_BG' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = out_path + 'T200_test_%s-band_BG' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = out_path + 'T200_test_%s-band_BG' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = out_path + 'T200_test_%s-band_BG' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = out_path + 'T200_test_%s-band_BG' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = out_path + 'T200_test_%s-band_BG' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = out_path + 'T200_test_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = out_path + 'T200_test_%s-band_BG' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = out_path + 'T200_test_%s-band_BG' % band_str + '_Mean_jack_pix-cont_z-ref.h5'


	sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
						sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
						rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False )

	# sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
	# 					sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	# 					rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, weit_img = mask_file )

print('%d-rank, Done' % rank )
