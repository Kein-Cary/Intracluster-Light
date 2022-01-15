## test record (2020.10.~2020.11.30)
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

from list_shuffle import find_unique_shuffle_lists as find_list
from img_jack_stack import jack_main_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']

home = '/media/xkchen/My Passport/data/SDSS/'
load = '/media/xkchen/My Passport/data/SDSS/'
#*************************************
## stacking imgs of A&B sub-sample at reference redshift
"""
dat = pds.read_csv('test_1000-to-AB_resamp_BCG-pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

set_ra = ra[:250]
set_dec = dec[:250]
set_z = z[:250]
set_x, set_y = clus_x[:250], clus_y[:250]

id_cen = 0
n_rbins = 100
N_bin = 25

size_arr = np.array([5, 30])

for mm in range(2):
	if mm == 0:
		d_file = home + '20_10_test/pix_resamp/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
	if mm == 1:
		d_file = home + 'tmp_stack/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	sub_img = load + '20_10_test_jack/A_clust_BCG-stack_sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	sub_pix_cont = load + '20_10_test_jack/A_clust_BCG-stack_sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	sub_sb = load + '20_10_test_jack/A_clust_BCG-stack_sub-%d_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	J_sub_img = load + '20_10_test_jack/A_clust_BCG-stack_jack-sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	J_sub_pix_cont = load + '20_10_test_jack/A_clust_BCG-stack_jack-sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	J_sub_sb = load + '20_10_test_jack/A_clust_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	jack_SB_file = load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	jack_img = load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	jack_cont_arr = load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	#jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	#	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	#	id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25,)

	jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_x, set_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25,)

"""

'''
## total-1000
dat = pds.read_csv('T1000_tot_resamp_BCG-pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

id_cen = 0
n_rbins = 100
N_bin = 25

d_file = home + 'tmp_stack/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

sub_img = load + '20_10_test_jack/T1000-tot_BCG-stack_sub-%d_img_z-ref.h5'
sub_pix_cont = load + '20_10_test_jack/T1000-tot_BCG-stack_sub-%d_pix-cont_z-ref.h5'
sub_sb = load + '20_10_test_jack/T1000-tot_BCG-stack_sub-%d_SB-pro_z-ref.h5'

J_sub_img = load + '20_10_test_jack/T1000-tot_BCG-stack_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = load + '20_10_test_jack/T1000-tot_BCG-stack_jack-sub-%d_pix-cont_z-ref.h5'
J_sub_sb = load + '20_10_test_jack/T1000-tot_BCG-stack_jack-sub-%d_SB-pro_z-ref.h5'

jack_SB_file = load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_SB-pro_z-ref.h5'
jack_img = load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_img_z-ref.h5'
jack_cont_arr = load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_pix-cont_z-ref.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25,)
'''

"""
## img select test (selected by A250, Bro-mode select)
dat = pds.read_csv('T1000_Bro-mode-select_resamp_BCG-pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

id_cen = 0
n_rbins = 100
N_bin = 30

size_arr = np.array([5, 10, 20, 30])

for mm in range( 4 ):

	if mm == 3:
		d_file = home + 'tmp_stack/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
	else:
		d_file = home + '20_10_test/pix_resamp/resamp-%s-ra%.3f-dec%.3f-redshift%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])

	sub_img = load + '20_10_test_jack/Bro-mode-select_BCG-stack_sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	sub_pix_cont = load + '20_10_test_jack/Bro-mode-select_BCG-stack_sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	sub_sb = load + '20_10_test_jack/Bro-mode-select_BCG-stack_sub-%d_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	J_sub_img = load + '20_10_test_jack/Bro-mode-select_BCG-stack_jack-sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	J_sub_pix_cont = load + '20_10_test_jack/Bro-mode-select_BCG-stack_jack-sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	J_sub_sb = load + '20_10_test_jack/Bro-mode-select_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	jack_SB_file = load + '20_10_test_jack/Bro-mode-select_BCG-stack_Mean_jack_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	jack_img = load + '20_10_test_jack/Bro-mode-select_BCG-stack_Mean_jack_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	jack_cont_arr = load + '20_10_test_jack/Bro-mode-select_BCG-stack_Mean_jack_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25,)
"""

"""
## other case ( 30*(FWHM/2) only )
file_lis = ['T1000_mean-select_resamp_BCG-pos.csv', 
			'T1000_mode-select_resamp_BCG-pos.csv', 
			'T1000_Bro-mean-select_resamp_BCG-pos.csv', 
			'T1000_Bro-mode-select_resamp_BCG-pos.csv']
pre_lis = ['mean-select', 'mode-select', 'Bro-mean-select', 'Bro-mode-select']

id_cen = 0
n_rbins = 100
N_bin = 30

d_file = home + 'tmp_stack/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
test_path = '/home/xkchen/mywork/ICL/code/SEX/'

for mm in range( 4 ):

	#dat = pds.read_csv( test_path + 'result/select_based_on_A250/' + file_lis[mm] ) ## selected based on A250
	dat = pds.read_csv( test_path + 'result/select_based_on_T1000/' + file_lis[mm] ) ## selected based on T1000

	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	sub_img = load + '20_10_test_jack/BCG-stack_sub-%d_img_z-ref.h5'
	sub_pix_cont = load + '20_10_test_jack/BCG-stack_sub-%d_pix-cont_z-ref.h5'
	sub_sb = load + '20_10_test_jack/BCG-stack_sub-%d_SB-pro_z-ref.h5'	
	'''
	## selected based on A250
	J_sub_img = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_img_30-FWHM-ov2_z-ref.h5'
	J_sub_pix_cont = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2_z-ref.h5'
	J_sub_sb = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_SB-pro_30-FWHM-ov2_z-ref.h5'

	jack_SB_file = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5'
	jack_img = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5'
	jack_cont_arr = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_pix-cont_30-FWHM-ov2_z-ref.h5'
	'''
	## selected based on T1000
	J_sub_img = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_img_30-FWHM-ov2_z-ref_selected-by-tot.h5'
	J_sub_pix_cont = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2_z-ref_selected-by-tot.h5'
	J_sub_sb = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_SB-pro_30-FWHM-ov2_z-ref_selected-by-tot.h5'

	jack_SB_file = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref_selected-by-tot.h5'
	jack_img = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref_selected-by-tot.h5'
	jack_cont_arr = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_pix-cont_30-FWHM-ov2_z-ref_selected-by-tot.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25, 
		id_S2N = True, S2N = 5,)

print('point 1')
"""

#*******************************
### stacking in angle coordinate (test for img selection)
id_cen = 0
n_rbins = 110
N_bin = 30

d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

'''
### total (1000 imgs)
dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000_no_select.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

sub_img = load + '20_10_test_jack/T1000_total_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/T1000_total_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/T1000_total_sub-%d_SB-pro.h5'

J_sub_img = load + '20_10_test_jack/T1000_total_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test_jack/T1000_total_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test_jack/T1000_total_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test_jack/T1000_total_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/T1000_total_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test_jack/T1000_total_Mean_jack_pix-cont.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

'''
### w/o C sample
n_main = np.array([250, 98, 193, 459])

ra, dec, z = np.array([]), np.array([]), np.array([])
img_x, img_y = np.array([]), np.array([])

for mm in (0, 1, 3):

	dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-%d_cat.csv' % n_main[mm],)
	ra = np.r_[ ra, np.array(dat.ra) ]
	dec = np.r_[ dec, np.array(dat.dec) ]
	z = np.r_[ z, np.array(dat.z) ]
	img_x = np.r_[ img_x, np.array(dat.bcg_x) ]
	img_y = np.r_[ img_y, np.array(dat.bcg_y) ]

sub_img = load + '20_10_test_jack/T1000_No-C_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/T1000_No-C_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/T1000_No-C_sub-%d_SB-pro.h5'

J_sub_img = load + '20_10_test_jack/T1000_No-C_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test_jack/T1000_No-C_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test_jack/T1000_No-C_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test_jack/T1000_No-C_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/T1000_No-C_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test_jack/T1000_No-C_Mean_jack_pix-cont.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, img_x, img_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

### w/o D sample
ra, dec, z = np.array([]), np.array([]), np.array([])
img_x, img_y = np.array([]), np.array([])

for mm in (0, 1, 2):

	dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-%d_cat.csv' % n_main[mm],)
	ra = np.r_[ ra, np.array(dat.ra) ]
	dec = np.r_[ dec, np.array(dat.dec) ]
	z = np.r_[ z, np.array(dat.z) ]
	img_x = np.r_[ img_x, np.array(dat.bcg_x) ]
	img_y = np.r_[ img_y, np.array(dat.bcg_y) ]

sub_img = load + '20_10_test_jack/T1000_No-D_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/T1000_No-D_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/T1000_No-D_sub-%d_SB-pro.h5'

J_sub_img = load + '20_10_test_jack/T1000_No-D_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test_jack/T1000_No-D_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test_jack/T1000_No-D_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test_jack/T1000_No-D_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/T1000_No-D_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test_jack/T1000_No-D_Mean_jack_pix-cont.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, img_x, img_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''


### selected imgs (4 sub-sample case)
n_main = np.array([250, 98, 193, 459]) ## for A, B, C, D

pre_lis = ['mean-select', 'mode-select', 'Bro-mean-select', 'Bro-mode-select']

## PS:
## 		mean-select*_cat -- select imgs based on mean of (mu_cen, sigma_cen) for given img sample + 
##							further brighter subpatches
##		mode-select*_cat -- select imgs based on mode point of (mu_cen, sigma_cen) for given img sample + 
##							further brighter subpatches

## 		Bro_mean-select*_cat -- select imgs based on mean of (mu_cen, sigma_cen) for given img sample, 
## 		Bro_mode-select*_cat -- select imgs based on mode point of (mu_cen, sigma_cen) for given img sample, 
"""
id_cen = 0
n_rbins = 110
N_bin = 30

d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

sub_img = load + '20_10_test_jack/T1000_select-test_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/T1000_select-test_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/T1000_select-test_sub-%d_SB-pro.h5'

for kk in range( 4 ):

	ra, dec, z = np.array([]), np.array([]), np.array([])
	clus_x, clus_y = np.array([]), np.array([])

	for mm in range( 4 ):
		'''
		p_dat = pds.read_csv('SEX/result/select_based_on_A250/' + pre_lis[kk] + 
			'_1000-to-%d_remain_cat_4.0-sigma.csv' % n_main[mm], ) ## selected based on A250
		'''
		p_dat = pds.read_csv('SEX/result/select_based_on_T1000/' + pre_lis[kk] + 
			'_1000-to-%d_remain_cat_4.0-sigma.csv' % n_main[mm], ) ## selected based on tot-1000

		ra = np.r_[ ra, np.array(p_dat.ra) ]
		dec = np.r_[ dec, np.array(p_dat.dec) ]
		z = np.r_[ z, np.array(p_dat.z) ]
		clus_x = np.r_[ clus_x, np.array(p_dat.bcg_x) ]
		clus_y = np.r_[ clus_y, np.array(p_dat.bcg_y) ]
	'''
	## selected based on A250
	J_sub_img = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_jack-sub-%d_img.h5'
	J_sub_pix_cont = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_jack-sub-%d_pix-cont.h5'
	J_sub_sb = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_jack-sub-%d_SB-pro.h5'

	jack_SB_file = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_Mean_jack_SB-pro.h5'
	jack_img = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_Mean_jack_img.h5'
	jack_cont_arr = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_Mean_jack_pix-cont.h5'
	'''
	## selected based on tot-1000
	J_sub_img = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_jack-sub-%d_img_selected-by-tot.h5'
	J_sub_pix_cont = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_jack-sub-%d_pix-cont_selected-by-tot.h5'
	J_sub_sb = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_jack-sub-%d_SB-pro_selected-by-tot.h5'

	jack_SB_file = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_Mean_jack_SB-pro_selected-by-tot.h5'
	jack_img = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_Mean_jack_img_selected-by-tot.h5'
	jack_cont_arr = load + '20_10_test_jack/T1000_' + pre_lis[kk] + '_Mean_jack_pix-cont_selected-by-tot.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = False, N_edg = None, id_Z0 = True, z_ref = None, id_S2N = True, S2N = 5,)

print('point 2')
"""

#*******************************
### radius bin test
from light_measure import jack_SB_func
from fig_out_module import cc_grid_img, grid_img
from light_measure_tmp import lim_SB_pros_func, zref_lim_SB_adjust_func

"""
n_rbins = 110
N_bin = 30
SN_lim = 5 #5, 10, 15, 20

img_lis = [ load + '20_10_test_jack/T1000_total_jack-sub-%d_img.h5',
			load + '20_10_test_jack/clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2.h5',
			load + '20_10_test_jack/T1000_mean-select_jack-sub-%d_img.h5',
			load + '20_10_test_jack/T1000_mode-select_jack-sub-%d_img.h5',
			load + '20_10_test_jack/T1000_Bro-mean-select_jack-sub-%d_img.h5',
			load + '20_10_test_jack/T1000_Bro-mode-select_jack-sub-%d_img.h5']

count_lis = [load + '20_10_test_jack/T1000_total_jack-sub-%d_pix-cont.h5',
			load + '20_10_test_jack/clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2.h5',
			load + '20_10_test_jack/T1000_mean-select_jack-sub-%d_pix-cont.h5',
			load + '20_10_test_jack/T1000_mode-select_jack-sub-%d_pix-cont.h5',
			load + '20_10_test_jack/T1000_Bro-mean-select_jack-sub-%d_pix-cont.h5',
			load + '20_10_test_jack/T1000_Bro-mode-select_jack-sub-%d_pix-cont.h5']

pro_lis = [ load + '20_10_test_jack/T1000_total_Mean_jack_SB-pro.h5',
			load + '20_10_test_jack/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5',
			load + '20_10_test_jack/T1000_mean-select_Mean_jack_SB-pro.h5',
			load + '20_10_test_jack/T1000_mode-select_Mean_jack_SB-pro.h5',
			load + '20_10_test_jack/T1000_Bro-mean-select_Mean_jack_SB-pro.h5',
			load + '20_10_test_jack/T1000_Bro-mode-select_Mean_jack_SB-pro.h5']

id_sam = 1

J_sub_img = img_lis[ id_sam ]
J_sub_pix_cont = count_lis[ id_sam ]
jack_SB_file = pro_lis[ id_sam ]

adjust_sub_sb = ['tmp_test/T1000_R-bin_SB_sub-%d.h5',
				'tmp_test/T1000_A250_R-bin_SB_sub-%d.h5',
				'tmp_test/T1000_mean-select_R-bin_SB_sub-%d.h5',
				'tmp_test/T1000_mode-select_R-bin_SB_sub-%d.h5',
				'tmp_test/T1000_Bro-mean-select_R-bin_SB_sub-%d.h5',
				'tmp_test/T1000_Bro-mode-select_R-bin_SB_sub-%d.h5']

adjust_jk_sb = ['tmp_test/T1000_R-bin_SB_test.h5',
			'tmp_test/T1000_A250_R-bin_SB_test.h5',
			'tmp_test/T1000_mean-select_R-bin_SB_test.h5',
			'tmp_test/T1000_mode-select_R-bin_SB_test.h5',
			'tmp_test/T1000_Bro-mean-select_R-bin_SB_test.h5',
			'tmp_test/T1000_Bro-mode-select_R-bin_SB_test.h5']

alter_sub_sb = adjust_sub_sb[ id_sam ]
alter_jk_sb = adjust_jk_sb[ id_sam ]

lim_SB_pros_func(J_sub_img, J_sub_pix_cont, adjust_sub_sb, alter_jk_sb, n_rbins, N_bin, SN_lim, edg_bins = 4,)

with h5py.File(pro_lis[id_sam], 'r') as f:
	pre_r = np.array(f['r'])
	pre_sb = np.array(f['sb'])
	pre_sb_err = np.array(f['sb_err'])

with h5py.File(alter_jk_sb, 'r') as f:
	new_r = np.array(f['r'])
	new_sb = np.array(f['sb'])
	new_sb_err = np.array(f['sb_err'])

	idnul = new_r < 1.
	new_r[ idnul ] = np.nan
	new_sb[ idnul ] = np.nan
	new_sb_err[ idnul ] = np.nan

plt.figure()
ax = plt.subplot(111)

ax.plot(pre_r, pre_sb, ls = '-', color = 'r', alpha = 0.8, label = 'before adjust',)
ax.fill_between(pre_r, y1 = pre_sb - pre_sb_err, y2 = pre_sb + pre_sb_err, color = 'r', alpha = 0.2,)

ax.plot(new_r, new_sb, ls = '--', color = 'g', alpha = 0.8, label = 'after adjust',)
ax.fill_between(new_r, y1 = new_sb - new_sb_err, y2 = new_sb + new_sb_err, color = 'g', alpha = 0.2,)

ax.set_ylim(1e-3, 8e-3)
ax.set_yscale('log')
ax.set_xlim(5e1, 1e3)
ax.set_xlabel('R [arcsec]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('tot-1000_R-bin_test.png', dpi = 300)
plt.close()
"""

'''
alter_jk_sb = [	'test/T1000_mean-select_R-bin_SB_test.h5',
				'test/T1000_mode-select_R-bin_SB_test.h5',
				'test/T1000_Bro-mean-select_R-bin_SB_test.h5',
				'test/T1000_Bro-mode-select_R-bin_SB_test.h5']

name_lis = ['$\\bar{\\sigma}, \\bar{\\mu}$ + brighter sub-patches', 
			'$Mode(\\sigma), Mode(\\mu)$ + brighter sub-patches', 
			'$\\bar{\\sigma}, \\bar{\\mu}$', 
			'$Mode(\\sigma), Mode(\\mu)$']

line_c = ['b', 'g', 'm', 'r',]

with h5py.File('test/T1000_R-bin_SB_test.h5', 'r') as f:
	tot_r = np.array(f['r'])
	tot_sb = np.array(f['sb'])
	tot_sb_err = np.array(f['sb_err'])

with h5py.File('test/T1000_A250_R-bin_SB_test.h5', 'r') as f:
	A250_r = np.array(f['r'])
	A250_sb = np.array(f['sb'])
	A250_sb_err = np.array(f['sb_err'])

pre_lis = ['mean-select', 'mode-select', 'Bro-mean-select', 'Bro-mode-select']

plt.figure()
ax = plt.subplot(111)

for mm in range( 4 ):

	jack_SB_file = load + '20_10_test_jack/T1000_' + pre_lis[mm] + '_Mean_jack_SB-pro_selected-by-tot.h5'
	jack_img = load + '20_10_test_jack/T1000_' + pre_lis[mm] + '_Mean_jack_img_selected-by-tot.h5'

	with h5py.File( jack_img, 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	with h5py.File( jack_SB_file, 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	with h5py.File(alter_jk_sb[mm], 'r') as f:
		pre_r_arr = np.array(f['r'])
		pre_sb_arr = np.array(f['sb'])
		pre_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.set_title( name_lis[mm] )
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'g', alpha = 0.8, label = 'selected by T1000')#label = 'New measurement',)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'g', alpha = 0.2,)

	ax1.plot(pre_r_arr, pre_sb_arr, ls = '--', color = 'r', alpha = 0.8, label = 'selected by A250')#label = 'previous',)
	ax1.fill_between(pre_r_arr, y1 = pre_sb_arr - pre_sb_err, y2 = pre_sb_arr + pre_sb_err, color = 'r', alpha = 0.2,)

	ax1.set_ylim(2e-3, 8e-3)
	ax1.set_yscale('log')
	ax1.set_xlim(5e1, 1e3)
	ax1.set_xlabel('R [arcsec]')
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.set_xscale('log')
	ax1.legend(loc = 3, frameon = False,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax1.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('%s_2D-grd_SB_30-FWHM-ov2.png' % pre_lis[mm], dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', color = line_c[mm], alpha = 0.8, label = name_lis[mm],)
	#ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = line_c[mm], alpha = 0.2,)

	idsb = c_sb_arr[-2]
	devi_sb = c_sb_arr - idsb
	ax.axhline(y = idsb, ls = ':', color = line_c[mm], alpha = 0.5,)
	ax.plot(c_r_arr, devi_sb, ls = '--', color = line_c[mm], alpha = 0.8,)
	#ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = line_c[mm], alpha = 0.2,)

ax.plot(A250_r, A250_sb, ls = '-', color = 'k', alpha = 0.8, label = 'A-250',)
ax.fill_between(A250_r, y1 = A250_sb - A250_sb_err, y2 = A250_sb + A250_sb_err, color = 'k', alpha = 0.2,)

idsb = A250_sb[-1]
devi_sb = A250_sb - idsb

ax.axhline(y = idsb, ls = ':', color = 'k', alpha = 0.5,)
ax.plot(A250_r, devi_sb, ls = '--', color = 'k', alpha = 0.8,)
ax.fill_between(A250_r, y1 = devi_sb - A250_sb_err, y2 = devi_sb + A250_sb_err, color = 'k', alpha = 0.2,)


ax.plot(tot_r, tot_sb, ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000',)
ax.fill_between(tot_r, y1 = tot_sb - tot_sb_err, y2 = tot_sb + tot_sb_err, color = 'c', alpha = 0.2,)

idsb = tot_sb[-2]
devi_sb = tot_sb - idsb

ax.axhline(y = idsb, ls = ':', color = 'c', alpha = 0.5,)
ax.plot(tot_r, devi_sb, ls = '--', color = 'c', alpha = 0.8,)
ax.fill_between(tot_r, y1 = devi_sb - tot_sb_err, y2 = devi_sb + tot_sb_err, color = 'c', alpha = 0.2,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(1e1, 1e3)
ax.set_xlabel('R [arcsec]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('R-bin_test_SB_compare.png', dpi = 300)
plt.close()
'''

#*******************************
### sample selection for reference redshift
n_rbins = 100
N_bin = 25
SN_lim = 5 #5, 10, 15, 20

J_sub_img = load + '20_10_test_jack/T1000-tot_BCG-stack_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = load + '20_10_test_jack/T1000-tot_BCG-stack_jack-sub-%d_pix-cont_z-ref.h5'

adjust_sub_sb = 'tmp_test/T1000-tot_R-bin_SB_sub-%d.h5'
adjust_jk_sb = 'tmp_test/T1000-tot_R-bin_SB_test.h5'

zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, adjust_sub_sb, adjust_jk_sb, n_rbins, N_bin, SN_lim, z_ref = 0.25, edg_bins = 4,)

with h5py.File(load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_SB-pro_z-ref.h5', 'r') as f:
	tot_r = np.array(f['r'])
	tot_sb = np.array(f['sb'])
	tot_sb_err = np.array(f['sb_err'])

with h5py.File(load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_img_z-ref.h5', 'r') as f:
	tot_img = np.array(f['a'])

with h5py.File('tmp_test/T1000-tot_R-bin_SB_test.h5', 'r') as f:
	alt_tot_r = np.array(f['r'])
	alt_tot_sb = np.array(f['sb'])
	alt_tot_sb_err = np.array(f['sb_err'])
	idnn = np.isnan(alt_tot_sb)
	idv = idnn == False
	alt_tot_r, alt_tot_sb_err, alt_tot_sb = alt_tot_r[idv], alt_tot_sb_err[idv], alt_tot_sb[idv]


J_sub_img = load + '20_10_test_jack/A_clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2_z-ref.h5'
J_sub_pix_cont = load + '20_10_test_jack/A_clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2_z-ref.h5'

adjust_sub_sb = 'tmp_test/A250_R-bin_SB_sub-%d.h5'
adjust_jk_sb = 'tmp_test/A250_R-bin_SB_test.h5'

zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, adjust_sub_sb, adjust_jk_sb, n_rbins, N_bin, SN_lim, z_ref = 0.25, edg_bins = 4,)

with h5py.File(load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5', 'r') as f:
	A250_r = np.array(f['r'])
	A250_sb = np.array(f['sb'])
	A250_sb_err = np.array(f['sb_err'])

with h5py.File(load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5', 'r') as f:
	A250_img = np.array(f['a'])

with h5py.File('tmp_test/A250_R-bin_SB_test.h5', 'r') as f:
	alt_A250_r = np.array(f['r'])
	alt_A250_sb = np.array(f['sb'])
	alt_A250_sb_err = np.array(f['sb_err'])
	idnn = np.isnan(alt_A250_sb)
	idv = idnn == False
	alt_A250_r, alt_A250_sb_err, alt_A250_sb = alt_A250_r[idv], alt_A250_sb_err[idv], alt_A250_sb[idv]

raise

n_rbins = 100
N_bin = 30
SN_lim = 5 #5, 10, 15, 20

size_arr = np.array([5, 10, 20, 30])
name_lis = ['5 (FWHM/2)', '10 (FWHM/2)', '20 (FWHM/2)', '30 (FWHM/2)']
line_c = ['b', 'g', 'm', 'r']
'''
alt_SB_pros = []

for kk in range( 4 ):

	J_sub_img = load + '20_10_test_jack/Bro-mode-select_BCG-stack_jack-sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % size_arr[kk]
	J_sub_pix_cont = load + '20_10_test_jack/Bro-mode-select_BCG-stack_jack-sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % size_arr[kk]

	adjust_sub_sb = 'test/Bro-mode-select_R-bin_SB_sub-%d_' + '%d-FWHM-ov2_z-ref.h5' % size_arr[kk]
	adjust_jk_sb = 'test/Bro-mode-select_R-bin_SB_test_%d-FWHM-ov2_z-ref.h5' % size_arr[kk]
	alt_SB_pros.append( adjust_jk_sb )

	zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, adjust_sub_sb, adjust_jk_sb, n_rbins, N_bin, SN_lim, z_ref = 0.25, edg_bins = 4,)

'''

'''
jack_SB_file = load + '20_10_test_jack/Bro-mode-select_BCG-stack_Mean_jack_SB-pro_%d-FWHM-ov2_z-ref.h5'
jack_img = load + '20_10_test_jack/Bro-mode-select_BCG-stack_Mean_jack_img_%d-FWHM-ov2_z-ref.h5'

plt.figure()
ax = plt.subplot(111)

for kk in range(4):

	with h5py.File(jack_img % size_arr[kk], 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	with h5py.File(jack_SB_file % size_arr[kk], 'r') as f:
		pre_r_arr = np.array(f['r'])
		pre_sb_arr = np.array(f['sb'])
		pre_sb_err = np.array(f['sb_err'])

	with h5py.File('test/Bro-mode-select_R-bin_SB_test_%d-FWHM-ov2_z-ref.h5' % size_arr[kk], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.set_title( name_lis[kk] )
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'g', alpha = 0.8, label = 'New measurement',)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'g', alpha = 0.2,)

	ax1.plot(pre_r_arr, pre_sb_arr, ls = '--', color = 'r', alpha = 0.8, label = 'previous',)
	ax1.fill_between(pre_r_arr, y1 = pre_sb_arr - pre_sb_err, y2 = pre_sb_arr + pre_sb_err, color = 'r', alpha = 0.2,)

	ax1.set_ylim(1e-3, 3e-2)
	ax1.set_yscale('log')
	ax1.set_xlim(5e1, 4e3)
	ax1.set_xlabel('R [kpc]')
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.set_xscale('log')
	ax1.legend(loc = 3, frameon = False, fontsize = 8)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax1.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('2D-grd_SB_%d-FWHM-ov2.png' % size_arr[kk], dpi = 300)
	plt.close()


	ax.plot(c_r_arr, c_sb_arr, ls = '-', color = line_c[kk], alpha = 0.8, label = name_lis[kk],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = line_c[kk], alpha = 0.1,)

	idr = c_r_arr > 1e3
	idsb = np.nanmin( c_sb_arr[idr] )
	devi_sb = c_sb_arr - idsb

	ax.axhline(y = idsb, ls = ':', color = line_c[kk], alpha = 0.5,)
	ax.plot(c_r_arr, devi_sb, ls = '--', color = line_c[kk], alpha = 0.8,)
	#ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = line_c[kk], alpha = 0.2,)

ax.plot(alt_A250_r, alt_A250_sb, ls = '-', color = 'k', alpha = 0.8, label = 'A-250 [30(FWHM/2)]',)
ax.fill_between(alt_A250_r, y1 = alt_A250_sb - alt_A250_sb_err, 
	y2 = alt_A250_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)

idr = alt_A250_r > 1e3
idsb = np.nanmin( alt_A250_sb[idr] )
devi_sb = alt_A250_sb - idsb

ax.axhline(y = idsb, ls = ':', color = 'k', alpha = 0.5,)
ax.plot(alt_A250_r, devi_sb, ls = '--', color = 'k', alpha = 0.8,)
ax.fill_between(alt_A250_r, y1 = devi_sb - alt_A250_sb_err, y2 = devi_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)


ax.plot(alt_tot_r, alt_tot_sb, ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000 [30(FWHM/2)]',)
ax.fill_between(alt_tot_r, y1 = alt_tot_sb - alt_tot_sb_err, 
	y2 = alt_tot_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

idr = alt_tot_r > 1e3
idsb = np.nanmin( alt_tot_sb[idr] )
devi_sb = alt_tot_sb - idsb

ax.axhline(y = idsb, ls = ':', color = 'c', alpha = 0.5,)
ax.plot(alt_tot_r, devi_sb, ls = '--', color = 'c', alpha = 0.8,)
ax.fill_between(alt_tot_r, y1 = devi_sb - alt_tot_sb_err, y2 = devi_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(5e1, 4e3)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('Bro-mode-select_z-ref_SB_compare.png', dpi = 300)
plt.close()
'''

### differ sub-samples (30(FWHM/2) case)
pre_lis = ['mean-select', 'mode-select', 'Bro-mean-select', 'Bro-mode-select']

name_lis = ['$\\bar{\\sigma}, \\bar{\\mu}$ + brighter sub-patches', 
			'$Mode(\\sigma), Mode(\\mu)$ + brighter sub-patches', 
			'$\\bar{\\sigma}, \\bar{\\mu}$', 
			'$Mode(\\sigma), Mode(\\mu)$']

id_cen = 0
n_rbins = 100
N_bin = 30
SN_lim = 5
'''
for mm in range( 3 ):

	J_sub_img = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_img_30-FWHM-ov2_z-ref.h5'
	J_sub_pix_cont = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2_z-ref.h5'

	jack_SB_file = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5'
	jack_img = load + '20_10_test_jack/' + pre_lis[mm] + '_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5'

	adjust_sub_sb = 'test/A250_R-bin_SB_sub-%d.h5'
	adjust_jk_sb = 'test/' + pre_lis[mm] + '_R-bin_SB_test_30-FWHM-ov2_z-ref.h5'

	zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, adjust_sub_sb, adjust_jk_sb, n_rbins, N_bin, SN_lim, z_ref = 0.25, edg_bins = 4,)

'''

plt.figure()
ax = plt.subplot(111)

for kk in range(4):
	'''
	## selected by A250
	jack_SB_file = load + '20_10_test_jack/' + pre_lis[kk] + '_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5'
	jack_img = load + '20_10_test_jack/' + pre_lis[kk] + '_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5'
	'''
	## selected by tot-1000
	jack_SB_file = load + '20_10_test_jack/' + pre_lis[kk] + '_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref_selected-by-tot.h5'
	jack_img = load + '20_10_test_jack/' + pre_lis[kk] + '_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref_selected-by-tot.h5'

	with h5py.File( jack_img, 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	with h5py.File( jack_SB_file, 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	with h5py.File('test/' + pre_lis[kk] + '_R-bin_SB_test_30-FWHM-ov2_z-ref.h5', 'r') as f:
		pre_r_arr = np.array(f['r'])
		pre_sb_arr = np.array(f['sb'])
		pre_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.set_title( name_lis[kk] )
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'g', alpha = 0.8, label = 'selected by tot-1000')#label = 'New measurement',)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'g', alpha = 0.2,)

	ax1.plot(pre_r_arr, pre_sb_arr, ls = '--', color = 'r', alpha = 0.8, label = 'selected by A-250')#label = 'previous',)
	ax1.fill_between(pre_r_arr, y1 = pre_sb_arr - pre_sb_err, y2 = pre_sb_arr + pre_sb_err, color = 'r', alpha = 0.2,)

	ax1.set_ylim(1e-3, 3e-2)
	ax1.set_yscale('log')
	ax1.set_xlim(5e1, 4e3)
	ax1.set_xlabel('R [kpc]')
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.set_xscale('log')
	ax1.legend(loc = 3, frameon = False,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax1.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('%s_2D-grd_SB_30-FWHM-ov2.png' % pre_lis[kk], dpi = 300)
	plt.close()


	ax.plot(c_r_arr, c_sb_arr, ls = '-', color = line_c[kk], alpha = 0.8, label = name_lis[kk],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = line_c[kk], alpha = 0.1,)

	idr = c_r_arr > 1e3
	idsb = np.nanmin( c_sb_arr[idr] )
	devi_sb = c_sb_arr - idsb

	ax.axhline(y = idsb, ls = ':', color = line_c[kk], alpha = 0.5,)
	ax.plot(c_r_arr, devi_sb, ls = '--', color = line_c[kk], alpha = 0.8,)
	#ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = line_c[kk], alpha = 0.2,)

ax.plot(alt_A250_r, alt_A250_sb, ls = '-', color = 'k', alpha = 0.8, label = 'A-250 [30(FWHM/2)]',)
ax.fill_between(alt_A250_r, y1 = alt_A250_sb - alt_A250_sb_err, 
	y2 = alt_A250_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)

idr = alt_A250_r > 1e3
idsb = np.nanmin( alt_A250_sb[idr] )
devi_sb = alt_A250_sb - idsb

ax.axhline(y = idsb, ls = ':', color = 'k', alpha = 0.5,)
ax.plot(alt_A250_r, devi_sb, ls = '--', color = 'k', alpha = 0.8,)
ax.fill_between(alt_A250_r, y1 = devi_sb - alt_A250_sb_err, y2 = devi_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)


ax.plot(alt_tot_r, alt_tot_sb, ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000 [30(FWHM/2)]',)
ax.fill_between(alt_tot_r, y1 = alt_tot_sb - alt_tot_sb_err, 
	y2 = alt_tot_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

idr = alt_tot_r > 1e3
idsb = np.nanmin( alt_tot_sb[idr] )
devi_sb = alt_tot_sb - idsb

ax.axhline(y = idsb, ls = ':', color = 'c', alpha = 0.5,)
ax.plot(alt_tot_r, devi_sb, ls = '--', color = 'c', alpha = 0.8,)
ax.fill_between(alt_tot_r, y1 = devi_sb - alt_tot_sb_err, y2 = devi_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(5e1, 4e3)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('total_selected_z-ref_SB_compare.png', dpi = 300)
plt.close()

