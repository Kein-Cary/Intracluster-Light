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


### === satellites match
def simple_match(ra_lis, dec_lis, tt_ra, tt_dec, id_choose = False):
	"""
	ra_lis, dec_lis: target satellite information~( rule out or select)

	"""

	dd_ra, dd_dec = [], []
	order_lis = []

	for kk in range( len(tt_ra) ):
		identi = ('%.4f' % tt_ra[kk] in ra_lis) * ('%.4f' % tt_dec[kk] in dec_lis)

		if id_choose == True:
			if identi == True:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				order_lis.append( kk )

			else:
				continue
		else:
			if identi == True:
				continue
			else:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				order_lis.append( kk )

	dd_ra = np.array( dd_ra)
	dd_dec = np.array( dd_dec)
	order_lis = np.array( order_lis )

	return order_lis


###********************************### image stacking
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

cat_path = load + 'Extend_Mbcg_richbin_sat_cat/'


id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

##. Background
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

# N_bin = 100
N_bin = 50


list_order = 0   ###. in range [0, 20)
print( 'list_order = ', list_order )


#### ==== #### entire samples

###... cut_BG_img with BCG
d_file = ( home + 'member_files/rich_binned_shufl_img/resamp_img/' + 
			'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]

mask_file = ( home + 'member_files/rich_binned_sat_test/resamp_img/' + 
			'Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits',)[0]

out_path = '/home/xkchen/figs/'

for ll in range( 3 ):

	for kk in range( 2,3 ):

		band_str = band[ kk ]

		##. satellite catalog
		dat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos-zref.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		print('N_sample = ', len( bcg_ra ) )


		##. N_g for weight
		pat = pds.read_csv('/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/shufl_cat/' + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_shufl-sat-Ng.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str, list_order),)

		orin_Ng = np.array( pat['orin_Ng'] )
		pos_Ng = np.array( pat['shufl_Ng'] )

		weit_Ng = pos_Ng / orin_Ng


		# XXX
		sub_img = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_img.h5'
		sub_pix_cont = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_pix-cont.h5'
		sub_sb = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_SB-pro.h5'
		# XXX

		J_sub_img = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5'
		J_sub_sb = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5'

		jack_SB_file = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5'
		jack_img = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_img_z-ref.h5'
		jack_cont_arr = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_pix-cont_z-ref.h5'

		sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
							sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
							rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, weit_img = mask_file )

		# sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
		# 					sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		# 					rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, weit_img = mask_file, 
		# 					Ng_weit = weit_Ng )

print('%d-rank, Done' % rank )

raise


###... cut_BG_img without BCG
out_path = '/home/xkchen/project/tmp_obj_cat/'

d_file = '/home/xkchen/project/tmp/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'
mask_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'


for ll in range( 1,2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##. satellite catalog
		dat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos-zref.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		print('N_sample = ', len( bcg_ra ) )


		##. N_g for weight
		pat = pds.read_csv('/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/shufl_cat/' + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_shufl-sat-Ng.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str, list_order),)

		orin_Ng = np.array( pat['orin_Ng'] )
		pos_Ng = np.array( pat['shufl_Ng'] )

		weit_Ng = pos_Ng / orin_Ng


		# XXX
		sub_img = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_img.h5'
		sub_pix_cont = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_pix-cont.h5'
		sub_sb = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_SB-pro.h5'
		# XXX

		J_sub_img = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5'
		J_sub_sb = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5'

		jack_SB_file = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5'
		jack_img = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_img_z-ref.h5'
		jack_cont_arr = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_pix-cont_z-ref.h5'

		sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
							sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
							rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, weit_img = mask_file, 
							Ng_weit = weit_Ng )

print('%d-rank, Done' % rank )


raise
