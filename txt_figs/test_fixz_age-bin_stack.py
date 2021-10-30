import time
import h5py
import numpy as np

import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.wcs as awc
import astropy.io.ascii as asc

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from fig_out_module import arr_jack_func
from fig_out_module import color_func
from light_measure import light_measure_weit
from img_pre_selection import WCS_to_pixel_func

from fig_out_module import zref_BCG_pos_func
from light_measure import flux_recal
from resample_modelu import sum_samp, down_samp

from img_jack_stack import jack_main_func
from img_jack_stack import SB_pros_func

from img_sky_jack_stack import sky_jack_main_func
from img_jack_stack import aveg_stack_img
from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

###.cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

L_wave = np.array( [ 6166, 4686, 7480 ] )
Mag_sun = [ 4.65, 5.11, 4.53 ]


### === ### cluster imgs
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

### stacking test
def simple_match(ra_lis, dec_lis, z_lis, ref_file, id_choose = False,):

	ref_dat = pds.read_csv( ref_file )
	tt_ra, tt_dec, tt_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)

	dd_ra, dd_dec, dd_z = [], [], []
	order_lis = []

	for kk in range( len(tt_z) ):
		identi = ('%.3f' % tt_ra[kk] in ra_lis) * ('%.3f' % tt_dec[kk] in dec_lis) # * ('%.3f' % tt_z[kk] in z_lis)

		if id_choose == True:
			if identi == True:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

			else:
				continue
		else:
			if identi == True:
				continue
			else:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

	dd_ra = np.array( dd_ra)
	dd_dec = np.array( dd_dec)
	dd_z = np.array( dd_z)
	order_lis = np.array( order_lis )

	return order_lis

id_cen = 0
n_rbins = 55
N_bin = 30

band_str = band[ rank ]

if band_str == 'r':
	out_ra = [ '164.740', '141.265', ]
	out_dec = [ '11.637', '11.376', ]
	out_z = [ '0.298', '0.288', ]

if band_str == 'g':
	out_ra = [ '206.511', '141.265', '236.438', ]
	out_dec = [ '38.731', '11.376', '1.767', ]
	out_z = [ '0.295', '0.288', '0.272', ]

d_file = home + '/photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'


#..fixed BCG stellar mass sub-sammples
cat_lis = [ 'low-age', 'hi-age' ]

"""
#. match the BCG position at z_ref ( with peak-position offset corrected )
for ll in range( 2 ):

	#. BCG position at z_ref 
	ref_dat = pds.read_csv( load + 'pkoffset_cat/' + 
							'low_BCG_star-Mass_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % band_str )

	ref_ra_0, ref_dec_0, ref_z_0 = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] ), np.array( ref_dat['z'] )
	ref_bcgx_0, ref_bcgy_0 = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )


	ref_dat = pds.read_csv( load + 'pkoffset_cat/' + 
							'high_BCG_star-Mass_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % band_str )

	ref_ra_1, ref_dec_1, ref_z_1 = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] ), np.array( ref_dat['z'] )
	ref_bcgx_1, ref_bcgy_1 = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )  


	ref_ra = np.r_[ ref_ra_0, ref_ra_1 ]
	ref_dec = np.r_[ ref_dec_0, ref_dec_1 ]
	ref_z = np.r_[ ref_z_0, ref_z_1 ]
	ref_bcgx = np.r_[ ref_bcgx_0, ref_bcgx_1 ]
	ref_bcgy = np.r_[ ref_bcgy_0, ref_bcgy_1 ]

	ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg,)


	#. lg_M20 divided samples
	dat = pds.read_csv( load + 'fixz_age_bin_cat/%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ll] )
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	sub_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg,)

	idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	print('N_samp = ', len(ra) )
	print('matched Ng = ', np.sum(id_lim) )

	mp_ra, mp_dec, mp_z = ra[ id_lim ], dec[ id_lim ], z[ id_lim ]
	mp_bcg_x, mp_bcg_y = ref_bcgx[ idx[ id_lim ] ], ref_bcgy[ idx[ id_lim ] ]

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ mp_ra, mp_dec, mp_z, mp_bcg_x, mp_bcg_y ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'fixz_age_bin_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str),)

commd.Barrier()


for ll in range( 2 ):

	## BCG position offset
	dat = pds.read_csv( load + 'fixz_age_bin_cat/' + 
						'%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'fixz_age_bin_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		ra, dec, z = ra[order_lis], dec[order_lis], z[order_lis]
		clus_x, clus_y = clus_x[order_lis], clus_y[order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	# XXX
	sub_img = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img_pk-off.h5'
	sub_pix_cont = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont_pk-off.h5'
	sub_sb = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro_pk-off.h5'
	# XXX

	J_sub_img = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref_pk-off.h5'
	J_sub_pix_cont = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_pk-off.h5'
	J_sub_sb = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref_pk-off.h5'

	jack_SB_file = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref_pk-off.h5'
	jack_img = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref_pk-off.h5'
	jack_cont_arr = '/home/xkchen/fig_tmp/fixz_age_bin_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref_pk-off.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

	print('%d, %s band finished !' % (ll, band_str) )
"""

### === ### match dust map on BCGs
#. dust map with the recalibration by Schlafly & Finkbeiner (2011)
import sfdmap
E_map = sfdmap.SFDMap('/home/xkchen/module/dust_map/sfddata_maskin')
from extinction_redden import A_wave
Rv = 3.1

for ll in range( 2 ):

	dat = pds.read_csv( load + 'fixz_age_bin_cat/%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ll] )
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	pos_deg = SkyCoord( ra, dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_deg )
	A_v = Rv * p_EBV

	Al_r = A_wave( L_wave[ 0 ], Rv) * A_v
	Al_g = A_wave( L_wave[ 1 ], Rv) * A_v
	Al_i = A_wave( L_wave[ 2 ], Rv) * A_v

	keys = [ 'ra', 'dec', 'z', 'E_bv', 'Al_r', 'Al_g', 'Al_i' ]
	values = [ ra, dec, z, p_EBV, Al_r, Al_g, Al_i ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/%s_BCG_dust_value.csv' % cat_lis[ll] )

