import time
import h5py
import numpy as np
import astropy.io.fits as fits

import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.wcs as awc
import astropy.io.ascii as asc

from astropy.coordinates import SkyCoord
from astropy import cosmology as apcy

from fig_out_module import arr_jack_func

from light_measure import light_measure_weit
from img_pre_selection import WCS_to_pixel_func

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

### === ### cluster imgs
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'


id_cen = 0
n_rbins = 55
N_bin = 30

band_str = band[ rank ]

##... except catalog
if band_str == 'r':
	out_ra = [ '164.740', '141.265', ]
	out_dec = [ '11.637', '11.376', ]
	out_z = [ '0.298', '0.288', ]

if band_str == 'g':
	out_ra = [ '206.511', '141.265', '236.438', ]
	out_dec = [ '38.731', '11.376', '1.767', ]
	out_z = [ '0.295', '0.288', '0.272', ]

d_file = home + '/photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

"""
#..fixed richness mass sub-samples
# cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

#..fixed BCG stellar mass sub-sammples
cat_lis = [ 'low-rich', 'hi-rich' ]


##... map BCG position with peak-offset correction
for ll in range( 2 ):

	#. BCG position at z_ref 
	ref_dat = pds.read_csv( load + 'Extend_Mbcg_cat/' + 
							'high_BCG_star-Mass_%s-band_photo-z-match_pk-offset_cat_z-ref.csv' % band_str )

	ref_ra_0, ref_dec_0, ref_z_0 = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] ), np.array( ref_dat['z'] )
	ref_bcgx_0, ref_bcgy_0 = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )


	ref_dat = pds.read_csv( load + 'Extend_Mbcg_cat/' + 
							'low_BCG_star-Mass_%s-band_photo-z-match_pk-offset_cat_z-ref.csv' % band_str )

	ref_ra_1, ref_dec_1, ref_z_1 = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] ), np.array( ref_dat['z'] )
	ref_bcgx_1, ref_bcgy_1 = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )  


	ref_ra = np.r_[ ref_ra_0, ref_ra_1 ]
	ref_dec = np.r_[ ref_dec_0, ref_dec_1 ]
	ref_z = np.r_[ ref_z_0, ref_z_1 ]
	ref_bcgx = np.r_[ ref_bcgx_0, ref_bcgx_1 ]
	ref_bcgy = np.r_[ ref_bcgy_0, ref_bcgy_1 ]

	ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg,)


	#. catalog use to stack
	# dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_BCG_cat.csv' % (cat_lis[ll], band_str),)
	dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[ll], band_str),)

	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	sub_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg,)

	idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	print('matched Ng = ', np.sum(id_lim) )

	mp_ra, mp_dec, mp_z = ra[ id_lim ], dec[ id_lim ], z[ id_lim ]
	mp_bcg_x, mp_bcg_y = ref_bcgx[ idx[ id_lim ] ], ref_bcgy[ idx[ id_lim ] ]

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ mp_ra, mp_dec, mp_z, mp_bcg_x, mp_bcg_y ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[ll], band_str),)

commd.Barrier()


##... stacking of subsamples
for ll in range( 2 ):

	## BCG position offset
	dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[ll], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		ra, dec, z = ra[order_lis], dec[order_lis], z[order_lis]
		clus_x, clus_y = clus_x[order_lis], clus_y[order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	# XXX
	sub_img = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img_pk-off.h5'
	sub_pix_cont = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont_pk-off.h5'
	sub_sb = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro_pk-off.h5'
	# XXX

	J_sub_img = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref_pk-off.h5'
	J_sub_pix_cont = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_pk-off.h5'
	J_sub_sb = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref_pk-off.h5'

	jack_SB_file = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref_pk-off.h5'
	jack_img = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref_pk-off.h5'
	jack_cont_arr = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref_pk-off.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

	print('%d, %s band finished !' % (ll, band_str) )

raise
"""

##... stacking of entire cluster sample
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

lo_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str),)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

if band_str != 'i':
	ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str)
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

	lo_ra, lo_dec, lo_z = lo_ra[order_lis], lo_dec[order_lis], lo_z[order_lis]
	lo_imgx, lo_imgy = lo_imgx[order_lis], lo_imgy[order_lis]

hi_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str),)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

if band_str != 'i':
	ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str)
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

	hi_ra, hi_dec, hi_z = hi_ra[order_lis], hi_dec[order_lis], hi_z[order_lis]
	hi_imgx, hi_imgy = hi_imgx[order_lis], hi_imgy[order_lis]

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]

print('N_sample = ', len(ra),)
print('band = %s' % band_str,)
print('rank is %d' % rank )

# XXX
sub_img = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_sub-%d_img.h5'
sub_pix_cont = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_sub-%d_pix-cont.h5'
sub_sb = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_sub-%d_SB-pro.h5'
# XXX

J_sub_img = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
J_sub_sb = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

jack_SB_file = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
jack_img = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
jack_cont_arr = '/home/xkchen/fig_tmp/Extend_Mbcg_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img, sub_pix_cont, 
				sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr, 
				id_cut = True, N_edg = 1, id_Z0 = False, z_ref = z_ref, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

