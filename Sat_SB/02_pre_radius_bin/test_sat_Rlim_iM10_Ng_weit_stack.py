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
from img_Ng_weit_stack import jack_main_func


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


##... stacking of entire cluster sample
dat = pds.read_csv( load + 'Extend_Mbcg_cat/photo-z_match_BCGM_%s-band_All_rgi-common_pk-offset_BCG-pos_z-ref.csv' % band_str )
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
clus_x, clus_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] ) 

if band_str != 'i':

	ref_file = load + 'Extend_Mbcg_cat/photo-z_match_BCGM_%s-band_All_rgi-common_pk-offset_BCG-pos_z-ref.csv' % band_str
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file )

	ra, dec, z = ra[ order_lis ], dec[ order_lis ], z[ order_lis ]
	clus_x, clus_y = clus_x[ order_lis ], clus_y[ order_lis ]

c_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )


###... weight with Ng (Pm >= 0.8) of subsamples, radius bin at fixed i_Mag_10
cat_lis = ['inner', 'middle', 'outer']

for mm in range( 3 ):

	pat = pds.read_csv( load + 'Extend_Mbcg_sat_cat/frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_Ng.csv' % cat_lis[ mm ],)

	n_ra, n_dec = np.array( pat['ra'] ), np.array( pat['dec'] )

	#. number with P_mem >= 0.8 cut
	n_Ng = np.array( pat['Ng_80'] )


	n_coord = SkyCoord( ra = n_ra * U.deg, dec = n_dec * U.deg )

	idx, sep, d3d = c_coord.match_to_catalog_sky( n_coord )
	id_lim = sep.value < 2.7e-4

	mp_Ng = n_Ng[ idx[ id_lim ] ]

	mp_ra, mp_dec, mp_z = ra[ id_lim ], dec[ id_lim ], z[ id_lim ]
	mp_imgx, mp_imgy = clus_x[ id_lim ], clus_y[ id_lim ]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)
	print('rank is %d' % rank )


	# XXX
	sub_img = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_sub-%d_img.h5'
	sub_pix_cont = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_sub-%d_pix-cont.h5'
	sub_sb = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = load + 'Extend_Mbcg_sat_stack/' + 'photo-z_match_tot-BCG-star-Mass_%s_%s-band' % (cat_lis[mm], band_str) + '_Mean_jack_pix-cont_z-ref.h5'

	#.. with Ng weight
	jack_main_func( id_cen, N_bin, n_rbins, mp_ra, mp_dec, mp_z, mp_imgx, mp_imgy, d_file, band_str, sub_img, sub_pix_cont, 
					sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr, 
					id_cut = True, N_edg = 1, id_Z0 = False, z_ref = z_ref, id_sub = True, N_weit = mp_Ng )

