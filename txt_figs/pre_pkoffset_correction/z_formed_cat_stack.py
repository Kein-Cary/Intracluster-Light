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
from light_measure import light_measure_weit
from light_measure import light_measure_rn_weit
from light_measure import jack_SB_func

from img_jack_stack import jack_main_func, zref_lim_SB_adjust_func
from img_jack_stack import SB_pros_func

from img_sky_jack_stack import sky_jack_main_func
from img_jack_stack import aveg_stack_img
from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

### === ### case 1 : divid into two sample, based on age [ lookback time of formed redshift - lookback time of observed redshift ]
# ... [ match image catalogs in g, r, i band respectively ( no-dependence between two bands ) ]

# id_cen = 0
# n_rbins = 90
# N_bin = 30

cat_lis = ['younger', 'older']

band_str = band[ rank ]

if band_str == 'r':
	out_ra = [ '164.740', '141.265', ]
	out_dec = [ '11.637', '11.376', ]
	out_z = [ '0.298', '0.288', ]

if band_str == 'g':
	out_ra = [ '206.511', '141.265', '236.438', ]
	out_dec = [ '38.731', '11.376', '1.767', ]
	out_z = [ '0.295', '0.288', '0.272', ]

'''
for ll in range( 2 ):

	dat = pds.read_csv( load + 'z_formed_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'z_formed_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		ra, dec, z = ra[ order_lis], dec[ order_lis], z[ order_lis]
		clus_x, clus_y = clus_x[ order_lis], clus_y[ order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	d_file = home + 'photo_files/resample_imgs/photo-z_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	# XXX
	sub_img = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = load + 'z_formed_stack/' + 'photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('%s band finished !' % band_str)
'''

'''
### === ### gri band common catalog stacking
for ll in range( 2 ):

	dat = pds.read_csv( load + 'z_formed_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'z_formed_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		ra, dec, z = ra[ order_lis], dec[ order_lis], z[ order_lis]
		clus_x, clus_y = clus_x[ order_lis], clus_y[ order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	d_file = home + 'photo_files/resample_imgs/photo-z_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	# XXX
	sub_img = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('%s band finished !' % band_str)
'''

## remeasure SB for color profile
id_cen = 0
n_rbins = 50 # 53
N_bin = 30

id_Z0 = False

for ll in range( 2 ):

	jk_sub_img = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	jk_sub_cont = load + 'z_formed_stack/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	jk_sub_sb = '/home/xkchen/figs/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'
	jk_mean_sb = '/home/xkchen/figs/' + 'photo-z_match_gri-common_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'

	SB_pros_func( jk_sub_img, jk_sub_cont, jk_sub_sb, N_bin, n_rbins, id_Z0, z_ref)

	tmp_sb = []
	tmp_r = []

	for nn in range( N_bin ):
		with h5py.File( jk_sub_sb % nn, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

		idvx = npix < 1.
		sb_arr[idvx] = np.nan
		r_arr[idvx] = np.nan

		tmp_sb.append(sb_arr)
		tmp_r.append(r_arr)

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func( tmp_sb, tmp_r, band_str, N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File( jk_mean_sb, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

print('%s band finished !' % band_str)

raise


"""
### === ### divid sample based on formation redshift only ( compare with case 1 )
'''
cat_lis = ['low_formed-z', 'high_formed-z']

for ll in range( 2 ):

	dat = pds.read_csv('/home/xkchen/figs/z_form_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	if band_str != 'i':
		ref_file = '/home/xkchen/figs/z_form_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		ra, dec, z = ra[ order_lis], dec[ order_lis], z[ order_lis]
		clus_x, clus_y = clus_x[ order_lis], clus_y[ order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	d_file = home + 'photo_files/resample_imgs/photo-z_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	# XXX
	sub_img = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('%s band finished !' % band_str)
'''
### common catalog of sample (divid sample based on formation redshift only) stacking test
identi_lis = ['low-z-form_young', 'high-z-form_old']

for ll in range( 2 ):

	if ll == 0:
		dat = pds.read_csv( '/home/xkchen/figs/z_form_cat/low_formed-z_and_young-age_%s-band_common_BCG-pos_cat_z-ref.csv' % band_str,)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		ref_file = '/home/xkchen/figs/z_form_cat/low_formed-z_and_young-age_%s-band_common_BCG-pos_cat_z-ref.csv' % band_str

	if ll == 1:
		dat = pds.read_csv( '/home/xkchen/figs/z_form_cat/high_formed-z_and_old-age_%s-band_common_BCG-pos_cat_z-ref.csv' % band_str,)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		ref_file = '/home/xkchen/figs/z_form_cat/high_formed-z_and_old-age_%s-band_common_BCG-pos_cat_z-ref.csv' % band_str

	if band_str != 'i':

		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)
		ra, dec, z = ra[ order_lis], dec[ order_lis], z[ order_lis]
		clus_x, clus_y = clus_x[ order_lis], clus_y[ order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	d_file = home + 'photo_files/resample_imgs/photo-z_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	# XXX
	sub_img = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = '/home/xkchen/figs/z_form_SBs/photo-z_match_' + identi_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('%s band finished !' % band_str)

"""
