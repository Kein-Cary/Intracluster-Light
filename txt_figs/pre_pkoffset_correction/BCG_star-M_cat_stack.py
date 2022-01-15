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
from light_measure import light_measure_Z0_weit
from light_measure import light_measure_rn_Z0_weit
from light_measure import jack_SB_func

from img_jack_stack import jack_main_func, zref_lim_SB_adjust_func
from img_jack_stack import SB_pros_func

from img_sky_jack_stack import sky_jack_main_func
from img_jack_stack import aveg_stack_img
from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func
from fig_out_module import zref_BCG_pos_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i']

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

"""
##### use r band, test err change with jackknife sample
dat = pds.read_csv( load + 'img_cat/com-star-Mass_r-band_BCG-pos_cat_z-ref.csv')
ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

band_str = band[0]

id_cen = 0
n_rbins = 100
N_bin = np.array([ 30, 60, 90, 120])

## test for 90 sub-samples case
rank_id = 2

# shuffle the image order
zN = len( z )
id_arr = np.arange(0, zN, 1)
np.random.shuffle( id_arr )
ra, dec, z = ra[ id_arr], dec[ id_arr], z[ id_arr]
clus_x, clus_y = clus_x[ id_arr], clus_y[ id_arr]


d_file = home + 'pix_resamp_imgs/z_ref_0.25/resamp-%s-ra%.3f-dec%.3f-redshift%.3f' + '30-FWHM-ov2.fits'

# XXX
sub_img = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_sub-%d_img_' + '%d-cpus.h5' % rank_id
sub_pix_cont = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_sub-%d_pix-cont_' + '%d-cpus.h5' % rank_id
sub_sb = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_sub-%d_SB-pro_' + '%d-cpus.h5' % rank_id
# XXX

J_sub_img = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_jack-sub-%d_img_z-ref_' + '%d-cpus.h5' % rank_id
J_sub_pix_cont = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_' + '%d-cpus.h5' % rank_id
J_sub_sb = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref_' + '%d-cpus.h5' % rank_id

jack_SB_file = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref_' + '%d-cpus.h5' % rank_id
jack_img = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_Mean_jack_img_z-ref_' + '%d-cpus.h5' % rank_id
jack_cont_arr = load + 'stack/BCG-star-M_jk-test_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref_' + '%d-cpus.h5' % rank_id

jack_main_func(id_cen, N_bin[ rank_id ], n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)
"""


##### gri-band img satcking (combine low and high BCG star Mass sample), at z-ref
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

band_str = band[ rank ]

lo_dat = pds.read_csv(load + 'img_cat/low_BCG_star-Mass_%s-band_remain_cat_resamp_BCG-pos.csv' % band_str,)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

hi_dat = pds.read_csv(load + 'img_cat/high_BCG_star-Mass_%s-band_remain_cat_resamp_BCG-pos.csv' % band_str,)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]

print('N_sample = ', len(ra),)
print('band = %s' % band_str,)

id_cen = 0
n_rbins = 100
N_bin = 30

'''
d_file = home + 'pix_resamp_imgs/z_ref_0.25/resamp-%s-ra%.3f-dec%.3f-redshift%.3f' + '30-FWHM-ov2.fits'
# XXX
sub_img = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_sub-%d_img.h5'
sub_pix_cont = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_sub-%d_pix-cont.h5'
sub_sb = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_sub-%d_SB-pro.h5'
# XXX

J_sub_img = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_jack-sub-%d_img_z-ref_with-selection.h5'
J_sub_pix_cont = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_with-selection.h5'
J_sub_sb = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'

jack_SB_file = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref_with-selection.h5'
jack_img = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_Mean_jack_img_z-ref_with-selection.h5'
jack_cont_arr = load + 'stack/com-BCG-star-Mass_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref_with-selection.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)
'''


##### gri-band img stacking (divid sample into high / low BCG star Mass sample), at z-ref
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

for ll in range( 2 ):

	band_str = band[ rank ]

	dat = pds.read_csv(load + 'img_cat/%s_%s-band_remain_cat_resamp_BCG-pos.csv' % (cat_lis[ll], band_str),)

	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	id_cen = 0
	n_rbins = 100
	N_bin = 30

	d_file = home + 'pix_resamp_imgs/z_ref_0.25/resamp-%s-ra%.3f-dec%.3f-redshift%.3f' + '30-FWHM-ov2.fits'
	# XXX
	sub_img = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_sub-%d_SB-pro.h5'
	# XXX

	J_sub_img = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref_with-selection.h5'
	J_sub_pix_cont = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_with-selection.h5'
	J_sub_sb = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'

	jack_SB_file = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref_with-selection.h5'
	jack_img = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_img_z-ref_with-selection.h5'
	jack_cont_arr = load + 'stack/' + cat_lis[ll] + '_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref_with-selection.h5'

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('%s band finished !' % band_str)

raise

#### stacking sky imgs
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

n_rbins = 100
N_bin = 30
z_ref = 0.25

N_edg = 1
id_mean = 2 #stacking median-subtracted sky imgs
d_file = home + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
"""
band_str = band[ rank ]

for kk in range( 2 ):

	dat = pds.read_csv(load + 'img_cat/%s_%s-band_remain_cat_resamp_BCG-pos.csv' % (cat_lis[kk], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	id_cen = 0 # centered on BCGs

	sub_img = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_BCG-stack_%s-band' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_BCG-stack_%s-band' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_BCG-stack_%s-band' % band_str + '_sub-%d_SB-pro.h5'

	J_sub_img = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_jack-sub-%d_img_z-ref_with-selection.h5'
	J_sub_pix_cont = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_jack-sub-%d_pix-cont_z-ref_with-selection.h5'
	J_sub_sb = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'

	jack_SB_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_Mean_jack_SB-pro_z-ref_with-selection.h5'
	jack_img = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_Mean_jack_img_z-ref_with-selection.h5'
	jack_cont_arr = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_Mean_jack_pix-cont_z-ref_with-selection.h5'

	print('start stacking')

	sky_jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_mean = id_mean, id_cut = True, N_edg = 1, id_Z0 = False, z_ref = z_ref, id_sub = True,)

print('finished %s band BCG-stack !' % band_str,)
"""

"""
## random center case
'''
for ll in range( 3 ):

	for kk in range( 2 ):

		dat = pds.read_csv(load + 'img_cat/%s_%s-band_remain_cat_resamp_BCG-pos.csv' % (cat_lis[kk], band[ll]),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		mock_x = np.random.choice(2048, size = len( z ),)
		mock_y = np.random.choice(1489, size = len( z ),)

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ ra, dec, z, mock_x, mock_y ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( '/home/xkchen/project/%s_%s-band_random-center_%d-rank.csv' % (cat_lis[kk], band[ll], rank),)

		cat_file = '/home/xkchen/project/%s_%s-band_random-center_%d-rank.csv' % (cat_lis[kk], band[ll], rank)
		out_file = '/home/xkchen/project/%s_%s-band_random-center_z-ref_%d-rank.csv' % (cat_lis[kk], band[ll], rank)
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)
'''
for ll in range( 3 ):

	for kk in range( 2 ):

		dat = pds.read_csv('/home/xkchen/project/%s_%s-band_random-center_z-ref_%d-rank.csv' % (cat_lis[kk], band[ll], rank),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		id_cen = 2 # random center stacking

		sub_img = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_rand-stack_sub-%d_img' + '_%d-rank_%s-band.h5' % (rank, band[ll] )
		sub_pix_cont = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_rand-stack_sub-%d_pix-cont' + '_%d-rank_%s-band.h5' % (rank, band[ll] )
		sub_sb = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_rand-stack_sub-%d_SB-pro' + '_%d-rank_%s-band.h5' % (rank, band[ll] )

		J_sub_img = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_img_z-ref_with-selection' + '_%d-rank.h5' % rank
		J_sub_pix_cont = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_pix-cont_z-ref_with-selection' + '_%d-rank.h5' % rank
		J_sub_sb = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_SB-pro_z-ref_with-selection' + '_%d-rank.h5' % rank

		jack_SB_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_SB-pro_z-ref_with-selection_%d-rank.h5' % rank
		jack_img = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_img_z-ref_with-selection_%d-rank.h5' % rank
		jack_cont_arr = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_pix-cont_z-ref_with-selection_%d-rank.h5' % rank

		sky_jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[ll], sub_img,
			sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
			id_mean = id_mean, id_cut = True, N_edg = 1, id_Z0 = False, z_ref = z_ref, id_sub = True,)

	print('finished %s band' % band[ll],)

commd.Barrier()

if rank == 0:
	for ll in range( 3 ):
		N_sample = 5 # random 5 times

		for kk in range( 2 ):

			for pp in range( N_bin ):

				data_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_img_z-ref_with-selection' % pp + '_%d-rank.h5'
				out_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_img_z-ref_with-selection' % pp + '-aveg.h5'
				aveg_stack_img(N_sample, data_file, out_file,)

				data_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_pix-cont_z-ref_with-selection' % pp + '_%d-rank.h5'
				out_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_jack-sub-%d_pix-cont_z-ref_with-selection' % pp + '-aveg.h5'
				aveg_stack_img(N_sample, data_file, out_file,)

			data_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_img_z-ref_with-selection' + '_%d-rank.h5'
			out_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_img_z-ref_with-selection' + '-aveg.h5'
			aveg_stack_img(N_sample, data_file, out_file,)

			data_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_pix-cont_z-ref_with-selection' + '_%d-rank.h5'
			out_file = load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band[ll] + '_Mean_jack_pix-cont_z-ref_with-selection-aveg.h5'
			aveg_stack_img(N_sample, data_file, out_file,)

print('rank 0 finished !')

print('finished rand-stack !')
"""

#### correction ( add ICL component in the residual sky)
band_str = band[ rank ]

n_rbins = 100
N_bin = 30
z_ref = 0.25

SN_lim = 5
edg_bins = 4

for kk in range( 2 ):

	tmp_r, tmp_sb = [], []

	for mm in range( N_bin ):

		with h5py.File(load + 'stack/' + cat_lis[kk] + '_%s-band' % band_str + '_jack-sub-%d_img_z-ref_with-selection.h5' % mm, 'r') as f:
			clust_img = np.array( f['a'] )

		with h5py.File(load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_BCG-stack' % band_str + '_jack-sub-%d_img_z-ref_with-selection.h5' % mm, 'r') as f:
			bcg_stack_sky = np.array( f['a'] )

		with h5py.File(load + 'stack/' + cat_lis[kk] + '_sky-sub-medi_%s-band_rand-stack' % band_str + '_jack-sub-%d_img_z-ref_with-selection-aveg.h5' % mm, 'r') as f:
			rand_stack_sky = np.array( f['a'] )

		diffi_sky = bcg_stack_sky - rand_stack_sky
		resi_sky_add_img = clust_img + diffi_sky

		with h5py.File(load + 'stack/' + cat_lis[kk] + '_%s-band' % band_str + '_jack-sub-%d_img-resi-sky-add_z-ref_with-selection.h5' % mm, 'w') as f:
			f['a'] = np.array( resi_sky_add_img )

for kk in range( 2 ):

	J_sub_img = load + 'stack/' + cat_lis[kk] + '_%s-band' % band_str + '_jack-sub-%d_img-resi-sky-add_z-ref_with-selection.h5'
	J_sub_pix_cont = load + 'stack/' + cat_lis[kk] + '_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref_with-selection.h5'

	alter_sub_sb = load + 'stack/' + cat_lis[kk] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_resi-sky-add_z-ref_with-selection.h5'
	alter_jk_sb = load + 'stack/' + cat_lis[kk] + '_%s-band' % band_str + '_Mean_jack_SB-pro_resi-sky-add_z-ref_with-selection.h5'

	id_band = band.index( band_str )

	zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, alter_sub_sb, alter_jk_sb, n_rbins, N_bin, SN_lim, z_ref, id_band, edg_bins = edg_bins,)

print( 'finished !' )

