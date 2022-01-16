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

from astropy.coordinates import SkyCoord
from astropy import cosmology as apcy

from fig_out_module import arr_jack_func
from light_measure import light_measure_weit
from img_jack_stack import jack_main_func
from img_jack_stack import SB_pros_func
from img_jack_stack import aveg_stack_img
from img_edg_cut_stack import cut_stack_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
L_wave = np.array( [ 6166, 4686, 7480 ] )
Mag_sun = [ 4.65, 5.11, 4.53 ]

def sersic_func(r, Ie, re, ndex):
	belta = 3 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

###=====### parameter for Galactic
Rv = 3.1
pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.25

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

### == ### divide sample with redshift span and check relative color deviation

cat_lis = [ 'low-age', 'hi-age' ]
# z_cri = [ 0.252, 0.256 ]
z_cri = [ 0.254, 0.254 ]

# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# z_cri = [ 0.25, 0.2564 ]
# z_cri = [ 0.253, 0.253 ]

# cat_lis = [ 'low-rich', 'hi-rich' ]
# z_cri = [ 0.254, 0.2535 ]
# z_cri = [ 0.254, 0.254 ]

"""
#. sample divided by z
for ll in range( 2 ):

	for kk in range( 3 ):

		###... bcg age binned sample
		lo_dat = pds.read_csv( load + 'pkoffset_cat/' + 
					'%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band[kk]),)
		lo_age_ra, lo_age_dec, lo_age_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
		lo_age_bcgx, lo_age_bcgy = np.array( lo_dat['bcg_x'] ), np.array( lo_dat['bcg_y'] )

		id_x0 = lo_age_z <= z_cri[ ll ]
		print('*' * 20 )
		print( np.sum(id_x0) )
		print( np.sum(id_x0 == False) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ lo_age_ra[ id_x0 ], lo_age_dec[ id_x0 ], lo_age_z[ id_x0 ], lo_age_bcgx[ id_x0 ], lo_age_bcgy[ id_x0 ] ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( load + 'half_z_cat/%s_bin_gri-common-cat_%s-band_low-z_zref-cat.csv' % (cat_lis[ll], band[kk]),)

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ lo_age_ra[ id_x0 == False], lo_age_dec[ id_x0 == False], lo_age_z[ id_x0 == False], 
					lo_age_bcgx[ id_x0 == False], lo_age_bcgy[ id_x0 == False] ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( load + 'half_z_cat/%s_bin_gri-common-cat_%s-band_hi-z_zref-cat.csv' % (cat_lis[ll], band[kk]),)

"""


###... stack test
band_str = band[ rank ]
z_samp = [ 'low-z', 'hi-z' ]

id_cen = 0
n_rbins = 55
N_bin = 30

for ll in range( 2 ):

	for kk in range( 2 ):

		d_file = home + 'photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

		## BCG position offset
		dat = pds.read_csv( load + 
							'half_z_cat/%s_bin_gri-common-cat_%s-band_%s_zref-cat.csv' % (cat_lis[ll], band_str, z_samp[kk]),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		# XXX
		sub_img = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_sub-%d_img.h5'
		sub_pix_cont = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_sub-%d_pix-cont.h5'
		sub_sb = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_sub-%d_SB-pro.h5'
		# XXX

		J_sub_img = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_jack-sub-%d_img_z-ref.h5'
		J_sub_pix_cont = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_jack-sub-%d_pix-cont_z-ref.h5'
		J_sub_sb = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_jack-sub-%d_SB-pro_z-ref.h5'

		jack_SB_file = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_Mean_jack_SB-pro_z-ref.h5'
		jack_img = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_Mean_jack_img_z-ref.h5'
		jack_cont_arr = load + 'half_z_stack/photo-z_match_gri-common_%s_%s-band_%s' % (cat_lis[ll], band_str, z_samp[kk]) + '_Mean_jack_pix-cont_z-ref.h5'

		jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band_str, sub_img,
			sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
			id_cut = True, N_edg = 1, id_Z0 = False, z_ref = 0.25, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

		print('%d, %s band finished !' % (ll, band_str) )

