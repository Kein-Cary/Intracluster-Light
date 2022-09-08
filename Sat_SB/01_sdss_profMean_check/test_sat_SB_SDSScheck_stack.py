import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

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
from scipy.stats import binned_statistic as binned

from fig_out_module import arr_jack_func
from light_measure import light_measure_weit


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### ===
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


home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

"""
### === satellite image stacking
from img_sat_jack_stack import jack_main_func
from img_sat_stack import cut_stack_func, stack_func

id_cen = 0

z_ref = 0.25

N_edg = 1
n_rbins = 35

d_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

band_str = band[ rank ]


N_bin = 30  ## number of jackknife subsample

dat = pds.read_csv('/home/xkchen/data/SDSS/member_files/shufl_img_woBCG/SDSS_profMean_check_cat_%s-band_pos_z-ref.csv' % band_str,)

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )

sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

print('N_sample = ', len( bcg_ra ) )


# XXX
sub_img = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_sub-%d_img.h5'
sub_pix_cont = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_sub-%d_pix-cont.h5'
sub_sb = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_sub-%d_SB-pro.h5'
# XXX

J_sub_img = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
J_sub_sb = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

jack_SB_file = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
jack_img = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_Mean_jack_img_z-ref.h5'
jack_cont_arr = '/home/xkchen/figs/' + 'sat_SB_check_%s-band' % band_str + '_Mean_jack_pix-cont_z-ref.h5'

jack_main_func( id_cen, N_bin, n_rbins, bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('%s band finished !' % band_str )

"""


### === background image stacking
from img_sat_BG_jack_stack import jack_main_func
from img_sat_BG_stack import cut_stack_func, stack_func

from img_sat_BG_jack_stack import weit_aveg_img
from img_sat_BG_jack_stack import SB_pros_func
from img_sat_BG_jack_stack import aveg_stack_img

from light_measure import jack_SB_func
from light_measure import light_measure_Z0_weit, light_measure_weit
from light_measure import light_measure_rn_Z0_weit, light_measure_rn_weit
from light_measure import lim_SB_pros_func, zref_lim_SB_adjust_func


id_cen = 0

z_ref = 0.25

N_edg = 1
n_rbins = 35

N_bin = 30

# ##... except catalog
# if band_str == 'r':
# 	out_ra = [ '164.740', '141.265', ]
# 	out_dec = [ '11.637', '11.376', ]
# 	out_z = [ '0.298', '0.288', ]

# if band_str == 'g':
# 	out_ra = [ '206.511', '141.265', '236.438', ]
# 	out_dec = [ '38.731', '11.376', '1.767', ]
# 	out_z = [ '0.295', '0.288', '0.272', ]


d_file = home + 'member_files/BG_imgs/clus_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_BG.fits'  ## at z-ref

#. use to apply the same weight as the masked images
mask_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'


for kk in range( 3 ):

	band_str = band[ kk ]

	#. catalog info
	dat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/shufl_img_woBCG/' + 
					'SDSS_profMean_check_cat_%s-band_pos_z-ref.csv' % band_str,)
	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	print('N_sample = ', len( bcg_ra ) )

	print('band = %s' % band_str,)


	#. XXX
	sub_img = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_sub-%d_img.h5'
	sub_pix_cont = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_sub-%d_pix-cont.h5'
	sub_sb = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_sub-%d_SB-pro.h5'
	#. XXX

	J_sub_img = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_jack-sub-%d_img_z-ref.h5'
	J_sub_pix_cont = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_jack-sub-%d_pix-cont_z-ref.h5'
	J_sub_sb = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

	jack_SB_file = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_Mean_jack_SB-pro_z-ref.h5'
	jack_img = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_Mean_jack_img_z-ref.h5'
	jack_cont_arr = '/home/xkchen/figs/' + 'sat_SB_check_%s-band_BG' % band_str + '_Mean_jack_pix-cont_z-ref.h5'


	zN = len( bcg_z )
	id_arr = np.arange(0, zN, 1)
	id_group = id_arr % N_bin

	for nn in range( rank, rank + 1 ):

		id_xbin = np.where( id_group == nn )[0]

		set_z = bcg_z[ id_xbin ]
		set_ra = bcg_ra[ id_xbin ]
		set_dec = bcg_dec[ id_xbin ]

		set_s_ra = sat_ra[ id_xbin ]
		set_s_dec = sat_dec[ id_xbin ]

		set_x = img_x[ id_xbin ]
		set_y = img_y[ id_xbin ]

		sub_img_file = sub_img % nn
		sub_cont_file = sub_pix_cont % nn

		cut_stack_func( d_file, sub_img_file, set_z, set_ra, set_dec, band_str, set_s_ra, set_s_dec, set_x, set_y, id_cen, 
						N_edg, pix_con_file = sub_cont_file, weit_img = mask_file)

	commd.Barrier()

	for nn in range( rank, rank + 1 ):

		id_arry = np.linspace(0, N_bin -1, N_bin)
		id_arry = id_arry.astype( int )
		jack_id = list( id_arry )
		jack_id.remove( jack_id[ nn ] )
		jack_id = np.array( jack_id )

		jack_img_file = J_sub_img % nn
		jack_cont_file = J_sub_pix_cont % nn

		weit_aveg_img(jack_id, sub_img, sub_pix_cont, jack_img_file, sum_weit_file = jack_cont_file,)

	commd.Barrier()

	if rank == 0:

		id_Z0 = False

		SB_pros_func(J_sub_img, J_sub_pix_cont, J_sub_sb, N_bin, n_rbins, id_Z0, z_ref)

		## final jackknife SB profile
		tmp_sb = []
		tmp_r = []
		for nn in range( N_bin ):
			with h5py.File(J_sub_sb % nn, 'r') as f:
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
		tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_bin)[4:]
		sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

		with h5py.File( jack_SB_file, 'w') as f:
			f['r'] = np.array(tt_jk_R)
			f['sb'] = np.array(tt_jk_SB)
			f['sb_err'] = np.array(tt_jk_err)
			f['lim_r'] = np.array(sb_lim_r)


		order_id = np.arange(0, N_bin, 1)
		order_id = order_id.astype( np.int32 )
		weit_aveg_img(order_id, sub_img, sub_pix_cont, jack_img, sum_weit_file = jack_cont_arr,)

	commd.Barrier()

	print('%s-band, done!' % band_str)

