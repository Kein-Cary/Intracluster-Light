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

#.
from light_measure import light_measure_weit
from img_sat_resamp import resamp_func

from img_sat_BG_extract_tmp import self_shufl_img_cut_func
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


###********************************### image preparing
list_order = 0   ### in range(0, 20)

### === tacking the stacking image of this cutout as background
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
img_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'


##. cluster subsamples
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

"""
out_path = '/home/xkchen/data/SDSS/member_files/rich_binned_self_shufl/shufl_cat/'

for tt in range( len(bin_rich) - 1 ):

	#. target cluster (want to know background of satellites)
	dat = pds.read_csv( cat_path + 
			'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[tt], bin_rich[tt + 1]),)
	targ_IDs = np.array( dat['clus_ID'] )

	set_IDs = np.array( list( set( targ_IDs ) ) )
	set_IDs = set_IDs.astype( int )

	N_cc = len( set_IDs )


	#. shuffle cluster IDs
	R_cut = 320   ##. pixels
	out_file = ( home + 
		'member_files/rich_binned_self_shufl/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]

	for kk in range( 3 ):

		band_str = band[ kk ]

		##. satellite position record table
		post_file = ( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_' % (bin_rich[tt], bin_rich[tt + 1]) + 
				'%s-band_origin-img_position.csv',)[0]

		#. shuffle cutout images
		m, n = divmod( N_cc, cpus )
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_clusID = set_IDs[N_sub0 : N_sub1]

		err_files = out_path + 'clust_rich_%d-%d_%s-band_err_in_symmetry-position.h5' % (bin_rich[tt], bin_rich[tt + 1], band_str)
		err_grops = "err_points"

		self_shufl_img_cut_func( post_file, img_file, band_str, sub_clusID, R_cut, pixel, out_file, 
							err_file = err_files, err_grop = err_grops)

	print('%s, %d-rank, cut Done!' %(sub_name[tt], rank), )

"""


#. resampling... 
"""
out_path = '/home/xkchen/data/SDSS/member_files/rich_binned_self_shufl/shufl_cat/'

for tt in range( len(bin_rich) - 1 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##. satellite table
		dat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos.csv' % 
				(bin_rich[tt], bin_rich[tt + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )


		##. error shuffle list
		err_files = ( out_path + 
			'clust_rich_%d-%d_%s-band_err_in_symmetry-position.h5' % (bin_rich[tt], bin_rich[tt + 1], band_str),)[0]

		err_grops = "err_points"


		with h5py.File( err_files, 'r') as f: 
			group = f["/%s/err_clusters/" % err_grops ][()]
			group_sat = f["/%s/err_sat/" % err_grops ][()]

		out_sat_ra = ['%.4f' % ll for ll in group_sat[0] ]
		out_sat_dec = ['%.4f' % ll for ll in group_sat[1] ]

		used_order = simple_match( out_sat_ra, out_sat_dec, sat_ra, sat_dec, id_choose = False)

		bcg_ra, bcg_dec, bcg_z = bcg_ra[ used_order ], bcg_dec[ used_order ], bcg_z[ used_order ]
		sat_ra, sat_dec = sat_ra[ used_order ], sat_dec[ used_order ]
		sat_cx, sat_cy = sat_cx[ used_order ], sat_cy[ used_order ]


		##.
		_Ns_ = len( sat_ra )

		m, n = divmod( _Ns_, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]
		img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]


		id_dimm = True
		d_file = ( home + 
			'member_files/rich_binned_self_shufl/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]
		out_file = ( home + 
			'member_files/rich_binned_self_shufl/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]

		resamp_func( d_file, sub_z, sub_ra, sub_dec, ra_set, dec_set, img_x, img_y, band_str, out_file, z_ref, id_dimm = id_dimm )

	print( '%d rank, done!' % rank )

"""



###********************************### image stacking
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

cat_path = load + 'Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/project/tmp_obj_cat/'


id_cen = 0
N_edg = 1   ##. rule out edge pixels
n_rbins = 35

##. Background
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

N_bin = 100


#. entire sample
d_file = home + 'member_files/rich_binned_self_shufl/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'
mask_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'


for ll in range( 3 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##. satellite catalog
		dat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos-zref.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		print('N_sample = ', len( bcg_ra ) )


		##. err_catalog in shuffle list
		err_files = ( '/home/xkchen/data/SDSS/member_files/rich_binned_self_shufl/shufl_cat/' + 
			'clust_rich_%d-%d_%s-band_err_in_symmetry-position.h5' % (bin_rich[ll], bin_rich[ll + 1], band_str),)[0]

		err_grops = "err_points"

		with h5py.File( err_files, 'r') as f: 
			group = f["/%s/err_clusters/" % err_grops ][()]
			group_sat = f["/%s/err_sat/" % err_grops ][()]

		out_sat_ra = ['%.4f' % ll for ll in group_sat[0] ]
		out_sat_dec = ['%.4f' % ll for ll in group_sat[1] ]

		used_order = simple_match( out_sat_ra, out_sat_dec, sat_ra, sat_dec, id_choose = False)

		bcg_ra, bcg_dec, bcg_z = bcg_ra[ used_order ], bcg_dec[ used_order ], bcg_z[ used_order ]
		sat_ra, sat_dec = sat_ra[ used_order ], sat_dec[ used_order ]
		img_x, img_y = img_x[ used_order ], img_y[ used_order ]


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
							rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, weit_img = mask_file,)

print('%d-rank, Done' % rank )

raise


#. sub-samples
##. R_bins = [ 0, 200, 400 ]   ## kpc, physical distance
cat_lis = ['inner', 'middle', 'outer']

for ll in range( 3 ):

	for tt in range( 3 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			dat = pds.read_csv(cat_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_pos-zref.csv' 
							% ( bin_rich[ ll ], bin_rich[ll + 1], cat_lis[tt], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

			print('N_sample = ', len( bcg_ra ) )

			# XXX
			sub_img = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_img.h5'
			sub_pix_cont = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_pix-cont.h5'
			sub_sb = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_sub-%d_SB-pro.h5'
			# XXX

			J_sub_img = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_img_z-ref.h5'
			J_sub_pix_cont = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_pix-cont_z-ref.h5'
			J_sub_sb = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_jack-sub-%d_SB-pro_z-ref.h5'

			jack_SB_file = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5'
			jack_img = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_img_z-ref.h5'
			jack_cont_arr = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_pix-cont_z-ref.h5'

			sat_BG_fast_stack_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, img_x, img_y, d_file, band_str, id_cen, N_bin, n_rbins, 
								sub_img, sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
								rank, id_cut = True, N_edg = N_edg, id_Z0 = False, z_ref = z_ref, id_sub = False, weit_img = mask_file)

print('%d-rank, Done' % rank )

