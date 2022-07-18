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


### ===
def shufl_list_match( cat_file, clust_file, N_shufl, shufl_list_file, out_shufl_file, 
					oirn_img_pos_file, out_pos_file, out_Ng_file):

	#. satellite catalog
	dat = pds.read_csv( cat_file,)
	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	N_sat = len( sat_ra )


	#. shuffle list match
	# for dd in range( N_shufl ):
	for dd in range( 13,14 ):

		#.
		pat = pds.read_csv( shufl_list_file % dd,)

		keys = pat.columns[1:]

		kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
		kk_IDs = np.array( pat['orin_cID'] )

		kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

		idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
		id_lim = sep.value < 2.7e-4

		N_ks = len( keys )

		#. record matched information
		tmp_arr = []
		N_ks = len( keys )

		for pp in range( N_ks ):

			sub_arr = np.array( pat['%s' % keys[pp] ] )
			tmp_arr.append( sub_arr[ idx[ id_lim ] ] )

		##.save
		fill = dict( zip( keys, tmp_arr ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_shufl_file % dd,)


	#. image cut information match
	# for dd in range( N_shufl ):
	for dd in range( 13,14 ):

		##. T200 catalog
		dat = pds.read_csv( out_shufl_file % dd,)
		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

		N_sat = len( sat_ra )

		host_IDs = np.array( dat['orin_cID'] )
		rand_IDs = np.array( dat['shufl_cID'] )

		host_IDs = host_IDs.astype( int )
		rand_IDs = rand_IDs.astype( int )


		##.==. origin image location match
		pat = pds.read_csv( oirn_img_pos_file,)
		keys = pat.columns[1:]

		kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
		kk_IDs = np.array( pat['clus_ID'] )

		kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

		idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
		id_lim = sep.value < 2.7e-4

		mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
		mp_IDs = kk_IDs[ idx[ id_lim ] ]

		#. record matched information
		tmp_arr = []
		N_ks = len( keys )

		for pp in range( N_ks ):

			sub_arr = np.array( pat['%s' % keys[pp] ] )
			tmp_arr.append( sub_arr[ idx[ id_lim ] ] )

		##.save
		fill = dict( zip( keys, tmp_arr ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_pos_file % dd,)


	#. count the number of satellite before and after shuffling
	# for dd in range( N_shufl ):
	for dd in range( 13,14 ):

		##. T200 catalog
		dat = pds.read_csv( out_shufl_file % dd,)
		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

		N_sat = len( sat_ra )

		host_IDs = np.array( dat['orin_cID'] )
		rand_IDs = np.array( dat['shufl_cID'] )

		host_IDs = host_IDs.astype( int )
		rand_IDs = rand_IDs.astype( int )


		##. entire cluster catalog
		cat = pds.read_csv( clust_file,)
		set_IDs = np.array( cat['clust_ID'] )	
		set_IDs = set_IDs.astype( int )

		N_cs = len( set_IDs )

		##. satellite number counts
		tmp_clus_Ng = np.zeros( N_sat, )
		tmp_shufl_Ng = np.zeros( N_sat, )

		for pp in range( N_cs ):

			sub_IDs = set_IDs[ pp ]

			id_vx = host_IDs == sub_IDs
			id_ux = rand_IDs == sub_IDs

			tmp_clus_Ng[ id_vx ] = np.sum( id_vx )
			tmp_shufl_Ng[ id_ux ] = np.sum( id_ux )

		#. save
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_Ng', 'shufl_Ng']
		values = [bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, tmp_clus_Ng, tmp_shufl_Ng ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_Ng_file % dd,)

	return


### === 
##. initial shuffle
# cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/radius_bin_table/'
# shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/'

# sub_name = ['inner', 'middle', 'outer']


##. more times for richness in range of (50~210)
# cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'
# shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/shufl_100_tables/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/cp_tables/'

# sub_name = ['inner', 'middle', 'outer']



##. rebinned radii subset~( based on R_cut = [200, 400] )
# cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
# shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/shufl_list/'

# #. rich < 30 or rich > 50
# R_bins_0 = np.arange(0, 350, 50)
# R_bins_1 = np.arange( R_bins_0[-1], 600, 100)
# R_bins_2 = np.arange( R_bins_1[-1], 1600, 400)
# R_bins = np.r_[ R_bins_0, R_bins_1[1:], R_bins_2[1:] ]


##. rebinned radii subset
cat_path = '/home/xkchen/figs_cp/cc_rich_rebin/cat/'
out_path = '/home/xkchen/figs_cp/cc_rich_rebin/shufl_list/'
shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/'

#. rich (30, 50)
# R_bins = np.array( [ 0, 300, 400, 550, 5000] )

##. rich < 30
R_bins = np.array([0, 150, 300, 500, 2000])

##. rich > 50
# R_bins = np.array([0, 400, 600, 750, 2000])


bin_rich = [ 20, 30, 50, 210 ]


# for pp in range( 1,2 ):
for pp in range( 1 ):

	for tt in range( len(R_bins) - 1 ):

		#. member file
		# sub_cat_file = ( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_cat.csv' % 
		# 				( bin_rich[pp], bin_rich[pp + 1], sub_name[tt]),)[0]

		sub_cat_file = ( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv' % 
								( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1]),)[0]


		#.cluster catalog
		clust_file = ( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/' + 
						'clust_rich_%d-%d_cat.csv' % (bin_rich[pp], bin_rich[pp + 1]),)[0]

		for kk in range( 3 ):

			band_str = band[ kk ]

			N_shufl = 20
			# N_shufl = 40
			# N_shufl = 100

			shufl_list_file = ( shufl_path + 'clust_rich_%d-%d_%s-band_sat-' % (bin_rich[pp], bin_rich[pp + 1], band_str) + 
								'shuffle-%d_position.csv',)[0]

			oirn_img_pos_file = ( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/' + 
								'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' % 
								(bin_rich[pp], bin_rich[pp + 1], band_str),)[0]

			##.
			# out_shufl_file = ( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_' 
			# 					% ( bin_rich[pp], bin_rich[pp + 1], sub_name[tt], band_str) + 'sat-shufl-%d_cat.csv',)[0]

			# out_pos_file = ( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_'
			# 					% ( bin_rich[pp], bin_rich[pp + 1], sub_name[tt], band_str) + 'sat-shufl-%d_origin-img_position.csv',)[0]

			# out_Ng_file = ( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_' 
			# 					% ( bin_rich[pp], bin_rich[pp + 1], sub_name[tt], band_str) + 'sat-shufl-%d_shufl-Ng.csv',)[0]


			##.
			out_shufl_file = ( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_%s-band_' 
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_cat.csv',)[0]

			out_pos_file = ( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_%s-band_'
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_origin-img_position.csv',)[0]

			out_Ng_file = ( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_%s-band_' 
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_shufl-Ng.csv',)[0]

			shufl_list_match( sub_cat_file, clust_file, N_shufl, shufl_list_file, out_shufl_file, 
						oirn_img_pos_file, out_pos_file, out_Ng_file )
