"""
shuffle each satellite individually, and each satellite must have a
shuffle-mapping location
"""
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

from img_sat_resamp import resamp_func
from img_sat_resamp import BG_resamp_func
from img_sat_BG_extract_tmp import origin_img_cut_func

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func

#.
import time
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
rad2arcsec = U.rad.to( U.arcsec )


### === shuffle table build

bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

# cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
# out_path = '/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/shufl_cat/'

# for kk in range( 20 ):

# 	for dd in range( 3 ):
# 		band_str = band[ dd ]   ## i-band as test

# 		for tt in range( 3 ):  ## medi-rich subsample


cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/figs/'

for kk in range( 100 ):

	for dd in range( 3 ):

		band_str = band[ dd ]   ## i-band as test

		for tt in range( 3 ):  ## medi-rich subsample

			##. cluster catalog
			cat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1]),)
			set_IDs = np.array( cat['clust_ID'] )	
			set_IDs = set_IDs.astype( int )


			##. cluster member catalog, satellite location table
			dat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' % 
					(bin_rich[tt], bin_rich[tt + 1], band_str ),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
			sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )
			sat_PA2bcg = np.array( dat['sat_PA2bcg'] )

			clus_IDs = np.array( dat['clus_ID'] )
			clus_IDs = clus_IDs.astype( int )

			N_sat = len( sat_ra )


			pre_R_sat = np.zeros( N_sat, )

			#. record the finally mapped cluster ID and the location of satellite in that image
			shufl_IDs = np.zeros( N_sat, dtype = int )
			shufl_ra, shufl_dec, shufl_z = np.zeros( N_sat,), np.zeros( N_sat,), np.zeros( N_sat,)

			shufl_sx, shufl_sy = np.zeros( N_sat, ), np.zeros( N_sat, )
			shufl_Rpix, shufl_PA =  np.zeros( N_sat, ), np.zeros( N_sat, )
			shufl_R_phy = np.zeros( N_sat, )    ###. in unit kpc

			#. record these points is symmetry (id_symP = 1,2,3) or not ( id_symP = 0 )
			id_symP = np.zeros( N_sat, dtype = int )

			#.
			dt0 = time.time()

			for qq in range( N_sat ):

				ra_g, dec_g, z_g = bcg_ra[ qq ], bcg_dec[ qq ], bcg_z[ qq ]
				sub_ra, sub_dec = sat_ra[ qq ], sat_dec[ qq ]

				x_cen, y_cen = bcg_x[ qq ], bcg_y[ qq ]
				x_sat, y_sat = sat_x[ qq ], sat_y[ qq ]

				sat_Rpix = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )  ### in pixel number
				sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

				Da_obs = Test_model.angular_diameter_distance( z_g ).value
				qq_R_sat = sat_Rpix * pixel * Da_obs * 1e3 / rad2arcsec     ##. kpc
				pre_R_sat[ qq ] = qq_R_sat


				##. random select cluster and do a map between satellite and the 'selected' cluster
				qq_ID = clus_IDs[ qq ]

				id_vx = set_IDs == qq_ID
				ID_dex = np.where( id_vx )[0][0]

				copy_IDs = np.delete( set_IDs, ID_dex )
				cp_Ns = len( copy_IDs )

				#. random order of index in the cluster array
				rand_arr = np.random.choice( cp_Ns, cp_Ns, replace = False )


				##. try to map satellite and the 'selected' cluster
				id_loop = 0
				id_live = 1

				while id_live > 0:

					rand_ID = copy_IDs[ rand_arr[ id_loop ] ]

					id_ux = clus_IDs == rand_ID
					cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]			

					cp_Da = Test_model.angular_diameter_distance( cp_z_g ).value

					cp_sat_Rpix = ( ( qq_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

					off_x, off_y = cp_sat_Rpix * np.cos( sat_theta ), cp_sat_Rpix * np.sin( sat_theta )

					cp_cx, cp_cy = bcg_x[ id_ux ][0], bcg_y[ id_ux ][0]

					cp_sx, cp_sy = cp_cx + off_x, cp_cy + off_y   ## new satellite location

					id_bond_x = ( cp_sx >= 0 ) & ( cp_sx < 2048 )
					id_bond_y = ( cp_sy >= 0 ) & ( cp_sy < 1489 )
					id_bond = id_bond_x & id_bond_y

					if id_bond:

						shufl_IDs[ qq ] = rand_ID
						shufl_sx[ qq ] = cp_sx
						shufl_sy[ qq ] = cp_sy
						shufl_Rpix[ qq ] = cp_sat_Rpix
						shufl_PA[ qq ] = sat_theta

						shufl_ra[ qq ] = cp_ra_g
						shufl_dec[ qq ] = cp_dec_g
						shufl_z[ qq ] = cp_z_g

						cp_R_phy = ( cp_sat_Rpix * pixel * cp_Da * 1e3 ) / rad2arcsec

						shufl_R_phy[ qq ] = cp_R_phy
						id_symP[ qq ] = 0

						id_live = id_live - 1
						break

					else:

						tmp_phi = np.array( [ 	np.pi + sat_theta, 
												np.pi - sat_theta, 
												np.pi * 2 - sat_theta ] )

						tmp_off_x = cp_sat_Rpix * np.cos( tmp_phi )
						tmp_off_y = cp_sat_Rpix * np.sin( tmp_phi )

						tmp_sx, tmp_sy = cp_cx + tmp_off_x, cp_cy + tmp_off_y				

						id_lim_x = ( tmp_sx >= 0 ) & ( tmp_sx < 2048 )
						id_lim_y = ( tmp_sy >= 0 ) & ( tmp_sy < 1489 )
						id_lim = id_lim_x & id_lim_y

						if np.sum( id_lim ) < 1 :  ## no points located in image frame

							id_loop = id_loop + 1
							continue

						else:

							rt0 = np.random.choice( np.sum( id_lim ), np.sum( id_lim), replace = False)

							shufl_IDs[ qq ] = rand_ID
							shufl_sx[ qq ] = tmp_sx[ id_lim ][ rt0[ 0 ] ]
							shufl_sy[ qq ] = tmp_sy[ id_lim ][ rt0[ 0 ] ]

							shufl_Rpix[ qq ] = cp_sat_Rpix
							shufl_PA[ qq ] = tmp_phi[ id_lim ][ rt0[ 0 ] ]

							shufl_ra[ qq ] = cp_ra_g
							shufl_dec[ qq ] = cp_dec_g
							shufl_z[ qq ] = cp_z_g

							cp_R_phy = ( cp_sat_Rpix * pixel * cp_Da * 1e3 ) / rad2arcsec

							shufl_R_phy[ qq ] = cp_R_phy

							#. symmetry point record
							ident_x0 = np.where( id_lim )[0]
							ident_x1 = ident_x0[ rt0[ 0 ] ]

							id_symP[ qq ] = ident_x1 + 1

							id_live = id_live - 1
							break


			##. save table	
			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_cID', 'orin_Rsat_phy', 
					 'shufl_cID', 'cp_sx', 'cp_sy', 'cp_PA', 'cp_Rpix', 'cp_Rsat_phy', 'is_symP',
					 'cp_bcg_ra', 'cp_bcg_dec', 'cp_bcg_z' ]

			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, clus_IDs, pre_R_sat, 
					shufl_IDs, shufl_sx, shufl_sy, shufl_PA, shufl_Rpix, shufl_R_phy, id_symP,
					shufl_ra, shufl_dec, shufl_z ]

			fill = dict( zip( keys, values ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + 'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % 
							(bin_rich[tt], bin_rich[tt + 1], band_str, kk),)



### ===
def count_shufl_N_sat():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

	bin_rich = [ 20, 30, 50, 210 ]

	##. radius binned satellite
	sub_name = ['low-rich', 'medi-rich', 'high-rich']

	##... 
	R_bins = [ 0, 200, 400 ]   ## kpc
	cat_lis = ['inner', 'middle', 'outer']

	N_shufl = 20


	#... number count for the entire sample
	for kk in range( 3 ):

		for tt in range( 3 ):

			band_str = band[ tt ]

			for dd in range( N_shufl ):

				##.
				dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/' + 
						'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, dd),)

				bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
				sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

				orin_IDs = np.array( dat['orin_cID'] )
				rand_IDs = np.array( dat['shufl_cID'] )

				orin_IDs = orin_IDs.astype( int )
				rand_IDs = rand_IDs.astype( int )


				##. entire all sample
				dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)
				clus_IDs = np.array( dat['clust_ID'] )
				clus_IDs = clus_IDs.astype( int )

				N_w = len( clus_IDs )
				N_ss = len( bcg_ra )

				##.
				pre_Ng = np.zeros( N_ss, )
				shufl_Ng = np.zeros( N_ss, )

				for mm in range( N_w ):

					sub_IDs = clus_IDs[ mm ]

					id_vx = orin_IDs == sub_IDs
					pre_Ng[ id_vx ] = np.sum( id_vx )

					id_ux = rand_IDs == sub_IDs
					shufl_Ng[ id_ux ] = np.sum( id_ux )

					print( np.sum( id_vx ) )

				##. save
				keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_Ng', 'shufl_Ng']
				values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, pre_Ng, shufl_Ng ]
				fill = dict( zip( keys, values ) )
				out_data = pds.DataFrame( fill )
				out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/' + 
						'clust_rich_%d-%d_%s-band_sat-shuffle-%d_shufl-sat-Ng.csv' % 
								(bin_rich[kk], bin_rich[kk + 1], band_str, dd),)



	#... number count for the subsamples sample
	for kk in range( 3 ):

		##. entire all sample
		dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)
		clus_IDs = np.array( dat['clust_ID'] )
		clus_IDs = clus_IDs.astype( int )

		N_w = len( clus_IDs )

		for tt in range( 3 ):

			band_str = band[ tt ]

			for dd in range( N_shufl ):

				##. shuffle list
				dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/' + 
						'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, dd),)

				bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
				sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

				pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

				orin_IDs = np.array( dat['orin_cID'] )
				rand_IDs = np.array( dat['shufl_cID'] )

				orin_IDs = orin_IDs.astype( int )
				rand_IDs = rand_IDs.astype( int )

				##. subsample catalog
				for nn in range( 3 ):

					pat = pds.read_csv( cat_path + 
							'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_cat.csv' % 
								( bin_rich[kk], bin_rich[kk + 1], cat_lis[ nn ] ),)

					p_ra, p_dec, p_z = np.array( pat['bcg_ra'] ), np.array( pat['bcg_dec'] ), np.array( pat['bcg_z'] )
					p_sat_ra, p_sat_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )

					nn_coord = SkyCoord( ra = p_sat_ra * U.deg, dec = p_sat_dec * U.deg )

					p_clus_IDs = np.array( pat['clus_ID'] )
					p_clus_IDs = p_clus_IDs.astype( int )

					N_ss = len( p_ra )


					idx, sep, d3d = nn_coord.match_to_catalog_sky( pre_coord )
					id_lim = sep.value < 2.7e-4

					mp_shufl_IDs = rand_IDs[ idx[ id_lim ] ]


					##.
					pre_Ng = np.zeros( N_ss, )
					shufl_Ng = np.zeros( N_ss, )

					for mm in range( N_w ):

						sub_IDs = clus_IDs[ mm ]

						id_vx = p_clus_IDs == sub_IDs
						pre_Ng[ id_vx ] = np.sum( id_vx )

						id_ux = mp_shufl_IDs == sub_IDs
						shufl_Ng[ id_ux ] = np.sum( id_ux )

					##.
					keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_Ng', 'shufl_Ng' ]
					values = [ p_ra, p_dec, p_z, p_sat_ra, p_sat_dec, pre_Ng, shufl_Ng ]
					fill = dict( zip( keys, values ) )
					out_data = pds.DataFrame( fill )
					out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/' + 
							'clust_rich_%d-%d_%s_%s-band_sat-shuffle-%d_shufl-sat-Ng.csv' % 
									(bin_rich[kk], bin_rich[kk + 1], cat_lis[ nn ], band_str, dd),)

	return

# count_shufl_N_sat()

raise