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

#.
from img_sat_resamp import resamp_func
from img_sat_resamp import BG_resamp_func
from img_sat_BG_extract_tmp import origin_img_cut_func

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


### === ###
"""
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/BG_imgs_nomock/shufl_cat/'

#.
N_shufl = 20

# for kk in range( N_shufl ):
for kk in range( rank, rank + 1 ):

	for dd in range( 3 ):

		band_str = band[ dd ]

		for tt in range( 3 ):

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

			#. record the finally mapped cluster ID and the location of satellite in that image
			pre_R_sat = np.zeros( N_sat, )
			shufl_IDs = np.zeros( N_sat, dtype = int )
			shufl_ra, shufl_dec, shufl_z = np.zeros( N_sat,), np.zeros( N_sat,), np.zeros( N_sat,)

			shufl_sx, shufl_sy = np.zeros( N_sat, ), np.zeros( N_sat, )
			shufl_Rpix, shufl_PA =  np.zeros( N_sat, ), np.zeros( N_sat, )
			shufl_R_phy = np.zeros( N_sat, )    ###. in unit kpc

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


				##. random order of index in the cluster array
				rand_arr = np.random.choice( cp_Ns, 1 )

				##. try to map satellite and the 'selected' cluster
				rand_ID = copy_IDs[ rand_arr ][0]

				id_ux = clus_IDs == rand_ID
				cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]			

				cp_Da = Test_model.angular_diameter_distance( cp_z_g ).value

				cp_sat_Rpix = ( ( qq_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel


				##. randomly select Position Angle between (-pi, pi)
				rand_PAs = np.linspace( -1, 1, 100 ) * np.pi   ##. unit : rad

				off_x, off_y = cp_sat_Rpix * np.cos( rand_PAs ), cp_sat_Rpix * np.sin( rand_PAs )

				cp_cx, cp_cy = bcg_x[ id_ux ][0], bcg_y[ id_ux ][0]

				cp_sx, cp_sy = cp_cx + off_x, cp_cy + off_y   ## new satellite location

				id_bond_x = ( cp_sx >= 0 ) & ( cp_sx < 2048 )
				id_bond_y = ( cp_sy >= 0 ) & ( cp_sy < 1489 )
				id_bond = id_bond_x & id_bond_y

				N_bond = np.sum( id_bond )
				id_tag = np.random.choice( N_bond, 1 )

				#.
				shufl_IDs[ qq ] = rand_ID
				shufl_sx[ qq ] = cp_sx[ id_bond ][ id_tag ]
				shufl_sy[ qq ] = cp_sy[ id_bond ][ id_tag ]
				shufl_Rpix[ qq ] = cp_sat_Rpix
				shufl_PA[ qq ] = rand_PAs[ id_bond ][ id_tag ]

				shufl_ra[ qq ] = cp_ra_g
				shufl_dec[ qq ] = cp_dec_g
				shufl_z[ qq ] = cp_z_g

				cp_R_phy = ( cp_sat_Rpix * pixel * cp_Da * 1e3 ) / rad2arcsec
				shufl_R_phy[ qq ] = cp_R_phy

			##. save table	
			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_cID', 'orin_Rsat_phy', 
					'shufl_cID', 'cp_sx', 'cp_sy', 'cp_PA', 'cp_Rpix', 'cp_Rsat_phy', 
					'cp_bcg_ra', 'cp_bcg_dec', 'cp_bcg_z' ]

			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, clus_IDs, pre_R_sat, 
					shufl_IDs, shufl_sx, shufl_sy, shufl_PA, shufl_Rpix, shufl_R_phy, 
					shufl_ra, shufl_dec, shufl_z ]

			fill = dict( zip( keys, values ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + 'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % 
							(bin_rich[tt], bin_rich[tt + 1], band_str, kk),)

raise
"""


### === z-ref cluster image extract catalog
shufl_path = '/home/xkchen/data/SDSS/member_files/BG_imgs_nomock/shufl_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/BG_imgs_nomock/zref_cut_cat/'

#.
N_shufl = 20
bin_rich = [ 20, 30, 50, 210 ]

#.
for dd in range( 3 ):

	band_str = band[ dd ]

	for kk in range( 3 ):

		for tt in range( N_shufl ):

			##.
			cat = pds.read_csv( shufl_path + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, tt),)

			clus_ID = np.array( cat['orin_cID'] )
			clus_ID = clus_ID.astype( int )

			bcg_ra, bcg_dec, bcg_z = np.array( cat['bcg_ra'] ), np.array( cat['bcg_dec'] ), np.array( cat['bcg_z'] )
			sat_ra, sat_dec = np.array( cat['sat_ra'] ), np.array( cat['sat_dec'] )

			Rsat_phy = np.array( cat['orin_Rsat_phy'] )


			##. satellite shuffle information
			cp_sx, cp_sy, cp_PA = np.array( cat['cp_sx'] ), np.array( cat['cp_sy'] ), np.array( cat['cp_PA'] )
			cp_Rpix, cp_Rsat_phy = np.array( cat['cp_Rpix'] ), np.array( cat['cp_Rsat_phy'] )

			cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array( cat['cp_bcg_ra'] ), np.array( cat['cp_bcg_dec'] ), np.array( cat['cp_bcg_z'] )
			cp_clus_ID = np.array( cat['shufl_cID'] )


			##. refer to information at z_ref
			R_cut = 320

			Da_z = Test_model.angular_diameter_distance( cp_bcg_z ).value
			Da_ref = Test_model.angular_diameter_distance( z_ref ).value

			L_ref = Da_ref * pixel / rad2arcsec
			L_z = Da_z * pixel / rad2arcsec
			eta = L_ref / L_z

			ref_sx = cp_sx / eta
			ref_sy = cp_sy / eta

			ref_R_cut = R_cut / eta
			ref_R_pix = cp_Rpix / eta

			##. 
			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_cID', 'orin_Rsat_phy', 
					'shufl_cID', 'cp_sx', 'cp_sy', 'cp_PA', 'cp_Rpix', 'cp_Rsat_phy', 
					'cp_bcg_ra', 'cp_bcg_dec', 'cp_bcg_z', 'cut_size']

			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, clus_ID, Rsat_phy, 
					cp_clus_ID, ref_sx, ref_sy, cp_PA, ref_R_pix, cp_Rsat_phy, 
					cp_bcg_ra, cp_bcg_dec, cp_bcg_z, ref_R_cut ]

			fill = dict( zip( keys, values ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, tt),)

