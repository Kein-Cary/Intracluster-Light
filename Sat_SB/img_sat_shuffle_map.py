"""
Use to record shuffle list of satellite table
"""
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


### ===== ### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

band = ['r', 'g', 'i']
rad2arcsec = U.rad.to( U.arcsec )



#### === ### no-alignment
def no_alin_shufl_func( clust_cat, member_sat_cat, out_file, pixel = 0.396, id_fixRs = False):
	"""
	clust_cat : .csv file, including cluster ( ra, dec, z, and ID in SDSS redMaPPer )
	member_sat_cat : .csv file, mapping SDSS redMapper catalog(have applied sample selection)
					to member catalog
	out_file : .csv file
	pixel : pixel size (unit : arcsec), by default is 0.396~( SDSS )

	----------------------------------------------------------------
	id_fixRs : fixed the distance of R_sat / R_v or not~( False, default case)
	"""

	##. cluster catalog
	cat = pds.read_csv( clust_cat )

	set_Rv = np.array( cat['R_vir'] )  ##. Mpc / h, physical
	set_Rv = set_Rv * 1e3 / h          ##. kpc

	set_IDs = np.array( cat['clust_ID'] )
	set_IDs = set_IDs.astype( int )


	##. cluster member catalog, satellite location table
	dat = pds.read_csv( member_sat_cat )

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
	for qq in range( N_sat ):

		ra_g, dec_g, z_g = bcg_ra[ qq ], bcg_dec[ qq ], bcg_z[ qq ]
		sub_ra, sub_dec = sat_ra[ qq ], sat_dec[ qq ]

		x_cen, y_cen = bcg_x[ qq ], bcg_y[ qq ]
		x_sat, y_sat = sat_x[ qq ], sat_y[ qq ]

		sat_Rpix = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )  ### in pixel number
		sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

		#.
		Da_obs = Test_model.angular_diameter_distance( z_g ).value
		qq_R_sat = sat_Rpix * pixel * Da_obs * 1e3 / rad2arcsec     ##. kpc
		pre_R_sat[ qq ] = qq_R_sat
		pre_R_vir = set_Rv[ set_IDs == clus_IDs[ qq ] ][ 0 ]


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

		cp_R_vir = set_Rv[ set_IDs == rand_ID ][0]

		cp_Da = Test_model.angular_diameter_distance( cp_z_g ).value

		#.
		if id_fixRs:

			_r_ratio = qq_R_sat / pre_R_vir
			_cp_R_sat = cp_R_vir * _r_ratio
			cp_sat_Rpix = ( ( _cp_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

		else:
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
	out_data.to_csv( out_file )

	return


### === ### align with BCG
def BCG_align_shufl_func( clust_cat, member_sat_cat, out_file, pixel = 0.396, id_fixRs = False):
	"""
	clust_cat : .csv file, including cluster ( ra, dec, z, and ID in SDSS redMaPPer )
	member_sat_cat : .csv file, mapping SDSS redMapper catalog(have applied sample selection)
					to member catalog
	out_file : .csv file
	pixel : pixel size (unit : arcsec), by default is 0.396~( SDSS )

	----------------------------------------------------------------
	id_fixRs : fixed the distance of R_sat / R_v or not~( False, default case)
	"""

	##. cluster catalog
	cat = pds.read_csv( clust_cat )

	set_Rv = np.array( cat['R_vir'] )  ##. Mpc / h, physical
	set_Rv = set_Rv * 1e3 / h          ##. kpc

	set_IDs = np.array( cat['clust_ID'] )
	set_IDs = set_IDs.astype( int )


	##. cluster member catalog, satellite location table
	dat = pds.read_csv( member_sat_cat )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
	sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	#.
	sat_PA2bcg = np.array( dat['sat_PA2bcg'] )   ##. sat PA to BCG but relative to the longer side~(2048 pixels)
	bcg_PAs = np.array( dat['BCG_PA'] )

	clus_IDs = np.array( dat['clus_ID'] )
	clus_IDs = clus_IDs.astype( int )

	N_sat = len( sat_ra )


	pre_R_sat = np.zeros( N_sat, )

	#. record the finally mapped cluster ID and the location of satellite in that image
	shufl_IDs = np.zeros( N_sat, dtype = int )
	shufl_ra, shufl_dec, shufl_z = np.zeros( N_sat,), np.zeros( N_sat,), np.zeros( N_sat,)

	shufl_sx, shufl_sy = np.zeros( N_sat, ), np.zeros( N_sat, )

	shufl_Rpix =  np.zeros( N_sat, )
	shufl_R_phy = np.zeros( N_sat, )    ###. in unit kpc

	shufl_PA = np.zeros( N_sat, )       ###. position angle to BCG major axis
	shufl_bcg_PA = np.zeros( N_sat, )

	#. record these points is symmetry (id_symP = 1,2,3) or not ( id_symP = 0 )
	id_symP = np.zeros( N_sat, dtype = int )

	#.
	for qq in range( N_sat ):

		ra_g, dec_g, z_g = bcg_ra[ qq ], bcg_dec[ qq ], bcg_z[ qq ]
		sub_ra, sub_dec = sat_ra[ qq ], sat_dec[ qq ]

		x_cen, y_cen = bcg_x[ qq ], bcg_y[ qq ]
		x_sat, y_sat = sat_x[ qq ], sat_y[ qq ]

		sat_Rpix = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )  ### in pixel number

		sat_chi2bcg = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )
		qq_bcg_chi = bcg_PAs[ qq ]

		delt_chi = sat_chi2bcg - qq_bcg_chi

		Da_obs = Test_model.angular_diameter_distance( z_g ).value
		qq_R_sat = sat_Rpix * pixel * Da_obs * 1e3 / rad2arcsec     ##. kpc

		pre_R_sat[ qq ] = qq_R_sat
		pre_R_vir = set_Rv[ set_IDs == clus_IDs[ qq ] ][ 0 ]


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

			cp_bcg_PA = bcg_PAs[ id_ux ][0]
			cp_R_vir = set_Rv[ set_IDs == rand_ID ][0]

			cp_Da = Test_model.angular_diameter_distance( cp_z_g ).value

			#.
			if id_fixRs:

				_r_ratio = qq_R_sat / pre_R_vir
				_cp_R_sat = cp_R_vir * _r_ratio
				cp_sat_Rpix = ( ( _cp_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

			else:
				cp_sat_Rpix = ( ( qq_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

			#.
			off_x = cp_sat_Rpix * np.cos( delt_chi + cp_bcg_PA )
			off_y = cp_sat_Rpix * np.sin( delt_chi + cp_bcg_PA )

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

				#.
				shufl_PA[ qq ] = delt_chi
				shufl_bcg_PA[ qq ] = cp_bcg_PA

				shufl_ra[ qq ] = cp_ra_g
				shufl_dec[ qq ] = cp_dec_g
				shufl_z[ qq ] = cp_z_g

				cp_R_phy = ( cp_sat_Rpix * pixel * cp_Da * 1e3 ) / rad2arcsec

				shufl_R_phy[ qq ] = cp_R_phy
				id_symP[ qq ] = 0

				id_live = id_live - 1
				break

			else:

				tmp_phi = np.array( [ 	np.pi + delt_chi, 
										np.pi - delt_chi, 
										np.pi * 2 - delt_chi ] )

				tmp_off_x = cp_sat_Rpix * np.cos( tmp_phi + cp_bcg_PA )
				tmp_off_y = cp_sat_Rpix * np.sin( tmp_phi + cp_bcg_PA )

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

					#.
					shufl_PA[ qq ] = tmp_phi[ id_lim ][ rt0[ 0 ] ]
					shufl_bcg_PA[ qq ] = cp_bcg_PA

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
			'shufl_cID', 'cp_sx', 'cp_sy', 'cp_PA2bcg', 'cp_bcgPA', 'cp_Rpix', 'cp_Rsat_phy', 'is_symP',
			'cp_bcg_ra', 'cp_bcg_dec', 'cp_bcg_z' ]

	values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, clus_IDs, pre_R_sat, 
			shufl_IDs, shufl_sx, shufl_sy, shufl_PA, shufl_bcg_PA, shufl_Rpix, shufl_R_phy, id_symP,
			shufl_ra, shufl_dec, shufl_z ]

	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_file )

	return


### === ### align with frame
def frame_align_shufll_func(  clust_cat, member_sat_cat, out_file, pixel = 0.396, id_fixRs = False):
	"""
	clust_cat : .csv file, including cluster ( ra, dec, z, and ID in SDSS redMaPPer )
	member_sat_cat : .csv file, mapping SDSS redMapper catalog(have applied sample selection)
					to member catalog
	out_file : .csv file
	pixel : pixel size (unit : arcsec), by default is 0.396~( SDSS )

	----------------------------------------------------------------
	id_fixRs : fixed the distance of R_sat / R_v or not~( False, default case)
	"""

	##. cluster catalog
	##. cluster catalog
	cat = pds.read_csv( clust_cat )

	set_Rv = np.array( cat['R_vir'] )  ##. Mpc / h, physical
	set_Rv = set_Rv * 1e3 / h          ##. kpc

	set_IDs = np.array( cat['clust_ID'] )
	set_IDs = set_IDs.astype( int )


	##. cluster member catalog, satellite location table
	dat = pds.read_csv( member_sat_cat )

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
		pre_R_vir = set_Rv[ set_IDs == clus_IDs[ qq ] ][ 0 ]


		##. random select cluster and do a map between satellite and the 'selected' cluster
		qq_ID = clus_IDs[ qq ]

		id_vx = set_IDs == qq_ID
		ID_dex = np.where( id_vx )[0][0]

		copy_IDs = np.delete( set_IDs, ID_dex )
		cp_Ns = len( copy_IDs )

		##. random order of index in the cluster array
		rand_arr = np.random.choice( cp_Ns, cp_Ns, replace = False )


		##. try to map satellite and the 'selected' cluster
		id_loop = 0
		id_live = 1

		while id_live > 0:

			rand_ID = copy_IDs[ rand_arr[ id_loop ] ]

			id_ux = clus_IDs == rand_ID
			cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]			

			cp_R_vir = set_Rv[ set_IDs == rand_ID ][0]
			cp_Da = Test_model.angular_diameter_distance( cp_z_g ).value

			#.
			if id_fixRs:

				_r_ratio = qq_R_sat / pre_R_vir
				_cp_R_sat = cp_R_vir * _r_ratio
				cp_sat_Rpix = ( ( _cp_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

			else:
				cp_sat_Rpix = ( ( qq_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

			#.
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
	out_data.to_csv( out_file )

	return


### === ### satellite number counts for shuffle mapping catalog
##.. i.e. once a satellite is stacked, we say that the cluster image is used once,
##.. Now we need to make sure the 'times of cluster images used' is the same for 
##.. satellite image stacking and background stacking
def N_shufl_count_func( shufl_cat, clust_cat, out_file ):
	"""
	shufl_cat : the shuffle mapping catalog
	clust_cat : all clusters whose satellites are list in the shufl_cat above
	out_file : output file~(.csv)
	"""

	##. member cat.
	dat = pds.read_csv( shufl_cat )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	orin_IDs = np.array( dat['orin_cID'] )
	rand_IDs = np.array( dat['shufl_cID'] )

	orin_IDs = orin_IDs.astype( int )
	rand_IDs = rand_IDs.astype( int )


	##. cluster cat.
	dat = pds.read_csv( clust_cat )

	clus_IDs = np.array( dat['clust_ID'] )
	clus_IDs = clus_IDs.astype( int )

	N_w = len( clus_IDs )
	N_ss = len( bcg_ra )

	##.
	pre_Ng = np.zeros( N_ss, )
	shufl_Ng = np.zeros( N_ss, )

	#.
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
	out_data.to_csv( out_file )

	return


### === ### zref_img extract for shuffle catalog
def zref_cut_func( shuffle_list, out_file, cut_size = 320, z_ref = 0.25, pixel = 0.396):
	"""
	shuffle_list : .csv file, including origin BCG_information~(ra, dec, z)
					Satellite_information~(ra, dec) and their location in origin image
					frame
	out_file : .csv file, record galaxies' position in resampled image
	cut_size : the cut_img size, by default is 320~(~500kpc)

	--------------------------------------------------------
	z_ref : reference redshift of galaxy images, by default is 0.25
			(roughly the median of cluster sample)
	pixel : pixel size (unit:arcsec) of image, by default is 0.396~(SDSS)
	"""

	##.
	cat = pds.read_csv( shuffle_list )

	clus_ID = np.array( cat['orin_cID'] )
	clus_ID = clus_ID.astype( int )

	bcg_ra, bcg_dec, bcg_z = np.array( cat['bcg_ra'] ), np.array( cat['bcg_dec'] ), np.array( cat['bcg_z'] )
	sat_ra, sat_dec = np.array( cat['sat_ra'] ), np.array( cat['sat_dec'] )

	Rsat_phy = np.array( cat['orin_Rsat_phy'] )


	##. satellite shuffle information
	cp_sx, cp_sy = np.array( cat['cp_sx'] ), np.array( cat['cp_sy'] )

	##. Position angle relative to BCG in image frame
	try:
		cp_PA = np.array( cat['cp_PA'] )

	except:
		cp_PA = np.array( cat['cp_PA2bcg'] )

	#.
	cp_Rpix, cp_Rsat_phy = np.array( cat['cp_Rpix'] ), np.array( cat['cp_Rsat_phy'] )

	cp_bcg_ra, cp_bcg_dec = np.array( cat['cp_bcg_ra'] ), np.array( cat['cp_bcg_dec'] )
	cp_bcg_z = np.array( cat['cp_bcg_z'] )

	cp_clus_ID = np.array( cat['shufl_cID'] )


	##. refer to information at z_ref
	R_cut = cut_size

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
	out_data.to_csv( out_file )

	return



### %%%%%%%%%%%%%%%%%%%%%%%% ###
##. control galaxy shuffle mapping
def field_galaxy_shufl_func(random_map_cat, contrl_galx_cat, N_shufl, out_file, 
							np_x = 2048, np_y = 1489):
	"""
	random_map_cat : the catalog of control galaxy < -- > map to random catalog in RedMaPPer
				(we use the images of random catalog as the observation of field galaxy)
	
	contrl_galx_cat : the entire catalog of mapped field galaxy with 
				the "5-band magnitude + (g-r) + (r-i) + (u-g)" space

	out_file : output the shuffle catalog (*.csv)

	N_shufl : times of randomly mapping field galaxies and images of random catalog in RedMaPPer

	np_x, np_y : columns and rows of image frame, for SDSS, np_x = 2048, np_y = 1489
	"""

	##.
	dat = pds.read_csv( random_map_cat )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec, sat_z = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] ), np.array( dat['sat_z'] )

	sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	coord_sat = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )


	##.
	ref_cat = pds.read_csv( contrl_galx_cat )

	cp_s_ra, cp_s_dec, cp_s_z = np.array( ref_cat['ra'] ), np.array( ref_cat['dec'] ), np.array( ref_cat['z'] )
	cp_clus_z = np.array( ref_cat['map_clus_z'] )

	cp_coord_sat = SkyCoord( ra = cp_s_ra * U.deg, dec = cp_s_dec * U.deg )

	idx, d2d, d3d = coord_sat.match_to_catalog_sky( cp_coord_sat )
	id_lim = d2d.value < 2.7e-4

	##. use for resampling ~ (save as 'z_bg' in shuffle catalog )
	ref_clus_z = cp_clus_z[ idx[id_lim] ]


	N_s = len( sat_ra )

	for dd in range( N_shufl ):

		##. 
		rand_dex = np.random.choice( N_s, N_s, replace = False )

		mp_ra, mp_dec, mp_z = bcg_ra[ rand_dex ], bcg_dec[ rand_dex ], bcg_z[ rand_dex ]
		mp_bg_z = ref_clus_z[ rand_dex ]

		##. use the symmetry points of origin image frame
		mp_gx = np_x - sat_x
		mp_gy = np_y - sat_y


		##.
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'orin_x', 'orin_y',
				'shfl_bcg_ra', 'shfl_bcg_dec', 'shfl_bcg_z', 'shfl_x', 'shfl_y', 'z_bg']
		values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, sat_z, sat_x, sat_y, 
				mp_ra, mp_dec, mp_z, mp_gx, mp_gy, mp_bg_z ]

		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_file % dd )

	return

