import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import scipy.stats as sts
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from Mass_rich_radius import rich2R_Simet
from img_sat_fig_out_mode import zref_sat_pos_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
z_ref = 0.25
pixel = 0.396
a_ref = 1 / (z_ref + 1)


### === ###
def sat_scaleR_binned():

	cat_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/cat/'
	out_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/cat_Rbin/'

	##. fixed R for all richness subsample
	# R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 0.6139, 1] )

	##.
	bin_rich = [ 20, 30, 50, 210 ]

	##... radius binned satellite
	for kk in range( 3 ):

		##.
		s_dat = pds.read_csv( cat_path + 
			'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

		bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
		p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

		p_Rsat = np.array( s_dat['R_sat'] )    ##. Mpc / h~(physical)
		p_R2Rv = np.array( s_dat['Rsat/Rv'] )
		clus_IDs = np.array( s_dat['clus_ID'] )


		##. division
		for nn in range( len( R_bins ) - 1 ):

			if nn == len( R_bins ) - 2:
				sub_N = p_R2Rv >= R_bins[ nn ]
			else:
				sub_N = (p_R2Rv >= R_bins[ nn ]) & (p_R2Rv < R_bins[ nn + 1])

			##. save
			out_c_ra, out_c_dec, out_c_z = bcg_ra[ sub_N ], bcg_dec[ sub_N ], bcg_z[ sub_N ]
			out_s_ra, out_s_dec = p_ra[ sub_N ], p_dec[ sub_N ]
			out_Rsat = p_Rsat[ sub_N ]
			out_R2Rv = p_R2Rv[ sub_N ]
			out_clus_ID = clus_IDs[ sub_N ]

			# print( len( out_c_ra ) )

			##.
			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID'] 
			values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
						% ( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

		# print( '*' * 10 )


	##... match the P_mem information
	pre_cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
	pre_table = pre_cat[1].data
	pre_ra, pre_dec = np.array( pre_table['RA'] ), np.array( pre_table['DEC'] )
	pre_Pm = np.array( pre_table['P'] )

	pre_coord = SkyCoord( ra = pre_ra * U.deg, dec = pre_dec * U.deg )

	for kk in range( 3 ):

		for nn in range( len( R_bins ) - 1 ):

			dat = pds.read_csv( out_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
						% ( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]), )

			nn_bcg_ra, nn_bcg_dec = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] )
			nn_bcg_z = np.array( dat['bcg_z'] )
			p_ra, p_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			##. Pm match
			p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			idx, sep, d3d = p_coord.match_to_catalog_sky( pre_coord )
			id_lim = sep.value < 2.7e-4 

			mp_Pm = pre_Pm[ idx[ id_lim ] ]

			##. save
			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'P_mem']
			values = [ nn_bcg_ra, nn_bcg_dec, nn_bcg_z, p_ra, p_dec, mp_Pm ] 
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_Pm_cat.csv' 
						% ( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]), )


	##... match with stacked information
	for pp in range( 3 ):

		for tt in range( len( R_bins ) - 1 ):

			s_dat = pds.read_csv( out_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv'
						% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1]), )

			bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
			p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

			pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			##.
			for kk in range( 3 ):

				#. z-ref position
				dat = pds.read_csv( '/home/xkchen/Pictures/BG_calib_SBs/sat_cat_z02_03/' + 
						'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk],)
				kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
				kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

				kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

				idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
				id_lim = sep.value < 2.7e-4

				mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
				mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

				keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
				values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
				fill = dict( zip( keys, values ) )
				data = pds.DataFrame( fill )
				data.to_csv( out_path + 
							'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem-%s-band_pos-zref.csv' 
							% (bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band[kk]),)

	return

##. origin frame location of subsample
##. for shuffle-mapping in Background measurement
def orin_frame_pos_map():

	cat_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/cat_Rbin/'
	out_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/cat_Rbin/'

	##. fixed R for all richness subsample
	# R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 0.6139, 1] )


	##.
	bin_rich = [ 20, 30, 50, 210 ]

	##... radius binned satellite
	for pp in range( 3 ):

		for tt in range( len( R_bins ) - 1 ):

			s_dat = pds.read_csv( cat_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv'
						% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1]), )

			bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
			p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

			pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			##.
			for kk in range( 3 ):

				#.
				pat = pds.read_csv('/home/xkchen/Pictures/BG_calib_SBs/sat_cat_z02_03/' + 
						'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band[kk],)

				keys = pat.columns[1:]

				kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
				kk_IDs = np.array( pat['clus_ID'] )

				kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

				#.
				idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
				id_lim = sep.value < 2.7e-4

				mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
				mp_IDs = kk_IDs[ idx[ id_lim ] ]

				#. record matched information
				tmp_arr = []
				N_ks = len( keys )

				#.
				for nn in range( N_ks ):

					sub_arr = np.array( pat['%s' % keys[ nn ] ] )
					tmp_arr.append( sub_arr[ idx[ id_lim ] ] )

				fill = dict( zip( keys, tmp_arr ) )
				data = pds.DataFrame( fill )
				data.to_csv( out_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_%s-band_orin-pos_cat.csv'
					% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band[ kk ] ),)

	return


### === ###
sat_scaleR_binned()

orin_frame_pos_map()

