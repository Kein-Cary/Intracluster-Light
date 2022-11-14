"""
rebinned satellite with BCG stellar mass
"""
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


### === 
def sat_rich_Rscal_bin():

	#.
	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'

	#.
	bin_rich = [ 20, 30, 50, 210 ]
	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

	##. fixed R for all richness subsample
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	##. radius binned satellite
	for pp in range( 2 ):

		for kk in range( 3 ):

			##.
			s_dat = pds.read_csv( cat_path + '%s_clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % 
								(cat_lis[ pp ], bin_rich[kk], bin_rich[kk + 1]), )

			bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
			p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

			p_Rsat = np.array( s_dat['R_cen'] )
			p_R2Rv = np.array( s_dat['Rcen/Rv'] )
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

				##.
				keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID'] 
				values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
				fill = dict( zip( keys, values ) )
				data = pds.DataFrame( fill )
				data.to_csv( out_path + '%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' % 
							(cat_lis[pp], bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)


	##... match with stacked information
	pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'

	for dd in range( 2 ):

		for pp in range( 3 ):

			for tt in range( len( R_bins ) - 1 ):

				s_dat = pds.read_csv( out_path + '%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' % 
										(cat_lis[ dd ], bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1]), )

				bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
				p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

				pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

				##.
				for kk in range( 3 ):

					#. z-ref position
					dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk])
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
					data.to_csv( out_path + '%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_%s-band_pos-zref.csv' % 
									( cat_lis[ dd ], bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band[kk]),)

	return


##. binned satellite with radius only
def sat_Rscal_bin():

	#.
	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'

	##. fixed R for all richness subsample
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	#.
	bin_rich = [ 20, 30, 50, 210 ]

	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

	##.
	for pp in range( 2 ):

		##. radius binned satellite
		for nn in range( len( R_bins ) - 1 ):

			dat = pds.read_csv( out_path + '%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_cat.csv' 
							% (cat_lis[pp], R_bins[nn], R_bins[nn + 1]),)
			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			R_sat, R2Rv = np.array( dat['R_sat'] ), np.array( dat['R2Rv'] )

			clust_ID = np.array( dat['clus_ID'] )

			#.
			for kk in range( 1,3 ):

				dat = pds.read_csv( out_path + '%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
								% (cat_lis[pp], bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

				bcg_ra = np.r_[ bcg_ra, np.array( dat['bcg_ra'] ) ]
				bcg_dec = np.r_[ bcg_dec, np.array( dat['bcg_dec'] ) ]
				bcg_z = np.r_[ bcg_z, np.array( dat['bcg_z'] ) ]
				
				sat_ra = np.r_[ sat_ra, np.array( dat['sat_ra'] ) ]
				sat_dec = np.r_[ sat_dec, np.array( dat['sat_dec'] ) ]
				
				R_sat = np.r_[ R_sat, np.array( dat['R_sat'] ) ]
				R2Rv = np.r_[ R2Rv, np.array( dat['R2Rv'] ) ]

				clust_ID = np.r_[ clust_ID, np.array( dat['clus_ID'] ) ]

			##.
			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID']
			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, R2Rv, clust_ID ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + '%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_cat.csv' 
						% (cat_lis[pp], R_bins[nn], R_bins[nn + 1]),)

	##... match with stacked information
	pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'

	for dd in range( 2 ):

		for tt in range( len( R_bins ) - 1 ):

			s_dat = pds.read_csv( out_path + '%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_cat.csv' 
								% (cat_lis[dd], R_bins[tt], R_bins[tt + 1]),)

			bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
			p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

			pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			##.
			for kk in range( 3 ):

				#. z-ref position
				dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk])
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
				data.to_csv( out_path + '%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_%s-band_pos-zref.csv' 
							% (cat_lis[dd], R_bins[tt], R_bins[tt + 1], band[kk]),)

	##... shuffle list mapping
	list_order = 13

	for xx in range( 3 ):

		band_str = band[ xx ]

		for pp in range( 2 ):

			##...
			for nn in range( len( R_bins ) - 1 ):

				dat = pds.read_csv( out_path + '%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_cat.csv' 
							% (cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

				keys = dat.columns[1:]
				N_ks = len( keys )

				tmp_arr = []

				for mm in range( N_ks ):

					sub_arr = np.array( dat[ keys[ mm ] ] )

					tmp_arr.append( sub_arr )

				#.
				for kk in range( 1,3 ):

					dat = pds.read_csv( out_path + 
									'%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_cat.csv' 
									% (cat_lis[pp], bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

					for mm in range( N_ks ):

						sub_arr = np.array( dat[ keys[ mm ] ] )

						tmp_arr[ mm ] = np.r_[ tmp_arr[ mm ], sub_arr ]

				##. save
				fill = dict( zip( keys, tmp_arr ) )
				data = pds.DataFrame( fill )
				data.to_csv( out_path + '%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_cat.csv' 
							% ( cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

			##...
			for nn in range( len( R_bins ) - 1 ):

				##.
				dat = pds.read_csv( out_path + 
							'%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_origin-img_position.csv'
							% (cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

				keys = dat.columns[1:]
				N_ks = len( keys )

				tmp_arr = []

				for mm in range( N_ks ):

					sub_arr = np.array( dat[ keys[ mm ] ] )

					tmp_arr.append( sub_arr )

				#.
				for kk in range( 1,3 ):

					dat = pds.read_csv( out_path + 
								'%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_origin-img_position.csv'
								% (cat_lis[pp], bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

					for mm in range( N_ks ):

						sub_arr = np.array( dat[ keys[ mm ] ] )

						tmp_arr[ mm ] = np.r_[ tmp_arr[ mm ], sub_arr ]

				##. save
				fill = dict( zip( keys, tmp_arr ) )
				data = pds.DataFrame( fill )
				data.to_csv( out_path + 
							'%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_origin-img_position.csv' 
							% ( cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

			##...
			for nn in range( len( R_bins ) - 1 ):

				##.
				dat = pds.read_csv( out_path + 
							'%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_shufl-Ng.csv'
							% (cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

				keys = dat.columns[1:]
				N_ks = len( keys )

				tmp_arr = []

				for mm in range( N_ks ):

					sub_arr = np.array( dat[ keys[ mm ] ] )

					tmp_arr.append( sub_arr )

				#.
				for kk in range( 1,3 ):

					dat = pds.read_csv( out_path + 
								'%s_clust_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_shufl-Ng.csv'
								% (cat_lis[pp], bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

					for mm in range( N_ks ):
						sub_arr = np.array( dat[ keys[ mm ] ] )
						tmp_arr[ mm ] = np.r_[ tmp_arr[ mm ], sub_arr ]

				##. save
				fill = dict( zip( keys, tmp_arr ) )
				data = pds.DataFrame( fill )
				data.to_csv( out_path + 
							'%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_%s-band_sat-shufl-%d_shufl-Ng.csv' 
							% ( cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str, list_order),)

	return


##.
# sat_rich_Rscal_bin()
sat_Rscal_bin()
raise



##. figs 
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

bin_rich = [ 20, 30, 50, 210 ]
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']
line_c = ['b', 'g', 'r']


#.
for pp in range( 2 ):

	#.
	fig = plt.figure()
	ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

	#.
	for kk in range( 3 ):

		##. fixed R for all richness subsample
		R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

		##.
		s_dat = pds.read_csv( cat_path + '%s_clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % 
								(cat_lis[ pp ], bin_rich[kk], bin_rich[kk + 1]),)

		bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
		p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

		p_Rsat = np.array( s_dat['R_cen'] )
		p_R2Rv = np.array( s_dat['Rcen/Rv'] )

		#.
		for qq in range( len(R_bins) - 1 ):

			sub_N0 = ( p_R2Rv >= R_bins[ qq ] ) & ( p_R2Rv < R_bins[ qq + 1 ] )
			print( np.sum( sub_N0 ) )

		print( '*' * 10 )

		#.
		R_edgs = np.logspace( -3, 0, 55 )

		ax.hist( p_R2Rv, bins = R_edgs, histtype = 'step', density = False, color = line_c[kk], label = line_name[kk],)

		for qq in range( 1, len(R_bins) ):

			ax.axvline( R_bins[ qq ], ls = ':', color = line_c[kk], alpha = 0.55,)

	ax.legend( loc = 2,)
	ax.set_xlabel('$R_{sat} \, / \, R_{200m}$')
	ax.set_xscale('log')
	ax.set_xlim( 1e-3, 1 )

	ax.set_ylabel('# of galaxies')
	ax.set_yscale('log')

	plt.savefig('/home/xkchen/%s_rich_R_rebin_hist.png' % cat_lis[ pp ], dpi = 300)
	plt.close()

