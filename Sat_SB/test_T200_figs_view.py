"""
compare the satellite image cut, background image cut,
and the position / location of satellite before and after shuffling
-----------------
use 200 cluster for testing
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


### === data load
def T200_select():

	bin_rich = [ 20, 30, 50, 210 ]
	sub_name = ['low-rich', 'medi-rich', 'high-rich']

	line_c = [ 'b', 'g', 'r']
	line_s = [ '--', '-', ':']

	line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']

	N_t0 = 200


	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'
	pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'

	out_path = '/home/xkchen/figs_cp/T200_test/cat/'

	"""
	##. cluster catalog
	for tt in range( 1,2 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			##. cluster catalog
			cat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1]),)
			set_IDs = np.array( cat['clust_ID'] )	
			set_IDs = set_IDs.astype( int )

			N_cs = len( set_IDs )


			##. select N_t0 from N_cs
			tt0 = np.random.choice( N_cs, N_t0, replace = False )


			##. shuffle catalog
			dat = pds.read_csv('/home/xkchen/Downloads/tables/' + 
					'clust_rich_%d-%d_%s-band_sat-shuffle-0_position.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			orin_IDs = np.array( dat['orin_cID'] )
			rand_IDs = np.array( dat['shufl_cID'] )

			orin_IDs = orin_IDs.astype( int )
			rand_IDs = rand_IDs.astype( int )


			keys = dat.columns[1:]
			N_ks = len( keys )


			tmp_arr = []
			##.
			id_vx = orin_IDs == set_IDs[ tt0[0] ]

			for pp in range( N_ks ):

				sub_arr = np.array( dat['%s' % keys[pp] ] )
				tmp_arr.append( sub_arr[ id_vx ] )

			##.
			for dd in range( 1, N_t0 ):

				id_vx = orin_IDs == set_IDs[ tt0[ dd ] ]

				for pp in range( N_ks ):

					sub_arr = np.array( dat['%s' % keys[pp] ] )
					tmp_arr[ pp ] = np.r_[ tmp_arr[ pp ], sub_arr[ id_vx ] ]

			##.
			fill = dict( zip( keys, tmp_arr ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_cat.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)

	"""


	##. origin image location match
	for tt in range( 1,2 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			##. T200 catalog
			dat = pds.read_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_cat.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

			N_sat = len( sat_ra )

			host_IDs = np.array( dat['orin_cID'] )
			rand_IDs = np.array( dat['shufl_cID'] )

			host_IDs = host_IDs.astype( int )
			rand_IDs = rand_IDs.astype( int )


			##. origin image location match
			pat = pds.read_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' % 
					(bin_rich[tt], bin_rich[tt + 1], band_str ),)

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
			data.to_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_origin-img_position.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)


	##. match with stacking information and count the number of satellite
	for tt in range( 1,2 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			##. T200 catalog
			dat = pds.read_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_cat.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

			N_sat = len( sat_ra )

			host_IDs = np.array( dat['orin_cID'] )
			rand_IDs = np.array( dat['shufl_cID'] )

			host_IDs = host_IDs.astype( int )
			rand_IDs = rand_IDs.astype( int )


			##. entire cluster catalog
			cat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1]),)
			set_IDs = np.array( cat['clust_ID'] )	
			set_IDs = set_IDs.astype( int )

			N_cs = len( set_IDs )


			##. satellite number counts
			tmp_clus_Ng = np.zeros( N_sat, )
			tmp_shufl_Ng = np.zeros( N_sat, )

			for dd in range( N_cs ):

				sub_IDs = set_IDs[ dd ]

				id_vx = host_IDs == sub_IDs
				id_ux = rand_IDs == sub_IDs

				tmp_clus_Ng[ id_vx ] = np.sum( id_vx )
				tmp_shufl_Ng[ id_ux ] = np.sum( id_ux )

			#. save
			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_Ng', 'shufl_Ng']
			values = [bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, tmp_clus_Ng, tmp_shufl_Ng ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_shufl-Ng.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)


			##. stacking information~(z-ref position)
			dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band_str )
			kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

			kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

			idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
			id_lim = sep.value < 2.7e-4

			mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
			mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, mp_imgx, mp_imgy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_sat-pos-zref.csv' % ( bin_rich[tt], bin_rich[tt + 1], band_str),)


			##. satellite cut_img information
			dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str,)
			kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			kk_imgx, kk_imgy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

			kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

			idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
			id_lim = sep.value < 2.7e-4

			mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
			mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, mp_imgx, mp_imgy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 
				'clust_rich_%d-%d_%s-band_sat-shufl_T200-test_sat-pos_pos.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)

	return

# T200_select()



### ===
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/T200_test/cat/'


##. without BCG-mask case
sat_cut_file = home + 'member_files/rich_binned_sat_test/mask_img/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'
sat_img_file = home + 'member_files/rich_binned_sat_test/resamp_img/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

bg_cut_file = home + 'member_files/rich_binned_shufl_img/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
bg_img_file = home + 'member_files/rich_binned_shufl_img/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'


##. with BCG-mask case
# sat_cut_file = home + 'member_files/mask_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'
# sat_img_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

# bg_cut_file = '/home/xkchen/project/tmp/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
# bg_img_file = '/home/xkchen/project/tmp/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'


for kk in range( 1 ):

	band_str = band[ kk ]

	##. T200 catalog
	dat = pds.read_csv( cat_path + 
		'clust_rich_30-50_%s-band_sat-shufl_T200-test_cat.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	R_sat_phy = np.array( dat['orin_Rsat_phy'] )

	clus_IDs = np.array( dat['orin_cID'] )
	clus_IDs = clus_IDs.astype( int )


	shufl_ra, shufl_dec, shufl_z = np.array( dat['cp_bcg_ra'] ), np.array( dat['cp_bcg_dec'] ), np.array( dat['cp_bcg_z'] )
	cp_sat_x, cp_sat_y = np.array( dat['cp_sx'] ), np.array( dat['cp_sy'] )
	cp_R_pix, cp_sat_PA = np.array( dat['cp_Rpix'] ), np.array( dat['cp_PA'] )
	cp_Rsat_phy = np.array( dat['cp_Rsat_phy'] )

	id_symP = np.array( dat['is_symP'] )
	shufl_IDs = np.array( dat['shufl_cID'] )
	shufl_IDs = shufl_IDs.astype( int )


	##. origin image location
	pat = pds.read_csv( cat_path + 
			'clust_rich_30-50_%s-band_sat-shufl_T200-test_origin-img_position.csv' % band_str,)
	
	p_sat_ra, p_sat_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
	p_sat_PA2bcg = np.array( pat['sat_PA2bcg'] )
	p_sat_x, p_sat_y = np.array( pat['sat_x'] ), np.array( pat['sat_y'] )
	p_bcg_x, p_bcg_y = np.array( pat['bcg_x'] ), np.array( pat['bcg_y'] )


	##. entire cluster catalog
	set_IDs = list( set( clus_IDs ) )

	N_cs = len( set_IDs )

	for pp in range( N_cs ):
	# for pp in range( 10 ):

		sub_IDs = set_IDs[ pp ]
		id_vx = clus_IDs == sub_IDs

		ra_g, dec_g, z_g = bcg_ra[ id_vx ][0], bcg_dec[ id_vx ][0], bcg_z[ id_vx ][0]
		s_ra, s_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]

		pre_sat_PA = p_sat_PA2bcg[ id_vx ]
		pre_sx, pre_sy = p_sat_x[ id_vx ], p_sat_y[ id_vx ]
		pre_bcg_x, pre_bcg_y = p_bcg_x[ id_vx ][0], p_bcg_y[ id_vx ][0]


		#. matched shuffled images
		mp_sat_x, mp_sat_y = cp_sat_x[ id_vx ], cp_sat_y[ id_vx ]
		mp_bcg_ra, mp_bcg_dec, mp_bcg_z = shufl_ra[ id_vx ], shufl_dec[ id_vx ], shufl_z[ id_vx ]
		mp_R_pix = cp_R_pix[ id_vx ]

		N_pp = len( s_ra )


		#. origin images
		orin_img = fits.open( '/home/xkchen/data/SDSS/photo_data/' + 
							'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g),)

		orin_arr = orin_img[0].data
		Header = orin_img[0].header

		# pre_wcs = awc.WCS( Header )
		# pre_sx, pre_sy = pre_wcs.all_world2pix( s_ra, s_dec, 0 )
		# pre_bcg_x, pre_bcg_y = pre_wcs.all_world2pix( ra_g, dec_g, 0 )


		#. random match images
		for mm in range( N_pp ):

			#.satellite
			mm_sat_ra, mm_sat_dec = s_ra[ mm ], s_dec[ mm ]
			mm_sat_x, mm_sat_y = pre_sx[ mm ], pre_sy[ mm ]

			mm_Rpix = np.sqrt( (mm_sat_x - pre_bcg_x)**2 + (mm_sat_y - pre_bcg_y)**2 )
			mm_sat_PA = pre_sat_PA[ mm ]

			#.
			kt_ra, kt_dec, kt_z = mp_bcg_ra[ mm ], mp_bcg_dec[ mm ], mp_bcg_z[ mm ]
			kt_sx, kt_sy = mp_sat_x[ mm ], mp_sat_y[ mm ]
			kt_R_pix = mp_R_pix[ mm ]


			#. full image frame
			shufl_img = fits.open( '/home/xkchen/data/SDSS/photo_data/' + 
							'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, kt_ra, kt_dec, kt_z),)
			shufl_arr = shufl_img[0].data
			pos_wcs = awc.WCS( shufl_img[0].header )

			pos_bcg_x, pos_bcg_y = pos_wcs.all_world2pix( kt_ra, kt_dec, 0 )
			pos_sx, pos_sy = pos_bcg_x + kt_R_pix * np.cos( mm_sat_PA ), pos_bcg_y + kt_R_pix * np.sin( mm_sat_PA )


			#. cutout satellite image
			cut_img = fits.open( sat_cut_file % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			sat_cut_arr = cut_img[ 0 ].data

			cut_img = fits.open( sat_img_file % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			sat_resamp_arr = cut_img[ 0 ].data

			#. cutout background image
			cut_img = fits.open( bg_cut_file % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			bg_cut_arr = cut_img[ 0 ].data

			cut_img = fits.open( bg_img_file % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			bg_resamp_arr = cut_img[ 0 ].data


			##. figs
			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
			ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

			ax0.set_title('orin_img: ra%.3f dec%.3f z%.3f, ra%.4f, dec%.4f' % (ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			ax0.imshow( orin_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
			ax0.scatter( mm_sat_x, mm_sat_y, s = 10, marker = 's', edgecolors = 'b', facecolors = 'none',)
			ax0.scatter( pre_bcg_x, pre_bcg_y, s = 10, marker = 'o', edgecolors = 'r', facecolors = 'none',)

			ax1.set_title('shufl_img: ra%.3f dec%.3f z%.3f' % (kt_ra, kt_dec, kt_z),)
			ax1.imshow( shufl_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
			ax1.scatter( kt_sx, kt_sy, s = 10, marker = 's', edgecolors = 'g', facecolors = 'none',)
			ax1.scatter( pos_sx, pos_sy, s = 10, marker = 's', edgecolors = 'b', facecolors = 'none',)
			ax1.scatter( pos_bcg_x, pos_bcg_y, s = 10, marker = 'o', edgecolors = 'r', facecolors = 'none',)

			plt.savefig('/home/xkchen/figs/' + 
				'clust_map_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f.png' % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			plt.close()


			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
			ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

			ax0.set_title('orin_img: ra%.3f dec%.3f z%.3f, ra%.4f, dec%.4f' % (ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			ax0.imshow( sat_cut_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-1,)

			ax1.set_title('shufl_img: ra%.3f dec%.3f z%.3f' % (kt_ra, kt_dec, kt_z),)
			ax1.imshow( bg_cut_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-1,)

			plt.savefig('/home/xkchen/figs/' + 
				'clust_map_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_cut.png' % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			plt.close()


			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
			ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])			

			ax0.set_title('orin_img: ra%.3f dec%.3f z%.3f, ra%.4f, dec%.4f' % (ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			ax0.imshow( sat_resamp_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-1,)

			ax1.set_title('shufl_img: ra%.3f dec%.3f z%.3f' % (kt_ra, kt_dec, kt_z),)
			ax1.imshow( bg_resamp_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-1,)

			plt.savefig('/home/xkchen/figs/' + 
				'clust_map_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_cut-resamp.png' % (band_str, ra_g, dec_g, z_g, mm_sat_ra, mm_sat_dec),)
			plt.close()

