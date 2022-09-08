"""
This file use to : 1) point target galaxy in the resampled image frame
				 : 2) cutout the image region centered on the pointed galaxy (at z-ref)
				 : 3) add mask on the sutout region
				 : 4) image scaling or reshape
"""
import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
import scipy.interpolate as interp


###... cosmology
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)


##... constant
rad2asec = U.rad.to(U.arcsec)


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


### === use mock BCG+ICL (without background subtraction) at zref
def BG_build_func( BG_R, BG_SB, zx, pix_size, R_max, out_file):
	"""
	BG_R, BG_SB : the surface brightness profile of background, 'nanomaggy / arcsec^2'
	z : redshfit
	pix_size : pixel scale, 'arcsec'
	R_max : the upper radius limit, creat a background within R_max, 'kpc'
	out_file : fits files
	"""

	tmp_SB_F = interp.interp1d( BG_R, BG_SB, kind = 'linear',)

	Da = Test_model.angular_diameter_distance( zx ).value # Mpc

	R_pix = ( R_max * 1e-3 / Da * rad2asec ) / pix_size
	R_pix = np.int( R_pix ) + 1

	L_wide = np.int( R_pix * 2 + 1 )  # width of mock image

	cen_x, cen_y = 0. , 0.

	Nx = np.linspace( -R_pix, R_pix, 2 * R_pix + 1 )	
	Ny = np.linspace( -R_pix, R_pix, 2 * R_pix + 1 )

	grid_x, grid_y = np.meshgrid( Nx, Ny )
	
	dR_pix = np.sqrt( ( (2 * grid_x + 1) / 2 - cen_x )**2 + ( (2 * grid_y + 1) / 2 - cen_y )**2 )

	dR_phy = dR_pix * pix_size * Da * 1e3 / rad2asec

	mod_bg = np.zeros( (L_wide, L_wide), dtype = np.float32 )

	id_vx = dR_phy >= R_max
	id_ux = dR_phy <= BG_R.min()

	id_lim = id_vx | id_ux

	#. mock flux
	mod_bg[ id_vx ] = tmp_SB_F( R_max ) * pix_size**2
	mod_bg[ id_ux ] = tmp_SB_F( BG_R.min() ) * pix_size**2

	mod_bg[ id_lim == False ] = tmp_SB_F( dR_phy[ id_lim == False ] ) * pix_size**2

	#. save the img file
	ref_pix_x = np.where( Nx == 0 )[0][0]
	ref_pix_y = np.where( Ny == 0 )[0][0]

	keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'Z', 'P_SCALE' ]
	value = ['T', 32, 2, L_wide, L_wide, ref_pix_x, ref_pix_y, zx, pix_size ]
	ff = dict( zip( keys, value) )
	fill = fits.Header( ff )
	fits.writeto( out_file, mod_bg, header = fill, overwrite = True)

	return


def sat_BG_extract_func( bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, sat_PA, band_str, zx, 
						lim_dx0, lim_dx1, lim_dy0, lim_dy1, pix_size, BG_file, out_file = None):
	"""
	cut out background from the mock 2D image~( given by the 1D BCG+ICL profile)
	or the stacked 2D image of BCG+ICL
	-------------------------
	R_sat : centric distance of satellites ( kpc )
	sat_PA : position angle of satellites, relative tp their BCG
	band_str : band ('r', 'g', 'i')
	lim_dx0, lim_dx1 : the deviation from central pixel along row direction
	lim_dy0, lim_dy1 : the deviation from central pixel along column direction
	out_file : the output file, in fits file
	BG_file : image array of background, fits file
	pix_size : pixel scale ('arcsec')
	"""

	BG_arry = fits.open( BG_file )

	BG_img = BG_arry[0].data

	BG_xn, BG_yn = BG_arry[0].header['CENTER_X'], BG_arry[0].header['CENTER_Y']

	pix_lx = np.linspace( 0, BG_img.shape[1] - 1, BG_img.shape[1] )
	pix_ly = np.linspace( 0, BG_img.shape[0] - 1, BG_img.shape[0] )
	pix_xy = np.meshgrid( pix_lx, pix_ly )

	Da = Test_model.angular_diameter_distance( zx ).value # Mpc

	R_pix = ( R_sat * 1e-3 / Da * rad2asec ) / pix_size

	off_xn, off_yn = R_pix * np.cos( sat_PA ), R_pix * np.sin( sat_PA )
	tag_xn, tag_yn = BG_xn + off_xn, BG_yn + off_yn


	if tag_xn - np.int( tag_xn ) >= 0.5:
		tag_xn = tag_xn + 1

	else:
		tag_xn = tag_xn + 0

	if tag_yn - np.int( tag_yn ) >= 0.5:
		tag_yn = tag_yn + 1

	else:
		tag_yn = tag_yn + 0


	la0, la1 = np.int( tag_xn - lim_dx0 ), np.int( tag_xn + lim_dx1 )
	lb0, lb1 = np.int( tag_yn - lim_dy0 ), np.int( tag_yn + lim_dy1 )

	cut_array = BG_img[ lb0: lb1,la0: la1 ]

	if out_file is not None:
		#. save the cutout image
		cp_cx, cp_cy = lim_dx0, lim_dy0

		Ny, Nx = cut_array.shape[0], cut_array.shape[1]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'Z_OBS', 'Z', 'P_SCALE']
		value = [ 'T', 32, 2, Nx, Ny, cp_cx, cp_cy, bcg_z, zx, pix_size ]
		ff = dict( zip( keys, value) )
		fill = fits.Header( ff )
		fits.writeto( out_file % (band_str, bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec), cut_array, header = fill, overwrite = True)

		return

	else:
		return tag_xn, tag_yn


### === extract Background cutout from the origin SDSS image frame
##. shuffle is among those clusters have simillar properties~( such as richness )
def origin_img_cut_func( clus_cat_file, img_file, band_str, sub_IDs, shufl_IDs, set_bcg_ra, set_bcg_dec, set_bcg_z, 
						set_sat_ra, set_sat_dec, shufl_sat_x, shufl_sat_y, R_cut, pix_size, out_file):
	"""
	pos_file : '.csv' file, record satellites location in their cluster image frame.
	img_file : '.fits' file, images will match the background patch cells
	band_str : filter information

	sub_IDs : the target clusters
	shufl_IDs : the cluster images
	-------------------------------
	aply the BCG and satellite position in sub_IDs to shufl_IDs, and then cutout images from shufl_IDs 

	out_file : '.fits', the output image
	R_cut : 0.5 width of cut region, in units of Kron radius or radius in sdss_photo_file, or width in units of pixel	
	pix_size : pixel scale, in units of arcsec

	"""
	dat = pds.read_csv( clus_cat_file )  ## files record satellites location in image frame

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	ref_IDs = np.array( dat['clust_ID'] )
	ref_IDs = ref_IDs.astype( int )


	N_ss = len( sub_IDs )

	for kk in range( N_ss ):

		kk_px, kk_py = shufl_sat_x[ kk ], shufl_sat_y[ kk ]

		kk_ra, kk_dec = set_sat_ra[ kk ], set_sat_dec[ kk ]

		ra_g, dec_g, z_g = set_bcg_ra[ kk ], set_bcg_dec[ kk ], set_bcg_z[ kk ]


		#. shuffle mapped cluster
		id_ux = ref_IDs == shufl_IDs[ kk ]
		cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]

		cp_img = fits.open( img_file % (band_str, cp_ra_g, cp_dec_g, cp_z_g),)
		cp_img_arr = cp_img[0].data


		#. cutout images
		dL = np.int( np.ceil( R_cut ) )
		cut_img = np.zeros( ( np.int( 2 * dL + 2 ), np.int( 2 * dL + 2 ) ), dtype = np.float32 ) + np.nan


		#. satellite region select
		d_x0 = np.max( [ kk_px - dL, 0 ] )
		d_x1 = np.min( [ kk_px + dL, cp_img_arr.shape[1] - 1 ] )

		d_y0 = np.max( [ kk_py - dL, 0 ] )
		d_y1 = np.min( [ kk_py + dL, cp_img_arr.shape[0] - 1 ] )

		d_x0 = np.int( d_x0 )
		d_x1 = np.int( d_x1 )

		d_y0 = np.int( d_y0 )
		d_y1 = np.int( d_y1 )

		pre_cut = cp_img_arr[ d_y0 : d_y1, d_x0 : d_x1 ]

		pre_cut_cx = kk_px - d_x0
		pre_cut_cy = kk_py - d_y0

		pre_cx = np.int( pre_cut_cx )
		pre_cy = np.int( pre_cut_cy )


		#. cutout image
		xn, yn = dL + 1, dL + 1

		pa0 = np.int( xn - pre_cx )
		pa1 = np.int( xn - pre_cx + pre_cut.shape[1] )

		pb0 = np.int( yn - pre_cy )
		pb1 = np.int( yn - pre_cy + pre_cut.shape[0] )

		cut_img[ pb0 : pb1, pa0 : pa1 ] = pre_cut + 0.

		_cx_off = pre_cut_cx - np.int( pre_cut_cx )
		_cy_off = pre_cut_cy - np.int( pre_cut_cy )

		cc_px, cc_py = xn + _cx_off, yn + _cy_off

		kk_Nx, kk_Ny = cut_img.shape[1], cut_img.shape[0]


		#. save fits files
		keys = [ 'SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 
									'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','BCG_Z', 'P_SCALE' ]
		value = [ 'T', 32, 2, kk_Nx, kk_Ny, cc_px, cc_py, kk_ra, kk_dec, ra_g, dec_g, z_g, pix_size ]
		ff = dict( zip( keys, value ) )
		fill = fits.Header(ff)
		fits.writeto( out_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec), cut_img, header = fill, overwrite = True)

	return


##. usimg symmetry points within the same cluster as background
def self_shufl_img_cut_func( pos_file, img_file, band_str, sub_IDs, R_cut, pix_size, out_file, err_file = None, err_grop = None):
	"""
	pos_file : '.csv' file, record satellites location in their cluster image frame.
	img_file : '.fits' file, images will match the background patch cells
	band_str : filter information	
	sub_IDs : the target clusters

	-------------------------------
	aply the BCG and satellite position in sub_IDs to shufl_IDs, and then cutout images from shufl_IDs 

	out_file : '.fits', the output image

	R_cut : 0.5 width of cut region, in units of Kron radius or radius in sdss_photo_file, or width in units of pixel	
	pix_size : pixel scale, in units of arcsec

	-------------------------------
	err_file : record information of satellites outside image frame when shuffle cluster and satellites
	err_grop : the shuffle list which has the err_catalog~("shufle_1", means the err record of the 1st group)
	"""

	if err_file is None:
		dat = pds.read_csv( pos_file % band_str )  ## files record satellites location in image frame

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
		sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		sat_PA = np.array( dat['sat_PA2bcg'] )
		ref_IDs = np.array( dat['clus_ID'] )

		ref_IDs = ref_IDs.astype( int )

	else:
		dat = pds.read_csv( pos_file % band_str )  ## files record satellites location in image frame

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
		sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		sat_PA = np.array( dat['sat_PA2bcg'] )
		ref_IDs = np.array( dat['clus_ID'] )

		ref_IDs = ref_IDs.astype( int )

		##.
		with h5py.File( err_file, 'r') as f: 
			group = f["/%s/err_clusters/" % err_grop ][()]
			group_sat = f["/%s/err_sat/" % err_grop ][()]

		out_sat_ra = ['%.4f' % ll for ll in group_sat[0] ]
		out_sat_dec = ['%.4f' % ll for ll in group_sat[1] ]

		used_order = simple_match( out_sat_ra, out_sat_dec, sat_ra, sat_dec, id_choose = False)

		bcg_ra, bcg_dec, bcg_z = bcg_ra[ used_order ], bcg_dec[ used_order ], bcg_z[ used_order ]
		sat_ra, sat_dec = sat_ra[ used_order ], sat_dec[ used_order ]

		bcg_x, bcg_y = bcg_x[ used_order ], bcg_y[ used_order ]
		sat_x, sat_y = sat_x[ used_order ], sat_y[ used_order ]

		sat_PA = sat_PA[ used_order ]
		ref_IDs = ref_IDs[ used_order ]


	N_ss = len( sub_IDs )

	for kk in range( N_ss ):

		#. target cluster satellites
		id_vx = ref_IDs == sub_IDs[ kk ]

		sub_ra, sub_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]
		ra_g, dec_g, z_g = bcg_ra[ id_vx ][0], bcg_dec[ id_vx ][0], bcg_z[ id_vx ][0]

		x_cen, y_cen = bcg_x[ id_vx ][0], bcg_y[ id_vx ][0]
		x_sat, y_sat = sat_x[ id_vx ], sat_y[ id_vx ]

		sat_R = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )
		sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

		off_x, off_y = sat_R * np.cos( sat_theta ), sat_R * np.sin( sat_theta )


		#. loop for satellites symmetry position adjust 
		N_pot = len( sat_R )

		tm_sx, tm_sy = np.zeros( N_pot,), np.zeros( N_pot,)
		id_err = np.zeros( N_pot,)  ##. records points cannot located in image frame

		for pp in range( N_pot ):

			tm_phi = np.array( [ np.pi + sat_theta[ pp ], 
								np.pi - sat_theta[ pp ], np.pi * 2 - sat_theta[ pp ] ] )

			tm_off_x = sat_R[ pp ] * np.cos( tm_phi )
			tm_off_y = sat_R[ pp ] * np.sin( tm_phi )

			tt_sx, tt_sy = x_cen + tm_off_x, y_cen + tm_off_y

			id_ux = ( tt_sx >= 0 ) & ( tt_sx < 2047 )
			id_uy = ( tt_sy >= 0 ) & ( tt_sy < 1488 )

			id_up = id_ux & id_uy

			if np.sum( id_up ) > 0:

				tm_sx[ pp ] = tt_sx[ id_up ][0]
				tm_sy[ pp ] = tt_sy[ id_up ][0]

			else:
				id_err[ pp ] = 1.

		#. 
		cp_sx, cp_sy = tm_sx + 0., tm_sy + 0.


		##. cluster image
		cp_img = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
		cp_img_arr = cp_img[0].data


		#. cutout images
		N_sat = len( cp_sx )
		dL = np.int( np.ceil( R_cut ) )

		for tt in range( N_sat ):

			cut_img = np.zeros( ( np.int( 2 * dL + 2 ), np.int( 2 * dL + 2 ) ), dtype = np.float32 ) + np.nan

			kk_px, kk_py = cp_sx[ tt ], cp_sy[ tt ]
			kk_ra, kk_dec = sub_ra[ tt ], sub_dec[ tt ]


			#. satellite region select
			d_x0 = np.max( [ kk_px - dL, 0 ] )
			d_x1 = np.min( [ kk_px + dL, cp_img_arr.shape[1] - 1 ] )

			d_y0 = np.max( [ kk_py - dL, 0 ] )
			d_y1 = np.min( [ kk_py + dL, cp_img_arr.shape[0] - 1 ] )

			d_x0 = np.int( d_x0 )
			d_x1 = np.int( d_x1 )

			d_y0 = np.int( d_y0 )
			d_y1 = np.int( d_y1 )

			pre_cut = cp_img_arr[ d_y0 : d_y1, d_x0 : d_x1 ]

			pre_cut_cx = kk_px - d_x0
			pre_cut_cy = kk_py - d_y0

			pre_cx = np.int( pre_cut_cx )
			pre_cy = np.int( pre_cut_cy )


			#. cutout image
			xn, yn = dL + 1, dL + 1

			pa0 = np.int( xn - pre_cx )
			pa1 = np.int( xn - pre_cx + pre_cut.shape[1] )

			pb0 = np.int( yn - pre_cy )
			pb1 = np.int( yn - pre_cy + pre_cut.shape[0] )

			cut_img[ pb0 : pb1, pa0 : pa1 ] = pre_cut + 0.

			_cx_off = pre_cut_cx - np.int( pre_cut_cx )
			_cy_off = pre_cut_cy - np.int( pre_cut_cy )

			cc_px, cc_py = xn + _cx_off, yn + _cy_off

			kk_Nx, kk_Ny = cut_img.shape[1], cut_img.shape[0]


			#. save fits files
			keys = [ 'SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 
										'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','BCG_Z', 'P_SCALE' ]
			value = [ 'T', 32, 2, kk_Nx, kk_Ny, cc_px, cc_py, kk_ra, kk_dec, ra_g, dec_g, z_g, pix_size ]
			ff = dict( zip( keys, value ) )
			fill = fits.Header(ff)
			fits.writeto( out_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec), cut_img, header = fill, overwrite = True)

	return


### === extract Background image from images after pixel resampling
def zref_img_cut_func( clus_cat_file, img_file, band_str, sub_IDs, shufl_IDs, set_bcg_ra, set_bcg_dec, set_bcg_z, 
						set_sat_ra, set_sat_dec, shufl_sat_x, shufl_sat_y, R_cut, pix_size, out_file):
	"""
	pos_file : '.csv' file, record satellites location in their cluster image frame.
	img_file : '.fits' file, images will match the background patch cells
	band_str : filter information
	
	sub_IDs : the target clusters
	shufl_IDs : the cluster images
	-------------------------------
	aply the BCG and satellite position in sub_IDs to shufl_IDs, and then cutout images from shufl_IDs 

	out_file : '.fits', the output image
	
	R_cut : 0.5 width of cut region, in units of Kron radius or radius in sdss_photo_file, or width in units of pixel
			(R_cut can be different among clusters and satellites)

	pix_size : pixel scale, in units of arcsec

	"""
	dat = pds.read_csv( clus_cat_file )  ## files record satellites location in image frame

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	ref_IDs = np.array( dat['clust_ID'] )
	ref_IDs = ref_IDs.astype( int )


	N_ss = len( sub_IDs )

	for kk in range( N_ss ):

		kk_px, kk_py = shufl_sat_x[ kk ], shufl_sat_y[ kk ]

		kk_ra, kk_dec = set_sat_ra[ kk ], set_sat_dec[ kk ]

		ra_g, dec_g, z_g = set_bcg_ra[ kk ], set_bcg_dec[ kk ], set_bcg_z[ kk ]


		#. shuffle mapped cluster
		id_ux = ref_IDs == shufl_IDs[ kk ]
		cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]

		cp_img = fits.open( img_file % (band_str, cp_ra_g, cp_dec_g, cp_z_g),)
		cp_img_arr = cp_img[0].data


		#. cutout images
		dL = np.int( np.ceil( R_cut[ kk ] ) )
		cut_img = np.zeros( ( np.int( 2 * dL + 2 ), np.int( 2 * dL + 2 ) ), dtype = np.float32 ) + np.nan


		#. satellite region select
		d_x0 = np.max( [ kk_px - dL, 0 ] )
		d_x1 = np.min( [ kk_px + dL, cp_img_arr.shape[1] - 1 ] )

		d_y0 = np.max( [ kk_py - dL, 0 ] )
		d_y1 = np.min( [ kk_py + dL, cp_img_arr.shape[0] - 1 ] )

		d_x0 = np.int( d_x0 )
		d_x1 = np.int( d_x1 )

		d_y0 = np.int( d_y0 )
		d_y1 = np.int( d_y1 )

		pre_cut = cp_img_arr[ d_y0 : d_y1, d_x0 : d_x1 ]

		pre_cut_cx = kk_px - d_x0
		pre_cut_cy = kk_py - d_y0

		pre_cx = np.int( pre_cut_cx )
		pre_cy = np.int( pre_cut_cy )


		#. cutout image
		xn, yn = dL + 1, dL + 1

		pa0 = np.int( xn - pre_cx )
		pa1 = np.int( xn - pre_cx + pre_cut.shape[1] )

		pb0 = np.int( yn - pre_cy )
		pb1 = np.int( yn - pre_cy + pre_cut.shape[0] )

		cut_img[ pb0 : pb1, pa0 : pa1 ] = pre_cut + 0.

		_cx_off = pre_cut_cx - np.int( pre_cut_cx )
		_cy_off = pre_cut_cy - np.int( pre_cut_cy )

		cc_px, cc_py = xn + _cx_off, yn + _cy_off

		kk_Nx, kk_Ny = cut_img.shape[1], cut_img.shape[0]


		#. save fits files
		keys = [ 'SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 
									'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','BCG_Z', 'P_SCALE' ]
		value = [ 'T', 32, 2, kk_Nx, kk_Ny, cc_px, cc_py, kk_ra, kk_dec, ra_g, dec_g, z_g, pix_size ]
		ff = dict( zip( keys, value ) )
		fill = fits.Header(ff)
		fits.writeto( out_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec), cut_img, header = fill, overwrite = True)

	return

