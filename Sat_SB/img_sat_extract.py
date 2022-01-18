"""
This file use to : 1) point target galaxy in the image frame
				 : 2) cutout the image region centered on the pointed galaxy
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

from fig_out_module import WCS_to_pixel_func, pixel_to_WCS_func


### ===
def cat_combine( cat_lis, ra, dec, z, alt_G_size, head_info, img_lis):

	Ns = len( cat_lis )

	tot_cx, tot_cy = np.array([]), np.array([])
	tot_a, tot_b = np.array([]), np.array([])
	tot_theta = np.array([])
	tot_Numb = 0

	for ll in range( Ns ):

		ext_cat = cat_lis[ll] % ( ra, dec, z )
		ext_img = img_lis[ll] % ( ra, dec, z )

		try:

			source = asc.read( ext_cat )
			Numb = np.array(source['NUMBER'][-1])
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])

			cx = np.array(source['X_IMAGE'])
			cy = np.array(source['Y_IMAGE'])

			#.. extra-image info.
			ext_data = fits.open( ext_img )
			ext_head = ext_data[0].header

			c_ra, c_dec = pixel_to_WCS_func( cx, cy, ext_head )

			#.. reverse to target filter
			m_cx, m_cy = WCS_to_pixel_func( c_ra, c_dec, head_info )

			if alt_G_size is not None:
				Kron = alt_G_size * 2.
			else:
				Kron = 16

			a = Kron * A
			b = Kron * B

			tot_cx = np.r_[tot_cx, m_cx]
			tot_cy = np.r_[tot_cy, m_cy]
			tot_a = np.r_[tot_a, a]
			tot_b = np.r_[tot_b, b]
			tot_theta = np.r_[tot_theta, theta]
			tot_Numb = tot_Numb + Numb

		except:
			print('error occur')
			continue

	return tot_Numb, tot_cx, tot_cy, tot_a, tot_b, tot_theta


def mask_with_G_tag( img_arry, cen_x, cen_y, cen_ar, cen_br, cen_cr, cen_chi, gal_arr):
	## cen_x, cen_y : location of target galaxy in image frame

	cx, cy, a, b, theta = gal_arr[:]

	ef1 = ( cx - cen_x ) * np.cos( cen_chi ) + ( cy - cen_y ) * np.sin( cen_chi )
	ef2 = ( cy - cen_y ) * np.cos( cen_chi ) - ( cx - cen_x ) * np.sin( cen_chi )
	er = ef1**2 / cen_ar**2 + ef2**2 / cen_br**2
	idx = er < 1

	if np.sum( idx ) >= 1:
		id_bcg = np.where( idx == True )[0]

	if np.sum( idx ) == 0:
		id_bcg = np.array( [] )

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	Numb = len( major )

	mask_path = np.ones( (img_arry.shape[0], img_arry.shape[1]), dtype = np.float32)
	ox = np.linspace(0, img_arry.shape[1] - 1, img_arry.shape[1] )
	oy = np.linspace(0, img_arry.shape[0] - 1, img_arry.shape[0] )

	basic_coord = np.array( np.meshgrid(ox, oy) )

	# masking 'galaxies'
	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		if k in id_bcg:
			continue

		else:
			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r + 1), img_arry.shape[1] ] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r + 1), img_arry.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img_arry

	return mask_img


### === 
def sate_Extract_func( d_file, bcg_ra, bcg_dec, bcg_z, ra_set, dec_set, band, gal_file, out_file, R_cut, 
						offset_file = None, sdss_phot_file = None, alt_G_size = None, pixel = 0.396):
	"""
	d_file : path where image data saved (include file-name structure:'/xxx/xxx/xxx.xxx')
	bcg_ra, bcg_dec, bcg_z : the cluster or BCG information (for image load)
	----------------
	ra_set, dec_set : the information of satellites (for location found in frame)
	assuming redshift of satellites are the same as cluster

	band : filter information
	gal_file : the source catalog based on SExTractor calculation
	out_file : '.fits' file, the extracout satellite images
	R_cut : 0.5 width of cut region, in units of Kron radius or radius in sdss_photo_file, or width in units of pixel

	offset_file : source position correction for current band images
	sdss_photo_file : photometry information (mainly includes shape, position, and size) derived in SDSS
					if it is 'None', use the SExTractor running results only
	pixel : pixel scale, in unit of 'arcsec'
	"""

	##. origin image 
	img_data = fits.open( d_file % (band, bcg_ra, bcg_dec, bcg_z),)
	img_arr = img_data[ 0 ].data

	Header = img_data[0].header
	wcs_lis = awc.WCS( Header )


	if offset_file is not None:
		off_dat = pds.read_csv( offset_file % (band, bcg_ra, bcg_dec, bcg_z), )

		x2pk_off_arr = np.array( off_dat[ 'devi_pk_x' ] )
		y2pk_off_arr = np.array( off_dat[ 'devi_pk_y' ] )

		medi_x2pk_off = np.median( x2pk_off_arr )
		medi_y2pk_off = np.median( y2pk_off_arr )

	else:
		medi_x2pk_off = 0.
		medi_y2pk_off = 0.


	##. satellite galaxy region
	s_xn, s_yn = WCS_to_pixel_func( ra_set, dec_set, Header )
	s_xn, s_yn = s_xn + medi_x2pk_off, s_yn + medi_y2pk_off

	N_sat = len( ra_set )


	## galaxy location in targ_filter
	source = asc.read( gal_file % (band, bcg_ra, bcg_dec, bcg_z), )
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])

	peak_x = np.array( source['XPEAK_IMAGE'])
	peak_y = np.array( source['YPEAK_IMAGE'])

	if alt_G_size is not None:
		Kron = alt_G_size * 2
	else:
		Kron = 16 # 8-R_kron

	a = Kron * A
	b = Kron * B

	##. cutout satellite images
	for pp in range( N_sat ):

		kk_ra, kk_dec = ra_set[ pp ], dec_set[ pp ]


		#. find target galaxy in source catalog
		pp_cx, pp_cy = s_xn[ pp ], s_yn[ pp ]

		d_cen_R = np.sqrt( (cx - pp_cx)**2 + (cy - pp_cy)**2 )
		id_xcen = d_cen_R == d_cen_R.min()
		id_order = np.where( id_xcen )[0][0]

		kk_px, kk_py = cx[ id_order ], cy[ id_order ]
		kk_major_R = a[ id_order ]


		cen_ar = A[ id_order ] * 3
		cen_br = B[ id_order ] * 3

		cen_cr = np.sqrt( cen_ar**2 - cen_br**2 )
		cen_chi = theta[ id_order ] * np.pi / 180


		##.. cut image for given cut size
		dL = np.int( np.ceil( R_cut ) )

		cut_img = np.zeros( ( np.int( 2 * dL + 2 ), np.int( 2 * dL + 2 ) ), dtype = np.float32 ) + np.nan

		d_x0 = np.max( [ kk_px - dL, 0 ] )
		d_x1 = np.min( [ kk_px + dL, img_arr.shape[1] - 1 ] )

		d_y0 = np.max( [ kk_py - dL, 0 ] )
		d_y1 = np.min( [ kk_py + dL, img_arr.shape[0] - 1 ] )

		d_x0 = np.int( d_x0 )
		d_x1 = np.int( d_x1 )

		d_y0 = np.int( d_y0 )
		d_y1 = np.int( d_y1 )

		#. cutout image
		pre_cut = img_arr[ d_y0 : d_y1, d_x0 : d_x1 ]
		pre_cut_cx = kk_px - d_x0
		pre_cut_cy = kk_py - d_y0

		pre_cx = np.int( pre_cut_cx )
		pre_cy = np.int( pre_cut_cy )


		xn, yn = dL + 1, dL + 1

		pa0 = np.int( xn - pre_cx )
		pa1 = np.int( xn - pre_cx + pre_cut.shape[1] )

		pb0 = np.int( yn - pre_cy )
		pb1 = np.int( yn - pre_cy + pre_cut.shape[0] )

		cut_img[ pb0 : pb1, pa0 : pa1 ] = pre_cut + 0.

		_cx_off = pre_cut_cx - np.int( pre_cut_cx )
		_cy_off = pre_cut_cy - np.int( pre_cut_cy )

		cc_px, cc_py = xn + _cx_off, yn + _cy_off


		##.. peak position
		_pkx, _pky = peak_x[ id_order ], peak_y[ id_order ]

		devi_x = _pkx - kk_px
		devi_y = _pky - kk_py

		cc_pkx, cc_pky = cc_px + devi_x, cc_py + devi_y


		kk_Nx, kk_Ny = cut_img.shape[1], cut_img.shape[0]

		#. save fits files
		keys = [ 'SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'PEAK_X', 'PEAK_Y', 
				'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','BCG_Z', 'P_SCALE' ]
		value = [ 'T', 32, 2, kk_Nx, kk_Ny, cc_px, cc_py, cc_pkx, cc_pky, kk_ra, kk_dec, bcg_ra, bcg_dec, bcg_z, pixel ]
		ff = dict( zip( keys, value ) )
		fil = fits.Header(ff)
		fits.writeto( out_file % (band, bcg_ra, bcg_dec, bcg_z, kk_ra, kk_dec ), cut_img, header = fil, overwrite = True)

	return


def sate_surround_mask_func(d_file, cat_file, bcg_ra, bcg_dec, bcg_z, ra_set, dec_set, band, gal_file, out_mask_file, R_cut, 
						offset_file = None, sdss_phot_file = None, 
						extra_cat = None, extra_img = None, 
						alter_fac = None, alt_bright_R = None, alt_G_size = None, stack_info = None, pixel = 0.396):
	"""
	d_file : path where image data saved (include file-name structure:'/xxx/xxx/xxx.xxx')
	cat_file : table of stars and saturated pixels
	bcg_ra, bcg_dec, bcg_z : the cluster or BCG information (for image load)
	----------------
	ra_set, dec_set : the information of satellites (for location found in frame)
	assuming redshift of satellites are the same as cluster

	band : filter information

	gal_file : the source catalog based on SExTractor calculation
	out_mask_file : '.fits' file, the extracout satellite images with surrounding masks
	R_cut : 0.5 width of cut region, in units of Kron radius or radius in sdss_photo_file, or in units of pixel

	offset_file : source position correction for current band images
	sdss_photo_file : photometry information (mainly includes shape, position, and size) derived in SDSS
					if it is 'None', use the SExTractor running results only

	extra_cat : extral galaxy catalog for masking adjust, (list type, .cat files)
	extra_img : images of extral catalog, use for matching wcs information

	alter_fac : size adjust for normal stars
	alt_bright_R : size adjust for saturated sources
	alt_G_size : size adjust for galaxy-like sources
	
	stack_info : record the position of stacking center (here is the center of satellite galaxies)
	pixel : pixel scale, in unit of 'arcsec'
	"""

	ra_g, dec_g, z_g = bcg_ra, bcg_dec, bcg_z

	##. origin image 
	img_data = fits.open( d_file % (band, bcg_ra, bcg_dec, bcg_z),)

	Header = img_data[0].header
	wcs_lis = awc.WCS( Header )


	if offset_file is not None:
		off_dat = pds.read_csv( offset_file % (band, ra_g, dec_g, z_g), )

		x2pk_off_arr = np.array( off_dat[ 'devi_pk_x' ] )
		y2pk_off_arr = np.array( off_dat[ 'devi_pk_y' ] )

		medi_x2pk_off = np.median( x2pk_off_arr )
		medi_y2pk_off = np.median( y2pk_off_arr )
	else:
		medi_x2pk_off = 0.
		medi_y2pk_off = 0.


	## ... galaxy location in targ_filter
	source = asc.read( gal_file % (band, ra_g, dec_g, z_g), )
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])

	theta = np.array(source['THETA_IMAGE'])

	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	peak_x = np.array( source['XPEAK_IMAGE'])
	peak_y = np.array( source['YPEAK_IMAGE'])

	if alt_G_size is not None:
		Kron = alt_G_size * 2
	else:
		Kron = 16 # 8-R_kron

	a = Kron * A
	b = Kron * B


	## ... extral catalog load
	if extra_cat is not None:
		Ecat_num, Ecat_x, Ecat_y, Ecat_a, Ecat_b, Ecat_chi = cat_combine( extra_cat, ra_g, dec_g, z_g, alt_G_size, Header, extra_img)
		Ecat_x, Ecat_y = Ecat_x + medi_x2pk_off, Ecat_y + medi_y2pk_off
	else:
		Ecat_num = 0
		Ecat_x, Ecat_y, Ecat_a, Ecat_b, Ecat_chi = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


	## ... stars
	mask = cat_file % ( z_g, ra_g, dec_g )
	cat = pds.read_csv( mask, skiprows = 1 )
	set_ra = np.array( cat['ra'] )
	set_dec = np.array( cat['dec'] )
	set_mag = np.array( cat['r'] )
	OBJ = np.array( cat['type'] )
	xt = cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = WCS_to_pixel_func( set_ra, set_dec, Header )
	x, y = x + medi_x2pk_off, y + medi_y2pk_off

	set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]
	if alter_fac is not None:
		sub_A0 = lr_iso[ic] * alter_fac
		sub_B0 = sr_iso[ic] * alter_fac
	else:
		sub_A0 = lr_iso[ic] * 30
		sub_B0 = sr_iso[ic] * 30
	sub_chi0 = set_chi[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	if alt_bright_R is not None:
		sub_A2 = lr_iso[ipx] * alt_bright_R
		sub_B2 = sr_iso[ipx] * alt_bright_R
	else:
		sub_A2 = lr_iso[ipx] * 75
		sub_B2 = sr_iso[ipx] * 75

	sub_chi2 = set_chi[ipx]

	comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0] ]
	comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0] ]
	Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0] ]
	Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0] ]
	phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0] ]
	N_star = len( comx )


	gal_x = np.r_[ cx, Ecat_x ]
	gal_y = np.r_[ cy, Ecat_y ]
	gal_a = np.r_[ a, Ecat_a ]
	gal_b = np.r_[ b, Ecat_b ]
	gal_chi = np.r_[ theta, Ecat_chi ]

	gal_arr = [ gal_x, gal_y, gal_a, gal_b, gal_chi ]


	## ... array to record satellite location
	s_xn, s_yn = WCS_to_pixel_func( ra_set, dec_set, Header )
	s_xn, s_yn = s_xn + medi_x2pk_off, s_yn + medi_y2pk_off

	N_sat = len( ra_set )

	tmp_ini_x, tmp_ini_y = np.array( [] ), np.array( [] )     ## location in the original image
	tmp_cut_x, tmp_cut_y = np.array( [] ), np.array( [] )     ## location in the cutout image

	for pp in range( N_sat ):

		kk_ra, kk_dec = ra_set[ pp], dec_set[ pp]

		pp_cx, pp_cy = s_xn[ pp ], s_yn[ pp ]


		#. find target galaxy in source catalog
		d_cen_R = np.sqrt( (cx - pp_cx)**2 + (cy - pp_cy)**2 )

		id_xcen = d_cen_R == d_cen_R.min()
		id_order = np.where( id_xcen )[0][0]

		#. use the position derived by SExtractor in each band
		kk_px, kk_py = cx[ id_order ], cy[ id_order ]

		tmp_ini_x = np.r_[ tmp_ini_x, kk_px ]
		tmp_ini_y = np.r_[ tmp_ini_y, kk_py ]


		##... select samller region (compare to 8 Kron radius in ICL mask)
		cen_ar = A[ id_order ] * 2 # 1.5
		cen_br = B[ id_order ] * 2 # 1.5

		cen_cr = np.sqrt( cen_ar**2 - cen_br**2 )
		cen_chi = theta[ id_order ] * np.pi / 180

		kk_major_R = a[ id_order ]


		##.. cutout surrounding region and processing it only
		img_arr = img_data[ 0 ].data

		dL = np.int( np.ceil( R_cut ) )

		cut_img = np.zeros( ( np.int( 2 * dL + 2 ), np.int( 2 * dL + 2 ) ), dtype = np.float32 ) + np.nan

		d_x0 = np.max( [ kk_px - dL, 0 ] )
		d_x1 = np.min( [ kk_px + dL, img_arr.shape[1] - 1 ] )

		d_y0 = np.max( [ kk_py - dL, 0 ] )
		d_y1 = np.min( [ kk_py + dL, img_arr.shape[0] - 1 ] )

		d_x0 = np.int( d_x0 )
		d_x1 = np.int( d_x1 )

		d_y0 = np.int( d_y0 )
		d_y1 = np.int( d_y1 )


		#. cutout image
		pre_cut_img = img_arr[ d_y0 : d_y1, d_x0 : d_x1 ]
		pre_cut_cx = kk_px - d_x0
		pre_cut_cy = kk_py - d_y0


		#. select sources located in / surrounding this region
		_off_x0, _off_x1 = gal_x + gal_a, gal_x - gal_a
		_off_y0, _off_y1 = gal_y + gal_a, gal_y - gal_a

		id_vx = ( _off_x0 >= d_x0 ) & ( _off_x1 <= d_x1 )
		id_vy = ( _off_y0 >= d_y0 ) & ( _off_y1 <= d_y1 )
		id_obj = id_vx & id_vy

		lim_obj_x, lim_obj_y = gal_x[ id_obj ], gal_y[ id_obj ]
		lim_a, lim_b = gal_a[ id_obj ], gal_b[ id_obj ]
		lim_chi = gal_chi[ id_obj ]

		lim_obj_x, lim_obj_y = lim_obj_x - d_x0, lim_obj_y - d_y0  ## offset relative to the cut box
		lim_gal_arr = [ lim_obj_x, lim_obj_y, lim_a, lim_b, lim_chi ]

		#. galaxy mask
		pre_mask_img = mask_with_G_tag( pre_cut_img, pre_cut_cx, pre_cut_cy, cen_ar, cen_br, cen_cr, cen_chi, lim_gal_arr )


		#. stars select surrounding this region
		_off_x0, _off_x1 = comx + Lr, comx - Lr
		_off_y0, _off_y1 = comy + Lr, comy - Lr

		id_sx = ( _off_x0 >= d_x0 ) & ( _off_x1 <= d_x1 )
		id_sy = ( _off_y0 >= d_y0 ) & ( _off_y1 <= d_y1 )

		id_sobj = id_sx & id_sy

		lim_sx, lim_sy = comx[ id_sobj ], comy[ id_sobj ]
		lim_Lr, lim_Sr = Lr[ id_sobj ], Sr[ id_sobj ]
		lim_phi = phi[ id_sobj ]

		lim_sx, lim_sy = lim_sx - d_x0, lim_sy - d_y0
		lim_Ns = len( lim_sx )


		#. masking stars
		mask_path = np.ones( ( pre_cut_img.shape[0], pre_cut_img.shape[1] ), dtype = np.float32)
		ox = np.linspace( 0, pre_cut_img.shape[1] - 1, pre_cut_img.shape[1] )
		oy = np.linspace( 0, pre_cut_img.shape[0] - 1, pre_cut_img.shape[0] )

		basic_coord = np.array( np.meshgrid(ox, oy) )

		for k in range( lim_Ns ):

			xc = lim_sx[k]
			yc = lim_sy[k]

			lr = lim_Lr[k] / 2
			sr = lim_Sr[k] / 2
			cr = np.sqrt( lr**2 - sr**2 )
			chi = lim_phi[k] * np.pi / 180

			set_r = np.int( np.ceil(1.2 * lr) )

			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r + 1), pre_cut_img.shape[1] ] )
			lb0 = np.max( [np.int(yc - set_r), 0] )
			lb1 = np.min( [np.int(yc + set_r + 1), pre_cut_img.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc) * np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc) * np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc) * np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc) * np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where( jx == True )
			iv = np.ones( ( jx.shape[0], jx.shape[1] ), dtype = np.float32 )

			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

		mask_img = mask_path * pre_mask_img


		#. 'put mask img in the putput array'
		pre_cx = np.int( pre_cut_cx )
		pre_cy = np.int( pre_cut_cy )

		xn, yn = dL + 1, dL + 1

		pa0 = np.int( xn - pre_cx )
		pa1 = np.int( xn - pre_cx + pre_cut_img.shape[1] )

		pb0 = np.int( yn - pre_cy )
		pb1 = np.int( yn - pre_cy + pre_cut_img.shape[0] )

		cut_img[ pb0 : pb1, pa0 : pa1 ] = mask_img + 0.


		_cx_off = pre_cut_cx - np.int( pre_cut_cx )
		_cy_off = pre_cut_cy - np.int( pre_cut_cy )

		cc_px, cc_py = xn + _cx_off, yn + _cy_off

		tmp_cut_x = np.r_[ tmp_cut_x, cc_px ]
		tmp_cut_y = np.r_[ tmp_cut_y, cc_py ]


		##.. peak position
		_pkx, _pky = peak_x[ id_order ], peak_y[ id_order ]

		devi_x = _pkx - kk_px
		devi_y = _pky - kk_py

		cc_pkx, cc_pky = cc_px + devi_x, cc_py + devi_y


		kk_Nx, kk_Ny = cut_img.shape[1], cut_img.shape[0]

		#. save fits files
		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'PEAK_X', 'PEAK_Y', 
				'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','BCG_Z', 'P_SCALE' ]
		value = ['T', 32, 2, kk_Nx, kk_Ny, cc_px, cc_py, cc_pkx, cc_pky, kk_ra, kk_dec, bcg_ra, bcg_dec, bcg_z, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto( out_mask_file % (band, bcg_ra, bcg_dec, bcg_z, kk_ra, kk_dec ), cut_img, header = fil, overwrite = True)

	##...
	if stack_info is not None:

		_pp_bcg_ra = np.ones( N_sat,) * bcg_ra
		_pp_bcg_dec = np.ones( N_sat,) * bcg_dec
		_pp_bcg_z = np.ones( N_sat,) * bcg_z

		keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'ori_imgx', 'ori_imgy', 'cut_cx', 'cut_cy' ]
		values = [ _pp_bcg_ra, _pp_bcg_dec, _pp_bcg_z, ra_set, dec_set, tmp_ini_x, tmp_ini_y, tmp_cut_x, tmp_cut_y ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( stack_info % ( band, ra_g, dec_g, z_g ),)

	return

