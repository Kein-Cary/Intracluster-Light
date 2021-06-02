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

from img_pre_selection import WCS_to_pixel_func, pixel_to_WCS_func
from groups import groups_find_func

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

def mask_with_BCG( img_file, cen_x, cen_y, cen_ar, cen_br, cen_cr, cen_chi, gal_arr, bcg_R_eff,):
	## cen_x, cen_y : BCG location in image frame

	data = fits.open( img_file )
	img = data[0].data

	cx, cy, a, b, theta = gal_arr[:]

	ef1 = ( cx - cen_x ) * np.cos( cen_chi ) + ( cy - cen_y ) * np.sin( cen_chi )
	ef2 = ( cy - cen_y ) * np.cos( cen_chi ) - ( cx - cen_x ) * np.sin( cen_chi )
	er = ef1**2 / cen_ar**2 + ef2**2 / cen_br**2
	idx = er < 1

	# tdr = np.sqrt( (cen_x - cx)**2 + (cen_y - cy)**2)
	# idx = tdr <= bcg_R_eff

	if np.sum( idx ) >= 1:

		id_bcg = np.where( idx == True )[0]

	if np.sum( idx ) == 0:
		id_bcg = np.array( [] )

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	Numb = len( major )

	mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))

	# masking 'galaxies'
	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		if k in id_bcg:
			# print('no mask')
			continue

		else:
			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img

	return mask_img

def adjust_mask_func( d_file, cat_file, z_set, ra_set, dec_set, band, gal_file, out_file, bcg_mask,
	offset_file = None, bcg_photo_file = None, extra_cat = None, extra_img = None, alter_fac = None, alt_bright_R = None,
	alt_G_size = None, stack_info = None, pixel = 0.396):
	"""
	after img masking, use this function to detection "light" region, which
	mainly due to nearby brightstars, for SDSS case: taking the brightness of
	img center region as a normal brightness (mu, with scatter sigma), and rule out all the sub-patches
	whose mean pixel flux is larger than mu + 3.5 * sigma
	------------
	d_file : path where image data saved (include file-name structure:
	'/xxx/xxx/xxx.xxx')
	cat_file : path where photometric data saved, the same structure as d_file
	gal_file : the source catalog based on SExTractor calculation
	set_ra, set_dec, set_z : ra, dec, z of will be masked imgs
	band: band of image data, 'str' type
	out_file : save the masking data
	bcg_mask : 0 : keep BCGs; 1 : BCGs will be masked
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	stack_info : path to save the information of stacking (ra, dec, z, img_x, img_y)
	including file-name: '/xxx/xxx/xxx.xxx'

	extra_cat : extral galaxy catalog for masking adjust, (list type, .cat files)
	alter_fac : size adjust for normal stars
	alt_bright_R : size adjust for bright stars (also for saturated sources)
	alt_G_size : size adjust for galaxy-like sources

	bcg_photo_file : files including BCG properties (effective radius,), .txt files,
		[default is None, for radnom img case, always set masking for BCGs]
	"""

	Nz = len(z_set)
	bcg_x, bcg_y = [], []

	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.open(file)
		img = data[0].data
		head = data[0].header

		wcs_lis = awc.WCS(head)
		xn, yn = WCS_to_pixel_func( ra_g, dec_g, head) ## SDSS EDR paper relation

		if offset_file is not None:	
			off_dat = pds.read_csv( offset_file % (band, ra_g, dec_g, z_g), )

			x2pk_off_arr = np.array( off_dat[ 'devi_pk_x' ] )
			y2pk_off_arr = np.array( off_dat[ 'devi_pk_y' ] )

			medi_x2pk_off = np.median( x2pk_off_arr )
			medi_y2pk_off = np.median( y2pk_off_arr )
		else:
			medi_x2pk_off = 0.
			medi_y2pk_off = 0.

		xn = xn + medi_x2pk_off
		yn = yn + medi_y2pk_off

		bcg_x.append( xn )
		bcg_y.append( yn )

		## galaxy location in targ_filter
		source = asc.read( gal_file % (band, ra_g, dec_g, z_g), )
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])
		p_type = np.array(source['CLASS_STAR'])

		if alt_G_size is not None:
			Kron = alt_G_size * 2
		else:
			Kron = 16 # 8-R_kron

		a = Kron * A
		b = Kron * B

		d_cen_R = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )
		id_xcen = d_cen_R == d_cen_R.min()
		
		cen_R = 1.5 ## 1.5 R_kron
		cen_ar = A[ id_xcen ] * cen_R
		cen_br = B[ id_xcen ] * cen_R
		cen_cr = np.sqrt( cen_ar**2 - cen_br**2 )
		cen_chi = theta[ id_xcen ] * np.pi / 180

		## extral catalog load
		if extra_cat is not None:
			Ecat_num, Ecat_x, Ecat_y, Ecat_a, Ecat_b, Ecat_chi = cat_combine( extra_cat, ra_g, dec_g, z_g, alt_G_size, head, extra_img)
			Ecat_x, Ecat_y = Ecat_x + medi_x2pk_off, Ecat_y + medi_y2pk_off
		else:
			Ecat_num = 0
			Ecat_x, Ecat_y, Ecat_a, Ecat_b, Ecat_chi = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

		## stars
		mask = cat_file % ( z_g, ra_g, dec_g )
		cat = pds.read_csv( mask, skiprows = 1 )
		set_ra = np.array( cat['ra'] )
		set_dec = np.array( cat['dec'] )
		set_mag = np.array( cat['r'] )
		OBJ = np.array( cat['type'] )
		xt = cat['Column1']
		flags = [str(qq) for qq in xt]

		x, y = WCS_to_pixel_func( set_ra, set_dec, head)
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

		tot_cx = np.r_[cx, comx, Ecat_x]
		tot_cy = np.r_[cy, comy, Ecat_y]
		tot_a = np.r_[a, Lr, Ecat_a]
		tot_b = np.r_[b, Sr, Ecat_b]
		tot_theta = np.r_[theta, phi, Ecat_chi]
		tot_Numb = Numb + N_star + Ecat_num

		# masking part
		if bcg_mask == 1:

			mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
			ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
			oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
			basic_coord = np.array(np.meshgrid(ox, oy))
			major = tot_a / 2
			minor = tot_b / 2
			senior = np.sqrt(major**2 - minor**2)

			for k in range(tot_Numb):
				xc = tot_cx[k]
				yc = tot_cy[k]

				lr = major[k]
				sr = minor[k]
				cr = senior[k]
				chi = tot_theta[k] * np.pi/180

				set_r = np.int(np.ceil(1.2 * lr))
				la0 = np.max( [np.int(xc - set_r), 0])
				la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
				lb0 = np.max( [np.int(yc - set_r), 0] ) 
				lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

				df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
				df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
				fr = df1**2 / lr**2 + df2**2 / sr**2
				jx = fr <= 1

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
				iv[iu] = np.nan
				mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

			mask_img = mask_path * img

			hdu = fits.PrimaryHDU()
			hdu.data = mask_img
			hdu.header = head
			hdu.writeto(out_file % (band, ra_g, dec_g, z_g), overwrite = True)

		### add BCG region back
		if bcg_mask == 0:

			img_file = d_file % (band, ra_g, dec_g, z_g)

			gal_x = np.r_[ cx, Ecat_x ]
			gal_y = np.r_[ cy, Ecat_y ]
			gal_a = np.r_[ a, Ecat_a ]
			gal_b = np.r_[ b, Ecat_b ]
			gal_chi = np.r_[ theta, Ecat_chi ]

			BCG_photo_cat = pds.read_csv( bcg_photo_file % (z_g, ra_g, dec_g), skiprows = 1)
			## effective radius, in unit of arcsec
			r_Reff = np.array( BCG_photo_cat['deVRad_r'] )[0]
			g_Reff = np.array( BCG_photo_cat['deVRad_g'] )[0]
			i_Reff = np.array( BCG_photo_cat['deVRad_i'] )[0]

			if band == 'r':
				bcg_R_eff = r_Reff / pixel
			if band == 'g':
				bcg_R_eff = g_Reff / pixel
			if band == 'i':
				bcg_R_eff = i_Reff / pixel

			gal_arr = [ gal_x, gal_y, gal_a, gal_b, gal_chi ]

			pre_mask_img = mask_with_BCG( img_file, xn, yn, cen_ar, cen_br, cen_cr, cen_chi, gal_arr, bcg_R_eff,)

			mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
			ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
			oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
			basic_coord = np.array(np.meshgrid(ox, oy))

			# masking stars
			for k in range( N_star ):
				xc = comx[k]
				yc = comy[k]

				lr = Lr[k] / 2
				sr = Sr[k] / 2
				cr = np.sqrt(lr**2 - sr**2)
				chi = phi[k] * np.pi / 180

				set_r = np.int(np.ceil(1.2 * lr))
				la0 = np.max( [np.int(xc - set_r), 0])
				la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
				lb0 = np.max( [np.int(yc - set_r), 0] ) 
				lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

				df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
				df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
				fr = df1**2 / lr**2 + df2**2 / sr**2
				jx = fr <= 1

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
				iv[iu] = np.nan
				mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

			mask_img = mask_path * pre_mask_img

			hdu = fits.PrimaryHDU()
			hdu.data = mask_img
			hdu.header = head
			hdu.writeto(out_file % (band, ra_g, dec_g, z_g), overwrite = True)

	bcg_x = np.array(bcg_x)
	bcg_y = np.array(bcg_y)

	if stack_info != None:
		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ra_set, dec_set, z_set, bcg_x, bcg_y]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv(stack_info)

	return

