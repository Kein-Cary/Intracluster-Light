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

def source_detect_func(d_file, z_set, ra_set, dec_set, band, out_file, stack_info = None,):
	"""
	d_file : path where image data saved (include file-name structure:
	'/xxx/xxx/xxx.xxx')
	cat_file : path where photometric data saved, the same structure as d_file
	set_ra, set_dec, set_z : ra, dec, z of will be masked imgs
	band: band of image data, 'str' type
	out_file : save sources information (detected by SExTractor)
	"""
	Nz = len(z_set)
	#param_A = 'default_mask_A.sex'
	param_A = 'default_mask_A_g.sex'
	out_param = 'default_mask_A.param'
	bcg_x, bcg_y = [], []

	for q in range( Nz ):

		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.open(file)
		img = data[0].data
		head = data[0].header

		wcs_lis = awc.WCS( head )
		# xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		xn, yn = WCS_to_pixel_func( ra_g, dec_g, head )

		bcg_x.append( xn )
		bcg_y.append( yn )

		hdu = fits.PrimaryHDU()
		hdu.data = img
		hdu.header = head
		hdu.writeto('/home/xkchen/project/tmp/img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g), overwrite = True)

		out_cat = out_file % (band, ra_g, dec_g, z_g)

		file_source = '/home/xkchen/project/tmp/img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)

		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_cat, out_param)
		a = subpro.Popen(cmd, shell = True)
		a.wait()

		continue

	if stack_info != None:
		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ra_set, dec_set, z_set, bcg_x, bcg_y]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv(stack_info)

	return

def mask_with_BCG( img_file, cen_x, cen_y, cen_ar, cen_br, cen_cr, cen_chi, gal_arr ):
	## cen_x, cen_y : BCG location in image frame
	data = fits.open( img_file )
	img = data[0].data

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

def mask_func(d_file, cat_file, z_set, ra_set, dec_set, band, out_file0, out_file1, bcg_mask, stack_info = None, pixel = 0.396, source_det = False):
	"""
	d_file : path where image data saved (include file-name structure:
	'/xxx/xxx/xxx.xxx')
	cat_file : path where photometric data saved, the same structure as d_file
	set_ra, set_dec, set_z : ra, dec, z of will be masked imgs
	band: band of image data, 'str' type

	out_file0 : save sources information (mainly for galaxies)
	out_file1 : save the masking data

	bcg_mask : 0 -- mask all sources except BCGs; 1 : BCGs also will be masked
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	stack_info : path to save the information of stacking (ra, dec, z, img_x, img_y)
					including file-name: '/xxx/xxx/xxx.xxx'

	"""
	Nz = len(z_set)
	#param_A = 'default_mask_A.sex'
	param_A = 'default_mask_A_g.sex'
	out_param = 'default_mask_A.param'
	bcg_x, bcg_y = [], []

	## source detection
	if source_det == True:
		source_detect_func(d_file, z_set, ra_set, dec_set, band, out_file0, stack_info,)

	## masking
	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.open(file)
		img = data[0].data
		head = data[0].header

		wcs_lis = awc.WCS(head)
		# xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		xn, yn = WCS_to_pixel_func( ra_g, dec_g, head )

		bcg_x.append( xn )
		bcg_y.append( yn )

		source = asc.read(out_file0 % (band, ra_g, dec_g, z_g), )
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])
		p_type = np.array(source['CLASS_STAR'])

		Kron = 16
		a = Kron * A
		b = Kron * B

		##.. find BCG source
		d_cen_R = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )
		id_xcen = d_cen_R == d_cen_R.min()
		
		cen_R = 1.5 ## 1.5 R_kron
		cen_ar = A[ id_xcen ] * cen_R
		cen_br = B[ id_xcen ] * cen_R
		cen_cr = np.sqrt( cen_ar**2 - cen_br**2 )
		cen_chi = theta[ id_xcen ] * np.pi / 180

		## stars
		mask = cat_file % (z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		flags = [str(qq) for qq in xt]

		# x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
		x, y = WCS_to_pixel_func( set_ra, set_dec, head )

		set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i'] ]) / pixel
		set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i'] ]) / pixel
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
		sub_A2 = lr_iso[ipx] * 75
		sub_B2 = sr_iso[ipx] * 75
		sub_chi2 = set_chi[ipx]

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]
		N_star = len(comx)

		tot_cx = np.r_[cx, comx]
		tot_cy = np.r_[cy, comy]
		tot_a = np.r_[a, Lr]
		tot_b = np.r_[b, Sr]
		tot_theta = np.r_[theta, phi]
		tot_Numb = Numb + N_star

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
			hdu.writeto(out_file1 % (band, ra_g, dec_g, z_g), overwrite = True)

		### add BCG region back
		if bcg_mask == 0:

			gal_arr = [ cx, cy, a, b, theta ]

			pre_mask_img = mask_with_BCG( img_file, xn, yn, cen_ar, cen_br, cen_cr, cen_chi, gal_arr )

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
				chi = phi[k] * np.pi/180

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
			hdu.writeto(out_file1 % (band, ra_g, dec_g, z_g), overwrite = True)

	bcg_x = np.array( bcg_x )
	bcg_y = np.array( bcg_y )

	if stack_info != None:
		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ra_set, dec_set, z_set, bcg_x, bcg_y]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv(stack_info)

	return
