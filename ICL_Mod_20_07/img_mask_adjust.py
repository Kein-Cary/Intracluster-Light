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

from groups import groups_find_func

def adjust_mask_func(d_file, cat_file, z_set, ra_set, dec_set, band, gal_file, out_file, bcg_mask, extra_cat, alter_fac, 
	stack_info = None, pixel = 0.396,):
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
	extra_cat : extral catalog for masking adjust
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
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		bcg_x.append(xn)
		bcg_y.append(yn)

		source = asc.read(gal_file % (band, ra_g, dec_g, z_g), )
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1
		p_type = np.array(source['CLASS_STAR'])

		Kron = 16
		a = Kron * A
		b = Kron * B
		'''
		##### selected gal_cat source
		gal_dat = pds.read_csv(extra_cat % (ra_g, dec_g, z_g), )
		gcat_x, gcat_y = np.array(gal_dat.imgx), np.array(gal_dat.imgy)

		gcat_a = np.ones( len(gcat_x) ) * np.nanmean( a ) * 7.0
		gcat_b = np.ones( len(gcat_x) ) * np.nanmean( a ) * 7.0 # use a circle to mask
		gcat_chi = np.ones( len(gcat_x) ) * 0.
		'''
		## stars
		mask = cat_file % (z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		flags = [str(qq) for qq in xt]

		x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)

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
		sub_A0 = lr_iso[ic] * alter_fac #30
		sub_B0 = sr_iso[ic] * alter_fac #30
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

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0] ]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0] ]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0] ]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0] ]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0] ]

		tot_cx = np.r_[cx, comx, ] #gcat_x]
		tot_cy = np.r_[cy, comy, ] #gcat_y]
		tot_a = np.r_[a, Lr, ] #gcat_a]
		tot_b = np.r_[b, Sr, ] #gcat_b]
		tot_theta = np.r_[theta, phi, ] #gcat_chi]
		tot_Numb = Numb + len(comx)# + len(gcat_x)

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

		## add back the BCG region
		if bcg_mask == 0:
			tdr = np.sqrt((xn - cx)**2 + (yn - cy)**2)
			idx = tdr == np.min(tdr)
			lr = A[idx] * 2.5 # 2.5 R_Kron

			set_r = np.int(np.ceil(1.0 * lr))
			la0 = np.max( [np.int(cx[idx] - set_r), 0])
			la1 = np.min( [np.int(cx[idx] + set_r +1), img.shape[1] ] )
			lb0 = np.max( [np.int(cy[idx] - set_r), 0] )
			lb1 = np.min( [np.int(cy[idx] + set_r +1), img.shape[0] ] )
			mask_img[lb0: lb1, la0: la1] = img[lb0: lb1, la0: la1]

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

def main():

	home = '/media/xkchen/My Passport/data/SDSS/'
	load = '/media/xkchen/My Passport/data/SDSS/'

	size_arr = np.array([5, 10, 15, 20, 25])
	for mm in range(5):
		## cluster
		#test_1000-to-250_cat-match.csv') A-type
		dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-98_cat.csv')
		set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

		cat_file = '/home/xkchen/mywork/ICL/data/star_dr12_reload/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
		gal_file = '/home/xkchen/mywork/ICL/data/source_find/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

		out_file = home + '20_10_test/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])

		bcg_mask = 1
		band = 'r'
		extra_cat = '/home/xkchen/mywork/ICL/data/source_find/clus_photo-G_match_ra%.3f_dec%.3f_z%.3f.csv'
		adjust_mask_func(d_file, cat_file, set_z, set_ra, set_dec, band, gal_file, out_file, bcg_mask, extra_cat, size_arr[mm],)
	raise

	"""
		## random
		#with h5py.File(load + 'random_cat/cat_select/rand_r_band_catalog.h5', 'r') as f:
		#	tmp_array = np.array(f['a'])
		#ra, dec, z = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2])

		dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_cat.csv')
		set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
		cat_file = home + 'random_cat/star_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
		gal_file = '/home/xkchen/mywork/ICL/data/source_find/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

		#out_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
		out_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_add-photo-G.fits'

		bcg_mask = 1
		band = 'r'
		stack_info = 'random_clus_1000-match_bcg-pos.csv'
		extra_cat = '/home/xkchen/mywork/ICL/data/source_find/photo-G_match_ra%.3f_dec%.3f_z%.3f.csv'
		adjust_mask_func(d_file, cat_file, set_z, set_ra, set_dec, band, gal_file, out_file, bcg_mask, extra_cat, stack_info,)
	"""

if __name__ == "__main__":
	main()

