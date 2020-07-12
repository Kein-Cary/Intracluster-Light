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

def mask_func(d_file, cat_file, z_set, ra_set, dec_set, band, out_file0, out_file1, bcg_mask, pixel = 0.396):
	"""
	d_file : path where image data saved (include file-name structure:
	'/xxx/xxx/xxx.xxx')
	cat_file : path where photometric data saved, the same structure as d_file
	set_ra, set_dec, set_z : ra, dec, z of will be masked imgs
	band: band of image data, 'str' type
	out_file0 : save sources information
	out_file1 : save the masking data
	bcg_mask : 0 : keep BCGs; 1 : BCGs will be masked
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	"""
	Nz = len(z_set)
	param_A = 'default_mask_A.sex'
	out_param = 'default_mask_A.param'

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

		hdu = fits.PrimaryHDU()
		hdu.data = img
		hdu.header = head
		hdu.writeto('tmp.fits', overwrite = True)

		out_cat = out_file0 + '%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band, ra_g, dec_g, z_g,)
		file_source = 'tmp.fits'
		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_cat, out_param)
		a = subpro.Popen(cmd, shell = True)
		a.wait()

		source = asc.read(out_cat)
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

		## stars
		mask = cat_file % (z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
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
		sub_A0 = lr_iso[ic] * 30
		sub_B0 = sr_iso[ic] * 30
		sub_chi0 = set_chi[ic]

		# saturated source(may not stars)
		xa = ['SATURATED' in qq for qq in xt]
		xv = np.array(xa)
		idx = xv == True
		ipx = (idx)

		sub_x2 = x[ipx]
		sub_y2 = y[ipx]
		sub_A2 = lr_iso[ipx] * 60
		sub_B2 = sr_iso[ipx] * 60
		sub_chi2 = set_chi[ipx]

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

		tot_cx = np.r_[cx, comx]
		tot_cy = np.r_[cy, comy]
		tot_a = np.r_[a, Lr]
		tot_b = np.r_[b, Sr]
		tot_theta = np.r_[theta, phi]
		tot_Numb = Numb + len(comx)

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

		if bcg_mask == 0:
			## add back the BCG region
			tdr = np.sqrt((xn - cx)**2 + (yn - cy)**2)
			idx = tdr == np.min(tdr)
			lr = A[idx] * 2.5

			set_r = np.int(np.ceil(1.0 * lr))
			la0 = np.max( [np.int(cx[idx] - set_r), 0])
			la1 = np.min( [np.int(cx[idx] + set_r +1), img.shape[1] ] )
			lb0 = np.max( [np.int(cy[idx] - set_r), 0] )
			lb1 = np.min( [np.int(cy[idx] + set_r +1), img.shape[0] ] )
			mask_img[lb0: lb1, la0: la1] = img[lb0: lb1, la0: la1]

		hdu = fits.PrimaryHDU()
		hdu.data = mask_img
		hdu.header = head
		hdu.writeto(out_file1 + 'mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band, ra_g, dec_g, z_g), overwrite = True)

	return

def main():

	dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
	ra, dec, z = dat.ra, dat.dec, dat.z
	Nz = 10
	set_ra, set_dec, set_z = ra[:10], dec[:10], z[:10]
	d_file = '/home/xkchen/mywork/ICL/data/sdss_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	cat_file = '/home/xkchen/mywork/ICL/data/star_dr12_reload/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
	out_file0 = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/'
	out_file1 = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/'
	bcg_mask = 0
	band = 'r'
	mask_func(d_file, cat_file, set_z, set_ra, set_dec, band, out_file0, out_file1, bcg_mask)

if __name__ == "__main__":
	main()
