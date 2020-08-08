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

def adjust_mask_func(d_file, cat_file, z_set, ra_set, dec_set, band, out_file0, out_file1, bcg_mask, stack_info = None, pixel = 0.396,):
	"""
	after img masking, use this function to detection "light" region, which
	mainly due to nearby brightstars, for SDSS case: taking the brightness of
	img center region as a normal brightness (mu, with scatter sigma), and rule out all the sub-patches
	whose mean pixel flux is larger than mu + 3.5 * sigma
	------------
	d_file : path where image data saved (include file-name structure:
	'/xxx/xxx/xxx.xxx')
	cat_file : path where photometric data saved, the same structure as d_file
	set_ra, set_dec, set_z : ra, dec, z of will be masked imgs
	band: band of image data, 'str' type
	out_file0 : save sources information
	out_file1 : save the masking data
	bcg_mask : 0 : keep BCGs; 1 : BCGs will be masked
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	stack_info : path to save the information of stacking (ra, dec, z, img_x, img_y)
	including file-name: '/xxx/xxx/xxx.xxx'
	"""
	Nz = len(z_set)
	param_A = 'default_mask_A.sex'
	out_param = 'default_mask_A.param'
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

		hdu = fits.PrimaryHDU()
		hdu.data = img
		hdu.header = head
		hdu.writeto('tmp.fits', overwrite = True)

		out_cat = out_file0 % (band, ra_g, dec_g, z_g)
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

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0] ]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0] ]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0] ]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0] ]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0] ]

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
		hdu.writeto(out_file1 % (band, ra_g, dec_g, z_g), overwrite = True)

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

	## cluster
	#with h5py.File(load + 'mpi_h5/r_band_sky_catalog.h5', 'r') as f:
	#	set_array = np.array(f['a'])
	#ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]

	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/clust-1000-select_cat.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	cat_file = '/home/xkchen/mywork/ICL/data/star_dr12_reload/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
	#cat_file = '/home/xkchen/mywork/ICL/data/tmp_img/tmp_stars/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' ### with larger query region

	out_file0 = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	out_file1 = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	bcg_mask = 1
	band = 'r'
	stack_info = 'clust-1000-select_cat_bcg-pos.csv'
	adjust_mask_func(d_file, cat_file, set_z, set_ra, set_dec, band, out_file0, out_file1, bcg_mask, stack_info)


	## random
	#with h5py.File(load + 'random_cat/cat_select/rand_r_band_catalog.h5', 'r') as f:
	#	tmp_array = np.array(f['a'])
	#ra, dec, z = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2])

	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_cat.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	cat_file = home + 'random_cat/star_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
	out_file0 = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	out_file1 = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	bcg_mask = 1
	band = 'r'
	stack_info = 'random_clus_1000-match_bcg-pos.csv'
	adjust_mask_func(d_file, cat_file, set_z, set_ra, set_dec, band, out_file0, out_file1, bcg_mask, stack_info)

if __name__ == "__main__":
	main()

"""
#### delete part
		## grid the masked img and rule out those 'too bright' region
		N_step = 100
		ca0, ca1 = np.int(img.shape[0] / 2), np.int(img.shape[1] / 2)
		cen_D = 500
		flux_cen = mask_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

		cen_lx = np.arange(0, 2 * cen_D + 100, N_step)
		cen_ly = np.arange(0, 2 * cen_D + 100, N_step)

		sub_pock_pix = np.zeros( (len(cen_ly) - 1, len(cen_lx) - 1), dtype = np.float)
		sub_pock_flux = np.zeros( (len(cen_ly) - 1, len(cen_lx) - 1), dtype = np.float)
		for nn in range( len(cen_ly) - 1 ):
			for tt in range( len(cen_lx) - 1 ):
				sub_flux = flux_cen[ cen_ly[nn]: cen_ly[nn+1], cen_lx[tt]: cen_lx[tt+1] ]
				id_nn = np.isnan(sub_flux)
				sub_pock_flux[nn,tt] = np.nanmean(sub_flux)
				sub_pock_pix[nn,tt] = len(sub_flux[id_nn == False])

		id_Nzero = sub_pock_pix > 100
		mu = np.nanmean( sub_pock_flux[id_Nzero] )
		sigm = np.nanstd( sub_pock_flux[id_Nzero] )

		ly = np.arange(0, img.shape[0], N_step)
		ly = np.r_[ly, img.shape[0] - N_step, img.shape[0] ]
		lx = np.arange(0, img.shape[1], N_step)
		lx = np.r_[lx, img.shape[1] - N_step, img.shape[1] ]

		patch_mean = np.zeros( (len(ly) - 1, len(lx) - 1), dtype = np.float )
		patch_pix = np.zeros( (len(ly) - 1, len(lx) - 1), dtype = np.float )
		for nn in range( len(ly) - 1 ):
			for tt in range( len(lx) - 1 ):
				if nn == len(ly) - 3:
					nn += 1
				if tt == len(lx) - 3:
					tt += 1
				sub_flux = mask_img[ly[nn]: ly[nn + 1], lx[tt]: lx[tt+1]]
				id_nn = np.isnan(sub_flux)
				patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
				patch_pix[nn,tt] = len( sub_flux[id_nn == False] )

		id_zeros = patch_pix == 0.
		patch_pix[id_zeros] = np.nan
		patch_mean[id_zeros] = np.nan
		over_sb = (patch_mean - mu) / sigm

		## regulate the 'over_sb' and (ly,lx) for region selection
		over_sb = np.delete(over_sb, -2, axis = 0)
		over_sb = np.delete(over_sb, -2, axis = 1)
		lx = np.delete(lx, -3)
		lx = np.delete(lx, -1)
		ly = np.delete(ly, -3)
		ly = np.delete(ly, -1)

		## rule out regions with mean flux brighter then X-sigma of the centeral mean value,
		## and also those region need to at least include 5 (or more) subpatches
		lim_sb = 5.5
		copy_arr = over_sb.copy()
		idnn = np.isnan(over_sb)
		copy_arr[idnn] = 100
		source_n, coord_x, coord_y = groups_find_func(copy_arr, lim_sb)

		lo_xs = lx[ [np.min( ll ) for ll in coord_x] ]
		hi_xs = lx[ [np.max( ll ) for ll in coord_x] ]
		lo_ys = ly[ [np.min( ll ) for ll in coord_y] ]
		hi_ys = ly[ [np.max( ll ) for ll in coord_y] ]

		idux = (lo_xs <= 500) | (2000 - hi_xs <= 500)
		iduy = (lo_ys <= 500) | (1400 - hi_ys <= 500)
		idu = idux | iduy

		idv = np.array(source_n) >= 5

		id_pat = idu & idv
		id_True = np.where(id_pat == True)[0]

		for qq in range( len(id_True) ):

			sub_edgx = np.array( lx[ coord_x[ id_True[qq] ] ] )
			sub_edgy = np.array( ly[ coord_y[ id_True[qq] ] ] )

			for mm in range( source_n[ id_True[qq] ] ):

				a0, a1 = sub_edgx[mm], sub_edgx[mm] + N_step
				b0, b1 = sub_edgy[mm], sub_edgy[mm] + N_step

				mask_img[b0: b1, a0: a1] = np.nan

		'''
		id_mean = over_sb >= 5.5
		id_pat = id_mean

		if np.sum(id_pat) > 0:
			idv = np.where(id_pat == True)
			for tt in range( np.sum(id_pat) ):

				xi, yi = lx[ idv[1][tt] ], ly[ idv[0][tt] ]

				da0, da1 = xi, 2000 - xi
				db0, db1 = yi, 1400 - yi

				ida = (da0 <= 400) | (da1 <= 500)
				idb = (db0 <= 400) | (db1 <= 500)
				if (ida | idb):
					mask_img[yi: yi + N_step, xi: xi + N_step] = np.nan
		'''
"""
